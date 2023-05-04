//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \file 01_LBM_PSM_CPU.cpp
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"
#include "blockforest/communication/UniformBufferedScheme.h"

#include "boundary/all.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/logging/all.h"
#include "core/math/all.h"
#include "core/timing/RemainingTimeLogger.h"

#include "domain_decomposition/SharedSweep.h"

#include "field/AddToStorage.h"
#include "field/communication/PackInfo.h"

#include "lbm/boundary/all.h"
#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/field/PdfField.h"
#include "lbm/lattice_model/D3Q19.h"
#include "lbm/vtk/all.h"

#include "lbm_mesapd_coupling/DataTypes.h"
#include "lbm_mesapd_coupling/mapping/ParticleMapping.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/PSMSweep.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/PSMUtility.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/ParticleAndVolumeFractionMapping.h"
#include "lbm_mesapd_coupling/utility/AddForceOnParticlesKernel.h"
#include "lbm_mesapd_coupling/utility/AddHydrodynamicInteractionKernel.h"
#include "lbm_mesapd_coupling/utility/LubricationCorrectionKernel.h"
#include "lbm_mesapd_coupling/utility/ParticleSelector.h"
#include "lbm_mesapd_coupling/utility/ResetHydrodynamicForceTorqueKernel.h"

#include "mesa_pd/collision_detection/AnalyticContactDetection.h"
#include "mesa_pd/data/DataTypes.h"
#include "mesa_pd/data/ParticleAccessorWithShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/ShapeStorage.h"
#include "mesa_pd/data/shape/HalfSpace.h"
#include "mesa_pd/data/shape/Sphere.h"
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/kernel/DoubleCast.h"
#include "mesa_pd/kernel/LinearSpringDashpot.h"
#include "mesa_pd/kernel/ParticleSelector.h"
#include "mesa_pd/kernel/VelocityVerlet.h"
#include "mesa_pd/mpi/ContactFilter.h"
#include "mesa_pd/mpi/ReduceProperty.h"
#include "mesa_pd/mpi/SyncNextNeighbors.h"
#include "mesa_pd/mpi/notifications/ForceTorqueNotification.h"
#include "mesa_pd/mpi/notifications/HydrodynamicForceTorqueNotification.h"
#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "timeloop/SweepTimeloop.h"

#include "vtk/all.h"

#include <functional>

namespace settling_sphere
{

///////////
// USING //
///////////

using namespace walberla;
using walberla::uint_t;

using LatticeModel_T = lbm::D3Q19< lbm::collision_model::TRT >;

using Stencil_T  = LatticeModel_T::Stencil;
using PdfField_T = lbm::PdfField< LatticeModel_T >;

using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

const uint_t FieldGhostLayers = 1;

///////////
// FLAGS //
///////////

const FlagUID Fluid_Flag("fluid");
const FlagUID NoSlip_Flag("no slip");

/////////////////////////////////////
// BOUNDARY HANDLING CUSTOMIZATION //
/////////////////////////////////////
template< typename ParticleAccessor_T >
class MyBoundaryHandling
{
 public:
   using NoSlip_T = lbm::NoSlip< LatticeModel_T, flag_t >;
   using Type     = BoundaryHandling< FlagField_T, Stencil_T, NoSlip_T >;

   MyBoundaryHandling(const BlockDataID& flagFieldID, const BlockDataID& pdfFieldID,
                      const shared_ptr< ParticleAccessor_T >& ac)
      : flagFieldID_(flagFieldID), pdfFieldID_(pdfFieldID), ac_(ac)
   {}

   Type* operator()(IBlock* const block, const StructuredBlockStorage* const /*storage*/) const
   {
      WALBERLA_ASSERT_NOT_NULLPTR(block);

      auto* flagField = block->getData< FlagField_T >(flagFieldID_);
      auto* pdfField  = block->getData< PdfField_T >(pdfFieldID_);

      const auto fluid =
         flagField->flagExists(Fluid_Flag) ? flagField->getFlag(Fluid_Flag) : flagField->registerFlag(Fluid_Flag);

      Type* handling =
         new Type("moving obstacle boundary handling", flagField, fluid, NoSlip_T("NoSlip", NoSlip_Flag, pdfField));

      handling->fillWithDomain(FieldGhostLayers);

      return handling;
   }

 private:
   const BlockDataID flagFieldID_;
   const BlockDataID pdfFieldID_;

   shared_ptr< ParticleAccessor_T > ac_;
};
//*******************************************************************************************************************

void createPlaneSetup(const shared_ptr< mesa_pd::data::ParticleStorage >& ps,
                      const shared_ptr< mesa_pd::data::ShapeStorage >& ss, const math::AABB& simulationDomain)
{
   // create bounding planes
   mesa_pd::data::Particle p0 = *ps->create(true);
   p0.setPosition(simulationDomain.minCorner());
   p0.setInteractionRadius(std::numeric_limits< real_t >::infinity());
   p0.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(0, 0, 1)));
   p0.setOwner(mpi::MPIManager::instance()->rank());
   p0.setType(0);
   mesa_pd::data::particle_flags::set(p0.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
   mesa_pd::data::particle_flags::set(p0.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

   mesa_pd::data::Particle p1 = *ps->create(true);
   p1.setPosition(simulationDomain.maxCorner());
   p1.setInteractionRadius(std::numeric_limits< real_t >::infinity());
   p1.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(0, 0, -1)));
   p1.setOwner(mpi::MPIManager::instance()->rank());
   p1.setType(0);
   mesa_pd::data::particle_flags::set(p1.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
   mesa_pd::data::particle_flags::set(p1.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

   mesa_pd::data::Particle p2 = *ps->create(true);
   p2.setPosition(simulationDomain.minCorner());
   p2.setInteractionRadius(std::numeric_limits< real_t >::infinity());
   p2.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(1, 0, 0)));
   p2.setOwner(mpi::MPIManager::instance()->rank());
   p2.setType(0);
   mesa_pd::data::particle_flags::set(p2.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
   mesa_pd::data::particle_flags::set(p2.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

   mesa_pd::data::Particle p3 = *ps->create(true);
   p3.setPosition(simulationDomain.maxCorner());
   p3.setInteractionRadius(std::numeric_limits< real_t >::infinity());
   p3.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(-1, 0, 0)));
   p3.setOwner(mpi::MPIManager::instance()->rank());
   p3.setType(0);
   mesa_pd::data::particle_flags::set(p3.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
   mesa_pd::data::particle_flags::set(p3.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

   mesa_pd::data::Particle p4 = *ps->create(true);
   p4.setPosition(simulationDomain.minCorner());
   p4.setInteractionRadius(std::numeric_limits< real_t >::infinity());
   p4.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(0, 1, 0)));
   p4.setOwner(mpi::MPIManager::instance()->rank());
   p4.setType(0);
   mesa_pd::data::particle_flags::set(p4.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
   mesa_pd::data::particle_flags::set(p4.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

   mesa_pd::data::Particle p5 = *ps->create(true);
   p5.setPosition(simulationDomain.maxCorner());
   p5.setInteractionRadius(std::numeric_limits< real_t >::infinity());
   p5.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(0, -1, 0)));
   p5.setOwner(mpi::MPIManager::instance()->rank());
   p5.setType(0);
   mesa_pd::data::particle_flags::set(p5.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
   mesa_pd::data::particle_flags::set(p5.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);
}

//////////
// MAIN //
//////////

//*******************************************************************************************************************
/*!\brief Simple tutorial application that simulates the settling of a sphere inside a rectangular column filled with
 * viscous fluid */
//*******************************************************************************************************************

int main(int argc, char** argv)
{
   mpi::Environment env(argc, argv);

   ///////////////////
   // Customization //
   ///////////////////

   std::string baseFolder = "vtk_out_SettlingSphere";
   uint_t vtkIOFreq       = 500;

   // numerical parameters
   uint_t numberOfCellsInHorizontalDirection = uint_t(15);
   uint_t numRPDSubCycles                    = uint_t(10);

   //////////////////////////////////////
   // SIMULATION PROPERTIES in SI units//
   //////////////////////////////////////

   // values are mainly taken from the reference paper
   const real_t diameter_SI      = real_t(7.5e-3);
   const real_t densitySphere_SI = real_t(1120);

   real_t densityFluid_SI;
   real_t dynamicViscosityFluid_SI;
   real_t expectedSettlingVelocity_SI;

   // Re_p around 1.5
   densityFluid_SI             = real_t(970);
   dynamicViscosityFluid_SI    = real_t(373e-3);
   expectedSettlingVelocity_SI = real_t(0.035986);

   const real_t kinematicViscosityFluid_SI = dynamicViscosityFluid_SI / densityFluid_SI;

   const real_t gravitationalAcceleration_SI = real_t(9.81);
   Vector3< real_t > domainSize_SI(real_t(15e-3), real_t(15e-3), real_t(20e-3));
   // shift starting gap a bit upwards to match the reported (plotted) values
   const real_t startingGapSize_SI = real_t(10e-3) + real_t(0.25) * diameter_SI;

   //////////////////////////
   // NUMERICAL PARAMETERS //
   //////////////////////////

   const real_t dx_SI = domainSize_SI[0] / real_c(numberOfCellsInHorizontalDirection);
   const Vector3< uint_t > domainSize(uint_c(floor(domainSize_SI[0] / dx_SI + real_t(0.5))),
                                      uint_c(floor(domainSize_SI[1] / dx_SI + real_t(0.5))),
                                      uint_c(floor(domainSize_SI[2] / dx_SI + real_t(0.5))));
   const real_t diameter     = diameter_SI / dx_SI;
   const real_t sphereVolume = math::pi / real_t(6) * diameter * diameter * diameter;

   const real_t expectedSettlingVelocity = real_t(0.01);
   const real_t dt_SI                    = expectedSettlingVelocity / expectedSettlingVelocity_SI * dx_SI;

   const real_t viscosity      = kinematicViscosityFluid_SI * dt_SI / (dx_SI * dx_SI);
   const real_t relaxationTime = real_t(1) / lbm::collision_model::omegaFromViscosity(viscosity);

   const real_t gravitationalAcceleration = gravitationalAcceleration_SI * dt_SI * dt_SI / dx_SI;

   const real_t densityFluid  = real_t(1);
   const real_t densitySphere = densityFluid * densitySphere_SI / densityFluid_SI;

   const real_t dx = real_t(1);

   const uint_t timesteps = 250000;

   WALBERLA_LOG_INFO_ON_ROOT("Setup (in simulation, i.e. lattice, units):");
   WALBERLA_LOG_INFO_ON_ROOT(" - domain size = " << domainSize);
   WALBERLA_LOG_INFO_ON_ROOT(" - sphere: diameter = " << diameter << ", density = " << densitySphere);

   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   Vector3< uint_t > numberOfBlocksPerDirection(uint_t(1), uint_t(1), uint_t(4));
   Vector3< uint_t > cellsPerBlockPerDirection(domainSize[0] / numberOfBlocksPerDirection[0],
                                               domainSize[1] / numberOfBlocksPerDirection[1],
                                               domainSize[2] / numberOfBlocksPerDirection[2]);

   for (uint_t i = 0; i < 3; ++i)
   {
      WALBERLA_CHECK_EQUAL(cellsPerBlockPerDirection[i] * numberOfBlocksPerDirection[i], domainSize[i],
                           "Unmatching domain decomposition in direction " << i << "!");
   }

   auto blocks = blockforest::createUniformBlockGrid(numberOfBlocksPerDirection[0], numberOfBlocksPerDirection[1],
                                                     numberOfBlocksPerDirection[2], cellsPerBlockPerDirection[0],
                                                     cellsPerBlockPerDirection[1], cellsPerBlockPerDirection[2], dx, 0,
                                                     false, false, false, false, false, // periodicity
                                                     false);

   // write domain decomposition to file
   if (vtkIOFreq > 0) { vtk::writeDomainDecomposition(blocks, "initial_domain_decomposition", baseFolder); }

   //////////////////
   // RPD COUPLING //
   //////////////////

   auto rpdDomain = std::make_shared< mesa_pd::domain::BlockForestDomain >(blocks->getBlockForestPointer());

   // init data structures
   auto ps                  = walberla::make_shared< mesa_pd::data::ParticleStorage >(1);
   auto ss                  = walberla::make_shared< mesa_pd::data::ShapeStorage >();
   using ParticleAccessor_T = mesa_pd::data::ParticleAccessorWithShape;
   auto accessor            = walberla::make_shared< ParticleAccessor_T >(ps, ss);

   // bounding planes
   createPlaneSetup(ps, ss, blocks->getDomain());

   // create sphere and store Uid
   Vector3< real_t > initialPosition(real_t(0.5) * real_c(domainSize[0]), real_t(0.5) * real_c(domainSize[1]),
                                     startingGapSize_SI / dx_SI + real_t(0.5) * diameter);
   auto sphereShape = ss->create< mesa_pd::data::Sphere >(diameter * real_t(0.5));
   ss->shapes[sphereShape]->updateMassAndInertia(densitySphere);
   walberla::id_t sphereUid = 0;
   if (rpdDomain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), initialPosition))
   {
      mesa_pd::data::Particle&& p = *ps->create();
      p.setPosition(initialPosition);
      p.setInteractionRadius(diameter * real_t(0.5));
      p.setOwner(mpi::MPIManager::instance()->rank());
      p.setShapeID(sphereShape);
      p.setType(1);
      sphereUid = p.getUid();
   }
   mpi::allReduceInplace(sphereUid, mpi::SUM);

   ///////////////////////
   // ADD DATA TO BLOCKS //
   ////////////////////////

   // create the lattice model
   LatticeModel_T latticeModel =
      LatticeModel_T(lbm::collision_model::TRT::constructWithMagicNumber(real_t(1) / relaxationTime));

   // add PDF field
   BlockDataID pdfFieldID = lbm::addPdfFieldToStorage< LatticeModel_T >(
      blocks, "pdf field (fzyx)", latticeModel, Vector3< real_t >(real_t(0)), real_t(1), uint_t(1), field::fzyx);
   // add flag field
   BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");

   // add boundary handling
   using BoundaryHandling_T       = MyBoundaryHandling< ParticleAccessor_T >::Type;
   BlockDataID boundaryHandlingID = blocks->addStructuredBlockData< BoundaryHandling_T >(
      MyBoundaryHandling< ParticleAccessor_T >(flagFieldID, pdfFieldID, accessor), "boundary handling");

   // set up RPD functionality
   std::function< void(void) > syncCall = [ps, rpdDomain]() {
      mesa_pd::mpi::SyncNextNeighbors syncNextNeighborFunc;
      syncNextNeighborFunc(*ps, *rpdDomain);
   };

   syncCall();

   mesa_pd::kernel::VelocityVerletPreForceUpdate vvIntegratorPreForce(real_t(1) / real_t(numRPDSubCycles));
   mesa_pd::kernel::VelocityVerletPostForceUpdate vvIntegratorPostForce(real_t(1) / real_t(numRPDSubCycles));

   mesa_pd::kernel::LinearSpringDashpot collisionResponse(2);
   collisionResponse.setStiffnessAndDamping(0, 1, real_t(0.97), real_t(60), real_t(0), densitySphere * sphereVolume);
   mesa_pd::mpi::ReduceProperty reduceProperty;

   // set up coupling functionality
   lbm_mesapd_coupling::RegularParticlesSelector sphereSelector;
   Vector3< real_t > gravitationalForce(real_t(0), real_t(0),
                                        -(densitySphere - densityFluid) * gravitationalAcceleration * sphereVolume);
   lbm_mesapd_coupling::AddForceOnParticlesKernel addGravitationalForce(gravitationalForce);
   lbm_mesapd_coupling::AddHydrodynamicInteractionKernel addHydrodynamicInteraction;
   lbm_mesapd_coupling::ResetHydrodynamicForceTorqueKernel resetHydrodynamicForceTorque;
   lbm_mesapd_coupling::LubricationCorrectionKernel lubricationCorrectionKernel(
      viscosity, [](real_t r) { return real_t(0.0016) * r; });
   lbm_mesapd_coupling::ParticleMappingKernel< BoundaryHandling_T > particleMappingKernel(blocks, boundaryHandlingID);

   ///////////////
   // TIME LOOP //
   ///////////////

   // map planes into the LBM simulation -> act as no-slip boundaries
   ps->forEachParticle(false, lbm_mesapd_coupling::GlobalParticlesSelector(), *accessor, particleMappingKernel,
                       *accessor, NoSlip_Flag);

   // map particles into the LBM simulation
   BlockDataID particleAndVolumeFractionFieldID =
      field::addToStorage< lbm_mesapd_coupling::psm::ParticleAndVolumeFractionField_T >(
         blocks, "particle and volume fraction field",
         std::vector< lbm_mesapd_coupling::psm::ParticleAndVolumeFraction_T >(), field::fzyx, 0);
   lbm_mesapd_coupling::psm::ParticleAndVolumeFractionMapping particleMapping(blocks, accessor, sphereSelector,
                                                                              particleAndVolumeFractionFieldID, 4);
   particleMapping();

   lbm_mesapd_coupling::psm::initializeDomainForPSM< LatticeModel_T, 1 >(*blocks, pdfFieldID,
                                                                         particleAndVolumeFractionFieldID, *accessor);

   // setup of the LBM communication for synchronizing the pdf field between neighboring blocks
   std::function< void() > commFunction;
   blockforest::communication::UniformBufferedScheme< Stencil_T > scheme(blocks);
   scheme.addPackInfo(make_shared< lbm::PdfFieldPackInfo< LatticeModel_T > >(pdfFieldID));
   commFunction = scheme;

   // create the timeloop
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   auto bhSweep  = BoundaryHandling_T::getBlockSweep(boundaryHandlingID);
   auto lbmSweep = lbm_mesapd_coupling::psm::makePSMSweep< LatticeModel_T, FlagField_T, 1, 1 >(
      pdfFieldID, particleAndVolumeFractionFieldID, blocks, accessor, flagFieldID, Fluid_Flag);

   timeloop.addFuncBeforeTimeStep(RemainingTimeLogger(timeloop.getNrOfTimeSteps()), "Remaining Time Logger");

   // vtk output
   if (vtkIOFreq != uint_t(0))
   {
      // spheres
      auto particleVtkOutput = make_shared< mesa_pd::vtk::ParticleVtkOutput >(ps);
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleInteractionRadius >("radius");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleLinearVelocity >("velocity");
      particleVtkOutput->setParticleSelector([sphereShape](const mesa_pd::data::ParticleStorage::iterator& pIt) {
         return pIt->getShapeID() == sphereShape &&
                !(mesa_pd::data::particle_flags::isSet(pIt->getFlags(), mesa_pd::data::particle_flags::GHOST));
      });

      auto particleVtkWriter =
         vtk::createVTKOutput_PointData(particleVtkOutput, "Particles", vtkIOFreq, baseFolder, "simulation_step");
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(particleVtkWriter), "VTK (sphere data)");

      // pdf field
      auto pdfFieldVTK = vtk::createVTKOutput_BlockData(blocks, "fluid_field", vtkIOFreq, 0, false, baseFolder);

      blockforest::communication::UniformBufferedScheme< stencil::D3Q27 > pdfGhostLayerSync(blocks);
      pdfGhostLayerSync.addPackInfo(make_shared< field::communication::PackInfo< PdfField_T > >(pdfFieldID));
      pdfFieldVTK->addBeforeFunction(pdfGhostLayerSync);

      pdfFieldVTK->addCellDataWriter(
         make_shared< lbm::VelocityVTKWriter< LatticeModel_T, float > >(pdfFieldID, "VelocityFromPDF"));
      pdfFieldVTK->addCellDataWriter(
         make_shared< lbm::DensityVTKWriter< LatticeModel_T, float > >(pdfFieldID, "DensityFromPDF"));

      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(pdfFieldVTK), "VTK (fluid field data)");
   }

   // add LBM communication function and boundary handling sweep (does the hydro force calculations and the no-slip
   // treatment)
   timeloop.add() << BeforeFunction(commFunction, "LBM Communication") << Sweep(bhSweep, "Boundary Handling");

   // stream + collide LBM step
   timeloop.add() << Sweep(makeSharedSweep(lbmSweep), "cell-wise LB sweep");

   ////////////////////////
   // EXECUTE SIMULATION //
   ////////////////////////

   WcTimingPool timeloopTiming;

   // time loop
   for (uint_t i = 0; i < timesteps; ++i)
   {
      timeloop.singleStep(timeloopTiming);

      timeloopTiming["RPD"].start();

      reduceProperty.operator()< mesa_pd::HydrodynamicForceTorqueNotification >(*ps);

      for (auto subCycle = uint_t(0); subCycle < numRPDSubCycles; ++subCycle)
      {
         ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor, vvIntegratorPreForce, *accessor);
         syncCall();

         ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor, addHydrodynamicInteraction, *accessor);
         ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor, addGravitationalForce, *accessor);

         // lubrication correction
         ps->forEachParticlePairHalf(
            false, mesa_pd::kernel::ExcludeInfiniteInfinite(), *accessor,
            [&lubricationCorrectionKernel, rpdDomain](const size_t idx1, const size_t idx2, auto& ac) {
               mesa_pd::collision_detection::AnalyticContactDetection acd;
               acd.getContactThreshold() = lubricationCorrectionKernel.getNormalCutOffDistance();
               mesa_pd::kernel::DoubleCast double_cast;
               mesa_pd::mpi::ContactFilter contact_filter;
               if (double_cast(idx1, idx2, ac, acd, ac))
               {
                  if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *rpdDomain))
                  {
                     double_cast(idx1, idx2, ac, lubricationCorrectionKernel, ac, acd.getContactNormal(),
                                 acd.getPenetrationDepth());
                  }
               }
            },
            *accessor);

         // collision response
         ps->forEachParticlePairHalf(
            false, mesa_pd::kernel::ExcludeInfiniteInfinite(), *accessor,
            [collisionResponse, rpdDomain, numRPDSubCycles](const size_t idx1, const size_t idx2, auto& ac) {
               mesa_pd::collision_detection::AnalyticContactDetection acd;
               mesa_pd::kernel::DoubleCast double_cast;
               mesa_pd::mpi::ContactFilter contact_filter;
               if (double_cast(idx1, idx2, ac, acd, ac))
               {
                  if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *rpdDomain))
                  {
                     collisionResponse(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), acd.getContactNormal(),
                                       acd.getPenetrationDepth(), real_t(1) / real_t(numRPDSubCycles));
                  }
               }
            },
            *accessor);

         reduceProperty.operator()< mesa_pd::ForceTorqueNotification >(*ps);

         ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor, vvIntegratorPostForce, *accessor);

         syncCall();
         particleMapping();
      }

      timeloopTiming["RPD"].end();

      // reset after logging here
      ps->forEachParticle(false, mesa_pd::kernel::SelectAll(), *accessor, resetHydrodynamicForceTorque, *accessor);
   }

   timeloopTiming.logResultOnRoot();

   return EXIT_SUCCESS;
}

} // namespace settling_sphere

int main(int argc, char** argv) { settling_sphere::main(argc, argv); }
