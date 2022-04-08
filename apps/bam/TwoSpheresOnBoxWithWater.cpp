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
//! \file
//! \author
//
//======================================================================================================================


#include <blockforest/Initialization.h>
#include <blockforest/StructuredBlockForest.h>

#include "core/timing/RemainingTimeLogger.h"
#include "core/SharedFunctor.h"

#include "domain_decomposition/SharedSweep.h"

#include "timeloop/SweepTimeloop.h"

#include "lbm/boundary/all.h"
#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/field/PdfField.h"
#include "lbm/lattice_model/D3Q19.h"
#include "lbm/sweeps/CellwiseSweep.h"

#include "lbm_mesapd_coupling/mapping/ParticleMapping.h"
#include "lbm_mesapd_coupling/momentum_exchange_method/MovingParticleMapping.h"
#include "lbm_mesapd_coupling/momentum_exchange_method/boundary/CurvedLinear.h"
#include "lbm_mesapd_coupling/momentum_exchange_method/reconstruction/PdfReconstructionManager.h"
#include "lbm_mesapd_coupling/utility/AddForceOnParticlesKernel.h"
#include "lbm_mesapd_coupling/utility/ParticleSelector.h"
#include "lbm_mesapd_coupling/DataTypes.h"
#include "lbm_mesapd_coupling/utility/AverageHydrodynamicForceTorqueKernel.h"
#include "lbm_mesapd_coupling/utility/AddHydrodynamicInteractionKernel.h"
#include "lbm_mesapd_coupling/utility/ResetHydrodynamicForceTorqueKernel.h"
#include "lbm_mesapd_coupling/utility/LubricationCorrectionKernel.h"

#include <mesa_pd/collision_detection/AnalyticContactDetection.h>

#include <mesa_pd/data/ParticleAccessorWithShape.h>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/ShapeStorage.h>
#include <mesa_pd/domain/BlockForestDomain.h>

#include <mesa_pd/kernel/DoubleCast.h>
#include <mesa_pd/kernel/CohesionInitialization.h>
#include <mesa_pd/kernel/Cohesion.h>
#include <mesa_pd/kernel/SpringDashpot.h>
#include <mesa_pd/kernel/SemiImplicitEuler.h>
#include <mesa_pd/mpi/ReduceContactHistory.h>
#include <mesa_pd/mpi/ReduceProperty.h>
#include <mesa_pd/mpi/SyncNextNeighbors.h>
#include <mesa_pd/mpi/notifications/ForceTorqueNotification.h>
#include <mesa_pd/mpi/ContactFilter.h>

#include <mesa_pd/vtk/OutputSelector.h>
#include <mesa_pd/vtk/ParticleVtkOutput.h>

#include "field/vtk/all.h"
#include "vtk/all.h"
#include "lbm/vtk/all.h"

#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <iostream>

#include "Utility.h"

namespace two_spheres_on_box_with_water {

using namespace walberla;
using namespace walberla::mesa_pd;

using walberla::uint_t;
using LatticeModel_T = lbm::D3Q19< lbm::collision_model::SRT>;
using Stencil_T = LatticeModel_T::Stencil;
using PdfField_T = lbm::PdfField<LatticeModel_T>;
using flag_t = walberla::uint8_t;
using FlagField_T = FlagField<flag_t>;
using ScalarField_T = GhostLayerField< real_t, 1>;
const uint_t FieldGhostLayers = 1;

///////////
// FLAGS //
///////////

const FlagUID Fluid_Flag( "fluid" );
const FlagUID NoSlip_Flag( "no slip" );
const FlagUID MO_Flag( "moving obstacle" );
const FlagUID FormerMO_Flag( "former moving obstacle" );

template <typename ParticleAccessor_T>
class MyBoundaryHandling
{
 public:

   using NoSlip_T = lbm::NoSlip< LatticeModel_T, flag_t >;
   using MO_T = lbm_mesapd_coupling::CurvedLinear< LatticeModel_T, FlagField_T, ParticleAccessor_T >;
   using Type = BoundaryHandling< FlagField_T, Stencil_T, NoSlip_T, MO_T >;

   MyBoundaryHandling( const BlockDataID & flagFieldID, const BlockDataID & pdfFieldID,
                       const BlockDataID & particleFieldID, const shared_ptr<ParticleAccessor_T>& ac) :
      flagFieldID_( flagFieldID ), pdfFieldID_( pdfFieldID ), particleFieldID_( particleFieldID ), ac_( ac ) {}

   Type * operator()( IBlock* const block, const StructuredBlockStorage* const storage ) const
   {
      WALBERLA_ASSERT_NOT_NULLPTR( block );
      WALBERLA_ASSERT_NOT_NULLPTR( storage );

      auto * flagField     = block->getData< FlagField_T >( flagFieldID_ );
      auto *  pdfField     = block->getData< PdfField_T > ( pdfFieldID_ );
      auto * particleField = block->getData< lbm_mesapd_coupling::ParticleField_T > ( particleFieldID_ );

      const auto fluid = flagField->flagExists( Fluid_Flag ) ? flagField->getFlag( Fluid_Flag ) : flagField->registerFlag( Fluid_Flag );

      Type * handling = new Type( "moving obstacle boundary handling", flagField, fluid,
                                  NoSlip_T( "NoSlip", NoSlip_Flag, pdfField ),
                                  MO_T( "MO", MO_Flag, pdfField, flagField, particleField, ac_, fluid, *storage, *block ) );

      // Add other boundary conditions here -> get cell interval

      handling->fillWithDomain( FieldGhostLayers ); // initialize flag field with "Fluid" flag

      return handling;
   }

 private:

   const BlockDataID flagFieldID_;
   const BlockDataID pdfFieldID_;
   const BlockDataID particleFieldID_;

   shared_ptr<ParticleAccessor_T> ac_;
};

int main( int argc, char ** argv )
{
   Environment env(argc, argv);
   WALBERLA_UNUSED(env);
   walberla::mpi::MPIManager::instance()->useWorldComm();

   std::string vtkOutputFolder = "vtk_out_TwoSpheresBoxWater";
   real_t cellsPerDiameter = real_t(10);
   real_t relaxationTime = real_t(0.65); // (0.5, \infty)
   Vector3<uint_t> numberOfBlocksPerDirection( uint_t(2), uint_t(2), uint_t(2) );
   uint_t vtkSpacing = uint_t(100);
   uint_t numberOfTimeSteps = uint_t(8000);
   uint_t numberOfMesapdSubCycles = uint_t(1);

   // SI parameters
   real_t diameter_SI = real_t(0.001); // m
   real_t densityParticle_SI = real_t(1010);
   Vector3<real_t> domainSize_SI(real_t(0.01),real_t(0.01),real_t(0.01));
   real_t densityFluid_SI = real_t(1000);
   real_t kinematicViscosity_SI = real_t(1e-6); // m**2 / s
   real_t frictionCoefficient = real_t(0.3);
   real_t gravitationalAcceleration_SI = real_t(10); // m / s**2
   real_t y_n_SI = 200_r; // N = tensile force, where bond breaks

   // unit conversion
   real_t dx_SI = diameter_SI / cellsPerDiameter; // m
   real_t omega = real_t(1) / relaxationTime;
   real_t kinematicViscosity = lbm::collision_model::viscosityFromOmega(omega);
   real_t dt_SI = (kinematicViscosity / kinematicViscosity_SI) * dx_SI * dx_SI; // s
   real_t gravitationalAcceleration = gravitationalAcceleration_SI * dt_SI * dt_SI / dx_SI;
   real_t densityRatio = densityParticle_SI / densityFluid_SI;
   real_t densityParticle = densityRatio;
   real_t densityFluid = real_t(1);

   real_t diameter = diameter_SI / dx_SI;
   real_t timeStepSize = real_t(1);
   Vector3<uint_t> domainSize( uint_c(domainSize_SI[0] / dx_SI),
                                uint_c(domainSize_SI[1] / dx_SI),
                                uint_c(domainSize_SI[2] / dx_SI));

   WALBERLA_LOG_INFO_ON_ROOT("dx_SI = " << dx_SI << " m");
   WALBERLA_LOG_INFO_ON_ROOT("dt_SI = " << dt_SI << " s");
   WALBERLA_LOG_INFO_ON_ROOT("gravitational acceleration lattice units = " << gravitationalAcceleration);
   WALBERLA_LOG_INFO_ON_ROOT("density ratio = " << densityRatio);
   WALBERLA_LOG_INFO_ON_ROOT("Domain size = " << domainSize);


   Vector3<uint_t> cellsPerBlockPerDirection( domainSize[0] / numberOfBlocksPerDirection[0],
                                              domainSize[1] / numberOfBlocksPerDirection[1],
                                              domainSize[2] / numberOfBlocksPerDirection[2] );
   for( uint_t i = 0; i < 3; ++i ) {
      WALBERLA_CHECK_EQUAL(cellsPerBlockPerDirection[i] * numberOfBlocksPerDirection[i], domainSize[i],
                           "Unmatching domain decomposition in direction " << i << "!");
   }

   auto domainAABB = math::AABB{Vector3<real_t>{0_r}, domainSize};
   auto blocks = blockforest::createUniformBlockGrid( numberOfBlocksPerDirection[0], numberOfBlocksPerDirection[1], numberOfBlocksPerDirection[2],
                                                      cellsPerBlockPerDirection[0], cellsPerBlockPerDirection[1], cellsPerBlockPerDirection[2], real_t(1),
                                                      0, false, false,
                                                      false, false, false, //periodicity
                                                      false );

   WALBERLA_LOG_INFO_ON_ROOT("Domain decomposition:");
   WALBERLA_LOG_INFO_ON_ROOT(" - blocks per direction = " << numberOfBlocksPerDirection );
   WALBERLA_LOG_INFO_ON_ROOT(" - cells per block = " << cellsPerBlockPerDirection );


   //write domain decomposition to file
   if( vtkSpacing > 0 )
   {
      walberla::vtk::writeDomainDecomposition( blocks, "initial_domain_decomposition", vtkOutputFolder );
   }


   // mesa pd
   auto mesapdDomain = std::make_shared<mesa_pd::domain::BlockForestDomain>(blocks->getBlockForestPointer());
   auto ps = std::make_shared<data::ParticleStorage>(2);
   auto ss = std::make_shared<data::ShapeStorage>();
   using ParticleAccessor_T = mesa_pd::data::ParticleAccessorWithShape;
   auto accessor = walberla::make_shared<ParticleAccessor_T >(ps, ss);

   real_t sphereRadius = diameter / 2_r;
   auto sphereShape = ss->create<data::Sphere>(sphereRadius);
   ss->shapes[sphereShape]->updateMassAndInertia(densityParticle);

   real_t boxEdgeLength = diameter;
   auto boxShape = ss->create<data::Box>(Vec3(boxEdgeLength));
   ss->shapes[boxShape]->updateMassAndInertia(densityParticle);

   Vector3<real_t> centerPoint = (domainAABB.maxCorner() - domainAABB.minCorner()) / real_t(2);

   std::vector<Vec3> spherePositions;
   spherePositions.push_back(centerPoint);
   spherePositions.push_back(centerPoint + Vec3(sphereRadius*1.9999, real_t(0), real_t(0)));
   //spherePositions.push_back(centerPoint + Vector3<real_t>(sphereRadius*1.9999_r*2_r, real_t(0), real_t(0)));

   for (uint_t i = 0; i < spherePositions.size(); ++i) {
      Vec3 pos = spherePositions[i];
      if (mesapdDomain->isContainedInProcessSubdomain(uint_c(walberla::mpi::MPIManager::instance()->rank()), pos)) {
         auto sphereParticle = ps->create();

         sphereParticle->setShapeID(sphereShape);
         sphereParticle->setType(0);
         sphereParticle->setPosition(pos);
         sphereParticle->setOwner(walberla::MPIManager::instance()->rank());
         sphereParticle->setInteractionRadius(sphereRadius);
         WALBERLA_LOG_INFO("sphere created");
      }
   }

   bool globalBox = true;
   std::vector<Vec3> boxPositions;
   boxPositions.push_back(Vec3{centerPoint[0], centerPoint[1], sphereRadius});

   for (uint_t i = 0; i < boxPositions.size(); ++i) {
      Vec3 pos = boxPositions[i];
      //if (mesapdDomain->isContainedInProcessSubdomain(uint_c(walberla::mpi::MPIManager::instance()->rank()), pos)) {
         auto particle = ps->create(globalBox);

         particle->setShapeID(boxShape);
         particle->setType(0);
         particle->setPosition(pos);
         particle->setOwner(walberla::MPIManager::instance()->rank());
         particle->setInteractionRadius(std::sqrt(3_r) * boxEdgeLength / 2_r);

         WALBERLA_LOG_INFO("box created");
      //}
   }

   createPlane(*ps, *ss, domainAABB.minCorner(), Vector3<real_t>(1_r,0,0));
   createPlane(*ps, *ss, domainAABB.minCorner(), Vector3<real_t>(0,1_r,0));
   createPlane(*ps, *ss, domainAABB.minCorner(), Vector3<real_t>(0,0,1_r));
   createPlane(*ps, *ss, domainAABB.maxCorner(), Vector3<real_t>(-1_r,0,0));
   createPlane(*ps, *ss, domainAABB.maxCorner(), Vector3<real_t>(0,-1_r,0));
   createPlane(*ps, *ss, domainAABB.maxCorner(), Vector3<real_t>(0,0,-1_r));



   // LBM parts
   LatticeModel_T latticeModel = LatticeModel_T(omega);

   // PDF field
   BlockDataID pdfFieldID = lbm::addPdfFieldToStorage< LatticeModel_T >( blocks, "pdf field (fzyx)", latticeModel,
                                                                         Vector3< real_t >( real_t(0) ), real_t(1),
                                                                         FieldGhostLayers, field::fzyx );
   // flag field
   BlockDataID flagFieldID = field::addFlagFieldToStorage<FlagField_T>( blocks, "flag field" );

   // particle field -> for coupling
   BlockDataID particleFieldID = field::addToStorage<lbm_mesapd_coupling::ParticleField_T>( blocks, "particle field", accessor->getInvalidUid(), field::fzyx, FieldGhostLayers );

   // add boundary handling
   using BoundaryHandling_T = MyBoundaryHandling<ParticleAccessor_T>::Type;
   BlockDataID boundaryHandlingID = blocks->addStructuredBlockData< BoundaryHandling_T >(MyBoundaryHandling<ParticleAccessor_T>( flagFieldID, pdfFieldID, particleFieldID, accessor), "boundary handling" );


   // kernels
   std::function<void(void)> syncCall = [ps,mesapdDomain](){
     const real_t overlap = real_t( 1.5 );
     mesa_pd::mpi::SyncNextNeighbors syncNextNeighborFunc;
     syncNextNeighborFunc(*ps, *mesapdDomain, overlap);
   };
   syncCall();

   // mesapd kernels
   kernel::CohesionInitialization cohesionInitKernel;
   kernel::Cohesion cohesionKernel(1);

   mesa_pd::mpi::ReduceContactHistory reduceAndSwapContactHistory;
   mesa_pd::mpi::ReduceProperty reduceProperty;
   mesa_pd::mpi::ContactFilter contactFilter;

   real_t timeStepSizeMesapd = timeStepSize / real_c(numberOfMesapdSubCycles);
   kernel::SemiImplicitEuler particleIntegration(timeStepSizeMesapd);
   SelectSphere sphereSelector;

   // coupling kernels
   lbm_mesapd_coupling::AddHydrodynamicInteractionKernel addHydrodynamicInteraction;
   lbm_mesapd_coupling::ResetHydrodynamicForceTorqueKernel resetHydrodynamicForceTorque;
   lbm_mesapd_coupling::AverageHydrodynamicForceTorqueKernel averageHydrodynamicForceTorque;

   //TDOO rethink parameterization: maybe use collision time instead of E_SI?

   real_t sphereRadius_SI = diameter_SI / 2_r;
   real_t sphereVolume_SI = 4_r/ 3_r * math::pi * sphereRadius_SI * sphereRadius_SI * sphereRadius_SI;
   real_t sphereMass_SI = densityParticle_SI * sphereVolume_SI;
   real_t E_SI = 1e3_r; // kg / (m * s^2)
   real_t en = 0.2_r; // coefficient of restitution
   real_t kn_SI = 2_r * E_SI * (sphereRadius_SI * sphereRadius_SI / (sphereRadius_SI + sphereRadius_SI));
   real_t meff_SI = sphereMass_SI * sphereMass_SI / (sphereMass_SI + sphereMass_SI);
   real_t damping = -std::log(en) / std::sqrt((std::log(en) * std::log(en) + math::pi * math::pi));
   real_t nun_SI = 2_r * std::sqrt(kn_SI * meff_SI) * damping;

   real_t kn = kn_SI / ( densityFluid_SI * dx_SI * dx_SI * dx_SI / ( dt_SI * dt_SI ));
   real_t nun = nun_SI / ( densityFluid_SI * dx_SI * dx_SI * dx_SI / ( dt_SI ));
   real_t y_n = y_n_SI / ( densityFluid_SI * dx_SI * dx_SI * dx_SI * dx_SI  / ( dt_SI * dt_SI ));

   real_t ksFactors = 0.5_r; // -
   real_t krFactors = 0.1_r; // -
   real_t koFactors = 0.1_r; // -

   real_t nusFactor = 0_r; // -
   real_t nurFactor = 0_r; // -
   real_t nuoFactor = 0_r; // -

   real_t y_s = 0.5_r * y_n;
   real_t y_r = 0.1_r * y_n / dx_SI; // TODO check -> torsion = N m
   real_t y_o = 0.1_r * y_n / dx_SI; // TODO check -> torsion = N m

   WALBERLA_LOG_INFO_ON_ROOT("kn = " << kn << ", nun = " << nun << ", yn = " << y_n);
   WALBERLA_LOG_INFO_ON_ROOT("Estimated maximum surface distance for rupture / radius= " << (y_n / kn) / sphereRadius);

   cohesionKernel.setKn(0,0,kn);
   cohesionKernel.setKsFactor(0,0,ksFactors);
   cohesionKernel.setKrFactor(0,0,krFactors);
   cohesionKernel.setKoFactor(0,0,koFactors);

   cohesionKernel.setNun(0,0,nun);
   cohesionKernel.setNusFactor(0,0,nusFactor);
   cohesionKernel.setNurFactor(0,0,nurFactor);
   cohesionKernel.setNuoFactor(0,0,nuoFactor);

   cohesionKernel.setFrictionCoefficient(0,0,frictionCoefficient);

   cohesionKernel.setYn(0,0,y_n);
   cohesionKernel.setYs(0,0,y_s);
   cohesionKernel.setYr(0,0,y_r);
   cohesionKernel.setYo(0,0,y_o);

   real_t volumeSphere = math::pi / real_t(6) * diameter * diameter * diameter;
   real_t massSphere = densityParticle * volumeSphere;
   Vector3<real_t> gravitationalForce(real_t(0), real_t(0), -gravitationalAcceleration * massSphere);
   Vector3<real_t> buoyancyForce(real_t(0), real_t(0), gravitationalAcceleration * densityFluid * volumeSphere);


   // vtk
   // sphere
   auto sphereVtkOutput = make_shared<mesa_pd::vtk::ParticleVtkOutput>(ps);
   sphereVtkOutput->addOutput<mesa_pd::data::SelectParticleInteractionRadius>("interactionRadius");
   sphereVtkOutput->addOutput<data::SelectParticleLinearVelocity>("velocity");
   sphereVtkOutput->setParticleSelector([sphereShape](const mesa_pd::data::ParticleStorage::iterator& pIt){
     return pIt->getShapeID() == sphereShape;
   });
   auto sphereVtkWriter = walberla::vtk::createVTKOutput_PointData(sphereVtkOutput, "spheres", vtkSpacing, vtkOutputFolder, "simulation_step");

   // box
   auto boxVtkOutput = make_shared<mesa_pd::vtk::ParticleVtkOutput>(ps);
   boxVtkOutput->addOutput<data::SelectParticleLinearVelocity>("velocity");
   boxVtkOutput->setParticleSelector( [boxShape](const mesa_pd::data::ParticleStorage::iterator& pIt) {return pIt->getShapeID() == boxShape;} ); //limit output to boxes
   auto boxVtkWriter = walberla::vtk::createVTKOutput_PointData(boxVtkOutput, "box", vtkSpacing, vtkOutputFolder, "simulation_step");

   // fluid
   auto pdfFieldVTK = walberla::vtk::createVTKOutput_BlockData( blocks, "fluid_field", vtkSpacing, 0, false, vtkOutputFolder );
   field::FlagFieldCellFilter< FlagField_T > fluidFilter( flagFieldID );
   fluidFilter.addFlag( Fluid_Flag );
   pdfFieldVTK->addCellInclusionFilter( fluidFilter );
   pdfFieldVTK->addCellDataWriter( make_shared< lbm::VelocityVTKWriter< LatticeModel_T, float > >( pdfFieldID, "VelocityFromPDF" ) );
   pdfFieldVTK->addCellDataWriter( make_shared< lbm::DensityVTKWriter < LatticeModel_T, float > >( pdfFieldID, "DensityFromPDF" ) );




   // create the timeloop
   SweepTimeloop timeloop( blocks->getBlockStorage(), numberOfTimeSteps );
   timeloop.addFuncBeforeTimeStep( RemainingTimeLogger( timeloop.getNrOfTimeSteps() ), "Remaining Time Logger" );
   timeloop.addFuncBeforeTimeStep( walberla::vtk::writeFiles( pdfFieldVTK ), "VTK (fluid field data)" );
   timeloop.addFuncBeforeTimeStep( walberla::vtk::writeFiles( sphereVtkWriter ), "VTK (sphere data)" );
   timeloop.addFuncBeforeTimeStep( walberla::vtk::writeFiles( boxVtkWriter ), "VTK (box field data)" );

   blockforest::communication::UniformBufferedScheme< Stencil_T > optimizedPDFCommunicationScheme( blocks );//meaning?
   optimizedPDFCommunicationScheme.addPackInfo( make_shared< lbm::PdfFieldPackInfo< LatticeModel_T > >( pdfFieldID ) ); // optimized sync

   // add LBM communication function (updates ghost layers) and boundary handling sweep (does the hydro force calculations and the no-slip treatment)
   auto boundaryHandlingSweep = BoundaryHandling_T::getBlockSweep( boundaryHandlingID );
   timeloop.add() << BeforeFunction( optimizedPDFCommunicationScheme, "LBM Communication" )
                  << Sweep(boundaryHandlingSweep, "Boundary Handling" );
   // add LBM part (stream + collide)
   auto lbmSweep = lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >( pdfFieldID, flagFieldID, Fluid_Flag );
   timeloop.add() << Sweep( makeSharedSweep( lbmSweep ), "cell-wise LB sweep" );


   SweepTimeloop timeloopAfterParticle( blocks->getBlockStorage(), numberOfTimeSteps );
   // update mapping: check if fluid -> moving obstacle, and moving obstacle -> former MO, when particles have moved
   timeloopAfterParticle.add() << Sweep( lbm_mesapd_coupling::makeMovingParticleMapping<PdfField_T, BoundaryHandling_T>(blocks, pdfFieldID, boundaryHandlingID, particleFieldID, accessor, MO_Flag, FormerMO_Flag,
                                                                                                                         SphereAndBoxSelector(), false), "Particle Mapping" );
   // reconstruct PDFs in former MO flags (former MO -> fluid)
   timeloopAfterParticle.add() << Sweep( makeSharedSweep(lbm_mesapd_coupling::makePdfReconstructionManager<PdfField_T,BoundaryHandling_T>(blocks, pdfFieldID, boundaryHandlingID, particleFieldID, accessor, FormerMO_Flag, Fluid_Flag, false) ), "PDF Restore" );


   bool openmp = false;

   // initialize fields
   // map planes into the LBM simulation -> act as no-slip boundaries
   lbm_mesapd_coupling::ParticleMappingKernel<BoundaryHandling_T> particleMappingKernel(blocks, boundaryHandlingID);
   ps->forEachParticle(openmp, lbm_mesapd_coupling::GlobalParticlesSelector(), *accessor, particleMappingKernel, *accessor, NoSlip_Flag);

   // map particles into the LBM simulation
   lbm_mesapd_coupling::MovingParticleMappingKernel<BoundaryHandling_T> movingParticleMappingKernel(blocks, boundaryHandlingID, particleFieldID);
   ps->forEachParticle(openmp, SphereAndBoxSelector(), *accessor, movingParticleMappingKernel, *accessor, MO_Flag);

   //cohesion init
   ps->forEachParticlePairHalf(openmp, sphereSelector, *accessor,
                               [&](const size_t idx1, const size_t idx2){
                                 mesa_pd::collision_detection::AnalyticContactDetection acd;
                                 mesa_pd::kernel::DoubleCast double_cast;
                                 if (double_cast(idx1, idx2, *accessor, acd, *accessor)) {
                                    // particles overlap
                                    if (contactFilter(acd.getIdx1(), acd.getIdx2(), *accessor, acd.getContactPoint(),  *mesapdDomain))
                                    {
                                       cohesionInitKernel(acd.getIdx1(), acd.getIdx2(), *accessor, acd.getPenetrationDepth());
                                    }
                                 }
                               });
   reduceAndSwapContactHistory(*ps);



   WcTimingPool timeloopTiming;
   for(uint_t t = 0; t < numberOfTimeSteps; ++t) {

      // LBM + boundary handling + coupling force evaluation
      timeloop.singleStep(timeloopTiming);

      // average hydrodynamic force over two time steps to avoid oscillations
      ps->forEachParticle(openmp, mesa_pd::kernel::SelectAll(), *accessor, averageHydrodynamicForceTorque, *accessor );


      // add sub cycling for particle simulation -> increase temporal resolution of contact detection and resolving
      for(uint_t subCycle = 0; subCycle < numberOfMesapdSubCycles; ++subCycle )
      {

         // take stored Fhyd values and add onto particles as force
         ps->forEachParticle(false, sphereSelector, *accessor, addHydrodynamicInteraction, *accessor );

         // cohesive and non-cohesive interaction
         ps->forEachParticlePairHalf(openmp, ExcludeGlobalGlobal(), *accessor,
                                     [&](size_t idx1, size_t idx2){
                                       mesa_pd::collision_detection::AnalyticContactDetection acd;
                                       mesa_pd::kernel::DoubleCast double_cast;
                                       bool contactExists = double_cast(idx1, idx2, *accessor, acd, *accessor);

                                       Vector3<real_t> filteringPoint;
                                       if (contactExists)  {
                                          filteringPoint = acd.getContactPoint();
                                       } else {
                                          filteringPoint = (accessor->getPosition(idx1) + accessor->getPosition(idx2)) / real_t(2);
                                       }

                                       if (contactFilter(idx1, idx2, *accessor, filteringPoint, *mesapdDomain))
                                       {
                                          bool contactTreatedByCohesionKernel = false;
                                          if (sphereSelector(idx1, idx2, *accessor))
                                          {
                                             if (cohesionKernel.isCohesiveBondActive(idx1, idx2, *accessor))
                                             { contactTreatedByCohesionKernel = cohesionKernel(idx1, idx2, *accessor, timeStepSizeMesapd); }
                                          }
                                          if (contactExists && !contactTreatedByCohesionKernel)
                                          {
                                             cohesionKernel.nonCohesiveInteraction(
                                                acd.getIdx1(), acd.getIdx2(), *accessor, acd.getContactPoint(),
                                                acd.getContactNormal(), acd.getPenetrationDepth(), timeStepSizeMesapd);
                                          }
                                       }

                                     });


         // synchronize collision information
         reduceAndSwapContactHistory(*ps);

         //ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor, [](const size_t idx, ParticleAccessor_T& ac){WALBERLA_LOG_INFO("vel = " << ac.getLinearVelocity(idx) << ", F = " << ac.getForce(idx))},*accessor);

         // add gravitational + buoyancy force
         ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor, [gravitationalForce](const size_t idx, ParticleAccessor_T& ac){mesa_pd::addForceAtomic(idx, ac, gravitationalForce);},*accessor);
         ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor, [buoyancyForce](const size_t idx, ParticleAccessor_T& ac){mesa_pd::addForceAtomic(idx, ac, buoyancyForce);},*accessor);

         // synchronize forces
         reduceProperty.operator()<mesa_pd::ForceTorqueNotification>(*ps);

         // update position and velocity
         ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor, particleIntegration, *accessor);

         // synchronize position and velocity
         syncCall();
      }



      // reset F hyd
      ps->forEachParticle(false, mesa_pd::kernel::SelectAll(), *accessor, resetHydrodynamicForceTorque, *accessor );

      // update mapping + PDF restore
      timeloopAfterParticle.singleStep(timeloopTiming);

   }

   timeloopTiming.logResultOnRoot();

   return EXIT_SUCCESS;
}

}

int main( int argc, char ** argv )
{
   return two_spheres_on_box_with_water::main(argc, argv);
}
