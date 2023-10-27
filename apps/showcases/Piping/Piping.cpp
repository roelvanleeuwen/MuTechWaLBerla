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
//! \file Piping.cpp
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"

#include "core/Environment.h"
#include "core/SharedFunctor.h"
#include "core/logging/all.h"
#include "core/timing/RemainingTimeLogger.h"

#include "field/AddToStorage.h"
#include "field/StabilityChecker.h"

#include "geometry/InitBoundaryHandling.h"

#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/communication/UniformGPUScheme.h"

#include "lbm/PerformanceLogger.h"
#include "lbm/vtk/all.h"

#include "lbm_mesapd_coupling/DataTypesGPU.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/cuda/PSMSweepCollectionGPU.h"

#include "mesa_pd/collision_detection/AnalyticContactDetection.h"
#include "mesa_pd/data/DataTypes.h"
#include "mesa_pd/data/ParticleAccessorWithBaseShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/shape/Sphere.h"
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/kernel/DoubleCast.h"
#include "mesa_pd/kernel/InsertParticleIntoLinkedCells.h"
#include "mesa_pd/kernel/LinearSpringDashpot.h"
#include "mesa_pd/kernel/VelocityVerlet.h"
#include "mesa_pd/mpi/ContactFilter.h"
#include "mesa_pd/mpi/ReduceContactHistory.h"
#include "mesa_pd/mpi/ReduceProperty.h"
#include "mesa_pd/mpi/SyncNextNeighbors.h"
#include "mesa_pd/mpi/notifications/ForceTorqueNotification.h"
#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "vtk/all.h"

#include "PSMPackInfo.h"
#include "PSMSweep.h"
#include "PSM_Density.h"
#include "PSM_InfoHeader.h"
#include "PSM_MacroGetter.h"
#include "PSM_NoSlip.h"
#include "Utility.h"

namespace walberla
{
namespace piping
{

///////////
// USING //
///////////

using namespace lbm_mesapd_coupling::psm::gpu;
typedef pystencils::PSMPackInfo PackInfo_T;

using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

///////////
// FLAGS //
///////////

const FlagUID Fluid_Flag("Fluid");
const FlagUID Density0_Flag("Density0");
const FlagUID Density1_Flag("Density1");
const FlagUID NoSlip_Flag("NoSlip");

//////////
// MAIN //
//////////

//*******************************************************************************************************************
/*!\brief CEEC LHC4: Localized erosion in offshore wind-turbine foundations
 *

 *
 */
//*******************************************************************************************************************
int main(int argc, char** argv)
{
   Environment env(argc, argv);
   auto cfgFile = env.config();
   if (!cfgFile) { WALBERLA_ABORT("Usage: " << argv[0] << " path-to-configuration-file \n"); }

   gpu::selectDeviceBasedOnMpiRank();

   WALBERLA_LOG_INFO_ON_ROOT("waLBerla revision: " << std::string(WALBERLA_GIT_SHA1).substr(0, 8));
   WALBERLA_LOG_INFO_ON_ROOT("compiler flags: " << std::string(WALBERLA_COMPILER_FLAGS));
   WALBERLA_LOG_INFO_ON_ROOT("build machine: " << std::string(WALBERLA_BUILD_MACHINE));
   WALBERLA_LOG_INFO_ON_ROOT(*cfgFile);

   // Read config file
   Config::BlockHandle domainParameters = cfgFile->getBlock("Domain");
   const Vector3< uint_t > domainSize   = domainParameters.getParameter< Vector3< uint_t > >("domainSize");
   const Vector3< uint_t > numBlocks    = domainParameters.getParameter< Vector3< uint_t > >("numBlocks");
   WALBERLA_CHECK_EQUAL(numBlocks[0] * numBlocks[1] * numBlocks[2], uint_t(MPIManager::instance()->numProcesses()),
                        "When using GPUs, the number of blocks (" << numBlocks[0] * numBlocks[1] * numBlocks[2]
                                                                  << ") has to match the number of MPI processes ("
                                                                  << uint_t(MPIManager::instance()->numProcesses())
                                                                  << ")");
   Vector3< uint_t > cellsPerBlock(domainSize[0] / numBlocks[0], domainSize[1] / numBlocks[1],
                                   domainSize[2] / numBlocks[2]);
   WALBERLA_CHECK_EQUAL(domainSize[0], cellsPerBlock[0] * numBlocks[0],
                        "number of cells in x of " << domainSize[0]
                                                   << " is not divisible by given number of blocks in x direction");
   WALBERLA_CHECK_EQUAL(domainSize[1], cellsPerBlock[1] * numBlocks[1],
                        "number of cells in y of " << domainSize[1]
                                                   << " is not divisible by given number of blocks in y direction");
   WALBERLA_CHECK_EQUAL(domainSize[2], cellsPerBlock[2] * numBlocks[2],
                        "number of cells in z of " << domainSize[2]
                                                   << " is not divisible by given number of blocks in z direction");
   const bool periodicInY = domainParameters.getParameter< bool >("periodicInY");
   if (periodicInY && numBlocks[1] == 1)
   {
      WALBERLA_ABORT("The number of blocks in periodic dimensions must be greater than 1.")
   }

   Config::BlockHandle physicsParameters     = cfgFile->getBlock("Physics");
   const uint_t timeSteps                    = physicsParameters.getParameter< uint_t >("timeSteps");
   const real_t hydraulicGradient            = physicsParameters.getParameter< real_t >("hydraulicGradient");
   const uint_t finalGradientTimeStep        = physicsParameters.getParameter< uint_t >("finalGradientTimeStep");
   const real_t kinematicViscosityFluid_SI   = physicsParameters.getParameter< real_t >("kinematicViscosityFluid_SI");
   const real_t dx_SI                        = physicsParameters.getParameter< real_t >("dx_SI");
   const real_t dt_SI                        = physicsParameters.getParameter< real_t >("dt_SI");
   const real_t gravitationalAcceleration_SI = physicsParameters.getParameter< real_t >("gravitationalAcceleration_SI");

   const real_t viscosity                 = kinematicViscosityFluid_SI * dt_SI / (dx_SI * dx_SI);
   const real_t omega                     = lbm::collision_model::omegaFromViscosity(viscosity);
   const real_t gravitationalAcceleration = gravitationalAcceleration_SI * dt_SI * dt_SI / dx_SI;
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(viscosity)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(omega)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(gravitationalAcceleration)

   Config::BlockHandle bucketParameters = cfgFile->getBlock("Bucket");
   const Vector3< real_t > bucketSizeFraction =
      bucketParameters.getParameter< Vector3< real_t > >("bucketSizeFraction");

   Config::BlockHandle particlesParameters     = cfgFile->getBlock("Particles");
   const std::string particleInFileName        = particlesParameters.getParameter< std::string >("inFileName");
   const real_t particleDensityRatio           = particlesParameters.getParameter< real_t >("densityRatio");
   const real_t particleFrictionCoefficient    = particlesParameters.getParameter< real_t >("frictionCoefficient");
   const real_t particleRestitutionCoefficient = particlesParameters.getParameter< real_t >("restitutionCoefficient");
   const uint_t particleNumSubCycles           = particlesParameters.getParameter< uint_t >("numSubCycles");
   const bool useLubricationCorrection         = particlesParameters.getParameter< bool >("useLubricationCorrection");
   const real_t poissonsRatio                  = particlesParameters.getParameter< real_t >("poissonsRatio");
   const Vector3< real_t > observationDomainFraction =
      particlesParameters.getParameter< Vector3< real_t > >("observationDomainFraction");
   const Vector3< real_t > observationDomainSize(real_c(domainSize[0]) * observationDomainFraction[0],
                                                 real_c(domainSize[1]) * observationDomainFraction[1],
                                                 real_c(domainSize[2]) * observationDomainFraction[2]);
   const uint_t numPreSteps           = particlesParameters.getParameter< uint_t >("numPreSteps");
   const real_t kappa                 = real_c(2) * (real_c(1) - poissonsRatio) / (real_c(2) - poissonsRatio);
   const real_t particleCollisionTime = real_t(10); // same resolution as in SettlingSpheres.prm
   bool useOpenMP                     = false;

   Config::BlockHandle outputParameters   = cfgFile->getBlock("Output");
   const uint_t vtkSpacing                = outputParameters.getParameter< uint_t >("vtkSpacing");
   const std::string vtkFolder            = outputParameters.getParameter< std::string >("vtkFolder");
   const bool fluidSlice                  = outputParameters.getParameter< bool >("fluidSlice");
   const uint_t performanceLogFrequency   = outputParameters.getParameter< uint_t >("performanceLogFrequency");
   const uint_t upliftSubsidenceFrequency = outputParameters.getParameter< uint_t >("upliftSubsidenceFrequency");

   Config::BlockHandle stabilityCheckerParameters = cfgFile->getBlock("StabilityChecker");
   const uint_t checkFrequency                    = stabilityCheckerParameters.getParameter< uint_t >("checkFrequency");

   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   shared_ptr< StructuredBlockForest > blocks = blockforest::createUniformBlockGrid(
      numBlocks[0], numBlocks[1], numBlocks[2], cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2], real_t(1),
      uint_t(0), false, false, false, periodicInY, false, // periodicity
      false);

   auto simulationDomain = blocks->getDomain();

   ////////////
   // MesaPD //
   ////////////

   auto rpdDomain = std::make_shared< mesa_pd::domain::BlockForestDomain >(blocks->getBlockForestPointer());

   // Init data structures
   auto ps       = std::make_shared< mesa_pd::data::ParticleStorage >(1);
   auto accessor = std::make_shared< mesa_pd::data::ParticleAccessorWithBaseShape >(ps);

   // Create bucket slice
   const Vector3< real_t > boxPosition(simulationDomain.xMax() * real_t(0.5), simulationDomain.yMax() * real_t(0.5),
                                       simulationDomain.zMax() * real_t(1 - bucketSizeFraction[2] / 2));
   const Vector3< real_t > boxEdgeLength(simulationDomain.xMax() * bucketSizeFraction[0],
                                         simulationDomain.yMax() * bucketSizeFraction[1],
                                         simulationDomain.zMax() * bucketSizeFraction[2]);
   createBox(*ps, boxPosition, boxEdgeLength);

   // Create planes
   createPlane(*ps, simulationDomain.minCorner(), Vector3< real_t >(0, 0, 1));
   createPlane(*ps, simulationDomain.maxCorner(), Vector3< real_t >(0, 0, -1));
   createPlane(*ps, simulationDomain.minCorner(), Vector3< real_t >(1, 0, 0));
   createPlane(*ps, simulationDomain.maxCorner(), Vector3< real_t >(-1, 0, 0));

   if (!periodicInY)
   {
      createPlane(*ps, simulationDomain.minCorner(), Vector3< real_t >(0, 1, 0));
      createPlane(*ps, simulationDomain.maxCorner(), Vector3< real_t >(0, -1, 0));
   }

   // Read spheres
   real_t maxParticleDiameter;
   initSpheresFromFile(particleInFileName, *ps, *rpdDomain, particleDensityRatio, simulationDomain, domainSize,
                       boxPosition, boxEdgeLength, maxParticleDiameter);

   // Set up RPD functionality
   // Synchronize particles between the blocks for the correct mapping of ghost particles
   // TODO: use overlap for synchronization due to lubrication
   mesa_pd::mpi::SyncNextNeighbors syncNextNeighborFunc;
   syncNextNeighborFunc(*ps, *rpdDomain);

   mesa_pd::kernel::LinearSpringDashpot collisionResponse(2);
   collisionResponse.setFrictionCoefficientDynamic(0, 0, particleFrictionCoefficient);

   real_t linkedCellWidth = 1.01_r * maxParticleDiameter;
   mesa_pd::data::LinkedCells linkedCells(rpdDomain->getUnionOfLocalAABBs().getExtended(linkedCellWidth),
                                          linkedCellWidth);

   settleParticles(
      numPreSteps, accessor, ps, *rpdDomain, linkedCells, syncNextNeighborFunc, collisionResponse, particleDensityRatio,
      particleRestitutionCoefficient, kappa,
      gravitationalAcceleration *
         real_t(10000), // this factor comes from the smaller time step size compared to the SettlingSpheres.cpp
      particleCollisionTime, useOpenMP);

   // Evaluate initial soil properties
   UpliftSubsidenceEvaluator upliftSubsidenceEvaluator(accessor, ps, boxPosition, boxEdgeLength, observationDomainSize);

   real_t seepageLength = computeSeepageLength(accessor, ps, boxPosition, boxEdgeLength);
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(seepageLength)

   // TODO: check formula again
   const real_t pressureDifference = hydraulicGradient * gravitationalAcceleration * seepageLength;
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(pressureDifference)

   ///////////////////////
   // ADD DATA TO BLOCKS //
   ////////////////////////

   // Setting initial PDFs to nan helps to detect bugs in the initialization/BC handling
   // TODO: setting pdf values to nan does not work because they propagate inside the domain from above the bucket
   BlockDataID pdfFieldID     = field::addToStorage< PdfField_T >(blocks, "pdf field (fzyx)", real_t(0), field::fzyx);
   BlockDataID pdfFieldGPUID  = gpu::addGPUFieldToStorage< PdfField_T >(blocks, pdfFieldID, "pdf field GPU");
   BlockDataID densityFieldID = field::addToStorage< DensityField_T >(blocks, "density field", real_t(0), field::fzyx);
   BlockDataID velFieldID  = field::addToStorage< VelocityField_T >(blocks, "velocity field", real_t(0), field::fzyx);
   BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");
   BlockDataID BFieldID    = field::addToStorage< BField_T >(blocks, "B field", real_t(0), field::fzyx);

   // Boundary handling
   assembleBoundaryBlock(domainSize, boxPosition, boxEdgeLength, periodicInY);

   auto boundariesCfgFile = Config();
   boundariesCfgFile.readParameterFile("boundaries.prm");
   auto boundariesConfig = boundariesCfgFile.getBlock("Boundaries");

   // map boundaries into the LBM simulation
   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldID, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldID, Fluid_Flag);
   lbm::PSM_Density density0_bc(blocks, pdfFieldGPUID, real_t(1.0));
   density0_bc.fillFromFlagField< FlagField_T >(blocks, flagFieldID, Density0_Flag, Fluid_Flag);
   lbm::PSM_Density density1_bc(blocks, pdfFieldGPUID, real_t(1.0));
   if (finalGradientTimeStep == 0) { density1_bc.bc_density_ = real_t(1.0) - pressureDifference; }
   density1_bc.fillFromFlagField< FlagField_T >(blocks, flagFieldID, Density1_Flag, Fluid_Flag);
   lbm::PSM_NoSlip noSlip(blocks, pdfFieldGPUID);
   noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldID, NoSlip_Flag, Fluid_Flag);

   ///////////////
   // TIME LOOP //
   ///////////////

   // Map particles into the fluid domain
   ParticleAndVolumeFractionSoA_T< Weighting > particleAndVolumeFractionSoA(blocks, omega);
   PSMSweepCollectionGPU psmSweepCollection(blocks, accessor, SphereSelector(), particleAndVolumeFractionSoA,
                                            uint_t(15));
   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      psmSweepCollection.particleMappingSweep(&(*blockIt));
   }

   real_t e_init =
      computeVoidRatio(blocks, BFieldID, particleAndVolumeFractionSoA.BFieldID, flagFieldID, Fluid_Flag, accessor, ps);
   WALBERLA_LOG_INFO_ON_ROOT("Void ratio e_init: " << e_init)

   // Initialize PDFs
   pystencils::InitializeDomainForPSM pdfSetter(
      particleAndVolumeFractionSoA.BsFieldID, particleAndVolumeFractionSoA.BFieldID,
      particleAndVolumeFractionSoA.particleVelocitiesFieldID, pdfFieldGPUID, real_t(0), real_t(0), real_t(0),
      real_t(1.0), real_t(0), real_t(0), real_t(0));

   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      // pdfSetter requires particle velocities at cell centers
      psmSweepCollection.setParticleVelocitiesSweep(&(*blockIt));
      pdfSetter(&(*blockIt));
   }

   // Setup of the LBM communication for synchronizing the pdf field between neighboring blocks
   // TODO: set sendDirectlyFromGPU to true for performance measurements on cluster
   gpu::communication::UniformGPUScheme< Stencil_T > com(blocks, false, false);
   com.addPackInfo(make_shared< PackInfo_T >(pdfFieldGPUID));
   auto communication = std::function< void() >([&]() { com.communicate(nullptr); });

   SweepTimeloop timeloop(blocks->getBlockStorage(), timeSteps);

   timeloop.addFuncBeforeTimeStep(RemainingTimeLogger(timeloop.getNrOfTimeSteps()), "Remaining Time Logger");

   pystencils::PSM_MacroGetter getterSweep(densityFieldID, pdfFieldID, velFieldID, real_t(0.0), real_t(0.0),
                                           real_t(0.0));
   // VTK output
   if (vtkSpacing != uint_t(0))
   {
      // Bucket slice
      auto bucketVtkOutput = make_shared< mesa_pd::vtk::ParticleVtkOutput >(ps);
      bucketVtkOutput->addOutput< mesa_pd::data::SelectParticleUid >("uid");
      bucketVtkOutput->addOutput< SelectBoxEdgeLength >("edgeLength");
      bucketVtkOutput->setParticleSelector([](const mesa_pd::data::ParticleStorage::iterator& pIt) {
         using namespace walberla::mesa_pd::data::particle_flags;
         return (pIt->getBaseShape()->getShapeType() == mesa_pd::data::Box::SHAPE_TYPE);
      });
      auto bucketVtkWriter = vtk::createVTKOutput_PointData(bucketVtkOutput, "bucket", vtkSpacing, vtkFolder);
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(bucketVtkWriter), "VTK (bucket data)");

      // Spheres
      auto particleVtkOutput = make_shared< mesa_pd::vtk::ParticleVtkOutput >(ps);
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleUid >("uid");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleLinearVelocity >("velocity");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleInteractionRadius >("radius");
      // Limit output to process-local spheres
      particleVtkOutput->setParticleSelector([](const mesa_pd::data::ParticleStorage::iterator& pIt) {
         using namespace walberla::mesa_pd::data::particle_flags;
         return (pIt->getBaseShape()->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE) &&
                !isSet(pIt->getFlags(), GHOST);
      });
      auto particleVtkWriter = vtk::createVTKOutput_PointData(particleVtkOutput, "particles", vtkSpacing, vtkFolder);
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(particleVtkWriter), "VTK (sphere data)");

      // Fields
      auto pdfFieldVTK = vtk::createVTKOutput_BlockData(blocks, "fluid", vtkSpacing, 0, false, vtkFolder);

      pdfFieldVTK->addBeforeFunction(communication);

      pdfFieldVTK->addBeforeFunction([&]() {
         gpu::fieldCpy< PdfField_T, gpu::GPUField< real_t > >(blocks, pdfFieldID, pdfFieldGPUID);
         gpu::fieldCpy< GhostLayerField< real_t, 1 >, BFieldGPU_T >(blocks, BFieldID,
                                                                    particleAndVolumeFractionSoA.BFieldID);
         for (auto& block : *blocks)
            getterSweep(&block);
      });

      pdfFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< VelocityField_T > >(velFieldID, "Velocity"));
      pdfFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< DensityField_T > >(densityFieldID, "Density"));
      pdfFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< BField_T > >(BFieldID, "OverlapFraction"));
      pdfFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< FlagField_T > >(flagFieldID, "FlagField"));

      AABB sliceAABB(real_t(0), real_c(domainSize[1]) * real_t(0.5) - real_t(1), real_t(0), real_c(domainSize[0]),
                     real_c(domainSize[1]) * real_t(0.5) + real_t(1), real_c(domainSize[2]));
      vtk::AABBCellFilter aabbSliceFilter(sliceAABB);
      field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagFieldID);
      fluidFilter.addFlag(Fluid_Flag);
      vtk::ChainedFilter combinedSliceFilter;
      combinedSliceFilter.addFilter(fluidFilter);
      if (fluidSlice) { combinedSliceFilter.addFilter(aabbSliceFilter); }
      pdfFieldVTK->addCellInclusionFilter(combinedSliceFilter);

      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(pdfFieldVTK), "VTK (fluid field data)");
   }

   if (vtkSpacing != uint_t(0)) { vtk::writeDomainDecomposition(blocks, "domain_decomposition", vtkFolder); }

   // Add performance logging
   // TODO: have a look why it causes a segmentation fault when called with performanceLogFrequency=0
   if (performanceLogFrequency > 0)
   {
      const lbm::PerformanceLogger< FlagField_T > performanceLogger(blocks, flagFieldID, Fluid_Flag,
                                                                    performanceLogFrequency);
      timeloop.addFuncAfterTimeStep(performanceLogger, "Evaluate performance logging");
   }

   // Add LBM communication function and boundary handling sweep
   // TODO: use split sweeps to hide communication
   timeloop.add() << BeforeFunction(communication, "LBM Communication")
                  << Sweep(deviceSyncWrapper(density0_bc.getSweep()), "Boundary Handling (Density0)");
   timeloop.add() << Sweep(deviceSyncWrapper(density1_bc.getSweep()), "Boundary Handling (Density1)");
   timeloop.add() << Sweep(deviceSyncWrapper(noSlip.getSweep()), "Boundary Handling (NoSlip)");

   // PSM kernel
   pystencils::PSMSweep PSMSweep(particleAndVolumeFractionSoA.BsFieldID, particleAndVolumeFractionSoA.BFieldID,
                                 particleAndVolumeFractionSoA.particleForcesFieldID,
                                 particleAndVolumeFractionSoA.particleVelocitiesFieldID, pdfFieldGPUID, real_t(0.0),
                                 real_t(0.0), real_t(0.0), omega);

   addPSMSweepsToTimeloop(timeloop, psmSweepCollection, PSMSweep);

   WcTimingPool timeloopTiming;

   for (uint_t timeStep = 0; timeStep < timeSteps; ++timeStep)
   {
      timeloop.singleStep(timeloopTiming);
      // If pressure difference did not yet reach the limit, decrease the pressure on the right hand side
      density1_bc.bc_density_ = std::max(real_t(1.0) - pressureDifference,
                                         density1_bc.bc_density_ - pressureDifference / real_t(finalGradientTimeStep));

      // LBM stability check (check for NaNs in the PDF field)
      timeloopTiming["LBM stability check"].start();
      if (checkFrequency > 0 && timeStep % checkFrequency == 0)
      {
         gpu::fieldCpy< PdfField_T, gpu::GPUField< real_t > >(blocks, pdfFieldID, pdfFieldGPUID);
         makeSharedFunctor(field::makeStabilityChecker< PdfField_T, FlagField_T >(cfgFile, blocks, pdfFieldID,
                                                                                  flagFieldID, Fluid_Flag))();
      }
      timeloopTiming["LBM stability check"].end();

      // Uplift/Subsidence evaluation
      timeloopTiming["Uplift/Subsidence evaluation"].start();
      if (upliftSubsidenceFrequency > 0 && timeStep % upliftSubsidenceFrequency == 0)
      {
         upliftSubsidenceEvaluator(hydraulicGradient * real_t(timeStep / finalGradientTimeStep), accessor, ps);
      }
      timeloopTiming["Uplift/Subsidence evaluation"].end();
   }

   timeloopTiming.logResultOnRoot();

   return EXIT_SUCCESS;
}

} // namespace piping
} // namespace walberla

int main(int argc, char** argv) { walberla::piping::main(argc, argv); }
