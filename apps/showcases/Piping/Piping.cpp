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
#include "lbm_mesapd_coupling/partially_saturated_cells_method/gpu/PSMSweepCollectionGPU.h"
#include "lbm_mesapd_coupling/utility/AddHydrodynamicInteractionKernel.h"
#include "lbm_mesapd_coupling/utility/AverageHydrodynamicForceTorqueKernel.h"
#include "lbm_mesapd_coupling/utility/InitializeHydrodynamicForceTorqueForAveragingKernel.h"
#include "lbm_mesapd_coupling/utility/LubricationCorrectionKernel.h"
#include "lbm_mesapd_coupling/utility/ResetHydrodynamicForceTorqueKernel.h"

#include "mesa_pd/collision_detection/AnalyticContactDetection.h"
#include "mesa_pd/data/DataTypes.h"
#include "mesa_pd/data/ParticleAccessorWithBaseShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/shape/Sphere.h"
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/kernel/AssocToBlock.h"
#include "mesa_pd/kernel/DoubleCast.h"
#include "mesa_pd/kernel/InsertParticleIntoLinkedCells.h"
#include "mesa_pd/kernel/LinearSpringDashpot.h"
#include "mesa_pd/kernel/VelocityVerlet.h"
#include "mesa_pd/mpi/ContactFilter.h"
#include "mesa_pd/mpi/ReduceContactHistory.h"
#include "mesa_pd/mpi/ReduceProperty.h"
#include "mesa_pd/mpi/SyncNextNeighbors.h"
#include "mesa_pd/mpi/notifications/ForceTorqueNotification.h"
#include "mesa_pd/mpi/notifications/HydrodynamicForceTorqueNotification.h"
#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "vtk/all.h"

#include "PSMPackInfo.h"
#include "PSMSweep.h"
#include "PSM_Density.h"
#include "PSM_FreeSlip.h"
#include "PSM_InfoHeader.h"
#include "PSM_MacroGetter.h"
#include "PSM_NoSlip.h"
#include "PSM_UBB.h"
#include "utility/BoundaryCondition.h"
#include "utility/ParticleUtility.h"
#include "utility/PipingEvaluators.h"

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
const FlagUID Velocity_Flag("Velocity");
const FlagUID NoSlip_Flag("NoSlip");
const FlagUID FreeSlip_Flag("FreeSlip");

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
   const bool pressureDrivenFlow             = physicsParameters.getParameter< bool >("pressureDrivenFlow");
   const real_t hydraulicGradient            = physicsParameters.getParameter< real_t >("hydraulicGradient");
   const real_t outflowVelocity_SI           = physicsParameters.getParameter< real_t >("outflowVelocity_SI");
   const uint_t maxSuctionTimeStep           = physicsParameters.getParameter< uint_t >("maxSuctionTimeStep");
   const real_t densityFluid_SI              = physicsParameters.getParameter< real_t >("densityFluid_SI");
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
   const bool movingBucket          = bucketParameters.getParameter< bool >("movingBucket");
   const real_t bucketFinalForce_SI = bucketParameters.getParameter< real_t >("finalForce_SI");
   const real_t bucketFinalForce =
      bucketFinalForce_SI * dt_SI * dt_SI / (densityFluid_SI * dx_SI * dx_SI * dx_SI * dx_SI);
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(bucketFinalForce)

   Config::BlockHandle particlesParameters     = cfgFile->getBlock("Particles");
   const std::string particleInFileName        = particlesParameters.getParameter< std::string >("inFileName");
   const real_t densityParticle_SI             = particlesParameters.getParameter< real_t >("densityParticle_SI");
   const real_t particleFrictionCoefficient    = particlesParameters.getParameter< real_t >("frictionCoefficient");
   const real_t particleRestitutionCoefficient = particlesParameters.getParameter< real_t >("restitutionCoefficient");
   const uint_t particleNumSubCycles           = particlesParameters.getParameter< uint_t >("numSubCycles");
   const Vector3< uint_t > numSubBlocks        = particlesParameters.getParameter< Vector3< uint_t > >("numSubBlocks");
   const bool useLubricationCorrection         = particlesParameters.getParameter< bool >("useLubricationCorrection");
   const real_t poissonsRatio                  = particlesParameters.getParameter< real_t >("poissonsRatio");
   const real_t particleDensityRatio           = densityParticle_SI / densityFluid_SI;
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particleDensityRatio)
   const Vector3< real_t > observationDomainFraction =
      particlesParameters.getParameter< Vector3< real_t > >("observationDomainFraction");
   const Vector3< real_t > observationDomainSize(real_c(domainSize[0]) * observationDomainFraction[0],
                                                 real_c(domainSize[1]) * observationDomainFraction[1],
                                                 real_c(domainSize[2]) * observationDomainFraction[2]);
   const uint_t numPreSteps   = particlesParameters.getParameter< uint_t >("numPreSteps");
   const bool movingParticles = particlesParameters.getParameter< bool >("movingParticles");
   const real_t kappa         = real_c(2) * (real_c(1) - poissonsRatio) / (real_c(2) - poissonsRatio);
   const real_t particleCollisionTime =
      real_t(100); // TODO: check why it works with this value but not with 10, depends on dt
   // Set useOpenMP always to true, will only have an effect once MesaPD is built with OpenMP
   bool useOpenMP = true;

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
                                       simulationDomain.zMax() * (real_t(0.5) + (real_t(1) - bucketSizeFraction[2])));
   // Bucket has full size in z direction to be long enough to move downwards (if movingBucket is true)
   const Vector3< real_t > boxEdgeLength(simulationDomain.xMax() * bucketSizeFraction[0],
                                         simulationDomain.yMax() * bucketSizeFraction[1], simulationDomain.zMax());
   const uint_t boxUid = createBox(*ps, boxPosition, boxEdgeLength);

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
   std::function< void(void) > syncCall = [&ps, &rpdDomain, &syncNextNeighborFunc]() {
      syncNextNeighborFunc(*ps, *rpdDomain);
   };

   real_t timeStepSizeRPD = real_t(1) / real_t(particleNumSubCycles);
   mesa_pd::kernel::VelocityVerletPreForceUpdate integratorPreForce(timeStepSizeRPD);
   // Wrap update of the total displacement around the pre force integration (which updates the position)
   auto vvIntegratorPreForce = [&integratorPreForce](const size_t idx, auto& ac) {
      auto oldPos = ac.getPosition(idx);
      integratorPreForce(idx, ac);
      ac.setTotalDisplacement(idx, ac.getTotalDisplacement(idx) + (ac.getPosition(idx) - oldPos).length());
   };
   mesa_pd::kernel::VelocityVerletPostForceUpdate vvIntegratorPostForce(timeStepSizeRPD);
   mesa_pd::kernel::LinearSpringDashpot collisionResponse(2);
   collisionResponse.setFrictionCoefficientDynamic(0, 0, particleFrictionCoefficient);
   // No friction between spheres and (artificial) bounding planes
   collisionResponse.setFrictionCoefficientDynamic(0, 1, particleFrictionCoefficient);

   // Set stiffness and damping globally
   const real_t maxParticleRadius  = maxParticleDiameter / real_t(2);
   const real_t invMaxParticleMass = real_t(1) / (real_t(4) / real_t(3) * maxParticleRadius * maxParticleRadius *
                                                  maxParticleRadius * math::pi * particleDensityRatio);
   collisionResponse.setStiffnessAndDamping(0, 0, particleRestitutionCoefficient, particleCollisionTime, kappa,
                                            real_t(1) / (invMaxParticleMass + invMaxParticleMass));
   collisionResponse.setStiffnessAndDamping(0, 1, particleRestitutionCoefficient, particleCollisionTime, kappa,
                                            real_t(1) / (invMaxParticleMass));

   WALBERLA_LOG_INFO_ON_ROOT("stiffnessN = " << collisionResponse.getStiffnessN(0, 0))
   WALBERLA_LOG_INFO_ON_ROOT("stiffnessT = " << collisionResponse.getStiffnessT(0, 0))
   WALBERLA_LOG_INFO_ON_ROOT("dampingN = " << collisionResponse.getDampingN(0, 0))
   WALBERLA_LOG_INFO_ON_ROOT("dampingT = " << collisionResponse.getDampingT(0, 0))

   mesa_pd::kernel::AssocToBlock assoc(blocks->getBlockForestPointer());
   mesa_pd::mpi::ReduceProperty reduceProperty;
   mesa_pd::mpi::ReduceContactHistory reduceAndSwapContactHistory;
   mesa_pd::kernel::InsertParticleIntoLinkedCells ipilc;
   lbm_mesapd_coupling::ResetHydrodynamicForceTorqueKernel resetHydrodynamicForceTorque;
   lbm_mesapd_coupling::AverageHydrodynamicForceTorqueKernel averageHydrodynamicForceTorque;
   lbm_mesapd_coupling::LubricationCorrectionKernel lubricationCorrectionKernel(
      viscosity, [](real_t r) { return (real_t(0.001 + real_t(0.00007) * r)) * r; });

   real_t linkedCellWidth = 1.01_r * maxParticleDiameter;
   mesa_pd::data::LinkedCells linkedCells(rpdDomain->getUnionOfLocalAABBs().getExtended(linkedCellWidth),
                                          linkedCellWidth);

   // Settle particles to the bucket wall
   settleParticles(numPreSteps, accessor, ps, *rpdDomain, linkedCells, syncNextNeighborFunc, collisionResponse,
                   particleDensityRatio, particleRestitutionCoefficient, kappa, gravitationalAcceleration,
                   particleCollisionTime, useOpenMP);

   // Evaluate initial soil properties
   UpliftSubsidenceEvaluator upliftSubsidenceEvaluator(accessor, ps, boxPosition, boxEdgeLength, observationDomainSize);

   real_t seepageLength = computeSeepageLength(accessor, ps, boxPosition, boxEdgeLength);
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(seepageLength)

   // TODO: check formula again
   const real_t pressureDifference = hydraulicGradient * gravitationalAcceleration * seepageLength;
   // TODO: check formula again
   const real_t densityDifference = pressureDifference * real_t(3); // d_p = d_rho * c_s^2
   const real_t outflowVelocity   = outflowVelocity_SI * dt_SI / dx_SI;
   if (pressureDrivenFlow)
   {
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(pressureDifference)
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(densityDifference)
   }
   else { WALBERLA_LOG_DEVEL_VAR_ON_ROOT(outflowVelocity) }

   ///////////////////////
   // ADD DATA TO BLOCKS //
   ////////////////////////

   // Setting initial PDFs to nan helps to detect bugs in the initialization/BC handling
   // Setting pdf values to nan (for debugging purposes) does not work because they propagate inside the domain from
   // above the bucket
   BlockDataID pdfFieldID     = field::addToStorage< PdfField_T >(blocks, "pdf field (fzyx)", real_t(0), field::fzyx);
   BlockDataID pdfFieldGPUID  = gpu::addGPUFieldToStorage< PdfField_T >(blocks, pdfFieldID, "pdf field GPU");
   BlockDataID densityFieldID = field::addToStorage< DensityField_T >(blocks, "density field", real_t(0), field::fzyx);
   BlockDataID velFieldID  = field::addToStorage< VelocityField_T >(blocks, "velocity field", real_t(0), field::fzyx);
   BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");
   BlockDataID BFieldID    = field::addToStorage< BField_T >(blocks, "B field", real_t(0), field::fzyx);

   // Boundary handling
   assembleBoundaryBlock(domainSize, boxPosition, boxEdgeLength, movingBucket, periodicInY, pressureDrivenFlow);

   auto boundariesCfgFile = Config();
   boundariesCfgFile.readParameterFile("boundaries.prm");
   auto boundariesConfig = boundariesCfgFile.getBlock("Boundaries");

   // map boundaries into the LBM simulation
   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldID, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldID, Fluid_Flag);
   lbm::PSM_Density density0_bc(blocks, pdfFieldGPUID, real_t(1.0));
   density0_bc.fillFromFlagField< FlagField_T >(blocks, flagFieldID, Density0_Flag, Fluid_Flag);
   lbm::PSM_Density density1_bc(blocks, pdfFieldGPUID, real_t(1.0));
   if (maxSuctionTimeStep == 0) { density1_bc.bc_density_ = real_t(1.0) - densityDifference; }
   density1_bc.fillFromFlagField< FlagField_T >(blocks, flagFieldID, Density1_Flag, Fluid_Flag);
   lbm::PSM_UBB velocity_bc(blocks, pdfFieldGPUID, real_t(0.0), real_t(0.0), real_t(0.0));
   if (maxSuctionTimeStep == 0) { velocity_bc.bc_velocity_2_ = outflowVelocity; }
   velocity_bc.fillFromFlagField< FlagField_T >(blocks, flagFieldID, Velocity_Flag, Fluid_Flag);
   lbm::PSM_NoSlip noSlip(blocks, pdfFieldGPUID);
   noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldID, NoSlip_Flag, Fluid_Flag);
   lbm::PSM_FreeSlip freeSlip(blocks, pdfFieldGPUID);
   freeSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldID, FreeSlip_Flag, Fluid_Flag);

   ///////////////
   // TIME LOOP //
   ///////////////

   // Map particles into the fluid domain
   ParticleAndVolumeFractionSoA_T< Weighting > particleAndVolumeFractionSoA(blocks, omega);
   auto psmSelector = PSMSelector(movingBucket);
   PSMSweepCollectionGPU psmSweepCollection(blocks, accessor, psmSelector, particleAndVolumeFractionSoA, numSubBlocks);
   BoxFractionMappingGPU boxFractionMapping(blocks, accessor, boxUid, boxEdgeLength, particleAndVolumeFractionSoA,
                                            psmSelector);
   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      psmSweepCollection.particleMappingSweep(&(*blockIt));
      if (movingBucket) { boxFractionMapping(&(*blockIt)); }
   }

   real_t e_init = computeVoidRatio(blocks, BFieldID, particleAndVolumeFractionSoA.BFieldID, flagFieldID, Fluid_Flag,
                                    accessor, ps, real_t(1) / omega);
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
   // sendDirectlyFromGPU should be true if GPUDirect is available
   gpu::communication::UniformGPUScheme< Stencil_T > com(blocks, false, false);
   com.addPackInfo(make_shared< PackInfo_T >(pdfFieldGPUID));
   auto communication = std::function< void() >([&]() { com.communicate(); });

   SweepTimeloop timeloop(blocks->getBlockStorage(), timeSteps);

   timeloop.addFuncBeforeTimeStep(RemainingTimeLogger(timeloop.getNrOfTimeSteps()), "Remaining Time Logger");

   pystencils::PSM_MacroGetter getterSweep(BFieldID, densityFieldID, pdfFieldID, velFieldID, real_t(0.0), real_t(0.0),
                                           real_t(0.0));
   // VTK output
   if (vtkSpacing != uint_t(0))
   {
      // Bucket slice
      auto bucketVtkOutput = make_shared< mesa_pd::vtk::ParticleVtkOutput >(ps);
      bucketVtkOutput->addOutput< mesa_pd::data::SelectParticleUid >("uid");
      bucketVtkOutput->addOutput< mesa_pd::data::SelectParticleLinearVelocity >("velocity");
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
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleTotalDisplacement >("totalDisplacement");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleLinearVelocity >("velocity");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleCollisionForceNorm >("collisionForceNorm");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleInteractionRadius >("radius");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleOldHydrodynamicForce >("forceHydro");
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
      pdfFieldVTK->setSamplingResolution(outputParameters.getParameter< real_t >("resolutionSpacing"));
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
   timeloop.add() << Sweep(deviceSyncWrapper(velocity_bc.getSweep()), "Boundary Handling (Velocity)");
   timeloop.add() << Sweep(deviceSyncWrapper(noSlip.getSweep()), "Boundary Handling (NoSlip)");
   timeloop.add() << Sweep(deviceSyncWrapper(freeSlip.getSweep()), "Boundary Handling (FreeSlip)");

   // PSM kernel
   pystencils::PSMSweep PSMSweep(particleAndVolumeFractionSoA.BsFieldID, particleAndVolumeFractionSoA.BFieldID,
                                 particleAndVolumeFractionSoA.particleForcesFieldID,
                                 particleAndVolumeFractionSoA.particleVelocitiesFieldID, pdfFieldGPUID, real_t(0.0),
                                 real_t(0.0), real_t(0.0), omega);

   // Particle mapping overwrites the fields, therefore bucket mapping has to be called second
   timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollection.particleMappingSweep), "Particle mapping");
   if (movingBucket) { timeloop.add() << Sweep(deviceSyncWrapper(boxFractionMapping), "Bucket mapping"); }
   timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollection.setParticleVelocitiesSweep), "Set particle velocities");
   timeloop.add() << Sweep(deviceSyncWrapper(PSMSweep), "PSM sweep");
   timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollection.reduceParticleForcesSweep), "Reduce particle forces");

   WcTimingPool timeloopTiming;

   for (uint_t timeStep = 0; timeStep < timeSteps; ++timeStep)
   {
      timeloop.singleStep(timeloopTiming);
      // If pressure difference did not yet reach the limit, decrease the pressure on the right hand side
      density1_bc.bc_density_ = std::max(real_t(1.0) - densityDifference,
                                         density1_bc.bc_density_ - densityDifference / real_t(maxSuctionTimeStep));
      velocity_bc.bc_velocity_2_ =
         std::min(outflowVelocity, velocity_bc.bc_velocity_2_ + outflowVelocity / real_t(maxSuctionTimeStep));

      if (movingParticles)
      {
         ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, assoc, *accessor);
         reduceProperty.operator()< mesa_pd::HydrodynamicForceTorqueNotification >(*ps);

         if (timeStep == 0)
         {
            lbm_mesapd_coupling::InitializeHydrodynamicForceTorqueForAveragingKernel
               initializeHydrodynamicForceTorqueForAveragingKernel;
            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor,
                                initializeHydrodynamicForceTorqueForAveragingKernel, *accessor);
         }

         // This call also sets the old hydrodynamic forces and torques that are used for the visualization
         ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, averageHydrodynamicForceTorque,
                             *accessor);

         for (auto subCycle = uint_t(0); subCycle < particleNumSubCycles; ++subCycle)
         {
            timeloopTiming["RPD"].start();

            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, vvIntegratorPreForce, *accessor);
            if (movingBucket) { vvIntegratorPreForce(accessor->uidToIdx(boxUid), *accessor); }
            syncCall();

            linkedCells.clear();
            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectAll(), *accessor, ipilc, *accessor, linkedCells);

            if (useLubricationCorrection)
            {
               // lubrication correction (currently only used for sphere-sphere interaction)
               linkedCells.forEachParticlePairHalf(
                  useOpenMP, SphereSphereSelector(), *accessor,
                  [&lubricationCorrectionKernel, &rpdDomain](const size_t idx1, const size_t idx2, auto& ac) {
                     mesa_pd::collision_detection::AnalyticContactDetection acd;
                     acd.getContactThreshold() = lubricationCorrectionKernel.getNormalCutOffDistance();
                     mesa_pd::kernel::DoubleCast double_cast;
                     mesa_pd::mpi::ContactFilter contact_filter;
                     if (double_cast(idx1, idx2, ac, acd, ac))
                     {
                        if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *rpdDomain))
                        {
                           double_cast(acd.getIdx1(), acd.getIdx2(), ac, lubricationCorrectionKernel, ac,
                                       acd.getContactNormal(), acd.getPenetrationDepth());
                        }
                     }
                  },
                  *accessor);
            }

            // Reset the sum over the collision force norms
            ps->forEachParticle(
               useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor,
               [](const size_t idx, auto& ac) { ac.setCollisionForceNorm(idx, real_t(0)); }, *accessor);

            // collision response
            linkedCells.forEachParticlePairHalf(
               useOpenMP, mesa_pd::kernel::ExcludeInfiniteInfinite(), *accessor,
               [&collisionResponse, &rpdDomain, timeStepSizeRPD, particleRestitutionCoefficient, particleCollisionTime,
                kappa](const size_t idx1, const size_t idx2, auto& ac) {
                  mesa_pd::collision_detection::AnalyticContactDetection acd;
                  mesa_pd::kernel::DoubleCast double_cast;
                  mesa_pd::mpi::ContactFilter contact_filter;
                  if (double_cast(idx1, idx2, ac, acd, ac))
                  {
                     if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *rpdDomain))
                     {
                        auto oldForce1 = ac.getForce(acd.getIdx1());
                        auto oldForce2 = ac.getForce(acd.getIdx2());
                        collisionResponse(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(),
                                          acd.getContactNormal(), acd.getPenetrationDepth(), timeStepSizeRPD);
                        // TODO: divide CollisionForceNorm by the number of collisions
                        ac.setCollisionForceNorm(acd.getIdx1(), ac.getCollisionForceNorm(acd.getIdx1()) +
                                                                   (oldForce1 - ac.getForce(idx1)).length());
                        ac.setCollisionForceNorm(acd.getIdx2(), ac.getCollisionForceNorm(acd.getIdx2()) +
                                                                   (oldForce2 - ac.getForce(idx2)).length());
                     }
                  }
               },
               *accessor);

            reduceAndSwapContactHistory(*ps);

            // add hydrodynamic force
            lbm_mesapd_coupling::AddHydrodynamicInteractionKernel addHydrodynamicInteraction;
            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, addHydrodynamicInteraction,
                                *accessor);

            ps->forEachParticle(
               useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor,
               [particleDensityRatio, gravitationalAcceleration](const size_t idx, auto& ac) {
                  mesa_pd::addForceAtomic(idx, ac,
                                          Vector3< real_t >(real_t(0), real_t(0),
                                                            -(particleDensityRatio - real_c(1)) * ac.getVolume(idx) *
                                                               gravitationalAcceleration));
               },
               *accessor);

            reduceProperty.operator()< mesa_pd::ForceTorqueNotification >(*ps);

            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, vvIntegratorPostForce, *accessor);
            if (movingBucket)
            {
               const size_t bucketIdx = accessor->uidToIdx(boxUid);
               accessor->setForce(bucketIdx, Vector3(real_t(0), real_t(0),
                                                     -std::min(bucketFinalForce, real_t(timeStep) * bucketFinalForce /
                                                                                    real_t(maxSuctionTimeStep))));
               vvIntegratorPostForce(bucketIdx, *accessor);
            }
            syncCall();

            timeloopTiming["RPD"].end();
         }
      }

      ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectAll(), *accessor, resetHydrodynamicForceTorque, *accessor);

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
         upliftSubsidenceEvaluator(hydraulicGradient * real_t(timeStep) / real_t(maxSuctionTimeStep), accessor, ps);
      }
      timeloopTiming["Uplift/Subsidence evaluation"].end();
   }

   timeloopTiming.logResultOnRoot();

   return EXIT_SUCCESS;
}

} // namespace piping
} // namespace walberla

int main(int argc, char** argv) { walberla::piping::main(argc, argv); }
