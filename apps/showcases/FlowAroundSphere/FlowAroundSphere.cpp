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
//! \file FlowAroundSphere.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#include "blockforest/AABBRefinementSelection.h"
#include "blockforest/SetupBlockForest.h"
#include "blockforest/StructuredBlockForest.h"
#include "blockforest/communication/NonUniformBufferedScheme.h"
#include "blockforest/loadbalancing/StaticCurve.h"

#include "core/Abort.h"
#include "core/DataTypes.h"
#include "core/SharedFunctor.h"
#include "core/debug/CheckFunctions.h"
#include "core/logging/Logging.h"
#include "core/logging/Initialization.h"
#include "core/math/Vector3.h"
#include "core/mpi/Environment.h"
#include "core/mpi/MPIManager.h"
#include "core/MemoryUsage.h"
#include "core/mpi/Reduce.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/timing/TimingPool.h"

#include "field/AddToStorage.h"
#include "field/CellCounter.h"
#include "field/FlagField.h"
#include "field/StabilityChecker.h"
#include "field/adaptors/AdaptorCreators.h"
#include "field/iterators/FieldIterator.h"
#include "field/vtk/VTKWriter.h"
#include "field/vtk/FlagFieldCellFilter.h"

#include "geometry/InitBoundaryHandling.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#   include "gpu/AddGPUFieldToStorage.h"
#   include "gpu/DeviceSelectMPI.h"
#   include "gpu/ErrorChecking.h"
#   include "gpu/FieldCopy.h"
#   include "gpu/HostFieldAllocator.h"
#   include "gpu/ParallelStreams.h"
#   include "gpu/communication/NonUniformGPUScheme.h"
#   include "gpu/communication/UniformGPUScheme.h"
#endif

#include "lbm_generated/communication/NonuniformGeneratedPdfPackInfo.h"
#include "lbm_generated/communication/UniformGeneratedPdfPackInfo.h"
#include "lbm_generated/evaluation/PerformanceEvaluation.h"
#include "lbm_generated/field/AddToStorage.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/refinement/BasicRecursiveTimeStep.h"
#include "lbm_generated/refinement/RefinementScaling.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#   include "lbm_generated/gpu/AddToStorage.h"
#   include "lbm_generated/gpu/BasicRecursiveTimeStepGPU.h"
#   include "lbm_generated/gpu/GPUPdfField.h"
#   include "lbm_generated/gpu/NonuniformGeneratedGPUPdfPackInfo.h"
#   include "lbm_generated/gpu/UniformGeneratedGPUPdfPackInfo.h"
#endif

#include "timeloop/SweepTimeloop.h"

#include "vtk/VTKOutput.h"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "Setup.h"
#include "Sphere.h"
#include "Evaluation.h"
#include "FlowAroundSphereInfoHeader.h"
#include "FlowAroundSphereStaticDefines.h"

using namespace walberla;

using StorageSpecification_T = lbm::FlowAroundSphereStorageSpecification;
using Stencil_T              = StorageSpecification_T::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;

using PdfField_T           = lbm_generated::PdfField< StorageSpecification_T >;
using FlagField_T          = FlagField< uint8_t >;
using BoundaryCollection_T = lbm::FlowAroundSphereBoundaryCollection< FlagField_T >;

using SweepCollection_T = lbm::FlowAroundSphereSweepCollection;

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
using GPUPdfField_T = lbm_generated::GPUPdfField< StorageSpecification_T >;
using gpu::communication::NonUniformGPUScheme;
using gpu::communication::UniformGPUScheme;

using lbm_generated::NonuniformGeneratedGPUPdfPackInfo;
using lbm_generated::UniformGeneratedGPUPdfPackInfo;
#else
using PdfField_T = lbm_generated::PdfField< StorageSpecification_T >;
using blockforest::communication::NonUniformBufferedScheme;
using blockforest::communication::UniformBufferedScheme;

using lbm_generated::NonuniformGeneratedPdfPackInfo;
using lbm_generated::UniformGeneratedPdfPackInfo;
#endif

using RefinementSelectionFunctor = SetupBlockForest::RefinementSelectionFunction;
using BlockExclusionFunction = SetupBlockForest::BlockExclusionFunction;

//////////////////////
// Parameter Struct //
//////////////////////

namespace
{
void workloadMemoryAndSUIDAssignment(SetupBlockForest& forest, const memory_t memoryPerBlock, const Setup& setup)
{
   for (auto block = forest.begin(); block != forest.end(); ++block)
   {
      block->setWorkload(numeric_cast< workload_t >(uint_t(1) << block->getLevel()));
      block->setMemory(memoryPerBlock);
   }
}

shared_ptr< SetupBlockForest >
   createSetupBlockForest(const blockforest::RefinementSelectionFunctions& refinementSelectionFunctions,
                          const BlockExclusionFunction& blockExclusionFunction, const Setup& setup,
                          uint_t numberOfProcesses, const memory_t memoryPerCell, const memory_t processMemoryLimit,
                          const bool outputSetupForest)
{
   shared_ptr< SetupBlockForest > forest = make_shared< SetupBlockForest >();

   const memory_t memoryPerBlock = numeric_cast< memory_t >((setup.zCells + uint_t(2) * setup.numGhostLayers) *
                                                            (setup.yCells + uint_t(2) * setup.numGhostLayers) *
                                                            (setup.xCells + uint_t(2) * setup.numGhostLayers)) *
                                   memoryPerCell;

   forest->addRefinementSelectionFunction(refinementSelectionFunctions);
   forest->addBlockExclusionFunction(blockExclusionFunction);
   forest->addWorkloadMemorySUIDAssignmentFunction(
      std::bind(workloadMemoryAndSUIDAssignment, std::placeholders::_1, memoryPerBlock, std::cref(setup)));

   forest->init(AABB(real_c(0), real_c(0), real_c(0), real_c(setup.xCells), real_c(setup.yCells), real_c(setup.zCells)),
                setup.xBlocks, setup.yBlocks, setup.zBlocks, false, false, true);

   MPIManager::instance()->useWorldComm();
   forest->balanceLoad(blockforest::StaticLevelwiseCurveBalanceWeighted(), numberOfProcesses);
   // forest->balanceLoad(blockforest::StaticLevelwiseCurveBalanceWeighted(true),
   //                     numberOfProcesses, real_t(0), processMemoryLimit, true, false);

   if (outputSetupForest)
   {
      forest->writeVTKOutput("domain_decomposition");
      forest->writeCSV("process_distribution");
   }

   WALBERLA_LOG_INFO_ON_ROOT("SetupBlockForest created successfully:\n" << *forest)

   return forest;
}

shared_ptr< blockforest::StructuredBlockForest >
   createStructuredBlockForest(const blockforest::RefinementSelectionFunctions& refinementSelectionFunctions,
                               const BlockExclusionFunction& blockExclusionFunction, const Setup& setup,
                               const memory_t memoryPerCell, const memory_t processMemoryLimit)
{
   //   if (configBlock.isDefined("sbffile"))
   //   {
   //      std::string sbffile = configBlock.getParameter< std::string >("sbffile");
   //
   //      WALBERLA_LOG_INFO_ON_ROOT("Creating the block structure: loading from file \'" << sbffile << "\' ...");
   //
   //      MPIManager::instance()->useWorldComm();
   //
   //      auto bf = std::make_shared< BlockForest >(uint_c(MPIManager::instance()->rank()), sbffile.c_str(), true, false);
   //
   //      auto sbf = std::make_shared< StructuredBlockForest >(bf, setup.xCells, setup.yCells, setup.zCells);
   //      sbf->createCellBoundingBoxes();
   //
   //      return sbf;
   //   }

   WALBERLA_LOG_INFO_ON_ROOT("Creating the block structure ...")

   shared_ptr< SetupBlockForest > sforest =
      createSetupBlockForest(refinementSelectionFunctions, blockExclusionFunction, setup,
                             uint_c(MPIManager::instance()->numProcesses()), memoryPerCell, processMemoryLimit, false);

   auto bf  = std::make_shared< blockforest::BlockForest >(uint_c(MPIManager::instance()->rank()), *sforest, false);
   auto sbf = std::make_shared< blockforest::StructuredBlockForest >(bf, setup.cellsPerBlock[0], setup.cellsPerBlock[1],
                                                                     setup.cellsPerBlock[2]);
   sbf->createCellBoundingBoxes();

   return sbf;
}

void consistentlySetBoundary(const std::shared_ptr< StructuredBlockForest >& blocks, Block& block,
                             FlagField_T* flagField, const uint8_t flag,
                             const std::function< bool(const Vector3< real_t >&) >& isBoundary)
{
   const uint_t level = blocks->getLevel(block);
   int ghostLayers    = int_c(flagField->nrOfGhostLayers());

   CellInterval cells = flagField->xyzSize();
   cells.expand(cell_idx_c(ghostLayers));

   std::vector< CellInterval > coarseRegions;
   for (auto dir = stencil::D3Q27::beginNoCenter(); dir != stencil::D3Q27::end(); ++dir)
   {
      const auto index = blockforest::getBlockNeighborhoodSectionIndex(dir.cx(), dir.cy(), dir.cz());
      if (block.neighborhoodSectionHasLargerBlock(index))
      {
         CellInterval coarseRegion(cells);
         for (uint_t i = 0; i != 3; ++i)
         {
            const auto c = stencil::c[i][*dir];

            if (c == -1)
               coarseRegion.max()[i] = coarseRegion.min()[i] + cell_idx_c(2 * ghostLayers - 1);
            else if (c == 1)
               coarseRegion.min()[i] = coarseRegion.max()[i] - cell_idx_c(2 * ghostLayers - 1);
         }
         coarseRegions.push_back(coarseRegion);
      }
   }

   for (auto cell = cells.begin(); cell != cells.end(); ++cell)
   {
      bool inCoarseRegion(false);
      for (auto region = coarseRegions.begin(); region != coarseRegions.end() && !inCoarseRegion; ++region)
         inCoarseRegion = region->contains(*cell);

      if (!inCoarseRegion)
      {
         Vector3< real_t > center;
         blocks->getBlockLocalCellCenter(block, *cell, center);
         blocks->mapToPeriodicDomain(center);

         if (isBoundary(center)) { flagField->addFlag(cell->x(), cell->y(), cell->z(), flag); }
      }
      else
      {
         Cell globalCell(*cell);
         blocks->transformBlockLocalToGlobalCell(globalCell, block);

         Cell coarseCell(globalCell);
         for (uint_t i = 0; i < 3; ++i)
         {
            if (coarseCell[i] < cell_idx_t(0)) { coarseCell[i] = -((cell_idx_t(1) - coarseCell[i]) >> 1); }
            else { coarseCell[i] >>= 1; }
         }

         Vector3< real_t > coarseCenter;
         blocks->getCellCenter(coarseCenter, coarseCell, level - uint_t(1));
         blocks->mapToPeriodicDomain(coarseCenter);

         if (isBoundary(coarseCenter)) { flagField->addFlag(cell->x(), cell->y(), cell->z(), flag); }
      }
   }
}

void setupBoundaryFlagField(const std::shared_ptr< StructuredBlockForest >& sbfs, const BlockDataID flagFieldID,
                            const std::function< bool(const Vector3< real_t >&) >& isObstacleBoundary)
{
   const FlagUID ubbFlagUID("UBB");
   const FlagUID outflowFlagUID("Outflow");
   const FlagUID freeSlipFlagUID("FreeSlip");
   const FlagUID noSlipFlagUID("NoSlip");

   for (auto bIt = sbfs->begin(); bIt != sbfs->end(); ++bIt)
   {
      Block& b             = dynamic_cast< Block& >(*bIt);
      uint_t level         = b.getLevel();
      auto flagField       = b.getData< FlagField_T >(flagFieldID);
      uint8_t ubbFlag      = flagField->registerFlag(ubbFlagUID);
      uint8_t outflowFlag  = flagField->registerFlag(outflowFlagUID);
      uint8_t freeSlipFlag = flagField->registerFlag(freeSlipFlagUID);
      uint8_t noSlipFlag   = flagField->registerFlag(noSlipFlagUID);

      consistentlySetBoundary(sbfs, b, flagField, noSlipFlag, isObstacleBoundary);

      for (auto cIt = flagField->beginWithGhostLayerXYZ(2); cIt != flagField->end(); ++cIt)
      {
         Cell localCell = cIt.cell();
         Cell globalCell(localCell);
         sbfs->transformBlockLocalToGlobalCell(globalCell, b);
         if (globalCell.x() < 0) { flagField->addFlag(localCell, ubbFlag); }
         else if (globalCell.x() >= cell_idx_c(sbfs->getNumberOfXCells(level)))
         {
            flagField->addFlag(localCell, outflowFlag);
         }
         else if (globalCell.y() < 0 || globalCell.y() >= cell_idx_c(sbfs->getNumberOfYCells(level)))
         {
            flagField->addFlag(localCell, freeSlipFlag);
         }
      }
   }
}
}


//////////////////////
// Parameter Struct //
//////////////////////

int main(int argc, char** argv)
{
   mpi::Environment env( argc, argv );
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   gpu::selectDeviceBasedOnMpiRank();
#endif

   shared_ptr< Config > config = make_shared< Config >();
   config->readParameterFile( argv[1] );

   logging::configureLogging(config);

   ///////////////////////
   /// PARAMETER INPUT ///
   ///////////////////////

   // read general simulation parameters
   auto parameters = config->getOneBlock("Parameters");

   const real_t machNumber = parameters.getParameter< real_t >("machNumber");
   const real_t reynoldsNumber = parameters.getParameter< real_t >("reynoldsNumber");
   const real_t diameterSphere       = parameters.getParameter< real_t >("diameterSphere");
   const real_t coarseMeshSize       = parameters.getParameter< real_t >("coarseMeshSize");
   const uint_t timesteps      = parameters.getParameter< uint_t >("timesteps");
   const uint_t simulationTime      = parameters.getParameter< uint_t >("simulationTime");

   const real_t speedOfSound = real_c(real_c(1.0) / std::sqrt( real_c(3.0) ));
   const real_t referenceVelocity = real_c(machNumber * speedOfSound);
   const real_t viscosity = real_c((referenceVelocity * diameterSphere)  / reynoldsNumber );
   const real_t omega = real_c(real_c(1.0) / (real_c(3.0) * viscosity + real_c(0.5)));
   const real_t referenceTime = real_c(diameterSphere / referenceVelocity);


   // read domain parameters
   auto domainParameters = config->getOneBlock("DomainSetup");
   const uint_t refinementLevels = domainParameters.getParameter< uint_t >("refinementLevels");
   const Vector3< uint_t > cellsPerBlock  = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
   const Vector3< real_t > domainSize = domainParameters.getParameter< Vector3< real_t > >("domainSize");

   const uint_t numGhostLayers = uint_c(2);
   const real_t fineMeshSize = real_c(coarseMeshSize) / real_c( 1 << refinementLevels );
   WALBERLA_LOG_INFO_ON_ROOT("Diameter of the Sphere is resolved with " << diameterSphere / fineMeshSize << " lattice cells.")

   auto loggingParameters         = config->getOneBlock("Logging");
   bool writeSetupForestAndReturn = loggingParameters.getParameter< bool >("writeSetupForestAndReturn", false);
   if (uint_c(MPIManager::instance()->numProcesses()) > 1) writeSetupForestAndReturn = false;

   const uint_t xBlocks = uint_c(std::ceil( (domainSize[0] / coarseMeshSize) / real_c(cellsPerBlock[0])));
   const uint_t yBlocks = uint_c(std::ceil( (domainSize[1] / coarseMeshSize) / real_c(cellsPerBlock[1])));
   const uint_t zBlocks = uint_c(std::ceil( (domainSize[2] / coarseMeshSize) / real_c(cellsPerBlock[2])));

   Setup setup;

   setup.xBlocks = xBlocks;
   setup.yBlocks = yBlocks;
   setup.zBlocks = zBlocks;

   setup.xCells = xBlocks * cellsPerBlock[0];
   setup.yCells = yBlocks * cellsPerBlock[1];
   setup.zCells = zBlocks * cellsPerBlock[2];

   setup.cellsPerBlock = cellsPerBlock;
   setup.numGhostLayers = numGhostLayers;

   setup.sphereXPosition = parameters.getParameter< real_t >("SphereXPosition") / coarseMeshSize;
   setup.sphereYPosition = real_c(setup.yCells) / real_c(2.0);;
   setup.sphereZPosition = real_c(setup.zCells) / real_c(2.0);
   setup.sphereRadius    = (diameterSphere / real_c(2.0)) / coarseMeshSize;

   setup.evaluateForceComponents = false;
   setup.nbrOfEvaluationPointsForCoefficientExtremas = 100;

   setup.evaluatePressure = parameters.getParameter< bool >("evaluatePressure");
   setup.pAlpha = parameters.getParameter< Vector3<real_t> >( "pAlpha" );
   setup.pOmega = parameters.getParameter< Vector3<real_t> >( "pOmega" );

   setup.evaluateStrouhal = parameters.getParameter< bool >("evaluateStrouhal");;
   setup.pStrouhal = parameters.getParameter< Vector3<real_t> >( "pStrouhal");

   setup.viscosity = viscosity;
   setup.rho = real_c(1.0);
   setup.inflowVelocity = referenceVelocity;
   setup.dx = coarseMeshSize;
   setup.dt = 1 / referenceTime;

   const uint_t valuesPerCell   = (Stencil_T::Q + VelocityField_T::F_SIZE + uint_c(2) * ScalarField_T::F_SIZE);
   const uint_t sizePerValue    = sizeof(PdfField_T::value_type);
   const memory_t memoryPerCell = memory_t(valuesPerCell * sizePerValue + uint_c(1));
   const memory_t processMemoryLimit = parameters.getParameter< memory_t >( "processMemoryLimit", memory_t( 512 ) ) * memory_t( 1024 * 1024  );

   ///////////////////////////
   /// Refinement ///
   ///////////////////////////

   blockforest::RefinementSelectionFunctions refinementSelectionFunctions;

   blockforest::AABBRefinementSelection aabbRefinementSelection( config->getOneBlock("AABBRefinementSelection") );
   refinementSelectionFunctions.add( aabbRefinementSelection );
   const real_t SphereRefinementBuffer = parameters.getParameter< real_t >( "SphereRefinementBuffer", real_t(0) );

   Sphere Sphere( setup );
   SphereRefinementSelection SphereRefinementSelection( Sphere, refinementLevels, SphereRefinementBuffer );
   SphereBlockExclusion SphereBlockExclusion( Sphere );

   refinementSelectionFunctions.add( SphereRefinementSelection );

   ///////////////////////////
   /// CREATE BLOCK FOREST ///
   ///////////////////////////

   if (writeSetupForestAndReturn)
   {
      std::string  sbffile = "sbfFlowAroundSphere.bfs";

      std::ostringstream infoString;
      infoString << "You have selected the option of just creating the block structure (= domain decomposition) and saving the result to file\n"
                    "by specifying the output file name \'" << sbffile << "\' AND also specifying \'saveToFile\'.\n";

      if( MPIManager::instance()->numProcesses() > 1 )
         WALBERLA_ABORT( infoString.str() << "In this mode you need to start " << argv[0] << " with just one process!" )

      WALBERLA_LOG_INFO_ON_ROOT( infoString.str() << "Creating the block structure ..." )

      const uint_t numberProcesses = domainParameters.getParameter< uint_t >("numberProcesses");

      shared_ptr< SetupBlockForest > sforest = createSetupBlockForest( refinementSelectionFunctions, SphereBlockExclusion,
                                                                       setup, numberProcesses,
                                                                       memoryPerCell, processMemoryLimit,
                                                                       true );
      sforest->saveToFile( sbffile.c_str() );

      WALBERLA_LOG_INFO_ON_ROOT( "Benchmark run data:"
                                "\n- simulation parameters:"
                                "\n   + collision model:  " << infoCollisionOperator <<
                                "\n   + stencil:          " << infoStencil <<
                                "\n   + streaming:        " << infoStreamingPattern <<
                                "\n   + compressible:     " << ( StorageSpecification_T::compressible ? "yes" : "no" ) <<
                                "\n   + mesh levels:      " << refinementLevels + uint_c(1) <<
                                "\n   + resolution:       " << coarseMeshSize << " - on the coarsest grid)" <<
                                "\n   + resolution:       " << fineMeshSize << " - on the finest grid)" <<
                                "\n- simulation properties:"
                                "\n   + sphere pos.(x):    " << setup.sphereXPosition << " [m]" <<
                                "\n   + sphere pos.(y):    " << setup.sphereYPosition << " [m]" <<
                                "\n   + sphere pos.(z):    " << setup.sphereZPosition << " [m]" <<
                                "\n   + sphere radius:     " << setup.sphereRadius << " [m]" <<
                                "\n   + kin. viscosity:      " << setup.viscosity << " [m^2/s] (" << setup.viscosity << " - on the coarsest grid)" <<
                                "\n   + omega:               " << omega << " - on the coarsest grid)" <<
                                "\n   + rho:                 " << setup.rho << " [kg/m^3]" <<
                                "\n   + inflow velocity:     " << setup.inflowVelocity << " [m/s] (" <<
                                "\n   + Reynolds number:     " << reynoldsNumber <<
                                "\n   + Mach number:         " << machNumber <<
                                "\n   + dx (coarsest grid):  " << setup.dx << " [m]" <<
                                "\n   + dt (coarsest grid):  " << setup.dt << " [s]")

      logging::Logging::printFooterOnStream();
      return EXIT_SUCCESS;
   }

   auto blocks = createStructuredBlockForest(refinementSelectionFunctions, SphereBlockExclusion,
                                             setup, memoryPerCell, processMemoryLimit );


   ////////////////////////////////////
   /// CREATE AND INITIALIZE FIELDS ///
   ////////////////////////////////////

   // create fields
   const StorageSpecification_T StorageSpec = StorageSpecification_T();

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   auto allocator = make_shared< gpu::HostFieldAllocator< real_t > >();
   const BlockDataID pdfFieldID =
      lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, numGhostLayers, field::fzyx);
   const BlockDataID velFieldID =
      field::addToStorage< VelocityField_T >(blocks, "velocity", real_c(0.0), field::fzyx, numGhostLayers);
   const BlockDataID densityFieldID =
      field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, numGhostLayers);
   const BlockDataID flagFieldID =
      field::addFlagFieldToStorage< FlagField_T >(blocks, "Boundary Flag Field", uint_c(2));

   const BlockDataID pdfFieldGPUID =
      lbm_generated::addGPUPdfFieldToStorage< PdfField_T >(blocks, pdfFieldID, StorageSpec, "pdfs on GPU", true);
   const BlockDataID velFieldGPUID =
      gpu::addGPUFieldToStorage< VelocityField_T >(blocks, velFieldID, "velocity on GPU", true);
   const BlockDataID densityFieldGPUID =
      gpu::addGPUFieldToStorage< ScalarField_T >(blocks, densityFieldID, "density on GPU", true);

   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#else
   const BlockDataID pdfFieldID =
      lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, numGhostLayers, field::fzyx);
   const BlockDataID velFieldID =
      field::addToStorage< VelocityField_T >(blocks, "vel", real_c(0.0), field::fzyx, numGhostLayers);
   const BlockDataID densityFieldID =
      field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, numGhostLayers);
   const BlockDataID flagFieldID =
      field::addFlagFieldToStorage< FlagField_T >(blocks, "Boundary Flag Field", uint_c(2));
#endif

   WALBERLA_MPI_BARRIER()

   const Cell innerOuterSplit =
      Cell(parameters.getParameter< Vector3< cell_idx_t > >("innerOuterSplit", Vector3< cell_idx_t >(1, 1, 1)));
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const Vector3< int64_t > gpuBlockSize = parameters.getParameter< Vector3< int64_t > >("gpuBlockSize");
   SweepCollection_T sweepCollection(blocks, pdfFieldGPUID, densityFieldGPUID, velFieldGPUID, gpuBlockSize[0],
                                     gpuBlockSize[1], gpuBlockSize[2], omega, innerOuterSplit);
   for (auto& block : *blocks)
   {
      sweepCollection.initialise(&block, cell_idx_c(numGhostLayers - uint_c(1)));
   }
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
#else
   SweepCollection_T sweepCollection(blocks, pdfFieldID, densityFieldID, velFieldID, omega, innerOuterSplit);
   for (auto& block : *blocks)
   {
      sweepCollection.initialise(&block, cell_idx_c(numGhostLayers));
   }
#endif

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_LOG_INFO_ON_ROOT("Setting up communication")

   std::shared_ptr< NonUniformGPUScheme< CommunicationStencil_T > >nonUniformCommunication = std::make_shared< NonUniformGPUScheme< CommunicationStencil_T > >(blocks);
   std::shared_ptr< NonuniformGeneratedGPUPdfPackInfo< GPUPdfField_T > >nonUniformPackInfo = lbm_generated::setupNonuniformGPUPdfCommunication< GPUPdfField_T >(blocks, pdfFieldGPUID);
   nonUniformCommunication->addPackInfo(nonUniformPackInfo);

   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#else
   WALBERLA_LOG_INFO_ON_ROOT("Setting up communication...")
   auto nonUniformCommunication = std::make_shared< NonUniformBufferedScheme< CommunicationStencil_T > >(blocks);
   auto nonUniformPackInfo      = lbm_generated::setupNonuniformPdfCommunication< PdfField_T >(blocks, pdfFieldID);
   nonUniformCommunication->addPackInfo(nonUniformPackInfo);

#endif
   WALBERLA_MPI_BARRIER()
   WALBERLA_LOG_INFO_ON_ROOT("Setting up communication done")

   /////////////////////////
   /// BOUNDARY HANDLING ///
   /////////////////////////
   WALBERLA_LOG_INFO_ON_ROOT("Start BOUNDARY HANDLING")
   // create and initialize boundary handling
   const FlagUID fluidFlagUID("Fluid");
   setupBoundaryFlagField(blocks, flagFieldID, Sphere);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldID, fluidFlagUID, cell_idx_c(numGhostLayers));

   std::function< real_t(const Cell&, const Cell&, const shared_ptr< StructuredBlockForest >&, IBlock&) >
      wallDistanceFunctor = wallDistance(Sphere);

   const real_t omegaFinestLevel = lbm_generated::relaxationRateScaling(omega, refinementLevels);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   BoundaryCollection_T boundaryCollection(blocks, flagFieldID, pdfFieldGPUID, fluidFlagUID, omegaFinestLevel,
                                           referenceVelocity, wallDistanceFunctor, pdfFieldID);
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#else
   BoundaryCollection_T boundaryCollection(blocks, flagFieldID, pdfFieldID, fluidFlagUID, omegaFinestLevel,
                                           referenceVelocity, wallDistanceFunctor);
#endif
   WALBERLA_MPI_BARRIER()
   WALBERLA_LOG_INFO_ON_ROOT("BOUNDARY HANDLING done")

   //////////////////////////////////
   /// SET UP SWEEPS AND TIMELOOP ///
   //////////////////////////////////
   WALBERLA_LOG_INFO_ON_ROOT("Start SWEEPS AND TIMELOOP")
   // flow evaluation
   auto EvaluationParameters      = config->getOneBlock("Evaluation");
   const uint_t evaluationCheckFrequency = EvaluationParameters.getParameter< uint_t >("evaluationCheckFrequency");
   const uint_t rampUpTime = EvaluationParameters.getParameter< uint_t >("rampUpTime");
   const bool evaluationLogToStream = EvaluationParameters.getParameter< bool >("logToStream");
   const bool evaluationLogToFile = EvaluationParameters.getParameter< bool >("logToFile");
   const std::string evaluationFilename = EvaluationParameters.getParameter< std::string >("filename");

   std::function<void ()> getFields = [&]()
   {
      for (auto& block : *blocks)
      {
         sweepCollection.calculateMacroscopicParameters(&block);
      }

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpu::fieldCpy< PdfField_T, GPUPdfField_T >(blocks, pdfFieldID, pdfFieldGPUID);
      gpu::fieldCpy< VelocityField_T, gpu::GPUField< real_t > >(blocks, velFieldID, velFieldGPUID);
      gpu::fieldCpy< ScalarField_T, gpu::GPUField< real_t > >(blocks, densityFieldID, densityFieldGPUID);
      WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
      WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif
   };

   shared_ptr< Evaluation > evaluation( new Evaluation( blocks, evaluationCheckFrequency, rampUpTime, getFields,
                                                        pdfFieldID, densityFieldID, velFieldID, flagFieldID, fluidFlagUID, FlagUID("NoSlip"),
                                                        setup, evaluationLogToStream, evaluationLogToFile, evaluationFilename));

   // create time loop
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   std::shared_ptr< lbm_generated::BasicRecursiveTimeStepGPU< GPUPdfField_T, SweepCollection_T, BoundaryCollection_T > > LBMRefinement = std::make_shared<lbm_generated::BasicRecursiveTimeStepGPU< GPUPdfField_T, SweepCollection_T, BoundaryCollection_T > >(blocks, pdfFieldGPUID, sweepCollection, boundaryCollection, nonUniformCommunication, nonUniformPackInfo);
   LBMRefinement->addPostBoundaryHandlingBlockFunction(evaluation->forceCalculationFunctor());
   LBMRefinement->addRefinementToTimeLoop(timeloop);
#else
   std::shared_ptr< lbm_generated::BasicRecursiveTimeStep< PdfField_T, SweepCollection_T, BoundaryCollection_T > >
      LBMRefinement;

   LBMRefinement = std::make_shared<
   lbm_generated::BasicRecursiveTimeStep< PdfField_T, SweepCollection_T, BoundaryCollection_T > >(
   blocks, pdfFieldID, sweepCollection, boundaryCollection, nonUniformCommunication, nonUniformPackInfo);
   LBMRefinement->addPostBoundaryHandlingBlockFunction(evaluation->forceCalculationFunctor());
   LBMRefinement->addRefinementToTimeLoop(timeloop);
#endif
   //////////////////
   /// VTK OUTPUT ///
   //////////////////
   WALBERLA_LOG_INFO_ON_ROOT("SWEEPS AND TIMELOOP done")

   auto VTKWriter                 = config->getOneBlock("VTKWriter");
   const uint_t vtkWriteFrequency  = VTKWriter.getParameter< uint_t >("vtkWriteFrequency", 0);
   const bool writeVelocity        = VTKWriter.getParameter< bool >("velocity");
   const bool writeDensity         = VTKWriter.getParameter< bool >("density");
   const bool writeFlag            = VTKWriter.getParameter< bool >("flag");
   const bool writeOnlySlice       = VTKWriter.getParameter< bool >("writeOnlySlice", true);
   const bool amrFileFormat        = VTKWriter.getParameter< bool >("amrFileFormat", false);
   const bool oneFilePerProcess    = VTKWriter.getParameter< bool >("oneFilePerProcess", false);
   const real_t samplingResolution = VTKWriter.getParameter< real_t >("samplingResolution", real_c(-1.0));

   auto finalDomain = blocks->getDomain();
   if (vtkWriteFrequency > 0)
   {
      auto vtkOutput =
         vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_FlowAroundSphere",
                                        "simulation_step", false, true, true, false, 0, amrFileFormat, oneFilePerProcess);

      vtkOutput->addBeforeFunction([&]() {
         for (auto& block : *blocks)
         {
            sweepCollection.calculateMacroscopicParameters(&block);
         }

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         gpu::fieldCpy< VelocityField_T, gpu::GPUField< real_t > >(blocks, velFieldID, velFieldGPUID);
         gpu::fieldCpy< ScalarField_T, gpu::GPUField< real_t > >(blocks, densityFieldID, densityFieldGPUID);
#endif
      });

      vtkOutput->setSamplingResolution(samplingResolution );

      field::FlagFieldCellFilter<FlagField_T> fluidFilter( flagFieldID );
      fluidFilter.addFlag( FlagUID("NoSlip") );
      vtkOutput->addCellExclusionFilter(fluidFilter);


      if (writeOnlySlice)
      {
         const AABB sliceAABB(finalDomain.xMin(), finalDomain.yMin(), finalDomain.center()[2] - coarseMeshSize,
                              finalDomain.xMax(), finalDomain.yMax(), finalDomain.center()[2] + coarseMeshSize);
         vtkOutput->addCellInclusionFilter(vtk::AABBCellFilter(sliceAABB));
      }

      if (writeVelocity)
      {
         auto velWriter = make_shared< field::VTKWriter< VelocityField_T, float32 > >(velFieldID, "velocity");
         vtkOutput->addCellDataWriter(velWriter);
      }
      if (writeDensity)
      {
         auto densityWriter = make_shared< field::VTKWriter< ScalarField_T, float32 > >(densityFieldID, "density");
         vtkOutput->addCellDataWriter(densityWriter);
      }
      if (writeFlag)
      {
         auto flagWriter = make_shared< field::VTKWriter< FlagField_T > >(flagFieldID, "flag");
         vtkOutput->addCellDataWriter(flagWriter);
      }
      timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   }

   // log remaining time
   const real_t remainingTimeLoggerFrequency =
      loggingParameters.getParameter< real_t >("remainingTimeLoggerFrequency", 3.0); // in seconds
   if (uint_c(remainingTimeLoggerFrequency) > 0)
   {
      timeloop.addFuncAfterTimeStep(
         timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
         "remaining time logger");
   }

   // LBM stability check
   auto CheckerParameters      = config->getOneBlock("StabilityChecker");
   const uint_t checkFrequency = CheckerParameters.getParameter< uint_t >("checkFrequency", uint_t(0));
   if (checkFrequency > 0)
   {
      auto checkFunction = [](PdfField_T::value_type value) {  return value < math::abs(PdfField_T::value_type(10)); };
      timeloop.addFuncAfterTimeStep(
         makeSharedFunctor(field::makeStabilityChecker< PdfField_T, FlagField_T >(
            config, blocks, pdfFieldID, flagFieldID, fluidFlagUID, checkFunction)),
         "Stability check");
   }

   timeloop.addFuncBeforeTimeStep( SharedFunctor< Evaluation >(evaluation), "evaluation" );
   timeloop.addFuncBeforeTimeStep( evaluation->resetForceFunctor(), "evaluation: reset force" );

   // WALBERLA_LOG_INFO_ON_ROOT("Execute single timestep to fully complete the preprocessing")
   // Do a single timestep to make sure all setups are completed before benchmarking
   // timeloop.singleStep();
   // timeloop.setCurrentTimeStepToZero();

   WALBERLA_MPI_BARRIER()
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif
   // WALBERLA_LOG_INFO_ON_ROOT("Execute single timestep to fully complete the preprocessing done")

   //////////////////////
   /// RUN SIMULATION ///
   //////////////////////
   const lbm_generated::PerformanceEvaluation< FlagField_T > performance(blocks, flagFieldID, fluidFlagUID);
   field::CellCounter< FlagField_T > fluidCells(blocks, flagFieldID, fluidFlagUID);
   fluidCells();

   WALBERLA_LOG_INFO_ON_ROOT("Blocks created: " << blocks->getNumberOfBlocks())
   for (uint_t level = 0; level <= refinementLevels; level++)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Level " << level << " Blocks: " << blocks->getNumberOfBlocks(level))
   }
   WALBERLA_LOG_INFO_ON_ROOT( "Benchmark run data:"
                             "\n- simulation parameters:"
                             "\n   + collision model:  " << infoCollisionOperator <<
                             "\n   + stencil:          " << infoStencil <<
                             "\n   + streaming:        " << infoStreamingPattern <<
                             "\n   + compressible:     " << ( StorageSpecification_T::compressible ? "yes" : "no" ) <<
                             "\n   + mesh levels:      " << refinementLevels + uint_c(1) <<
                             "\n   + resolution:       " << coarseMeshSize << " - on the coarsest grid)" <<
                             "\n   + resolution:       " << fineMeshSize << " - on the finest grid)" <<
                             "\n- simulation properties:"
                             "\n   + fluid cells:         " << fluidCells.numberOfCells() << " (in total on all levels)" <<
                             "\n   + sphere pos.(x):    " << setup.sphereXPosition << " [m]" <<
                             "\n   + sphere pos.(y):    " << setup.sphereYPosition << " [m]" <<
                             "\n   + sphere pos.(z):    " << setup.sphereZPosition << " [m]" <<
                             "\n   + sphere radius:     " << setup.sphereRadius << " [m]" <<
                             "\n   + kin. viscosity:      " << setup.viscosity << " [m^2/s] (" << setup.viscosity << " - on the coarsest grid)" <<
                             "\n   + omega:               " << omega << " - on the coarsest grid)" <<
                             "\n   + rho:                 " << setup.rho << " [kg/m^3]" <<
                             "\n   + inflow velocity:     " << setup.inflowVelocity << " [m/s] (" <<
                             "\n   + Reynolds number:     " << reynoldsNumber <<
                             "\n   + Mach number:         " << machNumber <<
                             "\n   + dx (coarsest grid):  " << setup.dx << " [m]" <<
                             "\n   + dt (coarsest grid):  " << setup.dt << " [s]" <<
                             "\n   + #time steps:         " << timeloop.getNrOfTimeSteps() << " (on the coarsest grid, " << ( real_t(1) / setup.dt ) << " for 1s of real time)" <<
                             "\n   + simulation time:     " << ( real_c( timeloop.getNrOfTimeSteps() ) * setup.dt ) << " [s]" )

   WALBERLA_LOG_INFO_ON_ROOT("Starting Simulation")
   WcTimingPool timeloopTiming;
   WcTimer simTimer;

   WALBERLA_MPI_BARRIER()
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif

   simTimer.start();
   timeloop.run(timeloopTiming);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif
   WALBERLA_MPI_BARRIER()
   simTimer.end();

   WALBERLA_LOG_INFO_ON_ROOT("Simulation finished")
   real_t time = simTimer.max();
   WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
   performance.logResultOnRoot(timesteps, time);

   const auto reducedTimeloopTiming = timeloopTiming.getReduced();
   WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)

   printResidentMemoryStatistics();

   return EXIT_SUCCESS;
}