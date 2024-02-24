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
//! \file FlowAroundCylinder.cpp
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
#include "core/MemoryUsage.h"
#include "core/SharedFunctor.h"
#include "core/debug/CheckFunctions.h"
#include "core/logging/Initialization.h"
#include "core/logging/Logging.h"
#include "core/math/Vector3.h"
#include "core/mpi/Environment.h"
#include "core/mpi/MPIManager.h"
#include "core/mpi/Reduce.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/timing/TimingPool.h"

#include "field/AddToStorage.h"
#include "field/CellCounter.h"
#include "field/FlagField.h"
#include "field/StabilityChecker.h"
#include "field/adaptors/AdaptorCreators.h"
#include "field/vtk/VTKWriter.h"

#include "geometry/InitBoundaryHandling.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#   include "gpu/AddGPUFieldToStorage.h"
#   include "gpu/DeviceSelectMPI.h"
#   include "gpu/ErrorChecking.h"
#   include "gpu/FieldCopy.h"
#   include "gpu/HostFieldAllocator.h"
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

#include "stencil/D3Q27.h"
#include "timeloop/SweepTimeloop.h"
#include "vtk/VTKOutput.h"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "Cylinder.h"
#include "Evaluation.h"
#include "FlowAroundCylinderInfoHeader.h"
#include "FlowAroundCylinderStaticDefines.h"
#include "Setup.h"
#include "Types.h"

using namespace walberla;

using StorageSpecification_T = lbm::FlowAroundCylinderStorageSpecification;
using Stencil_T              = StorageSpecification_T::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;

using PdfField_T           = lbm_generated::PdfField< StorageSpecification_T >;
using FlagField_T          = FlagField< uint8_t >;
using BoundaryCollection_T = lbm::FlowAroundCylinderBoundaryCollection< FlagField_T >;

using SweepCollection_T = lbm::FlowAroundCylinderSweepCollection;

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
using BlockExclusionFunction     = SetupBlockForest::BlockExclusionFunction;

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

   const memory_t memoryPerBlock = numeric_cast< memory_t >((setup.cellsPerBlock[0] + uint_t(2) * setup.numGhostLayers)  *
                                                               (setup.cellsPerBlock[1] + uint_t(2) * setup.numGhostLayers)  *
                                                               (setup.cellsPerBlock[2] + uint_t(2) * setup.numGhostLayers)) * memoryPerCell;

   forest->addRefinementSelectionFunction(refinementSelectionFunctions);
   forest->addBlockExclusionFunction(blockExclusionFunction);
   forest->addWorkloadMemorySUIDAssignmentFunction(
      std::bind(workloadMemoryAndSUIDAssignment, std::placeholders::_1, memoryPerBlock, std::cref(setup)));

   forest->init(AABB(real_c(0), real_c(0), real_c(0), real_c(setup.domainSize[0]), real_c(setup.domainSize[1]), real_c(setup.domainSize[2])),
                setup.xBlocks, setup.yBlocks, setup.zBlocks, setup.periodic[0], setup.periodic[1], setup.periodic[2]);

   MPIManager::instance()->useWorldComm();
   forest->balanceLoad(blockforest::StaticLevelwiseCurveBalanceWeighted(), numberOfProcesses);
   // forest->balanceLoad(blockforest::StaticLevelwiseCurveBalanceWeighted(true),
   //                     numberOfProcesses, real_t(0), processMemoryLimit, true, false);

   if (outputSetupForest)
   {
      forest->writeVTKOutput("domain_decomposition");
      forest->writeCSV("process_distribution");
   }

   WALBERLA_LOG_INFO_ON_ROOT("SetupBlockForest created successfully:\n" << *forest);

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
   //      auto bf = std::make_shared< BlockForest >(uint_c(MPIManager::instance()->rank()), sbffile.c_str(), true,
   //      false);
   //
   //      auto sbf = std::make_shared< StructuredBlockForest >(bf, setup.xCells, setup.yCells, setup.zCells);
   //      sbf->createCellBoundingBoxes();
   //
   //      return sbf;
   //   }

   WALBERLA_LOG_INFO_ON_ROOT("Creating the block structure ...");

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

void setupBoundaryCylinder(const std::shared_ptr< StructuredBlockForest >& sbfs, const BlockDataID flagFieldID,
                           const FlagUID& obstacleFlagUID, const std::function< bool(const Vector3< real_t >&) >& isObstacleBoundary)
{
   for (auto bIt = sbfs->begin(); bIt != sbfs->end(); ++bIt)
   {
      Block& b             = dynamic_cast< Block& >(*bIt);
      auto flagField       = b.getData< FlagField_T >(flagFieldID);
      uint8_t obstacleFlag = flagField->registerFlag(obstacleFlagUID);
      consistentlySetBoundary(sbfs, b, flagField, obstacleFlag, isObstacleBoundary);
   }
}

class InflowProfile
{
 public:

   InflowProfile( const real_t velocity, const real_t H, const uint_t inflowProfile ) : velocity_(velocity), H_( H ), inflowProfile_( inflowProfile )
   {
      uTerm_ = ( real_c(16.0) * velocity );
      HTerm_ = ( real_c(1.0) / ( H * H * H * H ) );
      tConstTerm_ = uTerm_ * HTerm_;
   }

   Vector3< real_t > operator()( const Cell& pos, const shared_ptr< StructuredBlockForest >& SbF, IBlock& block ) const;

 private:

   real_t velocity_;
   real_t H_;
   real_t uTerm_;
   real_t HTerm_;
   real_t tConstTerm_;
   uint_t inflowProfile_;
}; // class InflowProfile

Vector3< real_t > InflowProfile::operator()( const Cell& pos, const shared_ptr< StructuredBlockForest >& SbF, IBlock& block ) const
{
   if (inflowProfile_ == 1)
   {
      Cell globalCell;
      real_t x;
      real_t y;
      real_t z;
      const uint_t level = SbF->getLevel(block);

      SbF->transformBlockLocalToGlobalCell(globalCell, block, pos);
      SbF->getCellCenter(x, y, z, globalCell, level);

      return Vector3< real_t >(tConstTerm_ * y * z * ( H_ - y ) * ( H_ - z ), real_c(0.0), real_c(0.0) );
   }
   else if (inflowProfile_ == 2)
   {
      return Vector3< real_t >(velocity_, real_c(0.0), real_c(0.0) );
   }
   else
   {
      WALBERLA_ABORT("Inflow profile not implemented")
   }
}

} // namespace

//////////////////////
// Parameter Struct //
//////////////////////

int main(int argc, char** argv)
{
   mpi::Environment env(argc, argv);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   gpu::selectDeviceBasedOnMpiRank();
#endif

   shared_ptr< Config > config = make_shared< Config >();
   config->readParameterFile(argv[1]);

   logging::configureLogging(config);

   ///////////////////////
   /// PARAMETER INPUT ///
   ///////////////////////

   // read general simulation parameters
   auto parameters = config->getOneBlock("Parameters");

   const real_t kinViscosity        = parameters.getParameter< real_t >("kinViscosity");
   const real_t rho                 = parameters.getParameter< real_t >("rho");
   const real_t referenceVelocity   = parameters.getParameter< real_t >("inflowVelocity");
   const real_t maxLatticeVelocity  = parameters.getParameter< real_t >("maxLatticeVelocity");
   const uint_t inflowProfile       = parameters.getParameter< uint_t >("inflowProfile");

   const real_t diameterCylinder = parameters.getParameter< real_t >("diameterCylinder");
   const real_t coarseMeshSize   = parameters.getParameter< real_t >("coarseMeshSize");
   const uint_t timesteps        = parameters.getParameter< uint_t >("timesteps");

   const real_t dt               = maxLatticeVelocity / referenceVelocity * coarseMeshSize;
   const real_t latticeViscosity = kinViscosity / coarseMeshSize / coarseMeshSize * dt;
   const real_t omega            = real_c(real_c(1.0) / (real_c(3.0) * latticeViscosity + real_c(0.5)));
   const real_t uMean            = inflowProfile == 1 ? real_c(4.0) / real_c(9.0) * maxLatticeVelocity : maxLatticeVelocity;
   const real_t reynoldsNumber   = (uMean * (diameterCylinder / coarseMeshSize)) / latticeViscosity;

   // read domain parameters
   auto domainParameters     = config->getOneBlock("DomainSetup");
   const uint_t refinementLevels         = domainParameters.getParameter< uint_t >("refinementLevels");
   const Vector3< uint_t > cellsPerBlock = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
   const Vector3< real_t > domainSize    = domainParameters.getParameter< Vector3< real_t > >("domainSize");
   const Vector3< bool > periodic        = domainParameters.getParameter< Vector3< bool > >("periodic");

   const uint_t numGhostLayers = uint_c(2);
   const real_t fineMeshSize   = real_c(coarseMeshSize) / real_c(1 << refinementLevels);

   WALBERLA_LOG_INFO_ON_ROOT("Diameter of the Cylinder is resolved with " << diameterCylinder / fineMeshSize << " lattice cells.")

   auto loggingParameters         = config->getOneBlock("Logging");
   bool writeSetupForestAndReturn = loggingParameters.getParameter< bool >("writeSetupForestAndReturn", false);
   if (uint_c(MPIManager::instance()->numProcesses()) > 1) writeSetupForestAndReturn = false;

   for (uint_t i = 0; i < 3; ++i)
   {
      auto cells = uint_c(std::ceil((domainSize[i] / coarseMeshSize) / real_c(cellsPerBlock[i]))) * cellsPerBlock[i];
      if (cells > uint_c(std::ceil(domainSize[i] / coarseMeshSize)))
      {
         WALBERLA_LOG_WARNING_ON_ROOT("Total cells in direction "
                                      << i << ": " << uint_c(domainSize[i] / coarseMeshSize)
                                      << " is not dividable by cellsPerBlock in that direction. Domain was extended")
      }
   }

   const uint_t xBlocks = uint_c(std::ceil((domainSize[0] / coarseMeshSize) / real_c(cellsPerBlock[0])));
   const uint_t yBlocks = uint_c(std::ceil((domainSize[1] / coarseMeshSize) / real_c(cellsPerBlock[1])));
   const uint_t zBlocks = uint_c(std::ceil((domainSize[2] / coarseMeshSize) / real_c(cellsPerBlock[2])));

   Setup setup;

   setup.xBlocks = xBlocks;
   setup.yBlocks = yBlocks;
   setup.zBlocks = zBlocks;

   setup.xCells = xBlocks * cellsPerBlock[0];
   setup.yCells = yBlocks * cellsPerBlock[1];
   setup.zCells = zBlocks * cellsPerBlock[2];

   setup.cellsPerBlock = cellsPerBlock;
   setup.domainSize = domainSize;
   setup.periodic = periodic;

   setup.numGhostLayers = numGhostLayers;

   setup.H = domainSize[2];
   setup.L = domainSize[0];

   setup.cylinderXPosition    = parameters.getParameter< real_t >("cylinderXPosition");
   setup.cylinderYPosition    = parameters.getParameter< real_t >("cylinderYPosition");
   setup.cylinderRadius       = diameterCylinder / real_c(2.0);
   setup.circularCrossSection = parameters.getParameter< bool >("circularCrossSection");

   setup.evaluateForceComponents                     = false;
   setup.nbrOfEvaluationPointsForCoefficientExtremas = 100;

   setup.evaluatePressure = parameters.getParameter< bool >("evaluatePressure");
   setup.pAlpha           = parameters.getParameter< Vector3< real_t > >("pAlpha", Vector3< real_t >(real_c(0.45), real_c(0.2), real_c(0.205)));
   setup.pOmega = parameters.getParameter< Vector3< real_t > >("pOmega", Vector3< real_t >(real_c(0.55), real_c(0.2), real_c(0.205)));

   setup.evaluateStrouhal = parameters.getParameter< bool >("evaluateStrouhal");
   setup.pStrouhal = parameters.getParameter< Vector3< real_t > >("pStrouhal", Vector3< real_t >(real_c(1), real_c(0.325), real_c(0.205)));

   setup.kinViscosity      = kinViscosity;
   setup.rho               = rho;
   setup.inflowVelocity    = referenceVelocity;
   setup.uMean             = uMean;
   setup.dx                = coarseMeshSize;
   setup.dt                = dt;

   const uint_t valuesPerCell   = (Stencil_T::Q + VelocityField_T::F_SIZE + uint_c(2) * ScalarField_T::F_SIZE);
   const uint_t sizePerValue    = sizeof(PdfField_T::value_type);
   const memory_t memoryPerCell = memory_t(valuesPerCell * sizePerValue + uint_c(1));
   const memory_t processMemoryLimit = parameters.getParameter< memory_t >("processMemoryLimit", memory_t(512)) * memory_t(1024 * 1024);

   ///////////////////////////
   /// Refinement ///
   ///////////////////////////

   blockforest::RefinementSelectionFunctions refinementSelectionFunctions;

   blockforest::AABBRefinementSelection aabbRefinementSelection(config->getOneBlock("AABBRefinementSelection"));
   refinementSelectionFunctions.add(aabbRefinementSelection);
   const real_t cylinderRefinementBuffer = parameters.getParameter< real_t >("cylinderRefinementBuffer", real_t(0));

   Cylinder cylinder(setup);
   CylinderBlockExclusion cylinderBlockExclusion(cylinder);

   CylinderRefinementSelection cylinderRefinementSelection(cylinder, refinementLevels, cylinderRefinementBuffer);
   refinementSelectionFunctions.add(cylinderRefinementSelection);

   ///////////////////////////
   /// CREATE BLOCK FOREST ///
   ///////////////////////////

   if (writeSetupForestAndReturn)
   {
      std::string sbffile = "sbfFlowAroundCylinder.bfs";

      std::ostringstream infoString;
      infoString << "You have selected the option of just creating the block structure (= domain decomposition) and "
                    "saving the result to file\n"
                    "by specifying the output file name \'"
                 << sbffile << "\' AND also specifying \'saveToFile\'.\n";

      if (MPIManager::instance()->numProcesses() > 1)
         WALBERLA_ABORT(infoString.str() << "In this mode you need to start " << argv[0] << " with just one process!")

      WALBERLA_LOG_INFO_ON_ROOT(infoString.str() << "Creating the block structure ...")

      const uint_t numberProcesses = domainParameters.getParameter< uint_t >("numberProcesses");

      shared_ptr< SetupBlockForest > sforest =
         createSetupBlockForest(refinementSelectionFunctions, cylinderBlockExclusion, setup, numberProcesses,
                                memoryPerCell, processMemoryLimit, true);
      sforest->saveToFile(sbffile.c_str());

      WALBERLA_LOG_INFO_ON_ROOT("Benchmark run data:"
                                "\n- simulation parameters:"
                                "\n   + collision model:  "
                                << infoCollisionOperator << "\n   + stencil:          " << infoStencil
                                << "\n   + streaming:        " << infoStreamingPattern
                                << "\n   + compressible:     " << (StorageSpecification_T::compressible ? "yes" : "no")
                                << "\n   + mesh levels:      " << refinementLevels + uint_c(1)
                                << "\n   + resolution:       " << coarseMeshSize << " - on the coarsest grid"
                                << "\n   + resolution:       " << fineMeshSize << " - on the finest grid"
                                << "\n- simulation properties:"
                                   "\n   + H:                   "
                                << setup.H << " [m]"
                                << "\n   + L:                   " << setup.L << " [m]"
                                << "\n   + cylinder pos.(x):    " << setup.cylinderXPosition << " [m]"
                                << "\n   + cylinder pos.(y):    " << setup.cylinderYPosition << " [m]"
                                << "\n   + cylinder radius:     " << setup.cylinderRadius << " [m]"
                                << "\n   + circular profile:    " << (setup.circularCrossSection ? "yes" : "no (= box)")
                                << "\n   + kin. viscosity:      " << setup.kinViscosity << " [m^2/s] (" << setup.kinViscosity
                                << " - on the coarsest grid)"
                                << "\n   + omega:               " << omega << " on the coarsest grid"
                                << "\n   + rho:                 " << setup.rho << " [kg/m^3]"
                                << "\n   + inflow velocity:     " << setup.inflowVelocity << " [m/s]"
                                << "\n   + lattice velocity:    " << maxLatticeVelocity
                                << "\n   + Reynolds number:     " << reynoldsNumber
                                << "\n   + dt (coarsest grid):  " << setup.dt << " [s]"
                                << "\n   + #time steps:         " << (real_t(1) / setup.dt) << " (for 1s of real time)")

      logging::Logging::printFooterOnStream();
      return EXIT_SUCCESS;
   }

   auto blocks = createStructuredBlockForest(refinementSelectionFunctions, cylinderBlockExclusion, setup, memoryPerCell,
                                             processMemoryLimit);

   ////////////////////////////////////
   /// CREATE AND INITIALIZE FIELDS ///
   ////////////////////////////////////

   // create fields
   const StorageSpecification_T StorageSpec = StorageSpecification_T();

   IDs ids;
   ids.avgVelField      = field::addToStorage<VelocityField_T>(blocks, "average velocity", real_t(0.0), field::fzyx, numGhostLayers);
   ids.avgVelSqrField   = field::addToStorage<VelocityField_T>(blocks, "average velocity squared", real_t(0.0), field::fzyx, numGhostLayers);
   ids.avgPressureField = field::addToStorage<ScalarField_T>(blocks, "average pressure", real_t(0.0), field::fzyx, numGhostLayers);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   auto allocator = make_shared< gpu::HostFieldAllocator< real_t > >();
   ids.pdfField =      lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, numGhostLayers, field::fzyx);
   ids.velocityField = field::addToStorage< VelocityField_T >(blocks, "velocity", real_c(0.0), field::fzyx, numGhostLayers);
   ids.densityField =  field::addToStorage< ScalarField_T >(blocks, "density", setup.rho, field::fzyx, numGhostLayers);
   ids.flagField =     field::addFlagFieldToStorage< FlagField_T >(blocks, "Boundary Flag Field", uint_c(3));

   ids.pdfFieldGPU =      lbm_generated::addGPUPdfFieldToStorage< PdfField_T >(blocks, pdfFieldID, StorageSpec, "pdfs on GPU", true);
   ids.velocityFieldGPU = gpu::addGPUFieldToStorage< VelocityField_T >(blocks, velFieldID, "velocity on GPU", true);
   ids.densityFieldGPU =  gpu::addGPUFieldToStorage< ScalarField_T >(blocks, densityFieldID, "density on GPU", true);

   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#else
   ids.pdfField =      lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, numGhostLayers, field::fzyx);
   ids.velocityField = field::addToStorage< VelocityField_T >(blocks, "vel", real_c(0.0), field::fzyx, numGhostLayers);
   ids.densityField =  field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, numGhostLayers);
   ids.flagField =     field::addFlagFieldToStorage< FlagField_T >(blocks, "Boundary Flag Field", uint_c(3));
#endif

   WALBERLA_MPI_BARRIER()

   const Cell innerOuterSplit =
      Cell(parameters.getParameter< Vector3< cell_idx_t > >("innerOuterSplit", Vector3< cell_idx_t >(1, 1, 1)));
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const Vector3< int64_t > gpuBlockSize = parameters.getParameter< Vector3< int64_t > >("gpuBlockSize");
   SweepCollection_T sweepCollection(blocks, ids.pdfFieldGPU, ids.densityFieldGPU, ids.velocityFieldGPU, gpuBlockSize[0],
                                     gpuBlockSize[1], gpuBlockSize[2], omega, innerOuterSplit);
   for (auto& block : *blocks)
   {
      sweepCollection.initialise(&block, cell_idx_c(numGhostLayers - uint_c(1)));
   }
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
#else
   SweepCollection_T sweepCollection(blocks, ids.pdfField, ids.densityField, ids.velocityField, omega, innerOuterSplit);
   for (auto& block : *blocks)
   {
      sweepCollection.initialise(&block, cell_idx_c(numGhostLayers));
   }
#endif

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_LOG_INFO_ON_ROOT("Setting up communication")

   std::shared_ptr< NonUniformGPUScheme< CommunicationStencil_T > > nonUniformCommunication =
      std::make_shared< NonUniformGPUScheme< CommunicationStencil_T > >(blocks);
   std::shared_ptr< NonuniformGeneratedGPUPdfPackInfo< GPUPdfField_T > > nonUniformPackInfo =
      lbm_generated::setupNonuniformGPUPdfCommunication< GPUPdfField_T >(blocks, ids.pdfFieldGPU);
   nonUniformCommunication->addPackInfo(nonUniformPackInfo);
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#else
   WALBERLA_LOG_INFO_ON_ROOT("Setting up communication...")
   auto nonUniformCommunication = std::make_shared< NonUniformBufferedScheme< CommunicationStencil_T > >(blocks);
   auto nonUniformPackInfo      = lbm_generated::setupNonuniformPdfCommunication< PdfField_T >(blocks, ids.pdfField);
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
   const FlagUID obstacleFlagUID("Obstacle");

   auto boundariesConfig   = config->getBlock("Boundaries");
   geometry::initBoundaryHandling< FlagField_T >(*blocks, ids.flagField, boundariesConfig);
   setupBoundaryCylinder(blocks, ids.flagField, obstacleFlagUID, cylinder);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, ids.flagField, fluidFlagUID, cell_idx_c(0));

   std::function< real_t(const Cell&, const Cell&, const shared_ptr< StructuredBlockForest >&, IBlock&) >
      wallDistanceFunctor = wallDistance(cylinder);

   InflowProfile const velocityCallback{maxLatticeVelocity, setup.H, inflowProfile};
   std::function< Vector3< real_t >(const Cell&, const shared_ptr< StructuredBlockForest >&, IBlock&) >
      velocity_initialisation = velocityCallback;

   const real_t omegaFinestLevel = lbm_generated::relaxationRateScaling(omega, refinementLevels);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   BoundaryCollection_T boundaryCollection(blocks, ids.flagField, ids.pdfFieldGPU, fluidFlagUID, omegaFinestLevel,
                                           wallDistanceFunctor, velocity_initialisation, ids.pdfField);
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#else
   BoundaryCollection_T boundaryCollection(blocks, ids.flagField, ids.pdfField, fluidFlagUID, omegaFinestLevel,
                                           wallDistanceFunctor, velocity_initialisation);
#endif
   WALBERLA_MPI_BARRIER()
   WALBERLA_LOG_INFO_ON_ROOT("BOUNDARY HANDLING done")

   //////////////////////////////////
   /// SET UP SWEEPS AND TIMELOOP ///
   //////////////////////////////////
   WALBERLA_LOG_INFO_ON_ROOT("Start SWEEPS AND TIMELOOP")
   // flow evaluation
   auto EvaluationParameters             = config->getOneBlock("Evaluation");
   const uint_t evaluationCheckFrequency = EvaluationParameters.getParameter< uint_t >("evaluationCheckFrequency");
   const uint_t rampUpTime               = EvaluationParameters.getParameter< uint_t >("rampUpTime");
   const bool evaluationLogToStream      = EvaluationParameters.getParameter< bool >("logToStream");
   const bool evaluationLogToFile        = EvaluationParameters.getParameter< bool >("logToFile");
   const std::string evaluationFilename  = EvaluationParameters.getParameter< std::string >("filename");

   std::function< void() > getMacroFields = [&]() {
      for (auto& block : *blocks)
      {
         sweepCollection.calculateMacroscopicParameters(&block);
      }
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpu::fieldCpy< VelocityField_T, gpu::GPUField< real_t > >(blocks, ids.velocityField, ids.velocityFieldGPU);
      gpu::fieldCpy< ScalarField_T, gpu::GPUField< real_t > >(blocks, ids.densityField, ids.densityFieldGPU);
      WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
      WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif
   };

   std::function< void() > getPdfField = [&]() {
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpu::fieldCpy< PdfField_T, GPUPdfField_T >(blocks, ids.pdfField, ids.pdfFieldGPU);
      WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
      WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif
   };

   shared_ptr< Evaluation > evaluation(new Evaluation(blocks, evaluationCheckFrequency, rampUpTime, getMacroFields, getPdfField, ids,
      fluidFlagUID, obstacleFlagUID, setup, evaluationLogToStream, evaluationLogToFile, evaluationFilename));

   // create time loop
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);
   std::shared_ptr< lbm_generated::BasicRecursiveTimeStepGPU< GPUPdfField_T, SweepCollection_T, BoundaryCollection_T > >
      LBMRefinement;

   LBMRefinement = std::make_shared<
      lbm_generated::BasicRecursiveTimeStepGPU< GPUPdfField_T, SweepCollection_T, BoundaryCollection_T > >(
      blocks, ids.pdfFieldGPU, sweepCollection, boundaryCollection, nonUniformCommunication, nonUniformPackInfo);
   LBMRefinement->addPostBoundaryHandlingBlockFunction(evaluation->forceCalculationFunctor());
   LBMRefinement->addRefinementToTimeLoop(timeloop);
#else
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);
   std::shared_ptr< lbm_generated::BasicRecursiveTimeStep< PdfField_T, SweepCollection_T, BoundaryCollection_T > >
      LBMRefinement;

   LBMRefinement =
      std::make_shared< lbm_generated::BasicRecursiveTimeStep< PdfField_T, SweepCollection_T, BoundaryCollection_T > >(
         blocks, ids.pdfField, sweepCollection, boundaryCollection, nonUniformCommunication, nonUniformPackInfo);
   LBMRefinement->addPostBoundaryHandlingBlockFunction(evaluation->forceCalculationFunctor());
   LBMRefinement->addRefinementToTimeLoop(timeloop);
#endif
   //////////////////
   /// VTK OUTPUT ///
   //////////////////
   WALBERLA_LOG_INFO_ON_ROOT("SWEEPS AND TIMELOOP done")

   auto VTKWriter                 = config->getOneBlock("VTKWriter");
   const uint_t vtkWriteFrequency = VTKWriter.getParameter< uint_t >("vtkWriteFrequency", 0);
   const bool writeVelocity       = VTKWriter.getParameter< bool >("velocity");
   const bool writeDensity        = VTKWriter.getParameter< bool >("density");
   const bool writeAverageFields  = VTKWriter.getParameter< bool >("averageFields", false);
   const bool writeFlag           = VTKWriter.getParameter< bool >("flag");
   const bool writeOnlySlice      = VTKWriter.getParameter< bool >("writeOnlySlice", true);
   const bool amrFileFormat       = VTKWriter.getParameter< bool >("amrFileFormat", false);
   const bool oneFilePerProcess   = VTKWriter.getParameter< bool >("oneFilePerProcess", false);

   auto finalDomain = blocks->getDomain();
   if (vtkWriteFrequency > 0)
   {
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false,
                                                      "vtk_FlowAroundCylinder", "simulation_step", false, true, true,
                                                      false, 0, amrFileFormat, oneFilePerProcess);

      vtkOutput->addBeforeFunction([&]() {
         for (auto& block : *blocks)
         {
            sweepCollection.calculateMacroscopicParameters(&block);
         }

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         gpu::fieldCpy< VelocityField_T, gpu::GPUField< real_t > >(blocks, ids.velocityField, ids.velocityFieldGPU);
         gpu::fieldCpy< ScalarField_T, gpu::GPUField< real_t > >(blocks, ids.densityField, ids.densityFieldGPU);
#endif
      });

      if (writeOnlySlice)
      {
         const AABB sliceAABB(finalDomain.xMin(), finalDomain.yMin(), finalDomain.center()[2] - coarseMeshSize,
                              finalDomain.xMax(), finalDomain.yMax(), finalDomain.center()[2] + coarseMeshSize);
         vtkOutput->addCellInclusionFilter(vtk::AABBCellFilter(sliceAABB));
      }

      if (writeVelocity)
      {
         auto velWriter = make_shared< field::VTKWriter< VelocityField_T, float32 > >(ids.velocityField, "velocity");
         vtkOutput->addCellDataWriter(velWriter);
      }
      if (writeDensity)
      {
         auto densityWriter = make_shared< field::VTKWriter< ScalarField_T, float32 > >(ids.densityField, "density");
         vtkOutput->addCellDataWriter(densityWriter);
      }
      if (writeAverageFields)
      {
         auto avgVelWriter = make_shared< field::VTKWriter< VelocityField_T, float32 > >(ids.avgVelField, "avgVelocity");
         vtkOutput->addCellDataWriter(avgVelWriter);
         auto avgVelSqrWriter = make_shared< field::VTKWriter< VelocityField_T, float32 > >(ids.avgVelSqrField, "avgVelocitySqr");
         vtkOutput->addCellDataWriter(avgVelSqrWriter);
         auto avgPressureWriter = make_shared< field::VTKWriter< ScalarField_T, float32 > >(ids.avgPressureField, "avgPressure");
         vtkOutput->addCellDataWriter(avgPressureWriter);
      }
      {
         auto densityWriter = make_shared< field::VTKWriter< ScalarField_T, float32 > >(ids.densityField, "density");
         vtkOutput->addCellDataWriter(densityWriter);
      }


      if (writeFlag)
      {
         auto flagWriter = make_shared< field::VTKWriter< FlagField_T > >(ids.flagField, "flag");
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
      auto checkFunction = [](PdfField_T::value_type value) { return value < math::abs(PdfField_T::value_type(10)); };
      timeloop.addFuncAfterTimeStep(makeSharedFunctor(field::makeStabilityChecker< PdfField_T, FlagField_T >(
                                       config, blocks, ids.pdfField, ids.flagField, fluidFlagUID, checkFunction)),
                                    "Stability check");
   }

   timeloop.addFuncBeforeTimeStep(SharedFunctor< Evaluation >(evaluation), "evaluation");
   timeloop.addFuncBeforeTimeStep(evaluation->resetForceFunctor(), "evaluation: reset force");

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
   const lbm_generated::PerformanceEvaluation< FlagField_T > performance(blocks, ids.flagField, fluidFlagUID);
   field::CellCounter< FlagField_T > fluidCells(blocks, ids.flagField, fluidFlagUID);
   fluidCells();

   WALBERLA_LOG_INFO_ON_ROOT("Blocks created: " << blocks->getNumberOfBlocks())
   for (uint_t level = 0; level <= refinementLevels; level++)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Level " << level << " Blocks: " << blocks->getNumberOfBlocks(level))
   }
   WALBERLA_LOG_INFO_ON_ROOT(
      "Benchmark run data:"
      "\n- simulation parameters:"
      "\n   + collision model:  "
      << infoCollisionOperator << "\n   + stencil:          " << infoStencil << "\n   + streaming:        "
      << infoStreamingPattern << "\n   + compressible:     " << (StorageSpecification_T::compressible ? "yes" : "no")
      << "\n   + mesh levels:      " << refinementLevels + uint_c(1) << "\n   + resolution:       " << coarseMeshSize
      << " - on the coarsest grid"
      << "\n   + resolution:       " << fineMeshSize << " - on the finest grid"
      << "\n- simulation properties:"
         "\n   + fluid cells:         "
      << fluidCells.numberOfCells() << " (in total on all levels)"
      << "\n   + H:                   " << setup.H << " [m]"
      << "\n   + L:                   " << setup.L << " [m]"
      << "\n   + cylinder pos.(x):    " << setup.cylinderXPosition << " [m]"
      << "\n   + cylinder pos.(y):    " << setup.cylinderYPosition << " [m]"
      << "\n   + cylinder radius:     " << setup.cylinderRadius << " [m]"
      << "\n   + circular profile:    " << (setup.circularCrossSection ? "yes" : "no (= box)")
      << "\n   + kin. viscosity:      " << setup.kinViscosity << " [m^2/s] (on the coarsest grid)"
      << "\n   + omega:               " << omega << " (on the coarsest grid)"
      << "\n   + rho:                 " << setup.rho << " [kg/m^3]"
      << "\n   + inflow velocity:     " << setup.inflowVelocity << " [m/s]"
      << "\n   + lattice velocity:    " << maxLatticeVelocity
      << "\n   + Reynolds number:     " << reynoldsNumber
      << "\n   + dx (coarsest grid):  " << setup.dx << " [m]"
      << "\n   + dt (coarsest grid):  " << setup.dt << " [s]"
      << "\n   + #time steps:         " << timeloop.getNrOfTimeSteps() << " on the coarsest grid, "
      << (real_t(1) / setup.dt) << " for 1s of real time)"
      << "\n   + simulation time:     " << (real_c(timeloop.getNrOfTimeSteps()) * setup.dt) << " [s]")

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