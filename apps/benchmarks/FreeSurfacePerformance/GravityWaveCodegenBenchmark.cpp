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
//! \file GravityWaveCodegen.cpp
//! \author Christoph Schwarzmeier <christoph.schwarzmeier@fau.de>
//
// This showcase simulates a standing wave purely governed by gravity, i.e., without surface tension forces. The
// implementation uses an LBM kernel generated with lbmpy.
//======================================================================================================================

#include "blockforest/Initialization.h"

#include "core/Environment.h"

#include "field/Gather.h"

#include "lbm/PerformanceLogger.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/free_surface/LoadBalancing.h"
#include "lbm/free_surface/SurfaceMeshWriter.h"
#include "lbm/free_surface/TotalMassComputer.h"
#include "lbm/free_surface/VtkWriter.h"
#include "lbm/free_surface/bubble_model/Geometry.h"
#include "lbm/free_surface/dynamics/SurfaceDynamicsHandler.h"
#include "lbm/free_surface/surface_geometry/SurfaceGeometryHandler.h"
#include "lbm/free_surface/surface_geometry/Utility.h"
#include "lbm/lattice_model/D3Q19.h"

#include "GravityWaveLatticeModel.h"

#include "SQLProperties.h"
#include "sqlite/SQLite.h"

namespace walberla
{
namespace free_surface
{
namespace GravityWaveCodegen
{
using ScalarField_T          = GhostLayerField< real_t, 1 >;
using VectorField_T          = GhostLayerField< Vector3< real_t >, 1 >;
using VectorFieldFlattened_T = GhostLayerField< real_t, 3 >;

using LatticeModel_T        = lbm::GravityWaveLatticeModel;
using LatticeModelStencil_T = LatticeModel_T::Stencil;
using PdfField_T            = lbm::PdfField< LatticeModel_T >;
using PdfCommunication_T    = blockforest::SimpleCommunication< LatticeModelStencil_T >;

// the geometry computations in SurfaceGeometryHandler require meaningful values in the ghost layers in corner
// directions (flag field and fill level field); this holds, even if the lattice model uses a D3Q19 stencil
using CommunicationStencil_T =
   std::conditional_t< LatticeModel_T::Stencil::D == uint_t(2), stencil::D2Q9, stencil::D3Q27 >;
using Communication_T = blockforest::SimpleCommunication< CommunicationStencil_T >;

using flag_t                        = uint32_t;
using FlagField_T                   = FlagField< flag_t >;
using FreeSurfaceBoundaryHandling_T = FreeSurfaceBoundaryHandling< LatticeModel_T, FlagField_T, ScalarField_T >;

// write each entry in "vector" to line in a file; columns are separated by tabs
template< typename T >
void writeVectorToFile(const std::vector< T >& vector, const std::string& filename);

//----
using StateSweep = walberla::free_surface::BlockStateDetectorSweep< FlagField_T >;
// empty sweep required for using selectors (e.g. StateSweep::fullFreeSurface)
struct emptySweep
{
   void operator()(IBlock*) {}
};
//----

// function describing the global initialization profile
inline real_t initializationProfile(real_t x, real_t amplitude, real_t offset, real_t wavelength)
{
   return amplitude * std::cos(x / wavelength * real_c(2) * math::pi + math::pi) + offset;
}

// evaluate the symmetry of the fill level field along the y-axis located at the center in x-direction
// IMPORTANT REMARK: This implementation is very inefficient, as it gathers the field on a single process to perform the
// evaluation.
template< typename ScalarField_T >
class SymmetryXEvaluator
{
 public:
   SymmetryXEvaluator(const std::weak_ptr< StructuredBlockForest >& blockForest, const ConstBlockDataID& fillFieldID,
                      const Vector3< uint_t >& domainSize, uint_t interval,
                      const std::shared_ptr< real_t >& symmetryNorm)
      : blockForest_(blockForest), fillFieldID_(fillFieldID), domainSize_(domainSize), interval_(interval),
        symmetryNorm_(symmetryNorm), executionCounter_(uint_c(0))
   {
      auto blockForestPtr = blockForest_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(blockForestPtr)
   }

   void operator()()
   {
      auto blockForest = blockForest_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(blockForest)

      ++executionCounter_;

      // only evaluate in given intervals
      if (executionCounter_ % interval_ != uint_c(0) && executionCounter_ != uint_c(1)) { return; }

      real_t fillLevelSum2      = real_c(0); // sum of each cell's squared fill level
      real_t deltaFillLevelSum2 = real_c(0); // sum of each cell's fill level difference to its symmetrical counterpart

      // gather the fill level field on rank 0 (WARNING: simple, but very inefficient)
      std::shared_ptr< ScalarField_T > fillFieldGathered = nullptr;
      WALBERLA_ROOT_SECTION()
      {
         fillFieldGathered =
            std::make_shared< ScalarField_T >(domainSize_[0], domainSize_[1], domainSize_[2], uint_c(0));
      }
      field::gather< ScalarField_T, ScalarField_T >(*fillFieldGathered, blockForest, fillFieldID_);

      WALBERLA_ROOT_SECTION()
      {
         // get field's center-coordinate in x-direction
         const uint_t fieldXCenter = fillFieldGathered->xSize() / uint_c(2);

         WALBERLA_FOR_ALL_CELLS_XYZ(fillFieldGathered, {
            // skip cells in the right half of the field in x-direction, as they are treated
            // as mirrored cells later
            if (x >= cell_idx_c(fieldXCenter)) { continue; }

            // get this cell's x-distance to the field's center
            const uint_t cellDistXCenter = uint_c(std::abs(int_c(fieldXCenter) - int_c(x)));

            // get x-coordinate of (mirrored) cell in the right half of the field
            cell_idx_t fieldRightX = cell_idx_c(fieldXCenter + cellDistXCenter);
            if (fillFieldGathered->xSize() % 2 == uint_c(0))
            {
               fieldRightX -= cell_idx_c(1); // if xSize is even, the blocks on the right must be shifted by -1
            }

            // get fill level
            const real_t fillLevel   = fillFieldGathered->get(x, y, z);
            real_t fillLevelMirrored = real_c(0);
            fillLevelMirrored        = fillFieldGathered->get(fieldRightX, y, z);

            fillLevelSum2 += fillLevel * fillLevel;

            const real_t deltaFill = fillLevel - fillLevelMirrored;
            deltaFillLevelSum2 += deltaFill * deltaFill;
         }) // WALBERLA_FOR_ALL_CELLS_XYZ
      }

      // communicate values among all processes
      mpi::allReduceInplace< real_t >(fillLevelSum2, mpi::SUM);
      mpi::allReduceInplace< real_t >(deltaFillLevelSum2, mpi::SUM);

      // compute L2 norm evaluate symmetry
      *symmetryNorm_ = real_c(std::pow(deltaFillLevelSum2 / fillLevelSum2, real_c(0.5)));
   }

 private:
   std::weak_ptr< StructuredBlockForest > blockForest_;
   ConstBlockDataID fillFieldID_;
   Vector3< uint_t > domainSize_;
   uint_t interval_;
   std::shared_ptr< real_t > symmetryNorm_;
   uint_t executionCounter_;
}; // class SymmetryXEvaluator

// get interface position in y-direction at the specified (global) x-coordinate
template< typename FreeSurfaceBoundaryHandling_T >
class SurfaceYPositionEvaluator
{
 public:
   SurfaceYPositionEvaluator(const std::weak_ptr< const StructuredBlockForest >& blockForest,
                             const std::weak_ptr< const FreeSurfaceBoundaryHandling_T >& freeSurfaceBoundaryHandling,
                             const ConstBlockDataID& fillFieldID, const Vector3< uint_t >& domainSize,
                             cell_idx_t globalXCoordinate, uint_t frequency,
                             const std::shared_ptr< real_t >& surfaceYPosition)
      : blockForest_(blockForest), freeSurfaceBoundaryHandling_(freeSurfaceBoundaryHandling), fillFieldID_(fillFieldID),
        domainSize_(domainSize), globalXCoordinate_(globalXCoordinate), surfaceYPosition_(surfaceYPosition),
        frequency_(frequency), executionCounter_(uint_c(0))
   {}

   void operator()()
   {
      auto blockForest = blockForest_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(blockForest)

      auto freeSurfaceBoundaryHandling = freeSurfaceBoundaryHandling_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(freeSurfaceBoundaryHandling)

      ++executionCounter_;

      // only evaluate in given frequencies
      if (executionCounter_ % frequency_ != uint_c(0) && executionCounter_ != uint_c(1)) { return; }

      const BlockDataID flagFieldID = freeSurfaceBoundaryHandling->getFlagFieldID();
      const typename FreeSurfaceBoundaryHandling_T::FlagInfo_T& flagInfo = freeSurfaceBoundaryHandling->getFlagInfo();

      *surfaceYPosition_ = real_c(0);

      for (auto blockIt = blockForest->begin(); blockIt != blockForest->end(); ++blockIt)
      {
         real_t maxSurfaceYPosition = real_c(0);

         const CellInterval globalSearchInterval(globalXCoordinate_, cell_idx_c(0), cell_idx_c(0), globalXCoordinate_,
                                           cell_idx_c(domainSize_[1]), cell_idx_c(0));

         if (blockForest->getBlockCellBB(*blockIt).overlaps(globalSearchInterval))
         {
            // transform specified global x-coordinate into block local coordinate
            Cell localEvalCell = Cell(globalXCoordinate_, cell_idx_c(0), cell_idx_c(0));
            blockForest->transformGlobalToBlockLocalCell(localEvalCell, *blockIt);

            const FlagField_T* const flagField   = blockIt->template getData< const FlagField_T >(flagFieldID);
            const ScalarField_T* const fillField = blockIt->template getData< const ScalarField_T >(fillFieldID_);

            // searching from top ensures that the interface cell with the greatest y-coordinate is found first
            for (cell_idx_t y = cell_idx_c((flagField)->ySize() - uint_c(1)); y >= cell_idx_t(0); --y)
            {
               if (flagInfo.isInterface(flagField->get(localEvalCell[0], y, cell_idx_c(0))))
               {
                  const real_t fillLevel = fillField->get(localEvalCell[0], y, cell_idx_c(0));

                  // transform local y-coordinate to global coordinate
                  Cell localResultCell = localEvalCell;
                  localResultCell[1]   = y;
                  blockForest->transformBlockLocalToGlobalCell(localResultCell, *blockIt);
                  maxSurfaceYPosition = real_c(localResultCell[1]) + fillLevel;

                  break;
               }
            }
         }

         if (maxSurfaceYPosition > *surfaceYPosition_) { *surfaceYPosition_ = maxSurfaceYPosition; }
      }
      // communicate result among all processes
      mpi::allReduceInplace< real_t >(*surfaceYPosition_, mpi::MAX);
   }

 private:
   std::weak_ptr< const StructuredBlockForest > blockForest_;
   std::weak_ptr< const FreeSurfaceBoundaryHandling_T > freeSurfaceBoundaryHandling_;
   ConstBlockDataID fillFieldID_;
   Vector3< uint_t > domainSize_;
   cell_idx_t globalXCoordinate_;
   std::shared_ptr< real_t > surfaceYPosition_;

   uint_t frequency_;
   uint_t executionCounter_;
}; // class SurfaceYPositionEvaluator

int main(int argc, char** argv)
{
   Environment walberlaEnv(argc, argv);

   if (argc < 2) { WALBERLA_ABORT("Please specify a parameter file as input argument.") }

   // print content of parameter file
   WALBERLA_LOG_INFO_ON_ROOT(*walberlaEnv.config())

   // get block forest parameters from parameter file
   auto blockForestParameters              = walberlaEnv.config()->getOneBlock("BlockForestParameters");
   const Vector3< uint_t > cellsPerBlock   = blockForestParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
   const Vector3< bool > periodicity       = blockForestParameters.getParameter< Vector3< bool > >("periodicity");
   const uint_t loadBalancingFrequency     = blockForestParameters.getParameter< uint_t >("loadBalancingFrequency");
   const bool printLoadBalancingStatistics = blockForestParameters.getParameter< bool >("printLoadBalancingStatistics");

   // get domain parameters from parameter file
   auto domainParameters         = walberlaEnv.config()->getOneBlock("DomainParameters");
   Vector3< uint_t > domainSize   = domainParameters.getParameter< Vector3< uint_t > >("domainSize", Vector3< uint_t >(0, 0, 0));
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(domainSize)

   if (domainSize[0] == uint_t(0))
   {
      const uint_t nrOfProcesses = uint_c(MPIManager::instance()->numProcesses());
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(nrOfProcesses)

      Vector3< uint_t > blocksPerDimension;
      Vector3< uint_t > cellsPerBlockDummy;
      blockforest::calculateCellDistribution(cellsPerBlock, nrOfProcesses, blocksPerDimension, cellsPerBlockDummy);
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blocksPerDimension)
      //WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellsPerBlockDummy)

      // modify resulting block decomposition, s.t. #y-blocks = 1
      if(blocksPerDimension[1] != uint_c(1))
      {
         blocksPerDimension[0] *= blocksPerDimension[1];
         blocksPerDimension[1] = uint_c(1);
      }
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blocksPerDimension)

      // define domain size
      domainSize[0] = blocksPerDimension[0] * cellsPerBlock[0]; // domainWidth;
      domainSize[1] = blocksPerDimension[1] * cellsPerBlock[1]; // uint_c(liquidDepth * real_c(2));
      domainSize[2] = blocksPerDimension[2] * cellsPerBlock[2]; // uint_c(1);
   }
   const real_t liquidDepth = real_t(domainSize[1]) / 2;

   // compute number of blocks as defined by domainSize and cellsPerBlock
   Vector3< uint_t > numBlocks;
   numBlocks[0] = uint_c(std::ceil(real_c(domainSize[0]) / real_c(cellsPerBlock[0])));
   numBlocks[1] = uint_c(std::ceil(real_c(domainSize[1]) / real_c(cellsPerBlock[1])));
   numBlocks[2] = uint_c(std::ceil(real_c(domainSize[2]) / real_c(cellsPerBlock[2])));

   // get number of (MPI) processes
   const uint_t numProcesses = uint_c(MPIManager::instance()->numProcesses());
   WALBERLA_CHECK_LESS_EQUAL(numProcesses, numBlocks[0] * numBlocks[1] * numBlocks[2],
                             "The number of MPI processes is greater than the number of blocks as defined by "
                             "\"domainSize/cellsPerBlock\". This would result in unused MPI processes. Either decrease "
                             "the number of MPI processes or increase \"cellsPerBlock\".")

   //const uint_t nrWaves = domainParameters.getParameter< uint_t >("nrWaves", uint_t(1));
   const uint_t nrWaves = numBlocks[0];
   const real_t wavelength = real_t(domainSize[0]) / real_t(nrWaves);
   const real_t initialAmplitude = real_t(domainSize[1]) / real_t(4);
   //const real_t initialAmplitude = domainParameters.getParameter< real_t >("initialAmplitude");

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numProcesses)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellsPerBlock)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(domainSize)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numBlocks)
   //WALBERLA_LOG_DEVEL_VAR_ON_ROOT(domainWidth);
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(liquidDepth)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(initialAmplitude)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(wavelength)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(nrWaves)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(periodicity)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(loadBalancingFrequency)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(printLoadBalancingStatistics)

   // get physics parameters from parameter file
   auto physicsParameters = walberlaEnv.config()->getOneBlock("PhysicsParameters");
   const uint_t timesteps = physicsParameters.getParameter< uint_t >("timesteps");

   const real_t relaxationRate = physicsParameters.getParameter< real_t >("relaxationRate");
   const real_t viscosity      = real_c(1) / real_c(3) * (real_c(1) / relaxationRate - real_c(0.5));

   const real_t reynoldsNumber = physicsParameters.getParameter< real_t >("reynoldsNumber");
   const real_t waveNumber     = real_c(2) * math::pi / real_c(wavelength);
   const real_t waveFrequency  = reynoldsNumber * viscosity / real_c(wavelength) / initialAmplitude;
   const real_t accelerationY  = -(waveFrequency * waveFrequency) / waveNumber / std::tanh(waveNumber * liquidDepth);
   // accelerate fluid with half of the gravity in positive x-direction (otherwise, we get a perfect seperation of
   // gas and fluid after a few thousand time steps and the free surface does not move anymore)
   const shared_ptr< Vector3< real_t > > acceleration = std::make_shared< Vector3< real_t > >(abs(real_c(accelerationY/2)), accelerationY, real_c(0));

   const bool enableWetting  = physicsParameters.getParameter< bool >("enableWetting");
   const real_t contactAngle = physicsParameters.getParameter< real_t >("contactAngle");

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(reynoldsNumber)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(relaxationRate)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(enableWetting)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(contactAngle)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(timesteps)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(viscosity)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(*acceleration)

   // read model parameters from parameter file
   const auto modelParameters               = walberlaEnv.config()->getOneBlock("ModelParameters");
   const std::string pdfReconstructionModel = modelParameters.getParameter< std::string >("pdfReconstructionModel");
   const std::string pdfRefillingModel      = modelParameters.getParameter< std::string >("pdfRefillingModel");
   const std::string excessMassDistributionModel =
      modelParameters.getParameter< std::string >("excessMassDistributionModel");
   const walberla::free_surface::ExcessMassDistributionModel excessMassModel(excessMassDistributionModel);
   const std::string curvatureModel          = modelParameters.getParameter< std::string >("curvatureModel");
   const bool useSimpleMassExchange          = modelParameters.getParameter< bool >("useSimpleMassExchange");
   const real_t cellConversionThreshold      = modelParameters.getParameter< real_t >("cellConversionThreshold");
   const real_t cellConversionForceThreshold = modelParameters.getParameter< real_t >("cellConversionForceThreshold");
   const bool enableBubbleModel              = modelParameters.getParameter< bool >("enableBubbleModel");
   const bool enableBubbleSplits             = modelParameters.getParameter< bool >("enableBubbleSplits");

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(pdfReconstructionModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(pdfRefillingModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(excessMassDistributionModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(curvatureModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(useSimpleMassExchange)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellConversionThreshold)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellConversionForceThreshold)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(enableBubbleModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(enableBubbleSplits)

   // read evaluation parameters from parameter file
   const auto evaluationParameters      = walberlaEnv.config()->getOneBlock("EvaluationParameters");
   const uint_t performanceLogFrequency = evaluationParameters.getParameter< uint_t >("performanceLogFrequency");
   const uint_t evaluationFrequency     = evaluationParameters.getParameter< uint_t >("evaluationFrequency");
   const std::string filename           = evaluationParameters.getParameter< std::string >("filename");

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(performanceLogFrequency)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(evaluationFrequency)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(filename)

   const auto benchmarkParameters      = walberlaEnv.config()->getOneBlock("BenchmarkParameters");
   const bool benchmark  = benchmarkParameters.getParameter< bool >("benchmark", false);
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(benchmark)

   // create non-uniform block forest (non-uniformity required for load balancing)
   const std::shared_ptr< StructuredBlockForest > blockForest =
      createNonUniformBlockForest(domainSize, cellsPerBlock, numBlocks, periodicity);

   // add force field
   const BlockDataID forceDensityFieldID =
      field::addToStorage< VectorFieldFlattened_T >(blockForest, "Force field", real_c(0), field::fzyx, uint_c(1));

   // create lattice model
   LatticeModel_T latticeModel = LatticeModel_T(forceDensityFieldID, relaxationRate);

   // add pdf field
   const BlockDataID pdfFieldID = lbm::addPdfFieldToStorage(blockForest, "PDF field", latticeModel, field::fzyx);

   // add fill level field (initialized with 0, i.e., gas everywhere)
   const BlockDataID fillFieldID =
      field::addToStorage< ScalarField_T >(blockForest, "Fill level field", real_c(0.0), field::fzyx, uint_c(2));

   // add boundary handling
   const std::shared_ptr< FreeSurfaceBoundaryHandling_T > freeSurfaceBoundaryHandling =
      std::make_shared< FreeSurfaceBoundaryHandling_T >(blockForest, pdfFieldID, fillFieldID);
   const BlockDataID flagFieldID                                      = freeSurfaceBoundaryHandling->getFlagFieldID();
   const typename FreeSurfaceBoundaryHandling_T::FlagInfo_T& flagInfo = freeSurfaceBoundaryHandling->getFlagInfo();

   // samples used in the Monte-Carlo like estimation of the fill level
   const uint_t fillLevelInitSamples = uint_c(100); // actually there will be 101 since 0 is also included

   const uint_t numTotalPoints = (fillLevelInitSamples + uint_c(1)) * (fillLevelInitSamples + uint_c(1));
   const real_t stepsize       = real_c(1) / real_c(fillLevelInitSamples);

   // initialize sine profile such that there is exactly one period in the domain, i.e., with wavelength=domainSize[0];
   // every length is normalized with domainSize[0]
   for (auto blockIt = blockForest->begin(); blockIt != blockForest->end(); ++blockIt)
   {
      ScalarField_T* const fillField = blockIt->getData< ScalarField_T >(fillFieldID);

      WALBERLA_FOR_ALL_CELLS(fillFieldIt, fillField, {
         // cell in block-local coordinates
         const Cell localCell = fillFieldIt.cell();

         // get cell in global coordinates
         Cell globalCell = fillFieldIt.cell();
         blockForest->transformBlockLocalToGlobalCell(globalCell, *blockIt, localCell);

         // Monte-Carlo like estimation of the fill level:
         // create uniformly-distributed sample points in each cell and count the number of points below the sine
         // profile; this fraction of points is used as the fill level to initialize the profile
         uint_t numPointsBelow = uint_c(0);

         for (uint_t xSample = uint_c(0); xSample <= fillLevelInitSamples; ++xSample)
         {
            // value of the sine-function
            const real_t functionValue = initializationProfile(real_c(globalCell[0]) + real_c(xSample) * stepsize,
                                                               initialAmplitude, liquidDepth, real_c(wavelength));

            for (uint_t ySample = uint_c(0); ySample <= fillLevelInitSamples; ++ySample)
            {
               const real_t yPoint = real_c(globalCell[1]) + real_c(ySample) * stepsize;
               // with operator <, a fill level of 1 can not be reached when the line is equal to the cell's top border;
               // with operator <=, a fill level of 0 can not be reached when the line is equal to the cell's bottom
               // border
               if (yPoint < functionValue) { ++numPointsBelow; }
            }
         }

         // fill level is fraction of points below sine profile
         fillField->get(localCell) = real_c(numPointsBelow) / real_c(numTotalPoints);
      }) // WALBERLA_FOR_ALL_CELLS
   }

   // initialize domain boundary conditions from config file
   const auto boundaryParameters = walberlaEnv.config()->getOneBlock("BoundaryParameters");
   freeSurfaceBoundaryHandling->initFromConfig(boundaryParameters);

   // IMPORTANT REMARK: this must be only called after every solid flag has been set; otherwise, the boundary handling
   // might not detect solid flags correctly
   freeSurfaceBoundaryHandling->initFlagsFromFillLevel();

   // communication after initialization
   Communication_T communication(blockForest, flagFieldID, fillFieldID, forceDensityFieldID);
   communication();

   PdfCommunication_T pdfCommunication(blockForest, pdfFieldID);
   pdfCommunication();

   // add bubble model
   std::shared_ptr< bubble_model::BubbleModelBase > bubbleModel = nullptr;
   if (enableBubbleModel)
   {
      const std::shared_ptr< bubble_model::BubbleModel< LatticeModelStencil_T > > bubbleModelDerived =
         std::make_shared< bubble_model::BubbleModel< LatticeModelStencil_T > >(blockForest, enableBubbleSplits);
      bubbleModelDerived->initFromFillLevelField(fillFieldID);
      bubbleModelDerived->setAtmosphere(Cell(domainSize[0] - uint_c(1), domainSize[1] - uint_c(1), uint_c(0)),
                                        real_c(1));

      bubbleModel = std::static_pointer_cast< bubble_model::BubbleModelBase >(bubbleModelDerived);
   }
   else { bubbleModel = std::make_shared< bubble_model::BubbleModelConstantPressure >(real_c(1)); }

   // initialize hydrostatic pressure
   //initHydrostaticPressure< PdfField_T >(blockForest, pdfFieldID, *acceleration, liquidDepth);
   initHydrostaticPressure< PdfField_T >(blockForest, pdfFieldID, Vector3<real_t>(real_c(0), (*acceleration)[1], real_c(0)), liquidDepth);

   // initialize force density field
   initForceDensityFieldCodegen< PdfField_T, FlagField_T, VectorFieldFlattened_T, ScalarField_T >(
      blockForest, forceDensityFieldID, fillFieldID, pdfFieldID, flagFieldID, flagInfo, *acceleration);

   // set density in non-liquid or non-interface cells to 1 (after initializing with hydrostatic pressure)
   setDensityInNonFluidCellsToOne< FlagField_T, PdfField_T >(blockForest, flagInfo, flagFieldID, pdfFieldID);

   // create timeloop
   SweepTimeloop timeloop(blockForest, timesteps);

   const real_t surfaceTension = real_c(0);

   // Laplace pressure = 2 * surface tension * curvature; curvature computation is not necessary with no surface
   // tension
   bool computeCurvature = false;
   if (!realIsEqual(surfaceTension, real_c(0), real_c(1e-14))) { computeCurvature = true; }

   //--
   // add surface geometry handler
   //const SurfaceGeometryHandler< LatticeModel_T, FlagField_T, ScalarField_T, VectorField_T > geometryHandler(
   //   blockForest, freeSurfaceBoundaryHandling, fillFieldID, curvatureModel, computeCurvature, enableWetting,
   //   contactAngle);

   //geometryHandler.addSweeps(timeloop);

   //----
   auto blockStateUpdate = StateSweep(blockForest, flagInfo, flagFieldID);

   // add surface geometry handler
   BlockDataID curvatureFieldID =
      field::addToStorage< ScalarField_T >(blockForest, "Curvature field", real_c(0), field::fzyx, uint_c(1));
   BlockDataID normalFieldID = field::addToStorage< VectorField_T >(
      blockForest, "Normal field", Vector3< real_t >(real_c(0)), field::fzyx, uint_c(1));
   BlockDataID obstacleNormalFieldID = field::addToStorage< VectorField_T >(
      blockForest, "Obstacle normal field", Vector3< real_t >(real_c(0)), field::fzyx, uint_c(1));
   // add field for smoothed fill levels
   BlockDataID smoothFillFieldID =
      field::addToStorage< ScalarField_T >(blockForest, "Smooth fill level field", real_c(0), field::fzyx, uint_c(1));

   // smooth fill level field for decreasing error in finite difference normal and curvature computation (see
   // dissertation of S. Bogner, 2017 (section 4.4.2.1))
   walberla::free_surface::SmoothingSweep< CommunicationStencil_T, FlagField_T, ScalarField_T, VectorField_T >
      smoothingSweep(smoothFillFieldID, fillFieldID, flagFieldID,
                     walberla::free_surface::flagIDs::liquidInterfaceGasFlagIDs, flagInfo.getObstacleIDSet(),
                     enableWetting);
   // IMPORTANT REMARK: SmoothingSweep must be executed on all blocks, because the algorithm works on all liquid,
   // interface and gas cells. This is necessary since the normals are not only computed in interface cells, but also
   // in the neighborhood of interface cells. Therefore, meaningful values for the fill levels of the second
   // neighbors of interface cells are also required in NormalSweep.
   timeloop.add() << Sweep(smoothingSweep, "Sweep: fill level smoothing")
                  << AfterFunction(Communication_T(blockForest, smoothFillFieldID),
                                   "Communication: after smoothing sweep");

   // compute interface normals (using smoothed fill level field)
   walberla::free_surface::NormalSweep< CommunicationStencil_T, FlagField_T, ScalarField_T, VectorField_T > normalSweep(
      normalFieldID, smoothFillFieldID, flagFieldID, walberla::free_surface::flagIDs::interfaceFlagID,
      walberla::free_surface::flagIDs::liquidInterfaceGasFlagIDs, flagInfo.getObstacleIDSet(), true, false, true,
      false);
   timeloop.add() << Sweep(normalSweep, "Sweep: normal computation", StateSweep::fullFreeSurface)
                  << Sweep(emptySweep(), "Empty sweep: normal")
                  << AfterFunction(Communication_T(blockForest, normalFieldID), "Communication: after normal sweep");

   if (computeCurvature)
   {
      // compute interface curvature using finite differences according to Brackbill et al.
      walberla::free_surface::CurvatureSweepFiniteDifferences< CommunicationStencil_T, FlagField_T, ScalarField_T,
                                                               VectorField_T >
         curvSweep(curvatureFieldID, normalFieldID, obstacleNormalFieldID, flagFieldID,
                   walberla::free_surface::flagIDs::interfaceFlagID,
                   walberla::free_surface::flagIDs::liquidInterfaceGasFlagIDs, flagInfo.getObstacleIDSet(), false,
                   real_c(0));
      timeloop.add() << Sweep(curvSweep, "Sweep: curvature computation (finite difference method)",
                              StateSweep::fullFreeSurface)
                     << Sweep(emptySweep(), "Empty sweep: curvature")
                     << AfterFunction(Communication_T(blockForest, curvatureFieldID),
                                      "Communication: after curvature sweep");
   }
   //----
   //--

   //--
   // get fields created by surface geometry handler
   //const ConstBlockDataID curvatureFieldID = geometryHandler.getConstCurvatureFieldID();
   //const ConstBlockDataID normalFieldID    = geometryHandler.getConstNormalFieldID();

   // add boundary handling for standard boundaries and free surface boundaries
   //const SurfaceDynamicsHandler< LatticeModel_T, FlagField_T, ScalarField_T, VectorField_T, true,
   //                              VectorFieldFlattened_T >
   //   dynamicsHandler(blockForest, pdfFieldID, flagFieldID, fillFieldID, forceDensityFieldID, normalFieldID,
   //                   curvatureFieldID, freeSurfaceBoundaryHandling, bubbleModel, pdfReconstructionModel,
   //                   pdfRefillingModel, excessMassDistributionModel, relaxationRate, acceleration, surfaceTension,
   //                   useSimpleMassExchange, cellConversionThreshold, cellConversionForceThreshold);

   //dynamicsHandler.addSweeps(timeloop);

   //----
   // add surface dynamics handler

   // add standard waLBerla boundary handling
   timeloop.add() << Sweep(freeSurfaceBoundaryHandling->getBoundarySweep(), "Sweep: boundary handling",
                           Set< SUID >::emptySet(), StateSweep::onlyGasAndBoundary)
                  << Sweep(emptySweep(), "Empty sweep: boundary handling", StateSweep::onlyGasAndBoundary);

   // add sweep for weighting force in interface cells with fill level and density
   // different version for codegen because pystencils does not support 'Ghostlayerfield<Vector3(), 1>'
   const ForceDensityCodegenSweep< LatticeModel_T, FlagField_T, VectorFieldFlattened_T, ScalarField_T >
      forceDensityCodegenSweep(forceDensityFieldID, pdfFieldID, flagFieldID, fillFieldID, flagInfo, acceleration);
   timeloop.add() << Sweep(forceDensityCodegenSweep, "Sweep: force weighting", Set< SUID >::emptySet(),
                           StateSweep::onlyGasAndBoundary)
                  << Sweep(emptySweep(), "Empty sweep: force weighting", StateSweep::onlyGasAndBoundary)
                  << AfterFunction(Communication_T(blockForest, forceDensityFieldID),
                                   "Communication: after force weighting sweep");

   // sweep for
   // - reconstruction of PDFs in interface cells
   // - streaming of PDFs in interface cells (and liquid cells on the same block)
   // - advection of mass
   // - update bubble volumes
   // - marking interface cells for conversion
   const walberla::free_surface::StreamReconstructAdvectSweep<
      LatticeModel_T, typename FreeSurfaceBoundaryHandling_T::BoundaryHandling_T, FlagField_T,
      typename FreeSurfaceBoundaryHandling_T::FlagInfo_T, ScalarField_T, VectorField_T, true >
      streamReconstructAdvectSweep(surfaceTension, freeSurfaceBoundaryHandling->getHandlingID(), fillFieldID, flagFieldID,
                                   pdfFieldID, normalFieldID, curvatureFieldID, flagInfo, bubbleModel.get(),
                                   pdfReconstructionModel, useSimpleMassExchange, cellConversionThreshold,
                                   cellConversionForceThreshold);
   // sweep acts only on blocks with at least one interface cell (due to StateSweep::fullFreeSurface)
   timeloop.add() << Sweep(streamReconstructAdvectSweep, "Sweep: StreamReconstructAdvect", StateSweep::fullFreeSurface)
                  << Sweep(emptySweep(), "Empty sweep: StreamReconstructAdvect")
                  // do not communicate PDFs here:
                  // - stream on blocks with "StateSweep::fullFreeSurface" was performed here using post-collision PDFs
                  // - stream on other blocks is performed below and should also use post-collision PDFs
                  // => if PDFs were communicated here, the ghost layer of other blocks would have post-stream PDFs
                  << AfterFunction(Communication_T(blockForest, fillFieldID, flagFieldID),
                                   "Communication: after StreamReconstructAdvect sweep")
                  << AfterFunction(blockforest::UpdateSecondGhostLayer< ScalarField_T >(blockForest, fillFieldID),
                                   "Second ghost layer update: after StreamReconstructAdvect sweep (fill level field)")
                  << AfterFunction(blockforest::UpdateSecondGhostLayer< FlagField_T >(blockForest, flagFieldID),
                                   "Second ghost layer update: after StreamReconstructAdvect sweep (flag field)");

   auto lbmSweepGenerated = typename LatticeModel_T::Sweep(pdfFieldID);

   // temporary class for being able to call the LBM collision with operator()
   class CollideSweep
   {
    public:
      CollideSweep(const typename LatticeModel_T::Sweep& sweep) : sweep_(sweep){};

      void operator()(IBlock* const block, const uint_t numberOfGhostLayersToInclude = uint_t(0))
      {
         sweep_.collide(block, numberOfGhostLayersToInclude);
      }

    private:
      typename LatticeModel_T::Sweep sweep_;
   };

   timeloop.add() << Sweep(CollideSweep(lbmSweepGenerated), "Sweep: collision (generated)", StateSweep::fullFreeSurface)
                  << Sweep(lbmSweepGenerated, "Sweep: streamCollide (generated)", StateSweep::onlyLBM)
                  << Sweep(emptySweep(), "Empty sweep: streamCollide (generated)")
                  << AfterFunction(PdfCommunication_T(blockForest, pdfFieldID),
                                   "Communication: after streamCollide (generated)");

   // convert cells
   // - according to the flags from StreamReconstructAdvectSweep (interface -> gas/liquid)
   // - to ensure a closed layer of interface cells (gas/liquid -> interface)
   // - detect and register bubble merges/splits (bubble volumes are already updated in StreamReconstructAdvectSweep)
   // - convert cells and initialize PDFs near inflow boundaries
   const walberla::free_surface::CellConversionSweep< LatticeModel_T, FreeSurfaceBoundaryHandling_T::BoundaryHandling_T,
                                                      ScalarField_T >
      cellConvSweep(freeSurfaceBoundaryHandling->getHandlingID(), pdfFieldID, flagInfo, bubbleModel.get());
   timeloop.add() << Sweep(cellConvSweep, "Sweep: cell conversion", StateSweep::fullFreeSurface)
                  << Sweep(emptySweep(), "Empty sweep: cell conversion")
                  << AfterFunction(PdfCommunication_T(blockForest, pdfFieldID),
                                   "Communication: after cell conversion sweep (PDF field)")
                  // communicate the flag field also in corner directions
                  << AfterFunction(Communication_T(blockForest, flagFieldID),
                                   "Communication: after cell conversion sweep (flag field)")
                  << AfterFunction(blockforest::UpdateSecondGhostLayer< FlagField_T >(blockForest, flagFieldID),
                                   "Second ghost layer update: after cell conversion sweep (flag field)");

   // reinitialize PDFs, i.e., refill cells that were converted from gas to interface
   // - when the flag "convertedFromGasToInterface" has been set (by CellConversionSweep)
   // - according to the method specified with pdfRefillingModel_
   const walberla::free_surface::EquilibriumRefillingSweep< LatticeModel_T, FlagField_T > equilibriumRefillingSweep(
      pdfFieldID, flagFieldID, flagInfo, true);
   timeloop.add() << Sweep(equilibriumRefillingSweep, "Sweep: EquilibriumRefilling", StateSweep::fullFreeSurface)
                  << Sweep(emptySweep(), "Empty sweep: EquilibriumRefilling")
                  << AfterFunction(PdfCommunication_T(blockForest, pdfFieldID),
                                   "Communication: after EquilibriumRefilling sweep");

   // distribute excess mass:
   // - excess mass: mass that is free after conversion from interface to gas/liquid cells
   // - update the bubble model
   // IMPORTANT REMARK: this sweep computes the mass via the density, i.e., the PDF field must be up-to-date and the
   // PdfRefillingSweep must have been performed
   //if (excessMassModel.isEvenlyType())
   //{
      const walberla::free_surface::ExcessMassDistributionSweepInterfaceEvenly< LatticeModel_T, FlagField_T,
                                                                                ScalarField_T, VectorField_T >
         distributeMassSweep(excessMassModel, fillFieldID, flagFieldID, pdfFieldID, flagInfo);
      timeloop.add()
         << Sweep(distributeMassSweep, "Sweep: excess mass distribution", StateSweep::fullFreeSurface)
         << Sweep(emptySweep(), "Empty sweep: distribute excess mass")
         << AfterFunction(Communication_T(blockForest, fillFieldID),
                          "Communication: after excess mass distribution sweep")
         << AfterFunction(blockforest::UpdateSecondGhostLayer< ScalarField_T >(blockForest, fillFieldID),
                          "Second ghost layer update: after excess mass distribution sweep (fill level field)")
         // update bubble model, i.e., perform registered bubble merges/splits; bubble merges/splits are
         // already detected and registered by CellConversionSweep
         << AfterFunction(std::bind(&walberla::free_surface::bubble_model::BubbleModelBase::update, bubbleModel),
                          "Sweep: bubble model update");
   //}

   // reset all flags that signal cell conversions (except "keepInterfaceForWettingFlag")
   walberla::free_surface::ConversionFlagsResetSweep< FlagField_T > resetConversionFlagsSweep(flagFieldID, flagInfo);
   timeloop.add() << Sweep(resetConversionFlagsSweep, "Sweep: conversion flag reset", StateSweep::fullFreeSurface)
                  << Sweep(emptySweep(), "Empty sweep: conversion flag reset")
                  << AfterFunction(Communication_T(blockForest, flagFieldID),
                                   "Communication: after excess mass distribution sweep")
                  << AfterFunction(blockforest::UpdateSecondGhostLayer< FlagField_T >(blockForest, flagFieldID),
                                   "Second ghost layer update: after excess mass distribution sweep (flag field)");

   // update block states
   timeloop.add() << Sweep(blockStateUpdate, "Sweep: block state update");
   //----
   //--

   // add load balancing
   const LoadBalancer< FlagField_T, CommunicationStencil_T, LatticeModelStencil_T > loadBalancer(
      blockForest, communication, pdfCommunication, bubbleModel, uint_c(50), uint_c(10), uint_c(5),
      loadBalancingFrequency, printLoadBalancingStatistics);
   timeloop.addFuncAfterTimeStep(loadBalancer, "Sweep: load balancing");

   if (!benchmark)
   {
      // add sweep for evaluating the surface position in y-direction
      const std::shared_ptr< real_t > surfaceYPosition = std::make_shared< real_t >(real_c(0));
      const SurfaceYPositionEvaluator< FreeSurfaceBoundaryHandling_T > positionEvaluator(
         blockForest, freeSurfaceBoundaryHandling, fillFieldID, domainSize,
         cell_idx_c(real_c(domainSize[0]) * real_c(0.5)), evaluationFrequency, surfaceYPosition);
      timeloop.addFuncAfterTimeStep(positionEvaluator, "Evaluator: surface position");

      // add sweep for evaluating the symmetry of the fill level field in x-direction
      const std::shared_ptr< real_t > symmetryNorm = std::make_shared< real_t >(real_c(0));
      const SymmetryXEvaluator< ScalarField_T > symmetryEvaluator(blockForest, fillFieldID, domainSize,
                                                                  evaluationFrequency, symmetryNorm);
      timeloop.addFuncAfterTimeStep(symmetryEvaluator, "Evaluator: symmetry norm");

      // add evaluator for total and excessive mass (mass that is currently undistributed)
      const std::shared_ptr< real_t > totalMass  = std::make_shared< real_t >(real_c(0));
      const std::shared_ptr< real_t > excessMass = std::make_shared< real_t >(real_c(0));
      // const TotalMassComputer< FreeSurfaceBoundaryHandling_T, PdfField_T, FlagField_T, ScalarField_T > totalMassComputer(
      //    blockForest, freeSurfaceBoundaryHandling, pdfFieldID, fillFieldID, dynamicsHandler.getConstExcessMassFieldID(), evaluationFrequency, totalMass, excessMass);
      // timeloop.addFuncAfterTimeStep(totalMassComputer, "Evaluator: total mass");

      // add VTK output
      addVTKOutput< LatticeModel_T, FreeSurfaceBoundaryHandling_T, PdfField_T, FlagField_T, ScalarField_T,
                    VectorField_T, true, VectorFieldFlattened_T >(
         blockForest, timeloop, walberlaEnv.config(), flagInfo, pdfFieldID, flagFieldID, fillFieldID,
         forceDensityFieldID, curvatureFieldID, normalFieldID, obstacleNormalFieldID);

      // add triangle mesh output of free surface
      SurfaceMeshWriter< ScalarField_T, FlagField_T > surfaceMeshWriter(
         blockForest, fillFieldID, flagFieldID, flagIDs::liquidInterfaceGasFlagIDs, real_c(0), walberlaEnv.config());
      surfaceMeshWriter(); // write initial mesh
      timeloop.addFuncAfterTimeStep(surfaceMeshWriter, "Writer: surface mesh");

      // add logging for computational performance
      const lbm::PerformanceLogger< FlagField_T > performanceLogger(
         blockForest, flagFieldID, flagIDs::liquidInterfaceFlagIDs, performanceLogFrequency);
      timeloop.addFuncAfterTimeStep(performanceLogger, "Evaluator: performance logging");
   }

   WcTimingPool timingPool;

   if(benchmark) {
      for (uint_t t = uint_c(0); t != timesteps; ++t)
         timeloop.singleStep(timingPool, true);
   } else { // normal simulation
      for (uint_t t = uint_c(0); t != timesteps; ++t)
      {
         timeloop.singleStep(timingPool, true);

         if (t % performanceLogFrequency == uint_c(0) && t > uint_c(0)) { timingPool.logResultOnRoot(); }
      }
   }

   //uint_t runId = uint_c(-1);
   WALBERLA_LOG_INFO_ON_ROOT("*** SQL OUTPUT - START ***")
   auto tp_reduced = timingPool.getReduced();
   WALBERLA_ROOT_SECTION()
   {
      std::map< std::string, walberla::int64_t > integerProperties;
      std::map< std::string, double >            realProperties;
      std::map< std::string, std::string >       stringProperties;

      const std::string sqlFile = "GravityWaveCodegen.sqlite";

      // GENERAL INFORMATION
      stringProperties["tag"]                   = "free_surface";

      // BLOCKFOREST PARAMETERS
      integerProperties["loadBalancingFrequency"] = loadBalancingFrequency;

      // DOMAIN PARAMETERS
      integerProperties["nrWaves"] = nrWaves;
      realProperties["initialAmplitude"]        = initialAmplitude;

      // PHYSICS PARAMETERS
      realProperties["reynoldsNumber"] = reynoldsNumber;
      realProperties["relaxationRate"] = relaxationRate;
      integerProperties["enableWetting"] = enableWetting;
      if (enableWetting)
         realProperties["contactAngle"] = contactAngle;
      integerProperties["timesteps"] = timesteps;

      // MODEL PARAMETERS
      stringProperties["pdfReconstructionModel"] = pdfReconstructionModel;
      stringProperties["pdfRefillingModel"] = pdfRefillingModel;
      stringProperties["excessMassDistributionModel"] = excessMassDistributionModel;
      stringProperties["curvatureModel"] = curvatureModel;
      integerProperties["useSimpleMassExchange"] = useSimpleMassExchange;
      realProperties["cellConversionThreshold"] = cellConversionThreshold;
      realProperties["cellConversionForceThreshold"] = cellConversionForceThreshold;
      integerProperties["enableBubbleModel"] = enableBubbleModel;
      if (enableBubbleModel)
         integerProperties["enableBubbleSplits"] = enableBubbleSplits;

      addBuildInfoToSQL( integerProperties, realProperties, stringProperties );
      addDomainPropertiesToSQL(blockForest, integerProperties, realProperties, stringProperties);
      addSlurmPropertiesToSQL(integerProperties, realProperties, stringProperties);

      uint_t runId = sqlite::storeRunInSqliteDB( sqlFile, integerProperties, stringProperties, realProperties );
      sqlite::storeTimingPoolInSqliteDB( sqlFile, runId, *tp_reduced, "Timeloop" );
      WALBERLA_LOG_DEVEL_VAR(runId)
   }
   WALBERLA_LOG_INFO_ON_ROOT("*** SQL OUTPUT - END ***")

   return EXIT_SUCCESS;
}

template< typename T >
void writeVectorToFile(const std::vector< T >& vector, const std::string& filename)
{
   std::fstream file;
   file.open(filename, std::fstream::app);

   for (const auto i : vector)
   {
      file << "\t" << i;
   }

   file << "\n";
   file.close();
}

} // namespace GravityWaveCodegen
} // namespace free_surface
} // namespace walberla

int main(int argc, char** argv) { return walberla::free_surface::GravityWaveCodegen::main(argc, argv); }