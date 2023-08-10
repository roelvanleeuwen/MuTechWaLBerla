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
//! \file ViscousLDC.cpp
//! \author Jonas Plewinski <jonas.plewinski@fau.de>
//
// This showcase simulates a viscous mirrored lid driven cavity (velocity induced BC at the bottom), with a free surface
// at the top. The implementation uses an LBM kernel generated with lbmpy.
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

#include "ViscousLDCLatticeModel.h"

namespace walberla
{
namespace free_surface
{
namespace GravityWaveCodegen
{
using ScalarField_T          = GhostLayerField< real_t, 1 >;
using VectorField_T          = GhostLayerField< Vector3< real_t >, 1 >;
using VectorFieldFlattened_T = GhostLayerField< real_t, 3 >;

using LatticeModel_T        = lbm::ViscousLDCLatticeModel;
using LatticeModelStencil_T = LatticeModel_T::Stencil;
using PdfField_T            = lbm::PdfField< LatticeModel_T >;
using PdfCommunication_T    = blockforest::SimpleCommunication< LatticeModelStencil_T >;

// the geometry computations in SurfaceGeometryHandler require meaningful values in the ghost layers in corner
// directions (flag field and fill level field); this holds, even if the lattice model uses a D3Q19 stencil
using CommunicationStencil_T =
   typename std::conditional< LatticeModel_T::Stencil::D == uint_t(2), stencil::D2Q9, stencil::D3Q27 >::type;
using Communication_T = blockforest::SimpleCommunication< CommunicationStencil_T >;

using flag_t                        = uint32_t;
using FlagField_T                   = FlagField< flag_t >;
using FreeSurfaceBoundaryHandling_T = FreeSurfaceBoundaryHandling< LatticeModel_T, FlagField_T, ScalarField_T >;

// write each entry in "vector" to line in a file; columns are separated by tabs
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
      WALBERLA_CHECK_NOT_NULLPTR(blockForest);

      auto freeSurfaceBoundaryHandling = freeSurfaceBoundaryHandling_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(freeSurfaceBoundaryHandling);

      ++executionCounter_;

      // only evaluate in given frequencies
      if (executionCounter_ % frequency_ != uint_c(0) && executionCounter_ != uint_c(1)) { return; }

      const BlockDataID flagFieldID = freeSurfaceBoundaryHandling->getFlagFieldID();
      const typename FreeSurfaceBoundaryHandling_T::FlagInfo_T& flagInfo = freeSurfaceBoundaryHandling->getFlagInfo();

      *surfaceYPosition_ = real_c(0);

      for (auto blockIt = blockForest->begin(); blockIt != blockForest->end(); ++blockIt)
      {
         real_t maxSurfaceYPosition = real_c(0);

         CellInterval globalSearchInterval(globalXCoordinate_, cell_idx_c(0), cell_idx_c(0), globalXCoordinate_,
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

   // get domain parameters from parameter file
   auto domainParameters         = walberlaEnv.config()->getOneBlock("DomainParameters");
   const uint_t domainWidth      = domainParameters.getParameter< uint_t >("domainWidth");
   const real_t liquidDepth      = domainParameters.getParameter< real_t >("liquidDepth");
   const real_t initialAmplitude = domainParameters.getParameter< real_t >("initialAmplitude");

   // define domain size
   Vector3< uint_t > domainSize;
   domainSize[0] = domainWidth;
   domainSize[1] = uint_c(liquidDepth * real_c(2));
   domainSize[2] = uint_c(1);

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

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numProcesses)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellsPerBlock)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(domainSize)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numBlocks)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(domainWidth)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(liquidDepth)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(initialAmplitude)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(periodicity)

   // get physics parameters from parameter file
   auto physicsParameters = walberlaEnv.config()->getOneBlock("PhysicsParameters");
   const uint_t timesteps = physicsParameters.getParameter< uint_t >("timesteps");

   const real_t relaxationRate           = physicsParameters.getParameter< real_t >("relaxationRate");
   const CollisionModel_T collisionModel = CollisionModel_T(relaxationRate);
   const real_t viscosity                = collisionModel.viscosity();

   const real_t reynoldsNumber = physicsParameters.getParameter< real_t >("reynoldsNumber");
   const Vector3< real_t > acceleration(real_c(0), gravitationalAccelerationY, real_c(0));

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(reynoldsNumber)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(relaxationRate)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(timesteps)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(viscosity)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(acceleration)

   // read model parameters from parameter file
   const auto modelParameters               = walberlaEnv.config()->getOneBlock("ModelParameters");
   const std::string pdfReconstructionModel = modelParameters.getParameter< std::string >("pdfReconstructionModel");
   const std::string pdfRefillingModel      = modelParameters.getParameter< std::string >("pdfRefillingModel");
   const std::string excessMassDistributionModel =
      modelParameters.getParameter< std::string >("excessMassDistributionModel");
   const std::string curvatureModel          = modelParameters.getParameter< std::string >("curvatureModel");
   const bool useSimpleMassExchange          = modelParameters.getParameter< bool >("useSimpleMassExchange");
   const real_t cellConversionThreshold      = modelParameters.getParameter< real_t >("cellConversionThreshold");
   const real_t cellConversionForceThreshold = modelParameters.getParameter< real_t >("cellConversionForceThreshold");

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(pdfReconstructionModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(pdfRefillingModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(excessMassDistributionModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(curvatureModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(useSimpleMassExchange)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellConversionThreshold)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellConversionForceThreshold)

   // read evaluation parameters from parameter file
   const auto evaluationParameters      = walberlaEnv.config()->getOneBlock("EvaluationParameters");
   const uint_t performanceLogFrequency = evaluationParameters.getParameter< uint_t >("performanceLogFrequency");
   const uint_t evaluationFrequency     = evaluationParameters.getParameter< uint_t >("evaluationFrequency");
   const std::string filename           = evaluationParameters.getParameter< std::string >("filename");

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(performanceLogFrequency)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(evaluationFrequency)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(filename)

   // create non-uniform block forest (non-uniformity required for load balancing)
   const std::shared_ptr< StructuredBlockForest > blockForest =
      createNonUniformBlockForest(domainSize, cellsPerBlock, numBlocks, periodicity);

   // add force field
   const BlockDataID forceDensityFieldID = field::addToStorage< VectorField_T >(
      blockForest, "Force density field", Vector3< real_t >(real_c(0)), field::fzyx, uint_c(1));

   // create lattice model
   const LatticeModel_T latticeModel = LatticeModel_T(collisionModel, ForceModel_T(forceDensityFieldID));

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

   //--------------------------------- INITIALIZATION PROFILE ---------------------------------
}


} // namespace GravityWave
} // namespace free_surface
} // namespace walberla

int main(int argc, char** argv) { return walberla::free_surface::GravityWaveCodegen::main(argc, argv); }