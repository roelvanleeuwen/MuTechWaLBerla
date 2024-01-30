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
//! \file HybridBenchmark.cpp
//! \author Philipp Suffa philipp.suffa@fau.de
//
//======================================================================================================================

#include "blockforest/communication/UniformBufferedScheme.h"
#include "blockforest/Initialization.h"
#include "blockforest/loadbalancing/weight_assignment/WeightAssignmentFunctor.h"
#include "blockforest/loadbalancing/InfoCollection.h"
#include "blockforest/loadbalancing/DynamicCurve.h"

#include "core/Environment.h"
#include "core/logging/Initialization.h"
#include "core/SharedFunctor.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/MemoryUsage.h"

#include "field/AddToStorage.h"
#include "field/FlagField.h"
#include "field/StabilityChecker.h"
#include "field/vtk/VTKWriter.h"

#include "geometry/InitBoundaryHandling.h"

#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/PerformanceEvaluation.h"
#include "lbm/list/AddToStorage.h"
#include "lbm/list/ListVTK.h"
#include "lbm/vtk/all.h"

#include "mesh/blockforest/BlockExclusion.h"
#include "mesh/blockforest/BlockForestInitialization.h"
#include "mesh/blockforest/BlockWorkloadMemory.h"
#include "mesh/boundary/BoundaryInfo.h"
#include "mesh/boundary/BoundaryLocation.h"
#include "mesh/boundary/BoundaryLocationFunction.h"
#include "mesh/boundary/BoundarySetup.h"
#include "mesh/boundary/BoundaryUIDFaceDataSource.h"
#include "mesh/boundary/ColorToBoundaryMapper.h"

#include "mesh_common/DistanceComputations.h"
#include "mesh_common/DistanceFunction.h"
#include "mesh_common/MatrixVectorOperations.h"
#include "mesh_common/MeshIO.h"
#include "mesh_common/MeshOperations.h"
#include "mesh_common/TriangleMeshes.h"
#include "mesh_common/distance_octree/DistanceOctree.h"
#include "mesh_common/vtk/CommonDataSources.h"
#include "mesh_common/vtk/VTKMeshWriter.h"

#include "python_coupling/CreateConfig.h"
#include "python_coupling/PythonCallback.h"

#include "timeloop/SweepTimeloop.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#   include "gpu/AddGPUFieldToStorage.h"
#   include "gpu/DeviceSelectMPI.h"
#   include "gpu/HostFieldAllocator.h"
#   include "gpu/NVTX.h"
#   include "gpu/ParallelStreams.h"
#   include "gpu/communication/UniformGPUScheme.h"
#   include "gpu/lbm/CombinedInPlaceGpuPackInfo.h"
#else
#   include "lbm/communication/CombinedInPlaceCpuPackInfo.h"
#endif

#include "SparseLBMInfoHeader.h"
#include "DenseLBMInfoHeader.h"
#include "HybridPackInfoEven.h"
#include "HybridPackInfoOdd.h"
#include "InitSpherePacking.h"
#include "ReadParticleBoundaiesFromFile.h"
#include <iostream>
#include <fstream>

using namespace walberla;

uint_t numGhostLayers = uint_t(1);

using flag_t = walberla::uint8_t;
//using FlagField_T = FlagField<flag_t>;

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
using GPUField = gpu::GPUField< real_t >;
#endif

auto pdfFieldAdder = [](IBlock *const block, StructuredBlockStorage *const storage) {
   return new PdfField_T(storage->getNumberOfXCells(*block), storage->getNumberOfYCells(*block),
                         storage->getNumberOfZCells(*block), uint_t(1), field::fzyx,
                         make_shared<field::AllocateAligned<real_t, 64> >());
};


void setFlagFieldToPorosity(IBlock * block, const BlockDataID flagFieldId, const real_t porosity, const FlagUID noSlipFlagUID) {
   auto flagField    = block->getData< FlagField_T >(flagFieldId);
   auto noSlipFlag = flagField->getFlag(noSlipFlagUID);
   const real_t boundary_fraction = 1.0 - porosity;
   real_t nextBoundary = 0;
   for(auto it = flagField->begin(); it != flagField->end();++it) {
      if(nextBoundary  >= 1.0) {
         nextBoundary -= 1.0;
         addFlag(flagField->get(it), noSlipFlag);
      }
      nextBoundary += boundary_fraction;
   }
}

real_t calculatePorosity(IBlock * block, const BlockDataID flagFieldId, const FlagUID fluidFlagUID) {
   auto flagField = block->getData<FlagField_T>(flagFieldId);
   auto fluidFlag = flagField->getFlag(fluidFlagUID);
   uint_t fluidCells = 0;
   uint_t  numberOfCells = 0;
   for (auto it = flagField->begin(); it != flagField->end(); ++it) {
      if (isFlagSet(it, fluidFlag))
         fluidCells++;
      numberOfCells++;
   }
   return real_c(fluidCells) / real_c(numberOfCells);
}

void gatherAndPrintPorosityStats(weak_ptr<StructuredBlockForest> forest, BlockDataID flagFieldId, FlagUID fluidFlagUID, bool after) {
   auto blocks = forest.lock();
   real_t totalPorosityOnProcess = 0.0;
   real_t averagePorosity = 0.0;
   real_t minPorosity = 1.0;
   real_t maxPorosity = 0.0;
   for (auto& block : *blocks)
   {
      real_t blockPorosity = calculatePorosity(&block, flagFieldId, fluidFlagUID);
      if (blockPorosity < minPorosity) minPorosity = blockPorosity;
      if (blockPorosity > maxPorosity) maxPorosity = blockPorosity;
      totalPorosityOnProcess += blockPorosity;
   }
   averagePorosity = totalPorosityOnProcess / real_c(blocks->getNumberOfBlocks());
   WALBERLA_MPI_SECTION() {
      walberla::mpi::reduceInplace(minPorosity, walberla::mpi::MIN);
      walberla::mpi::reduceInplace(maxPorosity, walberla::mpi::MAX);
      walberla::mpi::reduceInplace(averagePorosity, walberla::mpi::SUM);
   }
   averagePorosity /= real_c(uint_c(MPIManager::instance()->numProcesses()));

   std::vector<real_t> porositiesPerProcess;
   porositiesPerProcess = mpi::gather( totalPorosityOnProcess);
   WALBERLA_ROOT_SECTION() {
      real_t sum = std::accumulate(porositiesPerProcess.begin(), porositiesPerProcess.end(), 0.0);
      real_t mean = sum / real_t(porositiesPerProcess.size());

      real_t sq_sum = std::inner_product(porositiesPerProcess.begin(), porositiesPerProcess.end(), porositiesPerProcess.begin(), 0.0);
      real_t stdev = std::sqrt(sq_sum / real_t(porositiesPerProcess.size()) - mean * mean);
      real_t minProcessPorosity = *std::min_element(porositiesPerProcess.begin(), porositiesPerProcess.end());
      real_t maxProcessPorosity = *std::max_element(porositiesPerProcess.begin(), porositiesPerProcess.end());
      if(after)
         WALBERLA_LOG_INFO_ON_ROOT("PROCESS Porosity AFTER Load balancing: mean = " << mean << ", stdev = " << stdev << ", min = " << minProcessPorosity << ", max = " << maxProcessPorosity )
      else {
         WALBERLA_LOG_INFO_ON_ROOT("BLOCK Porosity stats: average = " << averagePorosity << ", min = " << minPorosity << ", max = " << maxPorosity  )
         WALBERLA_LOG_INFO_ON_ROOT("PROCESS Porosity BEFORE Load balancing: mean = " << mean << ", stdev = " << stdev << ", min = " << minProcessPorosity << ", max = " << maxProcessPorosity )
      }

   }
}

void getBlocksWeights(weak_ptr<StructuredBlockForest> forest, blockforest::InfoCollection& ic, BlockDataID flagFieldId, FlagUID fluidFlagUID,
                      const Set< SUID > sweepSelectLowPorosity, const Set< SUID > sweepSelectHighPorosity) {
   auto blocks = forest.lock();
   for (auto &block : *blocks) {
      real_t porosity = calculatePorosity(&block, flagFieldId, fluidFlagUID);
      auto state = block.getState();

      const uint_t sparseAccess = 2 * Stencil_T::Q * 8 + Stencil_T::Q * 4;
      const uint_t denseAccess = 2 * Stencil_T::Q * 8;

      uint_t workload;
      if( selectable::isSetSelected( state, sweepSelectLowPorosity, sweepSelectHighPorosity ))
         workload = uint_c(1000.0 * real_c(sparseAccess) * porosity);
      else
         workload = 1000 * denseAccess;
      //WALBERLA_LOG_INFO("Workload for block " << block.getId() << " is " << workload)
      ic.insert( blockforest::InfoCollectionPair(static_cast<blockforest::Block*> (&block)->getId(), blockforest::BlockInfo(workload, 1)));
   }
}

template<typename MeshType>
void vertexToFaceColor(MeshType &mesh, const typename MeshType::Color &defaultColor) {
   WALBERLA_CHECK(mesh.has_vertex_colors())
   mesh.request_face_colors();

   for (auto faceIt = mesh.faces_begin(); faceIt != mesh.faces_end(); ++faceIt) {
      typename MeshType::Color vertexColor;

      bool useVertexColor = true;

      auto vertexIt = mesh.fv_iter(*faceIt);
      WALBERLA_ASSERT(vertexIt.is_valid())

      vertexColor = mesh.color(*vertexIt);

      ++vertexIt;
      while (vertexIt.is_valid() && useVertexColor) {
         if (vertexColor != mesh.color(*vertexIt)) useVertexColor = false;
         ++vertexIt;
      }

      mesh.set_color(*faceIt, useVertexColor ? vertexColor : defaultColor);
   }
}

auto deviceSyncWrapper = [](std::function< void(IBlock*) > sweep) {
   return [sweep](IBlock* b) {
      sweep(b);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpuDeviceSynchronize();
#endif
   };
};


int main(int argc, char **argv)
{
   walberla::Environment walberlaEnv(argc, argv);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   gpu::selectDeviceBasedOnMpiRank();
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif

   for (auto cfg = python_coupling::configBegin(argc, argv); cfg != python_coupling::configEnd(); ++cfg)
   {
      WALBERLA_MPI_WORLD_BARRIER()

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif

      auto config = *cfg;
      logging::configureLogging(config);
      logging::Logging::instance()->setLogLevel( logging::Logging::INFO );

      ///////////////////////
      /// PARAMETER INPUT ///
      ///////////////////////
      auto parameters = config->getBlock("Parameters");
      auto domainParameters = config->getBlock("DomainSetup");
      const Vector3< real_t > initialVelocity = parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >());
      const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(10));
      Vector3< int > InnerOuterSplit = parameters.getParameter< Vector3< int > >("innerOuterSplit", Vector3< int >(1, 1, 1));
      const bool weak_scaling = domainParameters.getParameter< bool >("weakScaling", false); // weak or strong scaling
      const Vector3<bool> periodic = domainParameters.getParameter< Vector3<bool> >("periodic", Vector3<bool>(false,false,false));

      const real_t remainingTimeLoggerFrequency = parameters.getParameter< real_t >("remainingTimeLoggerFrequency", 3.0); // in seconds
      const real_t omega = parameters.getParameter< real_t > ( "omega", real_c( 1.4 ) );
      const uint_t vtkWriteFrequency = parameters.getParameter< uint_t >("vtkWriteFrequency", uint_t(0));
      const bool runBoundaries = parameters.getParameter< bool >("runBoundaries", true);
      const std::string timeStepStrategy = parameters.getParameter< std::string >("timeStepStrategy", "noOverlap");
      const real_t porositySwitch = parameters.getParameter< real_t >("porositySwitch");
      const bool runHybrid = parameters.getParameter< bool >("runHybrid", false);
      const bool balanceLoad = parameters.getParameter< bool >("balanceLoad", false);


      Vector3< uint_t > cellsPerBlock;
      Vector3< uint_t > blocksPerDimension;
      uint_t nrOfProcesses = uint_c(MPIManager::instance()->numProcesses());

      if (!domainParameters.isDefined("blocks")) {
         if (weak_scaling) {
            Vector3< uint_t > cells = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
            blockforest::calculateCellDistribution(cells, nrOfProcesses, blocksPerDimension, cellsPerBlock);
            cellsPerBlock = cells;
         }
         else {
            Vector3< uint_t > cells = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
            blockforest::calculateCellDistribution(cells, nrOfProcesses, blocksPerDimension, cellsPerBlock);
         }
      }
      else {
         cellsPerBlock      = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
         blocksPerDimension = domainParameters.getParameter< Vector3< uint_t > >("blocks");
      }


      /////////////////////////
      /// BOUNDARY HANDLING ///
      /////////////////////////

      // create and initialize boundary handling
      const FlagUID fluidFlagUID("Fluid");
      const FlagUID noslipFlagUID("NoSlip");
      const FlagUID inflowUID("UBB");
      const FlagUID PressureOutflowUID("PressureOutflow");

      BlockDataID flagFieldId;
      shared_ptr< StructuredBlockForest > blocks;

      auto boundariesConfig = config->getOneBlock("Boundaries");
      const std::string geometrySetup = domainParameters.getParameter< std::string >("geometrySetup", "randomNoslip");
      bool pressureInflow = false;

      if (geometrySetup == "randomNoslip") {
         real_t dx = 1;
         /*blocks = walberla::blockforest::createUniformBlockGrid( blocksPerDimension[0], blocksPerDimension[1], blocksPerDimension[2], cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2], dx, 0, true, false, periodic[0], periodic[1], periodic[2], false);*/
         blocks = walberla::blockforest::createUniformBlockGrid(blocksPerDimension[0], blocksPerDimension[1], blocksPerDimension[2], cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2], dx,  true, periodic[0], periodic[1], periodic[2], false);

         flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");
         const real_t porosity = parameters.getParameter< real_t >("porosity");
         geometry::initBoundaryHandling<FlagField_T>(*blocks, flagFieldId, boundariesConfig);
         if(porosity < 1.0) {
            for (auto& block : *blocks) {
               setFlagFieldToPorosity(&block,flagFieldId,porosity,noslipFlagUID);
            }
         }
         geometry::setNonBoundaryCellsToDomain<FlagField_T>(*blocks, flagFieldId, fluidFlagUID);
      }
      else if (geometrySetup == "spheres") {
         mpi::MPIManager::instance()->useWorldComm();
         real_t dx = 1;
         blocks = walberla::blockforest::createUniformBlockGrid( blocksPerDimension[0], blocksPerDimension[1], blocksPerDimension[2], cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2], dx);
         flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");
         const real_t SpheresRadius = parameters.getParameter< real_t >("SpheresRadius");
         const real_t SphereShift = parameters.getParameter< real_t >("SphereShift");
         const Vector3<real_t> SphereFillDomainRatio = parameters.getParameter< Vector3<real_t> >("SphereFillDomainRatio", Vector3<real_t>(1.0));
         geometry::initBoundaryHandling<FlagField_T>(*blocks, flagFieldId, boundariesConfig);
         InitSpherePacking(blocks, flagFieldId, noslipFlagUID, SpheresRadius, SphereShift, SphereFillDomainRatio);
         geometry::setNonBoundaryCellsToDomain<FlagField_T>(*blocks, flagFieldId, fluidFlagUID);
      }
      else if (geometrySetup == "artery") {
         mpi::MPIManager::instance()->useWorldComm();
         std::string meshFile  = domainParameters.getParameter< std::string >("meshFile");
         WALBERLA_LOG_INFO_ON_ROOT("Using mesh from " << meshFile << ".")

         auto mesh = make_shared< mesh::TriangleMesh >();
         mesh->request_vertex_colors();
         mesh::readAndBroadcast(meshFile, *mesh);

         vertexToFaceColor(*mesh, mesh::TriangleMesh::Color(255, 255, 255));
         auto triDist = make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh);
         auto distanceOctree = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(triDist);
         distanceOctree->writeVTKOutput("distanceOctree");

         auto aabb = computeAABB(*mesh);
         //const Vector3< real_t > dx(scalingFactor, scalingFactor, scalingFactor);
         const Vector3< real_t > dx(0.05, 0.05, 0.05);

         mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, dx, mesh::makeExcludeMeshExterior(distanceOctree, dx[0]));

         auto meshWorkloadMemory = mesh::makeMeshWorkloadMemory( distanceOctree, dx[0] );
         meshWorkloadMemory.setInsideCellWorkload(1);
         meshWorkloadMemory.setOutsideCellWorkload(0);
         bfc.setWorkloadMemorySUIDAssignmentFunction( meshWorkloadMemory );

         blocks = bfc.createStructuredBlockForest(cellsPerBlock);

         flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");
         mesh::BoundarySetup boundarySetup(blocks, makeMeshDistanceFunction(distanceOctree), numGhostLayers);
         // write mesh info to file
         mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriter(mesh, "meshBoundaries", 1);
         for (auto& block : *blocks)
         {
            FlagField_T *flagField = block.getData<FlagField_T>(flagFieldId);
            flagField->registerFlag(fluidFlagUID);
            flagField->registerFlag(noslipFlagUID);
            flagField->registerFlag(inflowUID);
            flagField->registerFlag(PressureOutflowUID);
         }
         mesh::TriangleMesh::Color noSlipColor{255, 255, 255}; // White
         mesh::TriangleMesh::Color inflowColor;     // Yellow
         mesh::TriangleMesh::Color outflowColor;    // Light green

         if (meshFile == "coronary_colored_medium.obj") {
            inflowColor = mesh::TriangleMesh::Color(0, 255, 0);    // Light green
            outflowColor = mesh::TriangleMesh::Color(255, 0, 0);    // Red green
            pressureInflow = true;
         }
         else {
            inflowColor = mesh::TriangleMesh::Color(255, 255, 0);     // Yellow
            outflowColor = mesh::TriangleMesh::Color(0, 255, 0);    // Light green
         }
         const walberla::BoundaryUID OutflowBoundaryUID("PressureOutflow");
         const walberla::BoundaryUID InflowBoundaryUID("PressureInflow");
         static walberla::BoundaryUID wallFlagUID("NoSlip");
         mesh::ColorToBoundaryMapper< mesh::TriangleMesh > colorToBoundaryMapper((mesh::BoundaryInfo(wallFlagUID)));
         colorToBoundaryMapper.set(noSlipColor, mesh::BoundaryInfo(wallFlagUID));
         colorToBoundaryMapper.set(outflowColor, mesh::BoundaryInfo(OutflowBoundaryUID));
         colorToBoundaryMapper.set(inflowColor, mesh::BoundaryInfo(InflowBoundaryUID));
         auto boundaryLocations = colorToBoundaryMapper.addBoundaryInfoToMesh(*mesh);
         boundarySetup.setFlag<FlagField_T>(flagFieldId, fluidFlagUID, mesh::BoundarySetup::INSIDE);
         // set whole region outside the mesh to no-slip
         boundarySetup.setFlag<FlagField_T>(flagFieldId, FlagUID("NoSlip"), mesh::BoundarySetup::OUTSIDE);
         // set outflow flag to outflow boundary
         boundarySetup.setBoundaryFlag<FlagField_T>(flagFieldId, PressureOutflowUID, OutflowBoundaryUID, makeBoundaryLocationFunction(distanceOctree, boundaryLocations), mesh::BoundarySetup::OUTSIDE);
         // set inflow flag to inflow boundary
         boundarySetup.setBoundaryFlag<FlagField_T>(flagFieldId, inflowUID, InflowBoundaryUID, makeBoundaryLocationFunction(distanceOctree, boundaryLocations), mesh::BoundarySetup::OUTSIDE);
         meshWriter.addDataSource(make_shared< mesh::BoundaryUIDFaceDataSource< mesh::TriangleMesh > >(boundaryLocations));
         meshWriter.addDataSource(make_shared< mesh::ColorFaceDataSource< mesh::TriangleMesh > >());
         meshWriter.addDataSource(make_shared< mesh::ColorVertexDataSource< mesh::TriangleMesh > >());
         meshWriter();
      }
      else if (geometrySetup == "particleBed") {
         mpi::MPIManager::instance()->useWorldComm();
         const AABB  domainAABB = AABB(0.0, 0.0, 0.0, 0.1, 0.05, 0.1);
         const real_t dx = 0.0002;
         Vector3<uint_t> numCells(uint_c(domainAABB.xSize() / dx), uint_c(domainAABB.ySize() / dx), uint_c(domainAABB.zSize() / dx));
         Vector3<uint_t> numBlocks(uint_c(std::ceil(numCells[0] / cellsPerBlock[0])), uint_c(std::ceil(numCells[1] / cellsPerBlock[1])), uint_c(std::ceil(numCells[2] / cellsPerBlock[2])));

         blocks = blockforest::createUniformBlockGrid(domainAABB, numBlocks[0], numBlocks[1], numBlocks[2], cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2]);

         flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");
         geometry::initBoundaryHandling<FlagField_T>(*blocks, flagFieldId, boundariesConfig);
         const std::string filename = "/local/ed94aqyc/walberla_all/walberla/cmake-build-cpu_release/apps/showcases/Antidunes/spheres_out.dat";
         initSpheresFromFile(filename, blocks, flagFieldId, noslipFlagUID, dx);
         geometry::setNonBoundaryCellsToDomain<FlagField_T>(*blocks, flagFieldId, fluidFlagUID);
      }
      else {
         WALBERLA_ABORT_NO_DEBUG_INFO("Invalid value for 'geometrySetup'. Allowed values are 'randomNoslip', 'spheres', 'artery'")
      }

      WALBERLA_LOG_INFO_ON_ROOT("Number of cells is <" << blocks->getNumberOfXCells() << "," << blocks->getNumberOfYCells() << "," << blocks->getNumberOfZCells() << ">")
      WALBERLA_LOG_INFO_ON_ROOT("Number of blocks is <" << blocks->getXSize() << "," << blocks->getYSize() << "," << blocks->getZSize() << ">")
      WALBERLA_LOG_INFO_ON_ROOT("Is cartesian communicator used: " << mpi::MPIManager::instance()->hasCartesianSetup() << " or worldComm " << mpi::MPIManager::instance()->hasWorldCommSetup())


      if(timeStepStrategy != "noOverlap") {
         for (uint_t i = 0; i < 3; ++i) {
            if (int_c(cellsPerBlock[i]) <= InnerOuterSplit[i] * 2) {
               WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller or increase cellsPerBlock")
            }
         }
      }

      // Set state based on porosities
      const Set< SUID > sweepSelectHighPorosity("HighPorosity");
      const Set< SUID > sweepSelectLowPorosity("LowPorosity");
      for (auto& block : *blocks)
      {
         real_t blockPorosity = calculatePorosity(&block, flagFieldId, fluidFlagUID);
         if (blockPorosity > porositySwitch && runHybrid) {
            block.setState(sweepSelectHighPorosity);
         }
         else {
            block.setState(sweepSelectLowPorosity);
         }
      }

      gatherAndPrintPorosityStats(blocks, flagFieldId, fluidFlagUID, 0);

      //Balance load based on block porosity with Hilbert space filling curve
      if(balanceLoad) {
         //Get flags and FlagUIDs to later set them, because they get lost while loadBalancing
         std::vector<FlagUID> flagUIDs;
         std::vector<flag_t> flags;
         for (auto& block : *blocks)
         {
            auto flagfield = block.getData< FlagField_T >(flagFieldId);
            flagfield->getAllRegisteredFlags(flagUIDs);
            for( auto flagUId : flagUIDs) {
               flags.push_back(flagfield->getFlag(flagUId));
            }
            break;
         }

         //refresh here
         vtk::writeDomainDecomposition(blocks, "domain_decompositionDenseBeforeRefresh", "vtk_out", "write_call", true, true, 0, sweepSelectHighPorosity, sweepSelectLowPorosity);
         vtk::writeDomainDecomposition(blocks, "domain_decompositionSparseBeforeRefresh", "vtk_out", "write_call", true, true, 0, sweepSelectLowPorosity, sweepSelectHighPorosity);

         //Load Balancing 1
         auto & blockforest = blocks->getBlockForest();
         blockforest.recalculateBlockLevelsInRefresh( false );
         blockforest.alwaysRebalanceInRefresh( true ); //load balancing every time refresh is triggered
         blockforest.reevaluateMinTargetLevelsAfterForcedRefinement( false );
         blockforest.allowRefreshChangingDepth( false );
         blockforest.allowMultipleRefreshCycles( false ); // otherwise info collections are invalid
         blockforest.setRefreshPhantomBlockDataPackFunction( blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor() );
         blockforest.setRefreshPhantomBlockDataUnpackFunction( blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor() );
         blockforest.setRefreshPhantomBlockMigrationPreparationFunction( blockforest::DynamicCurveBalance<blockforest::WeightAssignmentFunctor::PhantomBlockWeight >( true, true, true ) );

         shared_ptr<blockforest::InfoCollection> weightsInfoCollection = make_shared<blockforest::InfoCollection>();
         getBlocksWeights(blocks, *weightsInfoCollection, flagFieldId, fluidFlagUID, sweepSelectLowPorosity, sweepSelectHighPorosity);
         blockforest::WeightAssignmentFunctor weightAssignmentFunctor(weightsInfoCollection, real_t(1));
         blockforest.setRefreshPhantomBlockDataAssignmentFunction(weightAssignmentFunctor);
         blocks->refresh();

         vtk::writeDomainDecomposition(blocks, "domain_decompositionDenseAfterRefresh", "vtk_out", "write_call", true, true, 0, sweepSelectHighPorosity, sweepSelectLowPorosity);
         vtk::writeDomainDecomposition(blocks, "domain_decompositionSparseAfterRefresh", "vtk_out", "write_call", true, true, 0, sweepSelectLowPorosity, sweepSelectHighPorosity);

         //Loadbalancing lost FlagUID information while communicating blocks, so set them back again
         for(auto &block : *blocks) {
            auto flagField = block.getData<FlagField_T>(flagFieldId);
            for(size_t i = 0; i < flagUIDs.size(); ++i) {
               if(!flagField->flagExists(flagUIDs[i]))
                  flagField->registerFlag(flagUIDs[i], uint_t(std::log2(int(flags[i]))));
            }
         }
         gatherAndPrintPorosityStats(blocks, flagFieldId, fluidFlagUID, 1);
      }

      ///////////////////////////////////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////    SETUP FIELDS      ///////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////////////////////////////////

      BlockDataID pdfListId = lbm::addListToStorage< List_T >(blocks, "LBM list (FIdx)", InnerOuterSplit, false, sweepSelectLowPorosity, sweepSelectHighPorosity);

      BlockDataID pdfFieldId     = blocks->addStructuredBlockData< PdfField_T >(pdfFieldAdder, "PDFs", sweepSelectHighPorosity, sweepSelectLowPorosity);
      BlockDataID velFieldId     = field::addToStorage< VelocityField_T >(blocks, "velocity", real_t(0), field::fzyx, uint_t(1), false, sweepSelectHighPorosity, sweepSelectLowPorosity);
      BlockDataID densityFieldId = field::addToStorage< ScalarField_T >(blocks, "density", real_t(1.0), field::fzyx, uint_t(1), false, sweepSelectHighPorosity, sweepSelectLowPorosity);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      BlockDataID pdfFieldIdGPU = gpu::addGPUFieldToStorage< PdfField_T >(blocks, pdfFieldId, "PDFs on GPU", true, sweepSelectHighPorosity, sweepSelectLowPorosity);
#endif

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////    SETUP KERNELS    //////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      const Vector3< int32_t > gpuBlockSize =
         parameters.getParameter< Vector3< int32_t > >("gpuBlockSize", Vector3< int32_t >(128, 1, 1));
      lbmpy::SparseLBSweep sparseKernel(pdfListId, omega, gpuBlockSize[0], gpuBlockSize[1], gpuBlockSize[2]);

      lbm::DenseLBSweep denseKernel(pdfFieldIdGPU, omega, gpuBlockSize[0], gpuBlockSize[1], gpuBlockSize[2], Cell(cell_idx_c(InnerOuterSplit[0]), cell_idx_c(InnerOuterSplit[1]), cell_idx_c(InnerOuterSplit[2])));
      lbm::DenseUBB denseUbb(blocks, pdfFieldIdGPU, initialVelocity[0]);
      lbm::DensePressure densePressureInflow(blocks, pdfFieldIdGPU, 1.01);
      lbm::DensePressure densePressureOutflow(blocks, pdfFieldIdGPU, 1.0);
      lbm::DenseNoSlip denseNoSlip(blocks, pdfFieldIdGPU);
#else
      lbmpy::SparseLBSweep sparseKernel(pdfListId, omega);

      lbm::DenseLBSweep denseKernel(pdfFieldId, omega, Cell(cell_idx_c(InnerOuterSplit[0]), cell_idx_c(InnerOuterSplit[1]), cell_idx_c(InnerOuterSplit[2])));
      lbm::DenseUBB denseUbb(blocks, pdfFieldId, initialVelocity[0]);
      lbm::DensePressure densePressureInflow(blocks, pdfFieldId, 1.01);
      lbm::DensePressure densePressureOutflow(blocks, pdfFieldId, 1.0);
      lbm::DenseNoSlip denseNoSlip(blocks, pdfFieldId);
#endif
      lbmpy::SparseUBB sparseUbb(blocks, pdfListId, initialVelocity[0]);
      lbmpy::SparsePressure sparsePressureInflow(blocks, pdfListId, 1.01);
      lbmpy::SparsePressure sparsePressureOutflow(blocks, pdfListId, 1.0);
      lbmpy::SparseMacroSetter sparseSetterSweep(pdfListId);

      pystencils::DenseMacroSetter denseSetterSweep(pdfFieldId);
      pystencils::DenseMacroGetter denseGetterSweep(densityFieldId, pdfFieldId, velFieldId);


      ///////////////////////////////////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////    INITFIELDS     //////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////////////////////////////////

      for (auto& block : *blocks)
      {
         if (block.getState() == sweepSelectLowPorosity) {
            auto* lbmList = block.getData< List_T >(pdfListId);
            WALBERLA_CHECK_NOT_NULLPTR(lbmList)
            lbmList->fillFromFlagField< FlagField_T >(block, flagFieldId, fluidFlagUID);

            if(pressureInflow)
               sparsePressureInflow.fillFromFlagField< FlagField_T >(&block, flagFieldId, inflowUID, fluidFlagUID);
            else
               sparseUbb.fillFromFlagField< FlagField_T >(&block, flagFieldId, inflowUID, fluidFlagUID);
            sparsePressureOutflow.fillFromFlagField< FlagField_T >(&block, flagFieldId, PressureOutflowUID, fluidFlagUID);

            sparseSetterSweep(&block);

         }
         else if (block.getState() == sweepSelectHighPorosity) {
            if(pressureInflow)
               densePressureInflow.fillFromFlagField< FlagField_T >(&block, flagFieldId, inflowUID, fluidFlagUID);
            else
               denseUbb.fillFromFlagField< FlagField_T >(&block, flagFieldId, inflowUID, fluidFlagUID);
            densePressureOutflow.fillFromFlagField< FlagField_T >(&block, flagFieldId, PressureOutflowUID, fluidFlagUID);
            denseNoSlip.fillFromFlagField< FlagField_T >(&block, flagFieldId, noslipFlagUID, fluidFlagUID);

            denseSetterSweep(&block);
         }

      }
      lbmpy::ListCommunicationSetup< FlagField_T, Stencil_T >(blocks, pdfListId, flagFieldId, fluidFlagUID, runHybrid, sweepSelectLowPorosity, sweepSelectHighPorosity);


      ///////////////////////////////////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////    SETUP COMMUNICATION   /////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////////////////////////////////

      auto tracker = make_shared<lbm::TimestepTracker>(0);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      const bool gpuEnabledMPI = parameters.getParameter< bool >("gpuEnabledMPI", true);
      auto packInfo = make_shared< lbm::CombinedInPlaceGpuPackInfo< lbmpy::HybridPackInfoEven, lbmpy::HybridPackInfoOdd > >(tracker, pdfFieldIdGPU, pdfListId, sweepSelectLowPorosity, sweepSelectHighPorosity);
      gpu::communication::UniformGPUScheme< Stencil_T > comm(blocks, gpuEnabledMPI);
#else
      auto packInfo = make_shared< lbm::CombinedInPlaceCpuPackInfo< lbmpy::HybridPackInfoEven, lbmpy::HybridPackInfoOdd > >(tracker, pdfFieldId, pdfListId, sweepSelectLowPorosity, sweepSelectHighPorosity);

      blockforest::communication::UniformBufferedScheme< Stencil_T > comm(blocks);

#endif
      WALBERLA_LOG_INFO_ON_ROOT("Finished setting up communication")
      comm.addPackInfo(packInfo);


      const auto emptySweep = [](IBlock*) {};
      SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

      if (timeStepStrategy == "noOverlap") {
         timeloop.add() << BeforeFunction(std::function< void() >([&]() { comm.communicate(); }), "communication") << Sweep(emptySweep);
         if(runBoundaries) {
            timeloop.add() << Sweep(deviceSyncWrapper(denseNoSlip.getSweep(tracker)), "denseNoslip", sweepSelectHighPorosity, sweepSelectLowPorosity);
            timeloop.add() << Sweep(deviceSyncWrapper(sparseUbb.getSweep(tracker)), "sparseUbb", sweepSelectLowPorosity, sweepSelectHighPorosity)
                           << Sweep(deviceSyncWrapper(denseUbb.getSweep(tracker)), "denseUbb", sweepSelectHighPorosity, sweepSelectLowPorosity);
            timeloop.add() << Sweep(deviceSyncWrapper(sparsePressureInflow.getSweep(tracker)), "sparsePressureInflow", sweepSelectLowPorosity, sweepSelectHighPorosity)
                           << Sweep(deviceSyncWrapper(densePressureInflow.getSweep(tracker)), "densePressureInflow", sweepSelectHighPorosity, sweepSelectLowPorosity);
            timeloop.add() << Sweep(deviceSyncWrapper(sparsePressureOutflow.getSweep(tracker)), "sparsePressureOutflow", sweepSelectLowPorosity, sweepSelectHighPorosity)
                           << Sweep(deviceSyncWrapper(densePressureOutflow.getSweep(tracker)), "densePressureOutflow", sweepSelectHighPorosity, sweepSelectLowPorosity);
         }
         timeloop.add() << BeforeFunction(tracker->getAdvancementFunction()) << Sweep(emptySweep);
         timeloop.add() << Sweep(deviceSyncWrapper(sparseKernel.getSweep(tracker)), "sparseKernel", sweepSelectLowPorosity, sweepSelectHighPorosity)
                        << Sweep(deviceSyncWrapper(denseKernel.getSweep(tracker)), "denseKernel", sweepSelectHighPorosity, sweepSelectLowPorosity);
      }
      else if (timeStepStrategy == "Overlap"){
         //start communication
         timeloop.add() << BeforeFunction(std::function< void() >([&]() { comm.startCommunication(); }), "communication") << Sweep(emptySweep);

         //run inner boundaries
         if(runBoundaries) {
            timeloop.add() << Sweep(deviceSyncWrapper(denseNoSlip.getSweep(tracker)), "denseNoslip", sweepSelectHighPorosity, sweepSelectLowPorosity);
            timeloop.add() << Sweep(deviceSyncWrapper(sparseUbb.getSweep(tracker)), "sparseUbb", sweepSelectLowPorosity, sweepSelectHighPorosity)
                           << Sweep(deviceSyncWrapper(denseUbb.getSweep(tracker)), "denseUbb", sweepSelectHighPorosity, sweepSelectLowPorosity);
            timeloop.add() << Sweep(deviceSyncWrapper(sparsePressureInflow.getSweep(tracker)), "sparsePressureInflow", sweepSelectLowPorosity, sweepSelectHighPorosity)
                           << Sweep(deviceSyncWrapper(densePressureInflow.getSweep(tracker)), "densePressureInflow", sweepSelectHighPorosity, sweepSelectLowPorosity);
            timeloop.add() << Sweep(deviceSyncWrapper(sparsePressureOutflow.getSweep(tracker)), "sparsePressureOutflow", sweepSelectLowPorosity, sweepSelectHighPorosity)
                           << Sweep(deviceSyncWrapper(densePressureOutflow.getSweep(tracker)), "densePressureOutflow", sweepSelectHighPorosity, sweepSelectLowPorosity);
         }
         //increase tracker and run inner LBM kernel
         timeloop.add() << BeforeFunction(tracker->getAdvancementFunction()) << Sweep(emptySweep);
         timeloop.add() << Sweep(deviceSyncWrapper(sparseKernel.getInnerSweep(tracker)), "sparseKernel inner", sweepSelectLowPorosity, sweepSelectHighPorosity)
                        << Sweep(deviceSyncWrapper(denseKernel.getInnerSweep(tracker)), "denseKernel inner", sweepSelectHighPorosity, sweepSelectLowPorosity);

         //decrease tracker and run wait communication and outer boundaries
         timeloop.add() << BeforeFunction(tracker->getAdvancementFunction()) << BeforeFunction(std::function< void() >([&]() { comm.wait(); }), "communication") << Sweep(emptySweep);
         /*if(runBoundaries) {
            timeloop.add() << Sweep(denseNoSlip.getOuterSweep(tracker), "denseNoslip outer", sweepSelectHighPorosity, sweepSelectLowPorosity);
            timeloop.add() << Sweep(sparseUbb.getOuterSweep(tracker), "sparseUbb outer", sweepSelectLowPorosity, sweepSelectHighPorosity)
                           << Sweep(denseUbb.getOuterSweep(tracker), "denseUbb outer", sweepSelectHighPorosity, sweepSelectLowPorosity);
            timeloop.add() << Sweep(sparsePressureOutflow.getOuterSweep(tracker), "sparsePressureOutflow outer", sweepSelectLowPorosity, sweepSelectHighPorosity)
                           << Sweep(densePressureOutflow.getOuterSweep(tracker), "densePressureOutflow outer", sweepSelectHighPorosity, sweepSelectLowPorosity);
         }*/
         //increase tracker again and run outer LBM kernel
         timeloop.add() << BeforeFunction(tracker->getAdvancementFunction()) << Sweep(emptySweep);
         timeloop.add() << Sweep(deviceSyncWrapper(sparseKernel.getOuterSweep(tracker)), "sparseKernel outer", sweepSelectLowPorosity, sweepSelectHighPorosity)
                        << Sweep(deviceSyncWrapper(denseKernel.getOuterSweep(tracker)), "denseKernel outer", sweepSelectHighPorosity, sweepSelectLowPorosity);
      }
      else if (timeStepStrategy == "kernelOnly")
      {
         WALBERLA_LOG_INFO_ON_ROOT("Running only compute kernel without boundary - this makes only sense for benchmarking!")
         timeloop.add() << BeforeFunction(tracker->getAdvancementFunction()) << Sweep(emptySweep);
         timeloop.add() << Sweep(deviceSyncWrapper(sparseKernel.getSweep(tracker)), "sparseKernel", sweepSelectLowPorosity, sweepSelectHighPorosity)
                        << Sweep(deviceSyncWrapper(denseKernel.getSweep(tracker)), "denseKernel", sweepSelectHighPorosity, sweepSelectLowPorosity);
      }
      else if (timeStepStrategy == "communicationOnly")
      {
         WALBERLA_LOG_INFO_ON_ROOT("Running only communication - this makes only sense for benchmarking!")
         timeloop.add() << BeforeFunction(tracker->getAdvancementFunction()) << Sweep(emptySweep);
         timeloop.add() << BeforeFunction(std::function< void() >([&]() { comm.startCommunication(); }), "communication Start") << Sweep(emptySweep);
         timeloop.add() << BeforeFunction(std::function< void() >([&]() { comm.wait(); }), "communication Wait") << Sweep(emptySweep);
      }
      else
      {
         WALBERLA_ABORT_NO_DEBUG_INFO("Invalid value for 'timeStepStrategy'. Allowed values are 'noOverlap', 'simpleOverlap', 'kernelOnly', 'communicationOnly'")
      }

      timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),"remaining time logger");


      ///////////////////////////////////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////    VTK OUT    ///////////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////////////////////////////////

      if (vtkWriteFrequency > 0)
      {
         auto sparseVtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtkSparse", vtkWriteFrequency, 0, false, "vtk_out", "simulation_step", false, false, true, false, 0);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         sparseVtkOutput->addBeforeFunction([&]() {
            for (auto& block : *blocks)
            {
               if (block.getState() == sweepSelectLowPorosity)
               {
                  List_T* lbmList = block.getData< List_T >(pdfListId);
                  lbmList->copyPDFSToCPU();
               }
            }
         });
#endif

         field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagFieldId);
         fluidFilter.addFlag(fluidFlagUID);
         sparseVtkOutput->addCellInclusionFilter(fluidFilter);

         auto velWriter = make_shared< lbm::ListVelocityVTKWriter< List_T, real_t > >(pdfListId, tracker, "velocity");
         auto densityWriter = make_shared< lbm::ListDensityVTKWriter< List_T, real_t > >(pdfListId, "density");

         sparseVtkOutput->addCellDataWriter(velWriter);
         sparseVtkOutput->addCellDataWriter(densityWriter);

         timeloop.addFuncBeforeTimeStep(vtk::writeFiles(sparseVtkOutput, true, 0, sweepSelectLowPorosity, sweepSelectHighPorosity), "VTK Output Sparse");
         vtk::writeDomainDecomposition(blocks, "domain_decompositionSparse", "vtk_out", "write_call", true, true, 0, sweepSelectLowPorosity, sweepSelectHighPorosity);


         auto denseVtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtkDense", vtkWriteFrequency, 0, false, "vtk_out", "simulation_step", false, false, true, false, 0);

         denseVtkOutput->addBeforeFunction([&]() {
            for (auto& block : *blocks)
            {
               if (block.getState() == sweepSelectHighPorosity) {
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
                  GPUField * dst = block.getData<GPUField>( pdfFieldIdGPU );
                  const PdfField_T * src = block.getData<PdfField_T>( pdfFieldId );
                  gpu::fieldCpy( *dst, *src );
#endif
                  denseGetterSweep(&block);
               }
            }
         });

         denseVtkOutput->addCellInclusionFilter(fluidFilter);

         auto denseVelWriter = make_shared<field::VTKWriter<VelocityField_T> >(velFieldId, "velocity");
         auto denseDensityWriter = make_shared<field::VTKWriter<ScalarField_T> >(densityFieldId, "density");

         denseVtkOutput->addCellDataWriter(denseVelWriter);
         denseVtkOutput->addCellDataWriter(denseDensityWriter);

         timeloop.addFuncBeforeTimeStep(vtk::writeFiles(denseVtkOutput, true, 0, sweepSelectHighPorosity, sweepSelectLowPorosity), "VTK Output Dense");
      }


      lbm::PerformanceEvaluation< FlagField_T > performance(blocks, flagFieldId, fluidFlagUID);

      WALBERLA_LOG_INFO_ON_ROOT("Simulating ListLBM:"
                                "\n timesteps:                  " << timesteps
                                << "\n relaxation rate:            " << omega)

      int warmupSteps     = parameters.getParameter< int >("warmupSteps", 2);
      for (int i = 0; i < warmupSteps; ++i)
         timeloop.singleStep();


      WcTimingPool timeloopTiming;
      WcTimer simTimer;

      WALBERLA_MPI_WORLD_BARRIER()
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      WALBERLA_GPU_CHECK(gpuPeekAtLastError())
      gpuDeviceSynchronize();
#endif

      simTimer.start();
      timeloop.run(timeloopTiming);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      WALBERLA_GPU_CHECK(gpuPeekAtLastError())
      gpuDeviceSynchronize();
#endif
      simTimer.end();

      WALBERLA_LOG_INFO_ON_ROOT("Simulation finished")
      real_t time = simTimer.max();
      //WALBERLA_LOG_INFO(performance.mflupsPerProcess(timesteps, time));
      WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
      performance.logResultOnRoot(timesteps, time);

      const auto reducedTimeloopTiming = timeloopTiming.getReduced();
      WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)

      WALBERLA_ROOT_SECTION(){
         std::ofstream myfile;
         myfile.open ("results.txt", std::ios::app);
         myfile << nrOfProcesses << " " << InnerOuterSplit  <<  " " << performance.mflupsPerProcess(timesteps, time) << " " << performance.mflups(timesteps, time) << std::endl;
         myfile.close();
      }
      //printResidentMemoryStatistics();
   }//config

   return EXIT_SUCCESS;
}
