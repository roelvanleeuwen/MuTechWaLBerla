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

#include "blockforest/communication/UniformBufferedScheme.h"
#include "blockforest/communication/NonUniformBufferedScheme.h"
#include "blockforest/Initialization.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/logging/Initialization.h"
#include "core/SharedFunctor.h"
#include "core/math/Vector3.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/MemoryUsage.h"

#include "field/AddToStorage.h"
#include "field/StabilityChecker.h"
#include "field/vtk/VTKWriter.h"

#include "geometry/InitBoundaryHandling.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/FieldCopy.h"
#include "gpu/ErrorChecking.h"
#include "gpu/HostFieldAllocator.h"
#include "gpu/ParallelStreams.h"
#include "gpu/communication/UniformGPUScheme.h"
#include "gpu/communication/NonUniformGPUScheme.h"
#endif

#include "lbm_generated/communication/NonuniformGeneratedPdfPackInfo.h"
#include "lbm_generated/communication/UniformGeneratedPdfPackInfo.h"
#include "lbm_generated/field/AddToStorage.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/refinement/BasicRecursiveTimeStep.h"
#include "lbm_generated/refinement/RefinementScaling.h"
#include "lbm_generated/evaluation/PerformanceEvaluation.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#include "lbm_generated/gpu/UniformGeneratedGPUPdfPackInfo.h"
#include "lbm_generated/gpu/NonuniformGeneratedGPUPdfPackInfo.h"
#include "lbm_generated/gpu/GPUPdfField.h"
#include "lbm_generated/gpu/AddToStorage.h"
#include "lbm_generated/gpu/BasicRecursiveTimeStepGPU.h"
#endif

#include "mesh/Utility.h"
#include "mesh/blockforest/BlockExclusion.h"
#include "mesh/blockforest/BlockForestInitialization.h"
#include "mesh/blockforest/RefinementSelection.h"
#include "mesh/boundary/BoundaryInfo.h"
#include "mesh/boundary/BoundaryLocation.h"
#include "mesh/boundary/BoundaryLocationFunction.h"
#include "mesh/boundary/BoundarySetup.h"
#include "mesh/boundary/BoundaryUIDFaceDataSource.h"
#include "mesh/boundary/ColorToBoundaryMapper.h"
#include "mesh/blockforest/BlockWorkloadMemory.h"

#include "timeloop/SweepTimeloop.h"

#include "mesh_common/DistanceComputations.h"
#include "mesh_common/DistanceFunction.h"
#include "mesh_common/MeshIO.h"
#include "mesh_common/MeshOperations.h"
#include "mesh_common/TriangleMeshes.h"
#include "mesh_common/distance_octree/DistanceOctree.h"
#include "mesh_common/vtk/CommonDataSources.h"
#include "mesh_common/vtk/VTKMeshWriter.h"

#include "FlowAroundSphereInfoHeader.h"
#include "wallDistance.h"

using namespace walberla;

using StorageSpecification_T = lbm::FlowAroundSphereStorageSpecification;
using Stencil_T              = StorageSpecification_T::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;

using PdfField_T           = lbm_generated::PdfField< StorageSpecification_T >;
using FlagField_T          = FlagField< uint8_t >;
using BoundaryCollection_T = lbm::FlowAroundSphereBoundaryCollection< FlagField_T >;

using SweepCollection_T = lbm::FlowAroundSphereSweepCollection;

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
using GPUPdfField_T           = lbm_generated::GPUPdfField< StorageSpecification_T >;
using gpu::communication::UniformGPUScheme;
using gpu::communication::NonUniformGPUScheme;

using lbm_generated::UniformGeneratedGPUPdfPackInfo;
using lbm_generated::NonuniformGeneratedGPUPdfPackInfo;
#else
using PdfField_T           = lbm_generated::PdfField< StorageSpecification_T >;
using blockforest::communication::UniformBufferedScheme;
using blockforest::communication::NonUniformBufferedScheme;

using lbm_generated::UniformGeneratedPdfPackInfo;
using lbm_generated::NonuniformGeneratedPdfPackInfo;
#endif

using RefinementSelectionFunctor = SetupBlockForest::RefinementSelectionFunction;

namespace{
void workloadAndMemoryAssignment(SetupBlockForest& forest, const memory_t memoryPerBlock)
{
   for (auto block = forest.begin(); block != forest.end(); ++block)
   {
      block->setWorkload(numeric_cast< workload_t >(uint_t(1) << block->getLevel()));
      block->setMemory(memoryPerBlock);
   }
}
}

//////////////////////
// Parameter Struct //
//////////////////////

struct Setup
{
   real_t Re;

   real_t viscosity; // on the coarsest grid
   real_t omega; // on the coarsest grid
   real_t inletVelocity;
   uint_t refinementLevels;

   void logSetup() const
   {
      WALBERLA_LOG_INFO_ON_ROOT( "FlowAroundSphere simulation properties:"
                                "\n   + Reynolds number:   " << Re <<
                                "\n   + lattice viscosity: " << viscosity << " (on the coarsest grid)" <<
                                "\n   + relaxation rate:   " << std::setprecision(16) << omega << " (on the coarsest grid)" <<
                                "\n   + inlet velocity:    " << inletVelocity << " (in lattice units)" <<
                                "\n   + refinement Levels: " << refinementLevels)
   }
};


int main(int argc, char **argv) {
   walberla::Environment walberlaEnv(argc, argv);
   mpi::MPIManager::instance()->useWorldComm();
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   gpu::selectDeviceBasedOnMpiRank();
#endif

   logging::configureLogging(walberlaEnv.config());

   ///////////////////////
   /// PARAMETER INPUT ///
   ///////////////////////

   // read general simulation parameters
   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");

   const real_t diameterSphere   = parameters.getParameter<real_t>("diameterSphere");
   const real_t reynoldsNumber   = parameters.getParameter<real_t>("reynoldsNumber");
   const real_t velocity   = parameters.getParameter<real_t>("velocity");
   const real_t timestepSize   = parameters.getParameter<real_t>("timestepSize");

   const uint_t timesteps                = parameters.getParameter<uint_t>("timesteps", 0);
   const real_t coarseMeshSize           = parameters.getParameter<real_t>("coarseMeshSize");
   auto resolution                       = Vector3<real_t>(coarseMeshSize);

   // read domain parameters
   auto domainParameters = walberlaEnv.config()->getOneBlock("DomainSetup");

   const std::string meshFile       = domainParameters.getParameter<std::string>("meshFile");
   const Vector3<uint_t> blockSize  = domainParameters.getParameter<Vector3<uint_t>>("cellsPerBlock");
   const Vector3<real_t> domainSize = domainParameters.getParameter<Vector3<real_t> >("domainSize");
   const Vector3<bool> periodicity  = domainParameters.getParameter<Vector3<bool> >("periodic", Vector3<bool>(false));
   const bool weakScaling           = domainParameters.getParameter<bool>("weakScaling", false);
   const uint_t refinementLevels    = domainParameters.getParameter< uint_t >( "refinementLevels");

   bool uniformGrid = true;
   auto numGhostLayers = uint_c(2);
   if (refinementLevels > 0) {uniformGrid = false; numGhostLayers = 2;}

   if(weakScaling)
   {
      auto blocks = math::getFactors3D( uint_c( MPIManager::instance()->numProcesses() ) );
      resolution[0] = domainSize[0] / real_c(blocks[0] * blockSize[0]);
      resolution[1] = domainSize[1] / real_c(blocks[1] * blockSize[1]);
      resolution[2] = domainSize[2] / real_c(blocks[2] * blockSize[2]);
      WALBERLA_LOG_INFO_ON_ROOT("Setting up a weak scaling benchmark with a resolution of: " << resolution)
   }

   const real_t fineMeshSize = resolution.min() / real_c(std::pow(2, refinementLevels));

   const real_t latticeVelocity = velocity * timestepSize / coarseMeshSize;
   const real_t latticeViscosity =  (diameterSphere / coarseMeshSize) * latticeVelocity / reynoldsNumber;
   const real_t omega = real_c(real_c(1.0) / (real_c(3.0) * latticeViscosity + real_c(0.5)));

   const real_t inletVelocity = latticeVelocity;

   const uint_t numProcesses = domainParameters.getParameter< uint_t >( "numberProcesses");

   auto loggingParameters = walberlaEnv.config()->getOneBlock("Logging");
   bool writeSetupForestAndReturn = loggingParameters.getParameter<bool>("writeSetupForestAndReturn", false);
   const bool writeDistanceOctree = loggingParameters.getParameter<bool>("writeDistanceOctree", false);
   const bool writeMeshBoundaries = loggingParameters.getParameter<bool>("writeMeshBoundaries", false);
   const bool readSetupFromFile = loggingParameters.getParameter<bool>("readSetupFromFile", false);
   if (uint_c( MPIManager::instance()->numProcesses() ) > 1)
      writeSetupForestAndReturn = false;

   const Setup setup{reynoldsNumber, latticeViscosity, omega, inletVelocity, refinementLevels};

   const uint_t valuesPerCell = (Stencil_T::Q + VelocityField_T::F_SIZE + uint_c(2) * ScalarField_T::F_SIZE);
   const uint_t sizePerValue = sizeof(PdfField_T::value_type);
   const memory_t memoryPerCell =  memory_t( valuesPerCell * sizePerValue + uint_c(1) );
   const Vector3<uint_t> blockSizeWithGL{blockSize[0] + uint_c(2) * numGhostLayers,
                                         blockSize[1] + uint_c(2) * numGhostLayers,
                                         blockSize[2] + uint_c(2) * numGhostLayers};
   const memory_t memoryPerBlock = numeric_cast< memory_t >( blockSizeWithGL[0] * blockSizeWithGL[1] * blockSizeWithGL[2] ) * memoryPerCell;

   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   WALBERLA_LOG_INFO_ON_ROOT("Using mesh from " << meshFile << ".")

   // read in mesh with vertex colors on a single process and broadcast it
   auto mesh = make_shared<mesh::TriangleMesh>();
   mesh->request_vertex_colors();
   mesh::readAndBroadcast(meshFile, *mesh);
   mesh::vertexToFaceColor(*mesh, mesh::TriangleMesh::Color(255, 255, 255));

   // building distanceOctree
   auto triDist = make_shared<mesh::TriangleDistance<mesh::TriangleMesh> >(mesh);
   auto distanceOctree = make_shared<mesh::DistanceOctree<mesh::TriangleMesh> >(triDist);

   // write distance octree to file
   if (writeDistanceOctree)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Writing distanceOctree to VTK file")
      distanceOctree->writeVTKOutput("distanceOctree");
   }

   ///////////////////////////
   /// CREATE BLOCK FOREST ///
   ///////////////////////////

   auto aabb = computeAABB(*mesh);
   auto domainScaling = Vector3<real_t> (domainSize[0] / aabb.xSize(), domainSize[1] / aabb.ySize(), domainSize[2] / aabb.zSize());
   aabb.scale( domainScaling );
   aabb.setCenter(Vector3<real_t >(real_c(5.0), real_c(0.0), real_c(0.0)));

   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, resolution);

   bfc.setRootBlockExclusionFunction(mesh::makeExcludeMeshInterior(distanceOctree, resolution.min()));
   bfc.setWorkloadMemorySUIDAssignmentFunction( std::bind( workloadAndMemoryAssignment, std::placeholders::_1, memoryPerBlock ) );
   bfc.setPeriodicity(periodicity);

   if( !uniformGrid ) {
      WALBERLA_LOG_INFO_ON_ROOT("Using " << refinementLevels << " refinement levels. The resolution around the object is " << fineMeshSize << " m")
      bfc.setRefinementSelectionFunction(mesh::RefinementSelection(distanceOctree, refinementLevels, real_c(0.0), resolution.min()));
      bfc.setBlockExclusionFunction(mesh::makeExcludeMeshInteriorRefinement(distanceOctree, fineMeshSize));
   }
   else {
      WALBERLA_LOG_INFO_ON_ROOT("Using a uniform Grid. The resolution around the object is " << fineMeshSize << " m")
   }

   if (writeSetupForestAndReturn)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Creating SetupBlockForest for " << numProcesses << " processes")
      auto setupForest = bfc.createSetupBlockForest( blockSize, numProcesses );
      WALBERLA_LOG_INFO_ON_ROOT("Writing SetupBlockForest to VTK file")
      WALBERLA_ROOT_SECTION() { setupForest->writeVTKOutput("SetupBlockForest"); }

      auto finalDomain = setupForest->getDomain();
      WALBERLA_LOG_INFO_ON_ROOT("Wind tunnel size in x-direction: " << finalDomain.xSize() << " m")
      WALBERLA_LOG_INFO_ON_ROOT("Wind tunnel size in y-direction: " << finalDomain.ySize() << " m")
      WALBERLA_LOG_INFO_ON_ROOT("Wind tunnel size in z-direction: " << finalDomain.zSize() << " m")

      WALBERLA_LOG_INFO_ON_ROOT("Blocks created: " << setupForest->getNumberOfBlocks())
      double totalCellUpdates( 0.0 );
      for (uint_t level = 0; level <= refinementLevels; level++)
      {
         const uint_t numberOfBlocks = setupForest->getNumberOfBlocks(level);
         const uint_t numberOfCells = numberOfBlocks * blockSize[0] * blockSize[1] * blockSize[2];
         totalCellUpdates += double_c( timesteps * math::uintPow2(level) ) * double_c( numberOfCells );
         WALBERLA_LOG_INFO_ON_ROOT("Level " << level << " Blocks: " << numberOfBlocks)
      }

      const uint_t totalNumberFluidCells = setupForest->getNumberOfBlocks() * blockSize[0] * blockSize[1] * blockSize[2];
      const uint_t totalNumberCells = setupForest->getNumberOfBlocks() * blockSizeWithGL[0] * blockSizeWithGL[1] * blockSizeWithGL[2];
      const uint_t conversionFactor = uint_c(1024) * uint_c(1024) * uint_c(1024);
      const uint_t expectedMemory = (totalNumberCells * valuesPerCell * sizePerValue) / conversionFactor;

      WALBERLA_LOG_INFO_ON_ROOT( "Total number of cells will be " << totalNumberFluidCells << " fluid cells (in total on all levels)")
      WALBERLA_LOG_INFO_ON_ROOT( "Total number of cells will be " << totalNumberCells << " cells (in total on all levels with Ghostlayers)")
      WALBERLA_LOG_INFO_ON_ROOT( "Expected total memory demand will be " << expectedMemory << " GB")
      WALBERLA_LOG_INFO_ON_ROOT( "The total cell updates after " << timesteps << " timesteps (on the coarse level) will be " << totalCellUpdates)

      setup.logSetup();

      std::ostringstream oss;
      oss << "BlockForest_" << std::to_string(numProcesses) <<  ".bfs";
      setupForest->saveToFile(oss.str().c_str());

      WALBERLA_LOG_INFO_ON_ROOT("Ending program")
      return EXIT_SUCCESS;

   }
   std::shared_ptr< StructuredBlockForest > blocks;
   if(readSetupFromFile)
   {
      std::ostringstream oss;
      oss << "BlockForest_" << std::to_string(uint_c( MPIManager::instance()->numProcesses() )) <<  ".bfs";
      const std::string setupBlockForestFilepath = oss.str();

      WALBERLA_LOG_INFO_ON_ROOT("Reading structured block forest from file")
      auto bfs = std::make_shared< BlockForest >(uint_c(MPIManager::instance()->worldRank()),
                                                 setupBlockForestFilepath.c_str(), false);
      blocks = std::make_shared< StructuredBlockForest >(bfs, blockSize[0], blockSize[1], blockSize[2]);
      blocks->createCellBoundingBoxes();
   }
   else
   {
      blocks = bfc.createStructuredBlockForest(blockSize);
   }


   auto finalDomain = blocks->getDomain();
   WALBERLA_LOG_INFO_ON_ROOT("Wind tunnel size in x-direction: " << finalDomain.xSize() << " m")
   WALBERLA_LOG_INFO_ON_ROOT("Wind tunnel size in y-direction: " << finalDomain.ySize() << " m")
   WALBERLA_LOG_INFO_ON_ROOT("Wind tunnel size in z-direction: " << finalDomain.zSize() << " m")

   WALBERLA_LOG_INFO_ON_ROOT("Blocks created: " << blocks->getNumberOfBlocks())
   for (uint_t level = 0; level <= refinementLevels; level++)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Level " << level << " Blocks: " << blocks->getNumberOfBlocks(level))
   }

   ////////////////////////////////////
   /// CREATE AND INITIALIZE FIELDS ///
   ////////////////////////////////////

   // create fields
   const StorageSpecification_T StorageSpec = StorageSpecification_T();

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   auto allocator = make_shared< gpu::HostFieldAllocator<real_t> >();
   const BlockDataID pdfFieldID = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, numGhostLayers, field::fzyx);
   const BlockDataID velFieldID = field::addToStorage< VelocityField_T >(blocks, "velocity", real_c(0.0), field::fzyx, numGhostLayers);
   const BlockDataID densityFieldID = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, numGhostLayers);
   const BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "Boundary Flag Field", uint_c(2));

   const BlockDataID pdfFieldGPUID = lbm_generated::addGPUPdfFieldToStorage< PdfField_T >(blocks, pdfFieldID, StorageSpec, "pdfs on GPU", true);
   const BlockDataID velFieldGPUID = gpu::addGPUFieldToStorage< VelocityField_T >(blocks, velFieldID, "velocity on GPU", true);
   const BlockDataID densityFieldGPUID = gpu::addGPUFieldToStorage< ScalarField_T >(blocks, densityFieldID, "density on GPU", true);

   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#else
   const BlockDataID pdfFieldID = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, numGhostLayers, field::fzyx);
   const BlockDataID velFieldID = field::addToStorage< VelocityField_T >(blocks, "vel", real_c(0.0), field::fzyx, numGhostLayers);
   const BlockDataID densityFieldID = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, numGhostLayers);
   const BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "Boundary Flag Field", uint_c(2));
#endif

   WALBERLA_MPI_BARRIER()

   const Cell innerOuterSplit = Cell(parameters.getParameter< Vector3< cell_idx_t > >("innerOuterSplit", Vector3< cell_idx_t >(1, 1, 1)));
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const Vector3<int64_t> gpuBlockSize = parameters.getParameter<Vector3<int64_t>>("gpuBlockSize");
   SweepCollection_T sweepCollection(blocks, pdfFieldGPUID, densityFieldGPUID, velFieldGPUID, gpuBlockSize[0], gpuBlockSize[1], gpuBlockSize[2], omega, innerOuterSplit);
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
   std::shared_ptr<UniformGPUScheme< CommunicationStencil_T >> uniformCommunication;
   std::shared_ptr<UniformGeneratedGPUPdfPackInfo< GPUPdfField_T >> uniformPackInfo;

   std::shared_ptr<NonUniformGPUScheme< CommunicationStencil_T >> nonUniformCommunication;
   std::shared_ptr<NonuniformGeneratedGPUPdfPackInfo<GPUPdfField_T >> nonUniformPackInfo;
   if (uniformGrid){
      uniformCommunication = std::make_shared< UniformGPUScheme< CommunicationStencil_T > >(blocks, false);
      uniformPackInfo = std::make_shared<lbm_generated::UniformGeneratedGPUPdfPackInfo< GPUPdfField_T >>(pdfFieldGPUID);
      uniformCommunication->addPackInfo(uniformPackInfo);
   } else {
      nonUniformCommunication = std::make_shared< NonUniformGPUScheme< CommunicationStencil_T > >(blocks);
      nonUniformPackInfo = lbm_generated::setupNonuniformGPUPdfCommunication< GPUPdfField_T >(blocks, pdfFieldGPUID);
      nonUniformCommunication->addPackInfo(nonUniformPackInfo);
   }
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#else
   WALBERLA_LOG_INFO_ON_ROOT("Setting up communication...")
   std::shared_ptr<UniformBufferedScheme< CommunicationStencil_T >> uniformCommunication;
   std::shared_ptr<UniformGeneratedPdfPackInfo< PdfField_T >> uniformPackInfo;

   std::shared_ptr<NonUniformBufferedScheme< CommunicationStencil_T >> nonUniformCommunication;
   std::shared_ptr<NonuniformGeneratedPdfPackInfo<PdfField_T >> nonUniformPackInfo;
   if (uniformGrid){
      uniformCommunication = std::make_shared< UniformBufferedScheme< CommunicationStencil_T > >(blocks);
      uniformPackInfo= std::make_shared<lbm_generated::UniformGeneratedPdfPackInfo< PdfField_T >>(pdfFieldID);
      uniformCommunication->addPackInfo(uniformPackInfo);
   } else {
      nonUniformCommunication = std::make_shared< NonUniformBufferedScheme< CommunicationStencil_T > >(blocks);
      nonUniformPackInfo = lbm_generated::setupNonuniformPdfCommunication< PdfField_T >(blocks, pdfFieldID);
      nonUniformCommunication->addPackInfo(nonUniformPackInfo);
   }
#endif
   WALBERLA_MPI_BARRIER()
   WALBERLA_LOG_INFO_ON_ROOT("Setting up communication done")


   /////////////////////////
   /// BOUNDARY HANDLING ///
   /////////////////////////
   WALBERLA_LOG_INFO_ON_ROOT("Start BOUNDARY HANDLING")
   // create and initialize boundary handling
   const FlagUID fluidFlagUID("Fluid");
   static const walberla::BoundaryUID wallFlagUID("NoSlip");
   for (auto &block: *blocks) {
      auto flagField = block.getData<FlagField_T>( flagFieldID );
      flagField->registerFlag(FlagUID("NoSlip"));
   }

   // write mesh info to file
   if (writeMeshBoundaries) {
      // set NoSlip UID to boundaries that we colored
      mesh::ColorToBoundaryMapper<mesh::TriangleMesh> colorToBoundaryMapper((mesh::BoundaryInfo(wallFlagUID)));
      colorToBoundaryMapper.set(mesh::TriangleMesh::Color(255, 255, 255), mesh::BoundaryInfo(wallFlagUID));
      // mark boundaries
      auto boundaryLocations = colorToBoundaryMapper.addBoundaryInfoToMesh(*mesh);

      mesh::VTKMeshWriter<mesh::TriangleMesh> meshWriter(mesh, "meshBoundaries", 1);
      meshWriter.addDataSource(make_shared<mesh::BoundaryUIDFaceDataSource<mesh::TriangleMesh> >(boundaryLocations));
      meshWriter.addDataSource(make_shared<mesh::ColorFaceDataSource<mesh::TriangleMesh> >());
      meshWriter.addDataSource(make_shared<mesh::ColorVertexDataSource<mesh::TriangleMesh> >());
      meshWriter();
   }

   WALBERLA_LOG_INFO_ON_ROOT("Voxelize mesh")
   mesh::BoundarySetup boundarySetup(blocks, makeMeshDistanceFunction(distanceOctree), numGhostLayers, false);
   WALBERLA_LOG_INFO_ON_ROOT("Voxelize mesh done")

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");
   geometry::initBoundaryHandling<FlagField_T>(*blocks, flagFieldID, boundariesConfig);
   boundarySetup.setFlag<FlagField_T>(flagFieldID, FlagUID("NoSlip"), mesh::BoundarySetup::INSIDE);
   geometry::setNonBoundaryCellsToDomain<FlagField_T>(*blocks, flagFieldID, fluidFlagUID, cell_idx_c(numGhostLayers));

   const wallDistance wallDistanceCallback{mesh};
   std::function<real_t(const Cell&, const Cell&, const shared_ptr< StructuredBlockForest >&, IBlock&) > wallDistanceFunctor = wallDistanceCallback;
   const real_t omegaFinestLevel = lbm_generated::relaxationRateScaling(omega, refinementLevels);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   BoundaryCollection_T boundaryCollection(blocks, flagFieldID, pdfFieldGPUID, fluidFlagUID, omegaFinestLevel, inletVelocity, wallDistanceFunctor, pdfFieldID);
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#else
   BoundaryCollection_T boundaryCollection(blocks, flagFieldID, pdfFieldID, fluidFlagUID, omegaFinestLevel, inletVelocity, wallDistanceFunctor);
#endif
   WALBERLA_MPI_BARRIER()
   WALBERLA_LOG_INFO_ON_ROOT("BOUNDARY HANDLING done")

   //////////////////////////////////
   /// SET UP SWEEPS AND TIMELOOP ///
   //////////////////////////////////
   WALBERLA_LOG_INFO_ON_ROOT("Start SWEEPS AND TIMELOOP")

   // create time loop
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);
   std::shared_ptr<lbm_generated::BasicRecursiveTimeStepGPU<GPUPdfField_T, SweepCollection_T, BoundaryCollection_T>> LBMRefinement;

   if (!uniformGrid) {
      LBMRefinement = std::make_shared<lbm_generated::BasicRecursiveTimeStepGPU<GPUPdfField_T , SweepCollection_T, BoundaryCollection_T> > (
         blocks, pdfFieldGPUID, sweepCollection, boundaryCollection, nonUniformCommunication,
         nonUniformPackInfo);
      LBMRefinement->addRefinementToTimeLoop(timeloop);

   }
   else {
      WALBERLA_LOG_INFO_ON_ROOT("Using uniform Grid")
      timeloop.add() << BeforeFunction(uniformCommunication->getCommunicateFunctor(), "Communication")
                     << Sweep(boundaryCollection.getSweep(BoundaryCollection_T::ALL), "Boundary Conditions");
      timeloop.add() << Sweep(sweepCollection.streamCollide(SweepCollection_T::ALL), "LBM StreamCollide");
   }
#else
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);
   std::shared_ptr<lbm_generated::BasicRecursiveTimeStep<PdfField_T, SweepCollection_T, BoundaryCollection_T>> LBMRefinement;

   if (!uniformGrid) {
      LBMRefinement = std::make_shared<lbm_generated::BasicRecursiveTimeStep<PdfField_T, SweepCollection_T, BoundaryCollection_T> > (
         blocks, pdfFieldID, sweepCollection, boundaryCollection, nonUniformCommunication,
         nonUniformPackInfo);
      LBMRefinement->addRefinementToTimeLoop(timeloop);

   }
   else {
      WALBERLA_LOG_INFO_ON_ROOT("Using uniform Grid")
      timeloop.add() << BeforeFunction(uniformCommunication->getCommunicateFunctor(), "Communication")
                     << Sweep(boundaryCollection.getSweep(BoundaryCollection_T::ALL), "Boundary Conditions");
      timeloop.add() << Sweep(sweepCollection.streamCollide(SweepCollection_T::ALL), "LBM StreamCollide");
   }
#endif
   //////////////////
   /// VTK OUTPUT ///
   //////////////////
   WALBERLA_LOG_INFO_ON_ROOT("SWEEPS AND TIMELOOP done")

   auto VTKWriter = walberlaEnv.config()->getOneBlock("VTKWriter");
   const uint_t vtkWriteFrequency = VTKWriter.getParameter<uint_t>("vtkWriteFrequency", 0);
   const bool writeVelocity = VTKWriter.getParameter<bool>("velocity");
   const bool writeDensity = VTKWriter.getParameter<bool>("density");
   const bool writeFlag = VTKWriter.getParameter<bool>("flag");
   const bool writeOnlySlice = VTKWriter.getParameter<bool>("writeOnlySlice", true);
   if (vtkWriteFrequency > 0)
   {
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_FlowAroundSphere",
                                                      "simulation_step", false, true, true, false, 0);

      vtkOutput->addBeforeFunction([&]() {
         for (auto& block : *blocks)
         {
            sweepCollection.calculateMacroscopicParameters(&block);
         }

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         gpu::fieldCpy< VelocityField_T, gpu::GPUField< real_t > >(blocks, velFieldID, velFieldGPUID);
         gpu::fieldCpy< ScalarField_T , gpu::GPUField< real_t > >(blocks, densityFieldID, densityFieldGPUID);
#endif
      });


      if (writeOnlySlice)
      {
         const AABB sliceAABB(finalDomain.xMin(), finalDomain.yMin(), finalDomain.center()[2] - resolution.min(), finalDomain.xMax(), finalDomain.yMax(), finalDomain.center()[2] + resolution.min());
         vtkOutput->addCellInclusionFilter(vtk::AABBCellFilter(sliceAABB));
      }

      if (writeVelocity)
      {
         auto velWriter = make_shared<field::VTKWriter<VelocityField_T> >(velFieldID, "velocity");
         vtkOutput->addCellDataWriter(velWriter);
      }
      if (writeDensity)
      {
         auto densityWriter = make_shared<field::VTKWriter<ScalarField_T> >(densityFieldID, "density");
         vtkOutput->addCellDataWriter(densityWriter);
      }
      if (writeFlag)
      {
         auto flagWriter = make_shared<field::VTKWriter<FlagField_T> >(flagFieldID, "flag");
         vtkOutput->addCellDataWriter(flagWriter);
      }
      timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   }

   // log remaining time
   const real_t remainingTimeLoggerFrequency = loggingParameters.getParameter<real_t>("remainingTimeLoggerFrequency", 3.0); // in seconds
   if (uint_c(remainingTimeLoggerFrequency) > 0)
   {
      timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency), "remaining time logger");
   }

   // LBM stability check
   auto CheckerParameters = walberlaEnv.config()->getOneBlock("StabilityChecker");
   const uint_t checkFrequency = CheckerParameters.getParameter<uint_t>("checkFrequency", uint_t(0));
   if (checkFrequency > 0) {
      auto checkFunction = [](PdfField_T::value_type value) {return value < math::abs(PdfField_T::value_type(10));};
      timeloop.addFuncAfterTimeStep( makeSharedFunctor( field::makeStabilityChecker< PdfField_T, FlagField_T >( walberlaEnv.config(), blocks, pdfFieldID, flagFieldID, fluidFlagUID, checkFunction) ),"Stability check" );
   }


   WALBERLA_LOG_INFO_ON_ROOT("Execute single timestep to fully complete the preprocessing")
   // Do a single timestep to make sure all setups are completed before benchmarking
   timeloop.singleStep();
   timeloop.setCurrentTimeStepToZero();

   WALBERLA_MPI_BARRIER()
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif
   WALBERLA_LOG_INFO_ON_ROOT("Execute single timestep to fully complete the preprocessing done")


   //////////////////////
   /// RUN SIMULATION ///
   //////////////////////
   const lbm_generated::PerformanceEvaluation<FlagField_T> performance(blocks, flagFieldID, fluidFlagUID);
   field::CellCounter< FlagField_T > fluidCells( blocks, flagFieldID, fluidFlagUID );
   fluidCells();

   WALBERLA_LOG_INFO_ON_ROOT( "Simulating FlowAroundSphere with " << fluidCells.numberOfCells() << " fluid cells (in total on all levels)")
   setup.logSetup();
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