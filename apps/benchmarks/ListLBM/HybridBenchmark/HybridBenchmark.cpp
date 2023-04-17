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
#include "python_coupling/DictWrapper.h"
#include "python_coupling/PythonCallback.h"


#include "timeloop/SweepTimeloop.h"

#if defined(WALBERLA_BUILD_WITH_CUDA)
#   include "cuda/AddGPUFieldToStorage.h"
#   include "cuda/DeviceSelectMPI.h"
#   include "cuda/HostFieldAllocator.h"
#   include "cuda/NVTX.h"
#   include "cuda/ParallelStreams.h"
#   include "cuda/communication/UniformGPUScheme.h"
#   include "cuda/lbm/CombinedInPlaceGpuPackInfo.h"
#else
#   include "lbm/communication/CombinedInPlaceCpuPackInfo.h"
#endif

#include "SparseLBMInfoHeader.h"
#include "DenseLBMInfoHeader.h"
#include "InitSpherePacking.h"
#include "SetupHybridCommunication.h"
#include <iostream>
#include <fstream>

using namespace walberla;

uint_t numGhostLayers = uint_t(1);

using flag_t = walberla::uint8_t;
using FlagField_T = FlagField<flag_t>;

#if defined(WALBERLA_BUILD_WITH_CUDA)
using GPUField = cuda::GPUField< real_t >;
#endif

auto pdfFieldAdder = [](IBlock *const block, StructuredBlockStorage *const storage) {
   return new PdfField_T(storage->getNumberOfXCells(*block), storage->getNumberOfYCells(*block),
                         storage->getNumberOfZCells(*block), uint_t(1), field::fzyx,
                         make_shared<field::AllocateAligned<real_t, 64> >());
};

void setFlagFieldToPorosity(IBlock * block, const BlockDataID flagFieldId, const real_t porosity, const FlagUID noSlipFlagUID) {
   auto flagField    = block->getData< FlagField_T >(flagFieldId);
   auto noSlipFlag = flagField->getFlag(noSlipFlagUID);
   real_t boundary_fraction = 1.0 - porosity;
   real_t nextBoundary = 0;
   for(auto it = flagField->begin(); it != flagField->end();++it) {
      if(nextBoundary  >= 1.0) {
         nextBoundary -= 1.0;
         addFlag(flagField->get(it), noSlipFlag);
      }
      nextBoundary += boundary_fraction;
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

int main(int argc, char **argv)
{
   walberla::Environment walberlaEnv(argc, argv);

#if defined(WALBERLA_BUILD_WITH_CUDA)
   cuda::selectDeviceBasedOnMpiRank();
   WALBERLA_CUDA_CHECK(cudaPeekAtLastError())
#endif

   mpi::MPIManager::instance()->useWorldComm();


   for (auto cfg = python_coupling::configBegin(argc, argv); cfg != python_coupling::configEnd(); ++cfg)
   {
      WALBERLA_MPI_WORLD_BARRIER()

#if defined(WALBERLA_BUILD_WITH_CUDA)
      WALBERLA_CUDA_CHECK(cudaPeekAtLastError())
#endif

      auto config = *cfg;
      logging::configureLogging(config);
      ///////////////////////
      /// PARAMETER INPUT ///
      ///////////////////////
      auto parameters = config->getBlock("Parameters");
      auto domainParameters = config->getBlock("DomainSetup");
      const Vector3< real_t > initialVelocity = parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >());
      const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(10));
      Vector3< int > InnerOuterSplit = parameters.getParameter< Vector3< int > >("innerOuterSplit", Vector3< int >(1, 1, 1));
      const bool weak_scaling = domainParameters.getParameter< bool >("weakScaling", false); // weak or strong scaling
      const real_t remainingTimeLoggerFrequency = parameters.getParameter< real_t >("remainingTimeLoggerFrequency", 3.0); // in seconds
      const real_t omega = parameters.getParameter< real_t > ( "omega", real_c( 1.4 ) );
      const uint_t vtkWriteFrequency = parameters.getParameter< uint_t >("vtkWriteFrequency", uint_t(0));

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

      if (geometrySetup == "randomNoslip") {
         real_t dx = 1;
         blocks = walberla::blockforest::createUniformBlockGrid( blocksPerDimension[0], blocksPerDimension[1], blocksPerDimension[2], cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2], dx);
         flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");
         const real_t porosity = parameters.getParameter< real_t >("porosity");
         geometry::initBoundaryHandling<FlagField_T>(*blocks, flagFieldId, boundariesConfig);
         for (auto& block : *blocks) {
            setFlagFieldToPorosity(&block,flagFieldId,porosity,noslipFlagUID);
         }
         geometry::setNonBoundaryCellsToDomain<FlagField_T>(*blocks, flagFieldId, fluidFlagUID);
      }
      else if (geometrySetup == "spheres") {
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
      else if (geometrySetup == "geometryFile") {
         std::string meshFile  = domainParameters.getParameter< std::string >("meshFile");
         const Vector3< uint_t > TotalCells(cellsPerBlock[0] * blocksPerDimension[0],
                                            cellsPerBlock[1] * blocksPerDimension[1],
                                            cellsPerBlock[2] * blocksPerDimension[2]);
         const real_t scalingFactor = 0.03;// (32.0 / real_c(TotalCells.min())) * 0.1;

         const Vector3< real_t > dx(scalingFactor, scalingFactor, scalingFactor);
         WALBERLA_LOG_INFO_ON_ROOT("Using mesh from " << meshFile << ".")

         auto mesh = make_shared< mesh::TriangleMesh >();
         mesh->request_vertex_colors();
         mesh::readAndBroadcast(meshFile, *mesh);

         vertexToFaceColor(*mesh, mesh::TriangleMesh::Color(255, 255, 255));
         auto triDist = make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh);
         auto distanceOctree = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(triDist);
         distanceOctree->writeVTKOutput("distanceOctree");

         auto aabb = computeAABB(*mesh);
         mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, dx, mesh::makeExcludeMeshInterior(distanceOctree, dx[0]));
         //bfc.setPeriodicity(periodicity);
         blocks = bfc.createStructuredBlockForest(cellsPerBlock, blocksPerDimension);
         flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");
         mesh::BoundarySetup boundarySetup(blocks, makeMeshDistanceFunction(distanceOctree), numGhostLayers);

         geometry::initBoundaryHandling<FlagField_T>(*blocks, flagFieldId, boundariesConfig);
         boundarySetup.setFlag<FlagField_T>(flagFieldId, FlagUID("NoSlip"), mesh::BoundarySetup::INSIDE);
         geometry::setNonBoundaryCellsToDomain<FlagField_T>(*blocks, flagFieldId, fluidFlagUID);
      }
      else {
         WALBERLA_ABORT_NO_DEBUG_INFO("Invalid value for 'geometrySetup'. Allowed values are 'randomNoslip', "
                                      "'spheres', 'geometryFile'")
      }






      const std::string timeStepStrategy = parameters.getParameter< std::string >("timeStepStrategy", "noOverlap");
      if(timeStepStrategy != "noOverlap") {
         for (uint_t i = 0; i < 3; ++i) {
            if (int_c(cellsPerBlock[i]) <= InnerOuterSplit[i] * 2) {
               WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller or increase cellsPerBlock")
            }
         }
      }

      //Calculate Poriosity
      uint_t FluidCellsOnProcess = 0;
      uint_t numberOfCells = 0;
      for (auto& block : *blocks)
      {
         auto* flagField = block.getData< FlagField_T >(flagFieldId);
         auto domainFlag = flagField->getFlag(fluidFlagUID);
         for (auto it = flagField->begin(); it != flagField->end(); ++it)
         {
            if (isFlagSet(it, domainFlag))
            {
               FluidCellsOnProcess++;
            }
            numberOfCells++;
         }
      }
      real_t blockPorosity = real_c(FluidCellsOnProcess) / real_c(numberOfCells);
      uint_t TotalNumberOfFluidCells = 0;
      WALBERLA_MPI_SECTION()
      {
         TotalNumberOfFluidCells = walberla::mpi::reduce(FluidCellsOnProcess, walberla::mpi::SUM, 0);
         walberla::mpi::broadcastObject( TotalNumberOfFluidCells, 0);
      }
      uint_t TotalNumberOfCells = blocks->getNumberOfXCells() * blocks->getNumberOfYCells() * blocks->getNumberOfZCells();
      const real_t realPorosity = real_c(TotalNumberOfFluidCells) / real_c(TotalNumberOfCells);
      WALBERLA_LOG_INFO_ON_ROOT("Porosity is " << realPorosity)



      const real_t porositySwitch = parameters.getParameter< real_t >("porositySwitch");
      bool runningIndirectAdressing;
      if(blockPorosity < porositySwitch)
         runningIndirectAdressing = true;
      else
         runningIndirectAdressing = false;

      if(runningIndirectAdressing)
      {
         ///////////////////////////////////////////////////////////////////////////////////////////////////////
         ////////////////////////////////////  INDIRECT ADDRESSING PART  ///////////////////////////////////////
         ///////////////////////////////////////////////////////////////////////////////////////////////////////

         WALBERLA_LOG_INFO("Block " << uint_c(MPIManager::instance()->rank()) << " is running Simulation with indirect addressing with porosity " << blockPorosity)
         BlockDataID pdfListId = lbm::addListToStorage< List_T >(blocks, "LBM list (FIdx)", InnerOuterSplit);
         WALBERLA_LOG_INFO_ON_ROOT("Start initialisation of the linked-list structure")

         WcTimer lbmTimer;
         for (auto& block : *blocks)
         {
            auto* lbmList = block.getData< List_T >(pdfListId);
            WALBERLA_CHECK_NOT_NULLPTR(lbmList)
            lbmList->fillFromFlagField< FlagField_T >(block, flagFieldId, fluidFlagUID);
         }

   #if defined(WALBERLA_BUILD_WITH_CUDA)
         const Vector3< int32_t > gpuBlockSize =
            parameters.getParameter< Vector3< int32_t > >("gpuBlockSize", Vector3< int32_t >(128, 1, 1));
         lbmpy::SparseLBSweep kernel(pdfListId, omega, gpuBlockSize[0], gpuBlockSize[1], gpuBlockSize[2]);
   #else
         lbmpy::SparseLBSweep kernel(pdfListId, omega);
   #endif

         auto tracker = make_shared< lbm::TimestepTracker >(0);
         lbmpy::SparseMacroSetter setterSweep(pdfListId);
         lbmpy::SparseUBB ubb(blocks, pdfListId, initialVelocity[0]);
         lbmpy::SparsePressure pressureOutflow(blocks, pdfListId, 1.0);
         // lbm::SparseNoSlip noSlip(blocks, pdfListId);

         ubb.fillFromFlagField< FlagField_T >(blocks, flagFieldId, inflowUID, fluidFlagUID);
         pressureOutflow.fillFromFlagField< FlagField_T >(blocks, flagFieldId, PressureOutflowUID, fluidFlagUID);
         // noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldId, noslipFlagUID, fluidFlagUID);
         for (auto& block : *blocks)
         {
            setterSweep(&block);
         }
         bool denseCommunication = true;
         lbmpy::ListCommunicationSetup< FlagField_T, Stencil_T >(pdfListId, flagFieldId, blocks, denseCommunication);
         lbmTimer.end();
         WALBERLA_LOG_INFO_ON_ROOT("Initialisation of the list structures needed " << lbmTimer.last() << " s")

   #if defined(WALBERLA_BUILD_WITH_CUDA)

         const bool cudaEnabledMPI = parameters.getParameter< bool >("cudaEnabledMPI", true);
         auto packInfo =
            make_shared< lbm::CombinedInPlaceGpuPackInfo< lbmpy::SparsePackInfoEven, lbmpy::SparsePackInfoOdd > >(
               tracker, pdfListId);
         cuda::communication::UniformGPUScheme< Stencil_T > comm(blocks, cudaEnabledMPI);
   #else
         auto packInfo =
            make_shared< lbm::CombinedInPlaceCpuPackInfo< lbmpy::SparsePackInfoEven, lbmpy::SparsePackInfoOdd > >(
               tracker, pdfListId);
         blockforest::communication::UniformBufferedScheme< Stencil_T > comm(blocks);
   #endif
         WALBERLA_LOG_INFO_ON_ROOT("Finished setting up communication")
         comm.addPackInfo(packInfo);
         auto communicate       = std::function< void() >([&]() { comm.communicate(); });
         auto start_communicate = std::function< void() >([&]() { comm.startCommunication(); });
         auto wait_communicate  = std::function< void() >([&]() { comm.wait(); });
         //////////////////////////////////
         /// SET UP SWEEPS AND TIMELOOP ///
         //////////////////////////////////

         const bool runBoundaries = parameters.getParameter< bool >("runBoundaries", true);

         auto normalTimeStep = [&]() {
            communicate();
            for (auto& block : *blocks)
            {
               if (runBoundaries)
               {
                  ubb(&block, tracker->getCounter());
                  pressureOutflow(&block, tracker->getCounter());
               }
               kernel(&block, tracker->getCounterPlusOne());
            }
            tracker->advance();
         };

         auto simpleOverlapTimeStep = [&]() {
            start_communicate();
            for (auto& block : *blocks)
            {
               if (runBoundaries)
               {
                  ubb(&block, tracker->getCounter());
                  pressureOutflow(&block, tracker->getCounter());
               }
               kernel.inner(&block, tracker->getCounterPlusOne());
            }
            wait_communicate();
            for (auto& block : *blocks)
            {
               kernel.outer(&block, tracker->getCounterPlusOne());
            }
            tracker->advance();
         };

         auto kernelOnlyFunc = [&]() {
            for (auto& block : *blocks) {
               if (runBoundaries)
               {
                  ubb(&block, tracker->getCounter());
                  pressureOutflow(&block, tracker->getCounter());
               }
               kernel(&block, tracker->getCounter());
            }
            tracker->advance();
         };

         std::function< void() > timeStep;
         if (timeStepStrategy == "noOverlap")
            timeStep = normalTimeStep;
         else if (timeStepStrategy == "Overlap")
            timeStep = simpleOverlapTimeStep;
         else if (timeStepStrategy == "kernelOnly")
         {
            WALBERLA_LOG_INFO_ON_ROOT(
               "Running only compute kernel without boundary - this makes only sense for benchmarking!")
            timeStep = kernelOnlyFunc;
         }
         else
         {
            WALBERLA_ABORT_NO_DEBUG_INFO("Invalid value for 'timeStepStrategy'. Allowed values are 'noOverlap', "
                                         "'simpleOverlap', 'kernelOnly'")
         }

         SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);
         timeloop.add() << BeforeFunction(timeStep, "Timestep") << Sweep([](IBlock*) {}, "Dummy");

         timeloop.addFuncAfterTimeStep(
            timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
            "remaining time logger");

         if (vtkWriteFrequency > 0)
         {
            auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtkSparse", vtkWriteFrequency, 0, false, "vtk_out",
                                                            "simulation_step", false, false, true, false, 0);

   #if defined(WALBERLA_BUILD_WITH_CUDA)
            vtkOutput->addBeforeFunction([&]() {
               for (auto& block : *blocks)
               {
                  List_T* lbmList = block.getData< List_T >(pdfListId);
                  lbmList->copyPDFSToCPU();
               }
            });
   #endif

            field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagFieldId);
            fluidFilter.addFlag(fluidFlagUID);
            vtkOutput->addCellInclusionFilter(fluidFilter);

            auto velWriter = make_shared< lbm::ListVelocityVTKWriter< List_T, real_t > >(pdfListId, tracker, "velocity");
            auto densityWriter = make_shared< lbm::ListDensityVTKWriter< List_T, real_t > >(pdfListId, "density");

            vtkOutput->addCellDataWriter(velWriter);
            vtkOutput->addCellDataWriter(densityWriter);

            timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
            vtk::writeDomainDecomposition(blocks, "domain_decomposition", "vtk_out", "write_call", true, true, 0);
         }

         lbm::PerformanceEvaluation< FlagField_T > performance(blocks, flagFieldId, fluidFlagUID);

         WALBERLA_LOG_INFO_ON_ROOT("Simulating ListLBM:"
                                   "\n timesteps:                  "
                                   << timesteps << "\n relaxation rate:            " << omega)

         int warmupSteps     = parameters.getParameter< int >("warmupSteps", 2);
         int outerIterations = parameters.getParameter< int >("outerIterations", 1);
         for (int i = 0; i < warmupSteps; ++i)
            timeloop.singleStep();

         for (int outerIteration = 0; outerIteration < outerIterations; ++outerIteration)
         {
            timeloop.setCurrentTimeStepToZero();

            WcTimingPool timeloopTiming;
            WcTimer simTimer;

            WALBERLA_MPI_WORLD_BARRIER()
   #if defined(WALBERLA_BUILD_WITH_CUDA)
            WALBERLA_CUDA_CHECK(cudaPeekAtLastError())
            cudaDeviceSynchronize();
   #endif

            simTimer.start();
            timeloop.run(timeloopTiming);
   #if defined(WALBERLA_BUILD_WITH_CUDA)
            WALBERLA_CUDA_CHECK(cudaPeekAtLastError())
            cudaDeviceSynchronize();
   #endif
            simTimer.end();

            WALBERLA_LOG_INFO_ON_ROOT("Simulation finished")
            real_t time = simTimer.max();
            WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
            performance.logResultOnRoot(timesteps, time);

            const auto reducedTimeloopTiming = timeloopTiming.getReduced();
            WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)

            WALBERLA_ROOT_SECTION(){
               std::ofstream myfile;
               myfile.open ("results.txt", std::ios::app);
               myfile << blockPorosity << " " << performance.mflupsPerProcess(timesteps, time) << " " << performance.mflups(timesteps, time) << std::endl;
               myfile.close();
            }
         }
         //printResidentMemoryStatistics();
      }
      else
      {
         ///////////////////////////////////////////////////////////////////////////////////////////////////////
         ////////////////////////////////////  DIRECT ADDRESSING PART  /////////////////////////////////////////
         ///////////////////////////////////////////////////////////////////////////////////////////////////////

         WALBERLA_LOG_INFO("Block " << uint_c(MPIManager::instance()->rank()) << " is running Simulation with direct addressing with porosity " << blockPorosity)

         BlockDataID pdfFieldId     = blocks->addStructuredBlockData< PdfField_T >(pdfFieldAdder, "PDFs");
         BlockDataID velFieldId     = field::addToStorage< VelocityField_T >(blocks, "velocity", real_t(0), field::fzyx);
         BlockDataID densityFieldId = field::addToStorage< ScalarField_T >(blocks, "density", real_t(1.0), field::fzyx);
         pystencils::DenseMacroSetter setterSweep(pdfFieldId);
         pystencils::DenseMacroGetter getterSweep(densityFieldId, pdfFieldId, velFieldId);

         for (auto& block : *blocks)
         {
            setterSweep(&block);
         }
         SetupHybridCommunication<FlagField_T, Stencil_T > ( blocks, flagFieldId, fluidFlagUID);

         auto tracker = make_shared< lbm::TimestepTracker >(0);

   #if defined(WALBERLA_BUILD_WITH_CUDA)
         BlockDataID pdfFieldIdGPU = cuda::addGPUFieldToStorage< PdfField_T >(blocks, pdfFieldId, "PDFs on GPU", true);
         const Vector3< int32_t > gpuBlockSize =
            parameters.getParameter< Vector3< int32_t > >("gpuBlockSize", Vector3< int32_t >(128, 1, 1));
         lbm::DenseLBSweep kernel(pdfFieldIdGPU, omega, gpuBlockSize[0], gpuBlockSize[1], gpuBlockSize[2], Cell(cell_idx_c(InnerOuterSplit[0]), cell_idx_c(InnerOuterSplit[1]), cell_idx_c(InnerOuterSplit[2])));
         lbm::DenseUBB ubb(blocks, pdfFieldIdGPU, initialVelocity[0]);
         lbm::DensePressure pressureOutflow(blocks, pdfFieldIdGPU, 1.0);
         lbm::DenseNoSlip noSlip(blocks, pdfFieldIdGPU);

         const bool cudaEnabledMPI = parameters.getParameter< bool >("cudaEnabledMPI", true);
         auto packInfo = make_shared< lbm::CombinedInPlaceGpuPackInfo< lbm::DensePackInfoEven, lbm::DensePackInfoOdd > >(tracker, pdfFieldIdGPU);
         cuda::communication::UniformGPUScheme< Stencil_T > comm(blocks, cudaEnabledMPI);
   #else
         lbm::DenseLBSweep kernel(pdfFieldId, omega, Cell(cell_idx_c(InnerOuterSplit[0]), cell_idx_c(InnerOuterSplit[1]), cell_idx_c(InnerOuterSplit[2])));
         lbm::DenseUBB ubb(blocks, pdfFieldId, initialVelocity[0]);
         lbm::DensePressure pressureOutflow(blocks, pdfFieldId, 1.0);
         lbm::DenseNoSlip noSlip(blocks, pdfFieldId);

         auto packInfo = make_shared< lbm::CombinedInPlaceCpuPackInfo< lbm::DensePackInfoEven, lbm::DensePackInfoOdd > >( tracker, pdfFieldId);
         blockforest::communication::UniformBufferedScheme< Stencil_T > comm(blocks);
   #endif

         ubb.fillFromFlagField< FlagField_T >(blocks, flagFieldId, inflowUID, fluidFlagUID);
         pressureOutflow.fillFromFlagField< FlagField_T >(blocks, flagFieldId, PressureOutflowUID, fluidFlagUID);
         noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldId, noslipFlagUID, fluidFlagUID);

         comm.addPackInfo(packInfo);
         auto communicate       = std::function< void() >([&]() { comm.communicate(); });
         auto start_communicate = std::function< void() >([&]() { comm.startCommunication(); });
         auto wait_communicate  = std::function< void() >([&]() { comm.wait(); });

         //////////////////////////////////
         /// SET UP SWEEPS AND TIMELOOP ///
         //////////////////////////////////

         const bool runBoundaries           = parameters.getParameter< bool >("runBoundaries", true);

         auto normalTimeStep = [&]() {
            communicate();
            for (auto& block : *blocks)
            {
               if (runBoundaries)
               {
                  ubb(&block, tracker->getCounter());
                  pressureOutflow(&block, tracker->getCounter());
                  noSlip(&block, tracker->getCounter());
               }
               kernel(&block, tracker->getCounterPlusOne());
            }
            tracker->advance();
         };

         auto simpleOverlapTimeStep = [&]() {
            start_communicate();
            for (auto& block : *blocks)
            {
               if (runBoundaries)
               {
                  ubb(&block, tracker->getCounter());
                  pressureOutflow(&block, tracker->getCounter());
                  noSlip(&block, tracker->getCounter());
               }
               kernel.inner(&block, tracker->getCounterPlusOne());
            }
            wait_communicate();
            for (auto& block : *blocks)
            {
               kernel.outer(&block, tracker->getCounterPlusOne());
            }
            tracker->advance();
         };

         auto kernelOnlyFunc = [&]() {
            for (auto& block : *blocks) {
               if (runBoundaries)
               {
                  ubb(&block, tracker->getCounter());
                  pressureOutflow(&block, tracker->getCounter());
                  noSlip(&block, tracker->getCounter());
               }
               kernel(&block, tracker->getCounter());
            }
            tracker->advance();
         };


         std::function< void() > timeStep;
         if (timeStepStrategy == "noOverlap")
            timeStep = normalTimeStep;
         else if (timeStepStrategy == "Overlap")
            timeStep = simpleOverlapTimeStep;
         else if (timeStepStrategy == "kernelOnly")
         {
            WALBERLA_LOG_INFO_ON_ROOT(
               "Running only compute kernel without boundary - this makes only sense for benchmarking!")
            timeStep = kernelOnlyFunc;
         }
         else
         {
            WALBERLA_ABORT_NO_DEBUG_INFO("Invalid value for 'timeStepStrategy'. Allowed values are 'noOverlap', "
                                         "'simpleOverlap', 'kernelOnly'")
         }

         SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);
         timeloop.add() << BeforeFunction(timeStep, "Timestep") << Sweep([](IBlock*) {}, "Dummy");

         timeloop.addFuncAfterTimeStep(
            timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
            "remaining time logger");

         if (vtkWriteFrequency > 0)
         {
            auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtkSparse", vtkWriteFrequency, 0, false, "vtk_out",
                                                            "simulation_step", false, false, true, false, 0);

   #if defined(WALBERLA_BUILD_WITH_CUDA)
            vtkOutput->addBeforeFunction([&]() {
               cuda::fieldCpy< PdfField_T , GPUField >(blocks, pdfFieldId, pdfFieldIdGPU);
               for (auto& block : *blocks)
               {
                  getterSweep(&block);
               }
            });
   #else
            vtkOutput->addBeforeFunction([&]() {
               for (auto& block : *blocks)
               {
                  getterSweep(&block);
               }
            });
   #endif

            field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagFieldId);
            fluidFilter.addFlag(fluidFlagUID);
            vtkOutput->addCellInclusionFilter(fluidFilter);

            auto velWriter = make_shared<field::VTKWriter<VelocityField_T> >(velFieldId, "velocity");
            auto densityWriter = make_shared<field::VTKWriter<ScalarField_T> >(densityFieldId, "density");

            vtkOutput->addCellDataWriter(velWriter);
            vtkOutput->addCellDataWriter(densityWriter);

            timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
            vtk::writeDomainDecomposition(blocks, "domain_decomposition", "vtk_out", "write_call", true, true, 0);

         }

         lbm::PerformanceEvaluation< FlagField_T > performance(blocks, flagFieldId, fluidFlagUID);

         WALBERLA_LOG_INFO_ON_ROOT("Simulating ListLBM:"
                                      "\n timesteps:                  " << timesteps
                                   << "\n relaxation rate:            " << omega)

         int warmupSteps     = parameters.getParameter< int >("warmupSteps", 2);
         int outerIterations = parameters.getParameter< int >("outerIterations", 1);
         for (int i = 0; i < warmupSteps; ++i)
            timeloop.singleStep();

         for (int outerIteration = 0; outerIteration < outerIterations; ++outerIteration)
         {
            timeloop.setCurrentTimeStepToZero();

            WcTimingPool timeloopTiming;
            WcTimer simTimer;

            WALBERLA_MPI_WORLD_BARRIER()
   #if defined(WALBERLA_BUILD_WITH_CUDA)
            WALBERLA_CUDA_CHECK(cudaPeekAtLastError())
            cudaDeviceSynchronize();
   #endif

            simTimer.start();
            timeloop.run(timeloopTiming);
   #if defined(WALBERLA_BUILD_WITH_CUDA)
            WALBERLA_CUDA_CHECK(cudaPeekAtLastError())
            cudaDeviceSynchronize();
   #endif
            simTimer.end();

            WALBERLA_LOG_INFO_ON_ROOT("Simulation finished")
            real_t time = simTimer.max();
            WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
            performance.logResultOnRoot(timesteps, time);

            const auto reducedTimeloopTiming = timeloopTiming.getReduced();
            WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)

            WALBERLA_ROOT_SECTION(){
               std::ofstream myfile;
               myfile.open ("results.txt", std::ios::app);
               myfile << blockPorosity << " " << performance.mflupsPerProcess(timesteps, time) << " " << performance.mflups(timesteps, time) << std::endl;
               myfile.close();
            }
         }
         //printResidentMemoryStatistics();
      } //direct addressing
   }//config

   return EXIT_SUCCESS;
}
