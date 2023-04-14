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
//! \file ListLBM.cpp
//! \author Philipp Suffa
//
//======================================================================================================================

#include "blockforest/communication/UniformBufferedScheme.h"
#include "blockforest/Initialization.h"

#include "core/Environment.h"
#include "core/SharedFunctor.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/MemoryUsage.h"

#include "field/AddToStorage.h"
#include "field/StabilityChecker.h"
#include "field/vtk/VTKWriter.h"

#include "geometry/InitBoundaryHandling.h"

#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/PerformanceEvaluation.h"
#include "lbm/list/AddToStorage.h"
#include "lbm/list/ListVTK.h"

#include "mesh/blockforest/BlockExclusion.h"
#include "mesh/blockforest/BlockForestInitialization.h"
#include "mesh/boundary/BoundaryInfo.h"
#include "mesh/boundary/BoundaryLocation.h"
#include "mesh/boundary/BoundaryLocationFunction.h"
#include "mesh/boundary/BoundarySetup.h"
#include "mesh/boundary/BoundaryUIDFaceDataSource.h"
#include "mesh/boundary/ColorToBoundaryMapper.h"

#include "timeloop/SweepTimeloop.h"

#include "mesh_common/DistanceComputations.h"
#include "mesh_common/DistanceFunction.h"
#include "mesh_common/MatrixVectorOperations.h"
#include "mesh_common/MeshIO.h"
#include "mesh_common/MeshOperations.h"
#include "mesh_common/TriangleMeshes.h"
#include "mesh_common/distance_octree/DistanceOctree.h"
#include "mesh_common/vtk/CommonDataSources.h"
#include "mesh_common/vtk/VTKMeshWriter.h"

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

#include "ListLBMInfoHeader.h"
#include <iostream>
#include <fstream>

using namespace walberla;
using PackInfoEven_T = lbmpy::PackInfoEven;
using PackInfoOdd_T = lbmpy::PackInfoOdd;

uint_t numGhostLayers = uint_t(1);

using flag_t = walberla::uint8_t;
using FlagField_T = FlagField<flag_t>;

#if defined(WALBERLA_BUILD_WITH_CUDA)
using GPUField = cuda::GPUField< real_t >;
#endif

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

   ///////////////////////
   /// PARAMETER INPUT ///
   ///////////////////////
   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");
   real_t reference_velocity = parameters.getParameter< real_t >("reference_velocity", real_c(0.9));
   real_t viscosity = parameters.getParameter< real_t >("viscosity", real_c(0.9));
   const Vector3< real_t > initialVelocity = parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >());
   const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(10));
   Vector3< int > InnerOuterSplit = parameters.getParameter< Vector3< int > >("innerOuterSplit", Vector3< int >(1, 1, 1));
   const bool weak_scaling = parameters.getParameter< bool >("weakScaling", false); // weak or strong scaling
   const real_t remainingTimeLoggerFrequency = parameters.getParameter< real_t >("remainingTimeLoggerFrequency", 3.0); // in seconds
   auto loggingParameters         = walberlaEnv.config()->getOneBlock("Logging");
   const bool WriteDistanceOctree = loggingParameters.getParameter< bool >("WriteDistanceOctree", false);
   auto domainParameters = walberlaEnv.config()->getOneBlock("DomainSetup");
   std::string meshFile  = domainParameters.getParameter< std::string >("meshFile");
   const Vector3< bool > periodicity = domainParameters.getParameter< Vector3< bool > >("periodic", Vector3< bool >(false));


   Vector3< uint_t > cellsPerBlock;
   Vector3< uint_t > blocksPerDimension;
   uint_t nrOfProcesses = uint_c(MPIManager::instance()->numProcesses());

   if (!domainParameters.isDefined("blocks"))
   {
      if (weak_scaling)
      {
         Vector3< uint_t > cells = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
         blockforest::calculateCellDistribution(cells, nrOfProcesses, blocksPerDimension, cellsPerBlock);
         cellsPerBlock = cells;
      }
      else
      {
         Vector3< uint_t > cells = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
         blockforest::calculateCellDistribution(cells, nrOfProcesses, blocksPerDimension, cellsPerBlock);
      }
   }
   else
   {
      cellsPerBlock      = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
      blocksPerDimension = domainParameters.getParameter< Vector3< uint_t > >("blocks");
   }

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellsPerBlock)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blocksPerDimension)

   const Vector3< uint_t > TotalCells(cellsPerBlock[0] * blocksPerDimension[0],
                                      cellsPerBlock[1] * blocksPerDimension[1],
                                      cellsPerBlock[2] * blocksPerDimension[2]);
   const real_t scalingFactor = 0.03;// (32.0 / real_c(TotalCells.min())) * 0.1;

   const Vector3< real_t > dx(scalingFactor, scalingFactor, scalingFactor);

   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   WALBERLA_LOG_INFO_ON_ROOT("Using mesh from " << meshFile << ".")

   // read in mesh with vertex colors on a single process and broadcast it
   auto mesh = make_shared< mesh::TriangleMesh >();
   mesh->request_vertex_colors();
   mesh::readAndBroadcast(meshFile, *mesh);

   vertexToFaceColor(*mesh, mesh::TriangleMesh::Color(255, 255, 255));
   auto triDist = make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh);
   auto distanceOctree = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(triDist);
   if (WriteDistanceOctree) { distanceOctree->writeVTKOutput("distanceOctree"); }

   const uint_t vtkWriteFrequency = parameters.getParameter< uint_t >("vtkWriteFrequency", uint_t(0));

   ///////////////////////////
   /// CREATE BLOCK FOREST ///
   ///////////////////////////

   auto aabb = computeAABB(*mesh);
   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, dx, mesh::makeExcludeMeshInterior(distanceOctree, dx[0]));

   bfc.setPeriodicity(periodicity);
   auto blocks = bfc.createStructuredBlockForest(cellsPerBlock, blocksPerDimension);
   WALBERLA_LOG_INFO_ON_ROOT("Created Blockforest")
   BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");

   uint_t cells_y          = blocks->getYSize() * cellsPerBlock[1];
   real_t maximum_veloctiy = initialVelocity[0];

   real_t Cu = reference_velocity / maximum_veloctiy;
   real_t Ct = dx.min() / Cu;

   real_t viscosityLattice = viscosity * Ct / (dx.min() * dx.min());
   real_t omega            = real_c(1.0 / (3.0 * viscosityLattice + 0.5));

   real_t reynolds_number = (real_t(cells_y) * maximum_veloctiy) / viscosityLattice;

   /////////////////////////
   /// BOUNDARY HANDLING ///
   /////////////////////////

   // create and initialize boundary handling
   const FlagUID fluidFlagUID("Fluid");
   const FlagUID noslipFlagUID("NoSlip");
   static walberla::BoundaryUID wallFlagUID("NoSlip");
   const FlagUID inflowUID("UBB");
   const FlagUID PressureOutflowUID("PressureOutflow");

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");

   // set NoSlip UID to boundaries that we colored
   mesh::ColorToBoundaryMapper< mesh::TriangleMesh > colorToBoundaryMapper((mesh::BoundaryInfo(wallFlagUID)));
   colorToBoundaryMapper.set(mesh::TriangleMesh::Color(255, 255, 255), mesh::BoundaryInfo(wallFlagUID));

   // mark boundaries
   auto boundaryLocations = colorToBoundaryMapper.addBoundaryInfoToMesh(*mesh);

   // write mesh info to file
   if(vtkWriteFrequency > 0) {
      mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriter(mesh, "meshBoundaries", 1);
      meshWriter.addDataSource(make_shared< mesh::BoundaryUIDFaceDataSource< mesh::TriangleMesh > >(boundaryLocations));
      meshWriter.addDataSource(make_shared< mesh::ColorFaceDataSource< mesh::TriangleMesh > >());
      meshWriter.addDataSource(make_shared< mesh::ColorVertexDataSource< mesh::TriangleMesh > >());
      meshWriter();
   }

   // voxelize mesh
   mesh::BoundarySetup boundarySetup(blocks, makeMeshDistanceFunction(distanceOctree), numGhostLayers);

   geometry::initBoundaryHandling<FlagField_T>(*blocks, flagFieldId, boundariesConfig);
   boundarySetup.setFlag<FlagField_T>(flagFieldId, FlagUID("NoSlip"), mesh::BoundarySetup::INSIDE);
   geometry::setNonBoundaryCellsToDomain<FlagField_T>(*blocks, flagFieldId, fluidFlagUID);

   for (uint_t i = 0; i < 3; ++i) {
      if (int_c(cellsPerBlock[i]) <= InnerOuterSplit[i] * 2) {
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller or increase cellsPerBlock")
      }
   }

   ////////////////////////////////////
   /// CREATE AND INITIALIZE FIELDS ///
   ////////////////////////////////////

   // create fields
   WALBERLA_LOG_INFO_ON_ROOT("Running Simulation with indirect addressing")
   BlockDataID pdfListId   = lbm::addListToStorage< List_T >(blocks, "LBM list (FIdx)", InnerOuterSplit);
   WALBERLA_LOG_INFO_ON_ROOT("Start initialisation of the linked-list structure")

   WcTimer lbmTimer;
   for (auto& block : *blocks)
   {
      auto* lbmList = block.getData< List_T >(pdfListId);
      WALBERLA_CHECK_NOT_NULLPTR(lbmList)
      lbmList->fillFromFlagField< FlagField_T >(block, flagFieldId, fluidFlagUID);
   }


#if defined(WALBERLA_BUILD_WITH_CUDA)
   const Vector3< int32_t > gpuBlockSize = parameters.getParameter< Vector3< int32_t > >("gpuBlockSize", Vector3< int32_t >(128, 1, 1));
   lbmpy::LBSweep kernel(pdfListId, omega, gpuBlockSize[0], gpuBlockSize[1], gpuBlockSize[2]);
#else
   lbmpy::LBSweep kernel(pdfListId, omega);
#endif


   auto tracker = make_shared<lbm::TimestepTracker>(0);
   lbmpy::MacroSetter setterSweep(pdfListId);
   lbmpy::UBB ubb(blocks, pdfListId, initialVelocity[0]);
   lbmpy::Pressure pressureOutflow(blocks, pdfListId, 1.0);
   //lbm::NoSlip noSlip(blocks, pdfListId);

   ubb.fillFromFlagField< FlagField_T >(blocks, flagFieldId, inflowUID, fluidFlagUID);
   pressureOutflow.fillFromFlagField< FlagField_T >(blocks, flagFieldId, PressureOutflowUID, fluidFlagUID);
   //noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldId, noslipFlagUID, fluidFlagUID);
   for (auto& block : *blocks)
   {
      setterSweep(&block);
   }
   lbmpy::ListCommunicationSetup(pdfListId, blocks);
   lbmTimer.end();
   WALBERLA_LOG_INFO_ON_ROOT("Initialisation of the list structures needed " << lbmTimer.last() << " s")

#if defined(WALBERLA_BUILD_WITH_CUDA)

   const bool cudaEnabledMPI = parameters.getParameter< bool >("cudaEnabledMPI", true);
   auto packInfo = make_shared<lbm::CombinedInPlaceGpuPackInfo<PackInfoEven_T, PackInfoOdd_T> >(tracker, pdfListId);
   cuda::communication::UniformGPUScheme< Stencil_T > comm(blocks, cudaEnabledMPI);
   comm.addPackInfo(packInfo);
   auto communicate = std::function< void() >([&]() { comm.communicate(nullptr); });
   auto start_communicate = std::function< void() >([&]() { comm.startCommunication(nullptr); });
   auto wait_communicate = std::function< void() >([&]() { comm.wait(nullptr); });
   WALBERLA_LOG_INFO_ON_ROOT("Finished setting up communication and start first communication")

   // TODO: Data for List LBM is synced at first communication. Should be fixed ...
   comm.communicate();
   WALBERLA_LOG_INFO_ON_ROOT("Finished first communication")
#else
   auto packInfo = make_shared<lbm::CombinedInPlaceCpuPackInfo<PackInfoEven_T, PackInfoOdd_T> >(tracker, pdfListId);
   blockforest::communication::UniformBufferedScheme< Stencil_T > comm(blocks);
   comm.addPackInfo(packInfo);
   auto communicate = std::function< void() >([&]() { comm.communicate(); });
   auto start_communicate = std::function< void() >([&]() { comm.startCommunication(); });
   auto wait_communicate = std::function< void() >([&]() { comm.wait(); });
#endif



   //////////////////////////////////
   /// SET UP SWEEPS AND TIMELOOP ///
   //////////////////////////////////

   const std::string timeStepStrategy = parameters.getParameter<std::string>("timeStepStrategy", "noOverlap");
   const bool runBoundaries = parameters.getParameter<bool>("runBoundaries", true);

   auto normalTimeStep = [&]() {
      communicate();
      for (auto& block : *blocks)
      {
         if (runBoundaries) {
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
         if (runBoundaries) {
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
      tracker->advance();
      for (auto& block : *blocks)
         kernel(&block, tracker->getCounter());
   };


   std::function< void() > timeStep;
   if (timeStepStrategy == "noOverlap")
      timeStep = normalTimeStep;
   else if (timeStepStrategy == "Overlap")
      timeStep = simpleOverlapTimeStep;
   else if (timeStepStrategy == "kernelOnly")
   {
      WALBERLA_LOG_INFO_ON_ROOT("Running only compute kernel without boundary - this makes only sense for benchmarking!")
      communicate();
      timeStep = kernelOnlyFunc;
   }
   else
   {
      WALBERLA_ABORT_NO_DEBUG_INFO("Invalid value for 'timeStepStrategy'. Allowed values are 'noOverlap', "
                                   "'simpleOverlap', 'kernelOnly'")
   }

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);
   timeloop.add() << BeforeFunction(timeStep, "Timestep") << Sweep([](IBlock*) {}, "Dummy");


   //////////////////////////////////
   ///       VTK AND STUFF        ///
   //////////////////////////////////


   // log remaining time
   timeloop.addFuncAfterTimeStep(
      timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
      "remaining time logger");

   //////////////////
   /// VTK OUTPUT ///
   //////////////////

   if (vtkWriteFrequency > 0)
   {
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtkList", vtkWriteFrequency, 0, false, "vtk_out",
                                                      "simulation_step", false, true, true, false, 0);

#if defined(WALBERLA_BUILD_WITH_CUDA)
      vtkOutput->addBeforeFunction([&]() {
         for (auto& block : *blocks)
         {
            List_T* lbmList = block.getData< List_T >(pdfListId);
            lbmList->copyPDFSToCPU();
         }
      });
#endif

      vtkOutput->addCellInclusionFilter(lbm::ListFluidFilter< List_T >(pdfListId));
      auto velWriter = make_shared< lbm::ListVelocityVTKWriter< List_T, float > >(pdfListId, tracker, "velocity");
      auto densityWriter = make_shared< lbm::ListDensityVTKWriter< List_T, float > >(pdfListId, "density");
      vtkOutput->addCellDataWriter(velWriter);
      vtkOutput->addCellDataWriter(densityWriter);
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   }

   //////////////////////
   /// RUN SIMULATION ///
   //////////////////////
   lbm::PerformanceEvaluation<FlagField_T> performance(blocks, flagFieldId, fluidFlagUID);

   WALBERLA_LOG_INFO_ON_ROOT("Simulating ListLBM:"
                             "\n timesteps:                  "
                             << timesteps << "\n simulation time in seconds: " << real_t(timesteps) * Ct
                             << "\n reynolds number:            " << reynolds_number
                             << "\n relaxation rate:            " << omega
                             << "\n maximum inflow velocity:    " << maximum_veloctiy)

   int warmupSteps = parameters.getParameter<int>("warmupSteps", 2);
   int outerIterations = parameters.getParameter<int>("outerIterations", 1);
   for (int i = 0; i < warmupSteps; ++i)
      timeloop.singleStep();

   for (int outerIteration = 0; outerIteration < outerIterations; ++outerIteration) {
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
   }
   //printResidentMemoryStatistics();
   return EXIT_SUCCESS;
}
