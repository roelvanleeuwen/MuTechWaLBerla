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
//! \file Lagoon.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#include "blockforest/communication/UniformBufferedScheme.h"
#include "blockforest/Initialization.h"

#include "core/Environment.h"
#include "core/SharedFunctor.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/math/Sample.h"
#include "core/MemoryUsage.h"

#include "field/AddToStorage.h"
#include "field/StabilityChecker.h"
#include "field/vtk/VTKWriter.h"
#include <fstream>

#include "geometry/InitBoundaryHandling.h"

#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/PerformanceEvaluation.h"
#include "lbm/all.h"

#include "domain_decomposition/all.h"


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

#include "Lagoon_InfoHeader.h"
#include "LBMSweepAA.h"


using namespace walberla;
using PackInfoEven_T = lbm::Lagoon_PackInfoEven;
using PackInfoOdd_T = lbm::Lagoon_PackInfoOdd;

uint_t numGhostLayers = uint_t(1);

using flag_t = walberla::uint8_t;
using FlagField_T = FlagField<flag_t>;
using LatticeModel_T = lbm::D3Q19<lbm::collision_model::SRT>;

#if defined(WALBERLA_BUILD_WITH_CUDA)
using GPUField = cuda::GPUField< real_t >;
#endif

auto pdfFieldAdder = [](IBlock *const block, StructuredBlockStorage *const storage) {
   return new PdfField_T(storage->getNumberOfXCells(*block), storage->getNumberOfYCells(*block),
                         storage->getNumberOfZCells(*block), uint_t(1), field::fzyx,
                         make_shared<field::AllocateAligned<real_t, 64> >());
};

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

int main(int argc, char **argv) {
   walberla::Environment walberlaEnv(argc, argv);

#if defined(WALBERLA_BUILD_WITH_CUDA)
   cuda::selectDeviceBasedOnMpiRank();
#endif

   mpi::MPIManager::instance()->useWorldComm();

   ///////////////////////
   /// PARAMETER INPUT ///
   ///////////////////////

   // read general simulation parameters
   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");

   real_t reference_velocity = parameters.getParameter<real_t>("reference_velocity", real_c(0.9));
   real_t viscosity = parameters.getParameter<real_t>("viscosity", real_c(0.9));
   const Vector3<real_t> initialVelocity =
      parameters.getParameter<Vector3<real_t> >("initialVelocity", Vector3<real_t>());
   const uint_t timesteps = parameters.getParameter<uint_t>("timesteps", uint_c(10));

   const real_t remainingTimeLoggerFrequency =
      parameters.getParameter<real_t>("remainingTimeLoggerFrequency", 3.0); // in seconds

   auto loggingParameters = walberlaEnv.config()->getOneBlock("Logging");
   const bool WriteDistanceOctree = loggingParameters.getParameter<bool>("WriteDistanceOctree");

   // read domain parameters
   auto domainParameters = walberlaEnv.config()->getOneBlock("DomainSetup");
   std::string meshFile = domainParameters.getParameter<std::string>("meshFile");
   const bool weakScaling = domainParameters.getParameter<bool>("weakScaling", false); // weak or strong scaling

   uint_t numProcesses = uint_c(MPIManager::instance()->numProcesses());

   const Vector3<bool> periodicity =
      domainParameters.getParameter<Vector3<bool> >("periodic", Vector3<bool>(false));

   Vector3<uint_t> cellsPerBlock;
   Vector3<uint_t> blocksPerDimension;

   if (!domainParameters.isDefined("blocks"))
   {
      if (weakScaling)
      {
         Vector3<uint_t> cells = domainParameters.getParameter<Vector3<uint_t> >("cellsPerBlock");
         blockforest::calculateCellDistribution(cells, numProcesses, blocksPerDimension, cellsPerBlock);
         cellsPerBlock = cells;
      }
      else
      {
         Vector3<uint_t> cells = domainParameters.getParameter<Vector3<uint_t> >("cellsPerBlock");
         blockforest::calculateCellDistribution(cells, numProcesses, blocksPerDimension, cellsPerBlock);
      }
   }
   else
   {
      cellsPerBlock = domainParameters.getParameter<Vector3<uint_t>>("cellsPerBlock");
      blocksPerDimension = domainParameters.getParameter<Vector3<uint_t>>("blocks");
   }

   const Vector3<uint_t> TotalCells(cellsPerBlock[0] * blocksPerDimension[0], cellsPerBlock[1] * blocksPerDimension[1],
                                      cellsPerBlock[2] * blocksPerDimension[2]);
   const real_t scalingFactor = 0.03; //(32.0 / real_c(TotalCells.min())) * 0.1;
   const Vector3<real_t> dx(scalingFactor, scalingFactor, scalingFactor);

   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   WALBERLA_LOG_INFO_ON_ROOT("Using mesh from " << meshFile << ".")

   // read in mesh with vertex colors on a single process and broadcast it
   auto mesh = make_shared<mesh::TriangleMesh>();
   mesh->request_vertex_colors();
   mesh::readAndBroadcast(meshFile, *mesh);

   // color faces according to vertices
   vertexToFaceColor(*mesh, mesh::TriangleMesh::Color(255, 255, 255));

   // add information to mesh that is required for computing signed distances from a point to a triangle
   auto triDist = make_shared<mesh::TriangleDistance<mesh::TriangleMesh> >(mesh);

   // building distance octree
   auto distanceOctree = make_shared<mesh::DistanceOctree<mesh::TriangleMesh> >(triDist);

   WALBERLA_LOG_INFO_ON_ROOT("Octree has height " << distanceOctree->height())

   // write distance octree to file
   if (WriteDistanceOctree) {
      distanceOctree->writeVTKOutput("distanceOctree");
   }

   ///////////////////////////
   /// CREATE BLOCK FOREST ///
   ///////////////////////////

   auto aabb = computeAABB(*mesh);

   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, dx,
                                                         mesh::makeExcludeMeshInterior(distanceOctree, dx[0]));

   bfc.setPeriodicity(periodicity);
   auto blocks = bfc.createStructuredBlockForest(cellsPerBlock, blocksPerDimension);

   uint_t cells_y = blocks->getYSize() * cellsPerBlock[1];
   real_t maximum_veloctiy = initialVelocity[0];

   real_t Cu = reference_velocity / maximum_veloctiy;
   real_t Ct = dx.min() / Cu;

   real_t viscosityLattice = viscosity * Ct / (dx.min() * dx.min());
   real_t omega = real_c(1.0 / (3.0 * viscosityLattice + 0.5));

   real_t reynolds_number = (real_t(cells_y) * maximum_veloctiy) / viscosityLattice;
   ////////////////////////////////////
   /// CREATE AND INITIALIZE FIELDS ///
   ////////////////////////////////////

   // create fields
   BlockDataID velFieldID = field::addToStorage<VelocityField_T>(blocks, "velocity", real_t(0), field::fzyx);
   BlockDataID densityFieldID = field::addToStorage<ScalarField_T>(blocks, "density", real_t(1), field::fzyx);

   BlockDataID pdfFieldID = blocks->addStructuredBlockData<PdfField_T>(pdfFieldAdder, "PDFs");
   BlockDataID flagFieldId = field::addFlagFieldToStorage<FlagField_T>(blocks, "flag field", numGhostLayers);

   pystencils::Lagoon_MacroSetter setterSweep(pdfFieldID/*, velFieldID*/);

   uint_t numberOfBlocksOnProcess = 0;
   for (auto &block: *blocks) {
      numberOfBlocksOnProcess++;
      setterSweep(&block);
   }

   uint_t total_number_of_blocks = walberla::mpi::reduce(numberOfBlocksOnProcess, walberla::mpi::SUM, 0);
   const int rank = walberla::MPIManager::instance()->rank();
   if (rank == 0) {WALBERLA_LOG_INFO("The work is delegated to " << total_number_of_blocks << " blocks")}

   WALBERLA_LOG_INFO_ON_ROOT("Cells in x direction: " << cellsPerBlock[0])
   WALBERLA_LOG_INFO_ON_ROOT("Cells in y direction: " << cellsPerBlock[1])
   WALBERLA_LOG_INFO_ON_ROOT("Cells in z direction: " << cellsPerBlock[2])

   WALBERLA_LOG_INFO_ON_ROOT("Blocks in x direction: " << blocks->getXSize())
   WALBERLA_LOG_INFO_ON_ROOT("Blocks in y direction: " << blocks->getYSize())
   WALBERLA_LOG_INFO_ON_ROOT("Blocks in z direction: " << blocks->getZSize())

#if defined(WALBERLA_BUILD_WITH_CUDA)
   BlockDataID pdfFieldIDGPU = cuda::addGPUFieldToStorage< PdfField_T >(blocks, pdfFieldID, "PDFs on GPU", true);
   BlockDataID velFieldIDGPU =
      cuda::addGPUFieldToStorage< VelocityField_T >(blocks, velFieldID, "velocity on GPU", true);
   BlockDataID densityFieldIDGPU =
      cuda::addGPUFieldToStorage< ScalarField_T >(blocks, densityFieldID, "density on GPU", true);
#endif



   /////////////////////////
   /// BOUNDARY HANDLING ///
   /////////////////////////

   // create and initialize boundary handling
   const FlagUID fluidFlagUID("Fluid");
   // const FlagUID wallFlagUID("NoSlip");
   static walberla::BoundaryUID wallFlagUID("NoSlip");


   auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");

   // set NoSlip UID to boundaries that we colored
   mesh::ColorToBoundaryMapper<mesh::TriangleMesh> colorToBoundaryMapper((mesh::BoundaryInfo(wallFlagUID)));
   colorToBoundaryMapper.set(mesh::TriangleMesh::Color(255, 255, 255), mesh::BoundaryInfo(wallFlagUID));

   // mark boundaries
   auto boundaryLocations = colorToBoundaryMapper.addBoundaryInfoToMesh(*mesh);

   // write mesh info to file
   mesh::VTKMeshWriter<mesh::TriangleMesh> meshWriter(mesh, "meshBoundaries", 1);
   meshWriter.addDataSource(make_shared<mesh::BoundaryUIDFaceDataSource<mesh::TriangleMesh> >(boundaryLocations));
   meshWriter.addDataSource(make_shared<mesh::ColorFaceDataSource<mesh::TriangleMesh> >());
   meshWriter.addDataSource(make_shared<mesh::ColorVertexDataSource<mesh::TriangleMesh> >());
   meshWriter();

   // voxelize mesh
   mesh::BoundarySetup boundarySetup(blocks, makeMeshDistanceFunction(distanceOctree), numGhostLayers);

   geometry::initBoundaryHandling<FlagField_T>(*blocks, flagFieldId, boundariesConfig);
   boundarySetup.setFlag<FlagField_T>(flagFieldId, FlagUID("NoSlip"), mesh::BoundarySetup::INSIDE);
   geometry::setNonBoundaryCellsToDomain<FlagField_T>(*blocks, flagFieldId, fluidFlagUID);

   Vector3<int> innerOuterSplit =
      parameters.getParameter<Vector3<int> >("innerOuterSplit", Vector3<int>(1, 1, 1));

   for (uint_t i = 0; i < 3; ++i) {
      if (int_c(cellsPerBlock[i]) <= innerOuterSplit[i] * 2) {
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller or increase cellsPerBlock")
      }
   }
   Cell innerOuterSplitCell(innerOuterSplit[0], innerOuterSplit[1], innerOuterSplit[2]);
   WALBERLA_LOG_INFO_ON_ROOT("innerOuterSplitCell: " << innerOuterSplitCell)


   auto tracker = make_shared<lbm::TimestepTracker>(0);

#if defined(WALBERLA_BUILD_WITH_CUDA)

   int streamHighPriority = 0;
   int streamLowPriority  = 0;
   WALBERLA_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&streamLowPriority, &streamHighPriority))
   auto defaultStream = cuda::StreamRAII::newPriorityStream(streamLowPriority);

   const bool cudaEnabledMPI = parameters.getParameter< bool >("cudaEnabledMPI", false);
   auto packInfo = make_shared<lbm::CombinedInPlaceGpuPackInfo<PackInfoEven_T, PackInfoOdd_T> >(tracker, pdfFieldIDGPU);
   cuda::communication::UniformGPUScheme< Stencil_T > comm(blocks, cudaEnabledMPI);
   comm.addPackInfo(packInfo);
   auto communicate = std::function< void() >([&]() { comm.communicate(defaultStream); });
   auto start_communicate = std::function< void() >([&]() { comm.startCommunication(defaultStream); });
   auto wait_communicate = std::function< void() >([&]() { comm.wait(defaultStream); });
#elif
   auto packInfo = make_shared<lbm::CombinedInPlaceCpuPackInfo<PackInfoEven_T, PackInfoOdd_T> >(tracker, pdfFieldIDGPU);
   blockforest::communication::UniformBufferedScheme<Stencil_T> communication(blocks);
   communication.addPackInfo(packInfo);
   auto start_communicate = std::function<void()>([&]() { communication.startCommunication(); });
   auto wait_communicate = std::function<void()>([&]() { communication.wait(); });
#endif

#if defined(WALBERLA_BUILD_WITH_CUDA)
   const Vector3< int32_t > gpuBlockSize = parameters.getParameter< Vector3< int32_t > >("gpuBlockSize", Vector3< int32_t >(128, 1, 1));
   lbm::Lagoon_LbSweep lbSweep(densityFieldIDGPU, pdfFieldIDGPU, velFieldIDGPU, omega, gpuBlockSize[0], gpuBlockSize[1],
                               gpuBlockSize[2], innerOuterSplitCell);
#elif
   lbm::Lagoon_LbSweep lbSweep(densityFieldIDGPU, pdfFieldIDGPU, velFieldIDGPU, omega, innerOuterSplitCell);
#endif

   lbm::Lagoon_UBB ubb(blocks, pdfFieldIDGPU);
   lbm::Lagoon_NoSlip noSlip(blocks, pdfFieldIDGPU);
   lbm::Lagoon_Outflow outflow(blocks, pdfFieldIDGPU, 1.0);


   ubb.fillFromFlagField<FlagField_T>(blocks, flagFieldId, FlagUID("UBB"), fluidFlagUID);
   noSlip.fillFromFlagField<FlagField_T>(blocks, flagFieldId, FlagUID("NoSlip"), fluidFlagUID);
   outflow.fillFromFlagField<FlagField_T>(blocks, flagFieldId, FlagUID("PressureOutflow"), fluidFlagUID);

   //////////////////////////////////
   /// SET UP SWEEPS AND TIMELOOP ///
   //////////////////////////////////

      const std::string timeStepStrategy = parameters.getParameter<std::string>("timeStepStrategy", "noOverlap");
      const bool runBoundaries = parameters.getParameter<bool>("runBoundaries", true);

      auto startCommunicationStep = [&]() {
         start_communicate();
         cudaStreamSynchronize(defaultStream);
      };

      auto waitCommunicationStep = [&]() {
         wait_communicate();
         cudaStreamSynchronize(defaultStream);
      };

      auto runInnerStep = [&]() {
         for (auto& block : *blocks)
         {
            lbSweep.inner(&block, tracker->getCounterPlusOne(), defaultStream);
            if (runBoundaries) {
               noSlip(&block, tracker->getCounter(), defaultStream);
               ubb(&block, tracker->getCounter(), defaultStream);
               outflow(&block, tracker->getCounter(), defaultStream);
            }
         }
         cudaStreamSynchronize(defaultStream);
      };

      auto runOuterStep = [&]() {
         for (auto& block : *blocks)
         {
            lbSweep.outer(&block, tracker->getCounterPlusOne(), defaultStream);
         }
         tracker->advance();
         cudaStreamSynchronize(defaultStream);
      };


      auto normalTimeStep = [&]() {
         communicate();
         for (auto& block : *blocks)
         {
            if (runBoundaries) {
               noSlip(&block, tracker->getCounter(), defaultStream);
               ubb(&block, tracker->getCounter(), defaultStream);
               outflow(&block, tracker->getCounter(), defaultStream);
            }
            lbSweep(&block, tracker->getCounterPlusOne(), defaultStream);
         }
         tracker->advance();
      };


      auto simpleOverlapTimeStep = [&]() {
         // Communicate post-collision values of previous timestep...
         start_communicate();
         for (auto& block : *blocks)
         {
            if (runBoundaries) {
               noSlip(&block, tracker->getCounter(), defaultStream);
               ubb(&block, tracker->getCounter(), defaultStream);
               outflow(&block, tracker->getCounter(), defaultStream);
            }
            lbSweep.inner(&block, tracker->getCounterPlusOne(), defaultStream);
         }
         wait_communicate();
         for (auto& block : *blocks)
         {
            lbSweep.outer(&block, tracker->getCounterPlusOne(), defaultStream);
         }

         tracker->advance();
      };

      auto kernelOnlyFunc = [&]() {
         tracker->advance();
         for (auto& block : *blocks)
            lbSweep(&block, tracker->getCounter(), defaultStream);
      };

      std::function< void() > timeStep;
      if (timeStepStrategy == "noOverlap")
         timeStep = std::function< void() >(normalTimeStep);
      else if (timeStepStrategy == "simpleOverlap")
         timeStep = simpleOverlapTimeStep;
      else if (timeStepStrategy == "kernelOnly")
      {
         WALBERLA_LOG_INFO_ON_ROOT("Running only compute kernel without boundary - this makes only sense for benchmarking!")
         comm.communicate();
         timeStep = kernelOnlyFunc;
      }
      else
      {
         WALBERLA_ABORT_NO_DEBUG_INFO("Invalid value for 'timeStepStrategy'. Allowed values are 'noOverlap', "
                                      "'simpleOverlap', 'kernelOnly'")
      }

      SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);
      //timeloop.add() << BeforeFunction(timeStep) << Sweep([](IBlock*) {}, "time step");
      timeloop.add() << BeforeFunction(startCommunicationStep, "startCommunicationStep") << Sweep([](IBlock*) {});
      timeloop.add() << BeforeFunction(runInnerStep, "runInnerStep") << Sweep([](IBlock*) {});
      timeloop.add() << BeforeFunction(waitCommunicationStep, "waitCommunicationStep") << Sweep([](IBlock*) {});
      timeloop.add() << BeforeFunction(runOuterStep, "runOuterStep") << Sweep([](IBlock*) {});



   // LBM stability check
   auto CheckerParameters = walberlaEnv.config()->getOneBlock("StabilityChecker");
   uint_t checkfrequency = CheckerParameters.getParameter<uint_t>("checkFrequency", uint_t(0));

   if (checkfrequency > 0) {
      timeloop.addFuncAfterTimeStep(makeSharedFunctor(field::makeStabilityChecker<VelocityField_T, FlagField_T>(
                                       walberlaEnv.config(), blocks, velFieldID, flagFieldId, fluidFlagUID)),
                                    "LBM stability check");
   }
   // log remaining time
   timeloop.addFuncAfterTimeStep(
      timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
      "remaining time logger");

   //////////////////
   /// VTK OUTPUT ///
   //////////////////

   //   pystencils::Lagoon_MacroGetter getterSweep( densityFieldID, pdfFieldId, velFieldID );

   uint_t vtkWriteFrequency = parameters.getParameter<uint_t>("vtkWriteFrequency", 0);
   if (vtkWriteFrequency > 0) {
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_out",
                                                      "simulation_step", false, true, true, false, 0);

#if defined(WALBERLA_BUILD_WITH_CUDA)
      vtkOutput->addBeforeFunction([&]() {
         cuda::fieldCpy< VelocityField_T, GPUField >(blocks, velFieldID, velFieldIDGPU);
         cuda::fieldCpy< ScalarField_T, GPUField >(blocks, densityFieldID, densityFieldIDGPU);
      });
#endif

      auto velWriter = make_shared<field::VTKWriter<VelocityField_T> >(velFieldID, "velocity");
      auto densityWriter = make_shared<field::VTKWriter<ScalarField_T> >(densityFieldID, "density");
      auto flagWriter = make_shared<field::VTKWriter<FlagField_T> >(flagFieldId, "flag");

      vtkOutput->addCellDataWriter(velWriter);
      vtkOutput->addCellDataWriter(densityWriter);
      vtkOutput->addCellDataWriter(flagWriter);

      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   }

   //////////////////////
   /// RUN SIMULATION ///
   //////////////////////
   lbm::PerformanceEvaluation<FlagField_T> performance(blocks, flagFieldId, fluidFlagUID);

   WALBERLA_LOG_INFO_ON_ROOT("Simulating Lagoon:"
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
   math::Sample memory;
   memory.castToRealAndInsert(getResidentMemorySize());
   memory.mpiGatherRoot();
   WALBERLA_LOG_INFO_ON_ROOT("resident memory: " << memory.format());
   WALBERLA_ROOT_SECTION() {
      std::ofstream outfile;
      outfile.open("memoryResults.txt", std::ios_base::app); // append instead of overwrite
      outfile << "Cores: " << numProcesses << ", Memory " << memory.format();
      outfile.close();
   };
   return EXIT_SUCCESS;
}
