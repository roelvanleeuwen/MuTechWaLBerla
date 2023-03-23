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
#include "core/MemoryUsage.h"

#include "field/AddToStorage.h"
#include "field/StabilityChecker.h"
#include "field/vtk/VTKWriter.h"

#include "geometry/InitBoundaryHandling.h"

#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/PerformanceEvaluation.h"
#include "lbm/list/AddToStorage.h"
#include "lbm/list/CellCounters.h"
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

using namespace walberla;
using PackInfo_T = pystencils::Lagoon_PackInfo;

uint_t numGhostLayers = uint_t(1);

using flag_t = walberla::uint8_t;
using FlagField_T = FlagField<flag_t>;
using ScalarField_T = field::GhostLayerField<real_t, 1>;

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

int main(int argc, char **argv) {
    walberla::Environment walberlaEnv(argc, argv);

#if defined(WALBERLA_BUILD_WITH_CUDA)
    cuda::selectDeviceBasedOnMpiRank();
    WALBERLA_CUDA_CHECK(gpuPeekAtLastError())
#endif

    mpi::MPIManager::instance()->useWorldComm();

    ///////////////////////
    /// PARAMETER INPUT ///
    ///////////////////////

    // read general simulation parameters
    auto parameters = walberlaEnv.config()->getOneBlock("Parameters");

    real_t ref_velocity = parameters.getParameter<real_t>("ref_velocity");
    real_t max_lattice_velocity = parameters.getParameter<real_t>("max_lattice_velocity");
    real_t ref_length = parameters.getParameter<real_t>("ref_length");
    // real_t ref_density = parameters.getParameter<real_t>("ref_density");
    real_t viscosity = parameters.getParameter<real_t>("viscosity");
    real_t sim_time = parameters.getParameter<real_t>("sim_time");
    real_t mesh_size = parameters.getParameter<real_t>("mesh_size");
    real_t omega_wall = parameters.getParameter<real_t>("omega_wall");
    cell_idx_t spongeZoneStart = parameters.getParameter<cell_idx_t>("sponge_zone_start");

    real_t reynolds_number = (ref_velocity * ref_length) / viscosity;


    real_t Cu = ref_velocity / max_lattice_velocity;
    real_t Ct = mesh_size / Cu;

    real_t inlet_velocity = ref_velocity / Cu;
    real_t viscosity_lattice = viscosity * Ct / (mesh_size * mesh_size);
    real_t omega = real_c(1.0 / (3.0 * viscosity_lattice + 0.5));
    const uint_t timesteps = uint_c(sim_time / Ct);


    const real_t remainingTimeLoggerFrequency =
        parameters.getParameter<real_t>("remainingTimeLoggerFrequency", 3.0); // in seconds

    auto loggingParameters = walberlaEnv.config()->getOneBlock("Logging");
    const bool WriteSetupForestAndReturn = loggingParameters.getParameter<bool>("WriteSetupForestAndReturn", false);
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

    const Vector3<real_t> dx(mesh_size, mesh_size, mesh_size);

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
    auto BoundingBoxCenter = aabb.center();
    aabb.setCenter(Vector3<real_t>(BoundingBoxCenter[0], -0.4, BoundingBoxCenter[2]));

    mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, dx,
                                                          mesh::makeExcludeMeshInterior(distanceOctree, dx[0]));

    if (WriteSetupForestAndReturn)
    {
      auto setupForest = bfc.createSetupBlockForest(cellsPerBlock, blocksPerDimension);
      WALBERLA_ROOT_SECTION() { setupForest->writeVTKOutput("SetupBlockForest"); }
      return EXIT_SUCCESS;
    }

    bfc.setPeriodicity(periodicity);
    auto blocks = bfc.createStructuredBlockForest(cellsPerBlock, blocksPerDimension);
    WALBERLA_LOG_INFO_ON_ROOT("Created Blockforest")
    BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");


    /////////////////////////
    /// BOUNDARY HANDLING ///
    /////////////////////////

    // create and initialize boundary handling
    const FlagUID fluidFlagUID("Fluid");
    // const FlagUID wallFlagUID("NoSlip");
    static walberla::BoundaryUID wallFlagUID("NoSlip");
    for (auto &block: *blocks) {
      auto flagField = block.getData<FlagField_T>( flagFieldId );
      flagField->registerFlag(FlagUID("NoSlip"));
    }



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

    Vector3<int> InnerOuterSplit = parameters.getParameter<Vector3<int> >("innerOuterSplit", Vector3<int>(1, 1, 1));
    for (uint_t i = 0; i < 3; ++i) {
        if (int_c(cellsPerBlock[i]) <= InnerOuterSplit[i] * 2) {
            WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller or increase cellsPerBlock")
        }
    }


    ////////////////////////////////////
    /// CREATE AND INITIALIZE FIELDS ///
    ////////////////////////////////////



    BlockDataID omegaFieldID = field::addToStorage<ScalarField_T>(blocks, "omega", omega, field::fzyx);
    real_t omega_diff = omega - omega_wall;
    real_t SizeX = real_c(cellsPerBlock[0] * blocks->getXSize());

    for (auto &block: *blocks) {

      auto omegaField = block.getData< ScalarField_T >(omegaFieldID);

      WALBERLA_FOR_ALL_CELLS_XYZ( omegaField,
          Cell globalCell;
          blocks->transformBlockLocalToGlobalCell(globalCell, block, Cell(x, y, z));

          if (globalCell[0] >= spongeZoneStart) {
            omegaField->get(x, y, z) = omega - omega_diff * ((real_c(globalCell[0]) - real_c(spongeZoneStart)) / (SizeX - real_c(spongeZoneStart)));
          }
      )
    }

    // create fields
    WALBERLA_LOG_INFO_ON_ROOT("Running Simulation with indirect addressing")
    BlockDataID pdfListId   = lbm::addListToStorage< List_T >(blocks, "LBM list (FIdx)", InnerOuterSplit);
    WALBERLA_LOG_INFO_ON_ROOT("Start initialisation of the linked-list structure")
    for (auto& block : *blocks)
    {
      auto* lbmList = block.getData< List_T >(pdfListId);
      WALBERLA_CHECK_NOT_NULLPTR(lbmList)
      lbmList->fillFromFlagField< FlagField_T >(block, flagFieldId, fluidFlagUID);
      lbmList->fillOmegasFromFlagField< FlagField_T, ScalarField_T >(block, flagFieldId, fluidFlagUID ,  omegaFieldID);
    }






#if defined(WALBERLA_BUILD_WITH_CUDA)

    int streamHighPriority = 0;
    int streamLowPriority  = 0;
    WALBERLA_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&streamLowPriority, &streamHighPriority))
    auto defaultStream = cuda::StreamRAII::newPriorityStream(streamLowPriority);

    const bool cudaEnabledMPI = parameters.getParameter< bool >("cudaEnabledMPI", false);
    cuda::communication::UniformGPUScheme< Stencil_T > comm(blocks, cudaEnabledMPI);
    WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cudaEnabledMPI)
    comm.addPackInfo(make_shared< PackInfo_T >(pdfListId, blocks));
    auto communicate = std::function< void() >([&]() { comm.communicate(defaultStream); });
    auto start_communicate = std::function< void() >([&]() { comm.startCommunication(defaultStream); });
    auto wait_communicate = std::function< void() >([&]() { comm.wait(defaultStream); });

    // TODO: Data for List LBM is synced at first communication. Should be fixed ...
    comm.communicate(defaultStream);

#else
    blockforest::communication::UniformBufferedScheme< Stencil_T > communicate(blocks);
    communicate.addPackInfo(make_shared< PackInfo_T >(pdfListId, blocks));
    auto start_communicate = std::function< void() >([&]() { communicate.startCommunication(); });
    auto wait_communicate = std::function< void() >([&]() { communicate.wait(); });
#endif

    //////////////////////////////////
    /// SET UP SWEEPS AND TIMELOOP ///
    //////////////////////////////////
    lbmpy::Lagoon_LbSweep kernel(pdfListId);
    lbm::Lagoon_UBB ubb(blocks, pdfListId, inlet_velocity);
    lbm::Lagoon_Pressure pressureInflow(blocks, pdfListId, 1.0001);
    lbm::Lagoon_Pressure pressureOutflow(blocks, pdfListId, 0.9999);


    lbmpy::Lagoon_MacroSetter setterSweep(pdfListId);
    const FlagUID inflowUID("UBB");
    const FlagUID PressureInflowUID("PressureInflow");
    const FlagUID PressureOutflowUID("PressureOutflow");
    ubb.fillFromFlagField< FlagField_T >(blocks, flagFieldId, inflowUID, fluidFlagUID);
    pressureInflow.fillFromFlagField< FlagField_T >(blocks, flagFieldId, PressureInflowUID, fluidFlagUID);
    pressureOutflow.fillFromFlagField< FlagField_T >(blocks, flagFieldId, PressureOutflowUID, fluidFlagUID);

    for (auto& block : *blocks)
    {
      setterSweep(&block);
    }

    WALBERLA_LOG_INFO_ON_ROOT("Finished initialisation of the linked-list structure")

    // create time loop
    SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

    auto InnerKernel = std::function< void(IBlock*) >([&](IBlock* b) { kernel.inner(b); });
    auto OuterKernel = std::function< void(IBlock*) >([&](IBlock* b) { kernel.outer(b); });

    const std::string timeStepStrategy = parameters.getParameter<std::string>("timeStepStrategy", "noOverlap");
    WALBERLA_LOG_DEVEL_VAR_ON_ROOT(timeStepStrategy)

    if (timeStepStrategy == "noOverlap") {
      timeloop.add() << BeforeFunction(communicate, "communication")
                     << Sweep(ubb, "UBB Boundary");
      timeloop.add() << Sweep(pressureInflow, "Pressure Inflow");
      timeloop.add() << Sweep(pressureOutflow, "Pressure Outflow");
      timeloop.add() << Sweep(kernel, "LB update rule");
    } else if (timeStepStrategy == "Overlap") {
        WALBERLA_LOG_INFO_ON_ROOT("Using timestep strategy with communication hiding")
        timeloop.add() << BeforeFunction(start_communicate, "Start Communication")
                       << Sweep(ubb, "UBB Boundary");
        timeloop.add() << Sweep(pressureInflow, "Pressure Inflow");
        timeloop.add() << Sweep(pressureOutflow, "Pressure Outflow");
        timeloop.add() << Sweep(InnerKernel, "LBM Inner Kernel");
        timeloop.add() << BeforeFunction(wait_communicate, "Wait for Communication")
                       << Sweep(OuterKernel, "LBM Outer Kernel");

    } else if (timeStepStrategy == "kernelOnly") {
        WALBERLA_LOG_INFO_ON_ROOT("Running only the compute kernel. This makes only sense for benchmarking")
        timeloop.add() << Sweep(kernel.getSweep(), "LBM complete Kernel");

    } else {
        WALBERLA_ABORT_NO_DEBUG_INFO("Invalid value for 'timeStepStrategy'")
    }

    // log remaining time
    timeloop.addFuncAfterTimeStep(
            timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
            "remaining time logger");

    //////////////////
    /// VTK OUTPUT ///
    //////////////////

    const uint_t vtkWriteFrequency = parameters.getParameter< uint_t >("vtkWriteFrequency", uint_t(0));


    auto VTKWriter = walberlaEnv.config()->getOneBlock("VTKWriter");
    const bool writeVelocity = VTKWriter.getParameter<bool>("velocity");
    const bool writeDensity = VTKWriter.getParameter<bool>("density");
    const bool writeOmega = VTKWriter.getParameter<bool>("omega");
    const bool writeFlag = VTKWriter.getParameter<bool>("flag");
    if (vtkWriteFrequency > 0)
    {
        auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtkList", vtkWriteFrequency, 0, false, "vtk_out",
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


        if (writeVelocity)
        {
          auto velWriter = make_shared< lbm::ListVelocityVTKWriter< List_T, real_t > >(pdfListId, "velocity");
          vtkOutput->addCellDataWriter(velWriter);
        }
        if (writeDensity)
        {
          auto densityWriter = make_shared< lbm::ListDensityVTKWriter< List_T, real_t > >(pdfListId, "density");
          vtkOutput->addCellDataWriter(densityWriter);
        }
        if (writeOmega)
        {
          auto omegaWriter = make_shared<field::VTKWriter<ScalarField_T> >(omegaFieldID, "omega");
          vtkOutput->addCellDataWriter(omegaWriter);
        }
        if (writeFlag)
        {
          auto flagWriter = make_shared<field::VTKWriter<FlagField_T> >(flagFieldId, "flag");
          vtkOutput->addCellDataWriter(flagWriter);
        }
        timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");

        //vtkOutput->addCellInclusionFilter(lbm::ListFluidFilter< List_T >(pdfListId));



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
                                      << "\n inflow velocity:    " << inlet_velocity)

    int warmupSteps = parameters.getParameter<int>("warmupSteps", 0);
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
    printResidentMemoryStatistics();

    return EXIT_SUCCESS;
}
