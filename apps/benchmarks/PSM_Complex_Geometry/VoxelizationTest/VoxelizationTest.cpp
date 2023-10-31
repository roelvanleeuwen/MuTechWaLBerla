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
//! \file VoxelizationTest_Test.cpp
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"
#include "blockforest/SetupBlockForest.h"
#include "blockforest/communication/UniformBufferedScheme.h"
#include "core/all.h"

#include "domain_decomposition/all.h"
#include "field/all.h"
#include "field/vtk/VTKWriter.h"
#include "geometry/all.h"

//#include "lbm_generated/communication/UniformGeneratedPdfPackInfo.h"
#include "stencil/D3Q19.h"

#include "timeloop/all.h"

#include "VoxelizationTest_InfoHeader.h"
#include "lbm_generated/evaluation/PerformanceEvaluation.h"
#include "lbm_generated/field/AddToStorage.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/communication/UniformGeneratedPdfPackInfo.h"


#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#include "lbm_generated/gpu/AddToStorage.h"
#include "lbm_generated/gpu/GPUPdfField.h"
#include "lbm_generated/gpu/UniformGeneratedGPUPdfPackInfo.h"
#include "gpu/communication/UniformGPUScheme.h"

#include "../ObjectRotatorGPUOpenLB.h"
#else
#include "../ObjectRotatorOpenLB.h"

#endif

namespace walberla
{
///////////////////////
/// Typedef Aliases ///
///////////////////////

using StorageSpecification_T = lbm::VoxelizationTestStorageSpecification;
using Stencil_T = lbm::VoxelizationTestStorageSpecification::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;
using PdfField_T = lbm_generated::PdfField< StorageSpecification_T >;
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
using GPUPdfField_T = lbm_generated::GPUPdfField< StorageSpecification_T >;
using gpu::communication::UniformGPUScheme;
#endif

typedef walberla::uint8_t flag_t;
typedef FlagField< flag_t > FlagField_T;
using BoundaryCollection_T = lbm::VoxelizationTestBoundaryCollection< FlagField_T >;
using SweepCollection_T = lbm::VoxelizationTestSweepCollection;
const FlagUID fluidFlagUID("Fluid");
using blockforest::communication::UniformBufferedScheme;

/////////////////////
/// Main Function ///
/////////////////////

auto deviceSyncWrapper = [](std::function< void(IBlock*) > sweep) {
   return [sweep](IBlock* b) {
      sweep(b);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      cudaDeviceSynchronize();
#endif
   };
};


int main(int argc, char** argv)
{
   walberla::Environment walberlaEnv(argc, argv);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   gpu::selectDeviceBasedOnMpiRank();
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif

   logging::Logging::instance()->setLogLevel( logging::Logging::INFO );

   mpi::MPIManager::instance()->useWorldComm();

   ///////////////////////
   /// PARAMETER INPUT ///
   ///////////////////////
   auto domainParameters = walberlaEnv.config()->getOneBlock("DomainSetup");
   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");


   const uint_t rotationFrequency = domainParameters.getParameter< uint_t >("rotationFrequency", uint_t(1));
   const Vector3< int > rotationAxis = domainParameters.getParameter< Vector3< int > >("rotationAxis", Vector3< int >(1,0,0));
   const real_t rotationSpeed = domainParameters.getParameter< real_t >("rotationSpeed");


   const Vector3< real_t > domainScaling = domainParameters.getParameter< Vector3< real_t > >("domainScaling", Vector3< real_t >(1.0));
   const Vector3< real_t > domainTransforming = domainParameters.getParameter< Vector3< real_t > >("domainTransforming", Vector3< real_t >(0.0));
   const Vector3< bool > periodicity = domainParameters.getParameter< Vector3< bool > >("periodic", Vector3< bool >(true));
   const Vector3< uint_t > cellsPerBlock = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
   const uint_t refinementDepth = domainParameters.getParameter< uint_t >("refinementDepth");


   const Vector3< real_t > initialVelocity = parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >(0.0));
   const uint_t timestepsFixed = parameters.getParameter< uint_t >("timesteps", uint_c(10));
   const uint_t VTKWriteFrequency = parameters.getParameter< uint_t >("VTKwriteFrequency", uint_c(10));
   const real_t remainingTimeLoggerFrequency = parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(3.0));
   const bool writeDomainDecompositionAndReturn = parameters.getParameter< bool >("writeDomainDecompositionAndReturn", false);
   const bool writeFractionFieldAndReturn = parameters.getParameter< bool >("writeFractionFieldAndReturn", false);
   const bool loadFractionFieldsFromFile = parameters.getParameter< bool >("loadFractionFieldsFromFile", false);
   const bool preProcessFractionFields = parameters.getParameter< bool >("preProcessFractionFields", false);
   const uint_t maxSuperSamplingDepth = parameters.getParameter< uint_t >("maxSuperSamplingDepth", uint_c(1));
   const std::string fractionFieldFolderName = "savedFractionFields";


   const std::string timeStepStrategy = parameters.getParameter<std::string>("timeStepStrategy", "noOverlap");
   const Cell innerOuterSplit = Cell(parameters.getParameter< Vector3<cell_idx_t> >("innerOuterSplit", Vector3<cell_idx_t>(1, 1, 1)));

   const real_t ref_lattice_velocity = 0.01;
   const real_t ref_velocity = parameters.getParameter<real_t>("ref_velocity");
   const real_t ref_length = parameters.getParameter<real_t>("ref_length");
   const real_t viscosity = parameters.getParameter<real_t>("viscosity");
   const real_t sim_time = parameters.getParameter<real_t>("sim_time");
   const real_t mesh_size = parameters.getParameter<real_t>("mesh_size");
   const real_t dx = mesh_size;
   const real_t reynolds_number = (ref_velocity * ref_length) / viscosity;
   const real_t Cu = ref_velocity / ref_lattice_velocity;
   const real_t Ct = mesh_size / Cu;

   const real_t viscosity_lattice = viscosity * Ct / (mesh_size * mesh_size);
   const real_t omega = real_c(1.0 / (3.0 * viscosity_lattice + 0.5));
   uint_t timesteps;
   if(sim_time > 0.0)
      timesteps = uint_c(sim_time / Ct);
   else
      timesteps = timestepsFixed;

   const real_t rotPerSec = rotationSpeed * (2 * M_PI) / 360;

   const real_t radPerTimestep = rotationSpeed * Ct;
   const real_t rotationAngle = radPerTimestep * real_c(rotationFrequency);


   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   auto meshBase = make_shared< mesh::TriangleMesh >();
   mesh::readAndBroadcast("../Meshfiles/CROR_base_downScaled10.obj", *meshBase);
   auto distanceOctreeMeshBase = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(meshBase));

   auto meshRotor = make_shared< mesh::TriangleMesh >();
   mesh::readAndBroadcast("../Meshfiles/CROR_rotor_downScaled10.obj", *meshRotor);
   auto distanceOctreeMeshRotor = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(meshRotor));

   //auto aabbBase = computeAABB(*meshBase);
   auto aabbBase = computeAABB(*meshRotor);

   AABB aabb = aabbBase;
   aabb.setCenter(aabb.center() - Vector3< real_t >(domainTransforming[0] * aabb.xSize(), domainTransforming[1] * aabb.ySize(), domainTransforming[2] * aabb.zSize()));
   aabb.scale(domainScaling);

   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, Vector3< real_t >(dx), mesh::makeExcludeMeshInterior(distanceOctreeMeshBase, dx), mesh::makeExcludeMeshInteriorRefinement(distanceOctreeMeshBase, dx));
   bfc.setPeriodicity(periodicity);

   auto setupForest = bfc.createSetupBlockForest( cellsPerBlock, 1 );

   const uint_t numCells = setupForest->getXSize() * setupForest->getYSize() * setupForest->getZSize() * cellsPerBlock[0] * cellsPerBlock[1] * cellsPerBlock[2];
   const AABB domainAABB = setupForest->getDomain();
   const real_t fullRefinedMeshSize = mesh_size / pow(2, real_c(refinementDepth));
   const uint_t numFullRefinedCell = uint_c((domainAABB.xSize() / fullRefinedMeshSize) * (domainAABB.ySize() / fullRefinedMeshSize) * (domainAABB.zSize() / fullRefinedMeshSize));


   WALBERLA_LOG_INFO_ON_ROOT("Simulation Parameter \n"
                             << "Domain Decomposition <" << setupForest->getXSize() << "," << setupForest->getYSize() << "," << setupForest->getZSize() << "> = " << setupForest->getXSize() * setupForest->getYSize() * setupForest->getZSize()  << " root Blocks \n"
                             << "Number of blocks is " << setupForest->getNumberOfBlocks() << " \n"
                             << "Cells per Block " << cellsPerBlock << " \n"
                             << "Number of cells "  << numCells << ", number of potential Cells (full refined) " << numFullRefinedCell <<  ", Saved computation " << (1.0 - (real_c(numCells) / real_c(numFullRefinedCell))) * 100 << "% \n"
                             << "Mesh_size " << mesh_size << " m \n"
                             << "Refined Mesh size " << fullRefinedMeshSize << " m \n"
                             << "Timestep Strategy " << timeStepStrategy << " \n"
                             << "Inner Outer Split " << innerOuterSplit << " \n"
                             << "Reynolds_number " << reynolds_number << "\n"
                             << "Inflow velocity " << ref_velocity << " m/s \n"
                             << "Lattice velocity " << initialVelocity[0] << "\n"
                             << "Viscosity " << viscosity << "\n"
                             << "Simulation time " << sim_time << " s \n"
                             << "Timesteps " << timesteps << "\n"
                             << "Omega " << omega << "\n"
                             << "Cu " << Cu << " \n"
                             << "Ct " << Ct << " \n"
                             << "Rotation Speed " << rotationSpeed << " rad/s \n"
                             << "Rotations per second " << rotPerSec << " 1/s \n"
                             << "Rad per second " << radPerTimestep << " rad \n"
                             << "Rotation Angle per Rotation " << rotationAngle / (2 * M_PI) * 360  << " ° \n"
   )

   if(writeDomainDecompositionAndReturn) {
      WALBERLA_ROOT_SECTION() { setupForest->writeVTKOutput("SetupBlockForest"); }
      //get maximum rotation per timestep
      return EXIT_SUCCESS;
   }
   auto blocks = bfc.createStructuredBlockForest(cellsPerBlock);

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   /////////////////////////
   /// Boundary Handling ///
   /////////////////////////



#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   auto allocator = make_shared< gpu::HostFieldAllocator<real_t> >(); // use pinned memory allocator for faster CPU-GPU memory transfers
#else
   auto allocator = make_shared< field::AllocateAligned< real_t, 64 > >();
#endif

   const BlockDataID fractionFieldId = field::addToStorage< FracField_T >(blocks, "fractionField", real_t(0.0), field::fzyx, uint_c(1));
   const BlockDataID objectVelocitiesFieldId = field::addToStorage< VectorField_T >(blocks, "particleVelocitiesField", real_c(0.0), field::fzyx, uint_c(1), allocator);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const BlockDataID fractionFieldGPUId = gpu::addGPUFieldToStorage< FracField_T >(blocks, fractionFieldId, "fractionFieldGPU", true);
#endif

   //Setting up Object Rotator
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   //auto objectRotatorBase = make_shared<ObjectRotatorGPUOpenLB> (blocks, meshBase, fractionFieldGPUId, objectVelocitiesFieldId, Vector3<real_t>(0,0,0), rotationAngle, rotationFrequency, rotationAxis,  distanceOctreeMeshBase, "base", false);
   auto objectRotatorRotor = make_shared<ObjectRotatorGPUOpenLB> (blocks, meshRotor, fractionFieldGPUId, objectVelocitiesFieldId,
                                                                   Vector3<real_t>(0,-0 ,0), rotationAngle, rotationFrequency,
                                                                   rotationAxis,  distanceOctreeMeshRotor, "rotor", true);
#else
   //auto objectRotatorBase = make_shared<ObjectRotatorOpenLB> (blocks, meshBase, fractionFieldId, objectVelocitiesFieldId, Vector3<real_t>(0,0,0), rotationAngle, rotationFrequency, rotationAxis,  distanceOctreeMeshBase, "base", false);
   auto objectRotatorRotor = make_shared<ObjectRotatorOpenLB> (blocks, meshRotor, fractionFieldId, objectVelocitiesFieldId,
                                                                Vector3<real_t>(0,-0.1 * dx,0), rotationAngle, rotationFrequency,
                                                                rotationAxis,  distanceOctreeMeshRotor, "rotor", true);

#endif

   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriterBase(meshBase, "meshBase", VTKWriteFrequency);
   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriterRotor(meshRotor, "meshRotor", VTKWriteFrequency);

   const std::function< void() > meshWritingFunc = [&]() {
      meshWriterBase();
      meshWriterRotor();
      objectRotatorRotor->moveTriangleMesh(timeloop.getCurrentTimeStep(), VTKWriteFrequency);
   };

   /////////////////////////
   /// Fields Creation   ///
   /////////////////////////

   const StorageSpecification_T StorageSpec = StorageSpecification_T();
   const BlockDataID pdfFieldId  = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, uint_c(1), field::fzyx, allocator);
   const BlockDataID velocityFieldId = field::addToStorage< VectorField_T >(blocks, "velocity", real_t(0.0), field::fzyx, uint_c(1), allocator);
   const BlockDataID densityFieldId = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, uint_c(1), allocator);
   const BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flagField");

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const BlockDataID pdfFieldGPUId = lbm_generated::addGPUPdfFieldToStorage< PdfField_T >(blocks, pdfFieldId, StorageSpec, "pdf field on GPU", true);
   const BlockDataID velocityFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, velocityFieldId, "velocity field on GPU", true);
   const BlockDataID densityFieldGPUId = gpu::addGPUFieldToStorage< ScalarField_T >(blocks, densityFieldId, "density field on GPU", true);
   const BlockDataID objectVelocitiesFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, objectVelocitiesFieldId, "object velocity field on GPU", true);
#endif


   /////////////////////////
   /// Rotation Calls    ///
   /////////////////////////

   const std::function< void() > objectRotatorFunc = [&]() {
      objectRotatorRotor->resetFractionField();
      (*objectRotatorRotor)(timeloop.getCurrentTimeStep());
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
#endif
   };

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   gpu::fieldCpy< gpu::GPUField< real_t >, VectorField_T >(blocks, objectVelocitiesFieldGPUId, objectVelocitiesFieldId);

#endif

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");
   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldId, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldId, fluidFlagUID);

   /////////////
   /// Sweep ///
   /////////////

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   SweepCollection_T sweepCollection(blocks, fractionFieldGPUId, objectVelocitiesFieldGPUId, pdfFieldGPUId, densityFieldGPUId, velocityFieldGPUId, omega, innerOuterSplit);
   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldGPUId, fluidFlagUID);
#else
   SweepCollection_T sweepCollection(blocks, fractionFieldId, objectVelocitiesFieldId, pdfFieldId, densityFieldId, velocityFieldId, omega, innerOuterSplit);
   //SweepCollection_T sweepCollection(blocks, pdfFieldId, densityFieldId, velocityFieldId, omega, innerOuterSplit);
   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldId, fluidFlagUID);
#endif

   for (auto& block : *blocks)
   {
      auto velField = block.getData<VectorField_T>(velocityFieldId);
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(velField, velField->get(x,y,z,0) = initialVelocity[0];)
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpu::GPUField<real_t> * dst = block.getData<gpu::GPUField<real_t>>( velocityFieldGPUId );
      const VectorField_T * src = block.getData<VectorField_T>( velocityFieldId );
      gpu::fieldCpy( *dst, *src );
#endif
      sweepCollection.initialise(&block, 1);
   }


   // Communication
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const bool sendDirectlyFromGPU = false;
   UniformGPUScheme< Stencil_T > communication(blocks, sendDirectlyFromGPU);
   auto packInfo = std::make_shared<lbm_generated::UniformGeneratedGPUPdfPackInfo< GPUPdfField_T >>(pdfFieldGPUId);
   communication.addPackInfo(packInfo);
#else
   UniformBufferedScheme< Stencil_T > communication(blocks);
   auto packInfo = std::make_shared<lbm_generated::UniformGeneratedPdfPackInfo< PdfField_T >>(pdfFieldId);
   //blockforest::communication::UniformBufferedScheme< Stencil_T > communication(blocks);
   //auto packInfo = std::make_shared<lbm_generated::UniformGeneratedPdfPackInfo< PdfField_T >>(pdfFieldId);
   communication.addPackInfo(packInfo);
#endif


   /////////////////
   /// Time Loop ///
   /////////////////

   const auto emptySweep = [](IBlock*) {};
   if( rotationFrequency > 0) {
      timeloop.add() << BeforeFunction(meshWritingFunc, "Meshwriter") <<  Sweep(emptySweep);
      timeloop.add() << BeforeFunction(objectRotatorFunc, "ObjectRotator") <<  Sweep(emptySweep);
   }
   timeloop.add() << BeforeFunction(communication.getCommunicateFunctor(), "Communication")
                  << Sweep(deviceSyncWrapper(boundaryCollection.getSweep(BoundaryCollection_T::ALL)), "Boundary Conditions");
   timeloop.add() << Sweep(deviceSyncWrapper(sweepCollection.streamCollide()), "PSMSweep");

   // Time logger
   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
                                 "remaining time logger");

   //timeloop.addFuncAfterTimeStep( makeSharedFunctor( field::makeStabilityChecker< PdfField_T >( blocks, pdfFieldId, VTKWriteFrequency ) ), "LBM stability check" );

   if (VTKWriteFrequency > 0)
   {
      const std::string path = "vtk_out";
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "fields", VTKWriteFrequency, 0,
                                                              false, path, "simulation_step", false, true, true, false, 0);

      vtkOutput->addBeforeFunction([&]() {
         for (auto& block : *blocks)
            sweepCollection.calculateMacroscopicParameters(&block);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         gpu::fieldCpy< VectorField_T, gpu::GPUField< real_t > >(blocks, velocityFieldId, velocityFieldGPUId);
         gpu::fieldCpy< FracField_T, gpu::GPUField< real_t > >(blocks, fractionFieldId, fractionFieldGPUId);
#endif
      });

      auto velWriter = make_shared< field::VTKWriter< VectorField_T > >(velocityFieldId, "Velocity");
      //auto flagWriter = make_shared< field::VTKWriter< FlagField_T > >(flagFieldId, "Flag");
      auto fractionFieldWriter = make_shared< field::VTKWriter< FracField_T > >(fractionFieldId, "FractionField");
      //auto objVeldWriter = make_shared< field::VTKWriter< VectorField_T > >(objectVelocitiesFieldId, "objectVelocity");

      vtkOutput->addCellDataWriter(velWriter);
      //vtkOutput->addCellDataWriter(flagWriter);
      vtkOutput->addCellDataWriter(fractionFieldWriter);
      //vtkOutput->addCellDataWriter(objVeldWriter);

      timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
      vtk::writeDomainDecomposition(blocks, "domain_decomposition", "vtk_out", "write_call", true, true, 0);
   }

   lbm_generated::PerformanceEvaluation<FlagField_T> const performance(blocks, flagFieldId, fluidFlagUID);
   WcTimingPool timeloopTiming;
   WcTimer simTimer;

   simTimer.start();
   timeloop.run(timeloopTiming, true);
   simTimer.end();

   double time = simTimer.max();
   WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
   performance.logResultOnRoot(timesteps, time);

   const auto reducedTimeloopTiming = timeloopTiming.getReduced();
   WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)

   //printResidentMemoryStatistics();

   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv) { return walberla::main(argc, argv); }
