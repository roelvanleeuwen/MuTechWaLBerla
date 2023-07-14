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
//! \file PSM_Test.cpp
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================



#include "core/all.h"
#include "core/MemoryUsage.h"

#include "domain_decomposition/all.h"
#include "field/all.h"
#include "field/vtk/VTKWriter.h"
#include "geometry/all.h"

#include "lbm_generated/communication/UniformGeneratedPdfPackInfo.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/field/AddToStorage.h"

#include "lbm_generated/evaluation/PerformanceEvaluation.h"

#include "stencil/D3Q19.h"
#include "timeloop/all.h"

#include "ObjectRotator.h"
#include "PSM_InfoHeader.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#include "lbm_generated/gpu/AddToStorage.h"
#include "lbm_generated/gpu/GPUPdfField.h"
#include "lbm_generated/gpu/UniformGeneratedGPUPdfPackInfo.h"
#endif

namespace walberla
{
///////////////////////
/// Typedef Aliases ///
///////////////////////

using StorageSpecification_T = lbm::PSMStorageSpecification;
using Stencil_T = lbm::PSMStorageSpecification::Stencil;
using PdfField_T = lbm_generated::PdfField< StorageSpecification_T >;
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
using GPUPdfField_T = lbm_generated::GPUPdfField< StorageSpecification_T >;
#endif

typedef walberla::uint8_t flag_t;
typedef FlagField< flag_t > FlagField_T;
using BoundaryCollection_T = lbm::PSMBoundaryCollection< FlagField_T >;

using SweepCollection_T = lbm::PSMSweepCollection;

const FlagUID fluidFlagUID("Fluid");
const FlagUID noSlipFlagUID("NoSlip");


auto deviceSyncWrapper = [](std::function< void(IBlock*) > sweep) {
   return [sweep](IBlock* b) {
      sweep(b);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpuDeviceSynchronize();
#endif
   };
};

/////////////////////
/// Main Function ///
/////////////////////

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

   const uint_t rotationFrequency = domainParameters.getParameter< uint_t >("rotationFrequency", uint_t(1));
   const Vector3< int > rotationAxis = domainParameters.getParameter< Vector3< int > >("rotationAxis", Vector3< int >(1,0,0));
   const bool preProcessFractionFields = domainParameters.getParameter< bool >("preProcessFractionFields", false);
   const real_t rotationSpeed = domainParameters.getParameter< real_t >("rotationSpeed");


   const Vector3< real_t > domainScaling = domainParameters.getParameter< Vector3< real_t > >("domainScaling", Vector3< real_t >(1.0));
   const Vector3< real_t > domainTransforming = domainParameters.getParameter< Vector3< real_t > >("domainTransforming", Vector3< real_t >(1.0));
   const Vector3< bool > periodicity = domainParameters.getParameter< Vector3< bool > >("periodic", Vector3< bool >(true));
   const Vector3< uint_t > cellsPerBlock = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");

   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");

   const Vector3< real_t > initialVelocity = parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >(0.0));
   const uint_t timestepsFixed = parameters.getParameter< uint_t >("timesteps", uint_c(10));
   const uint_t VTKWriteFrequency = parameters.getParameter< uint_t >("VTKwriteFrequency", uint_c(10));
   const real_t remainingTimeLoggerFrequency = parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(3.0));

   const std::string timeStepStrategy = parameters.getParameter<std::string>("timeStepStrategy", "noOverlap");
   const Cell innerOuterSplit = Cell(parameters.getParameter< Vector3<cell_idx_t> >("innerOuterSplit", Vector3<cell_idx_t>(1, 1, 1)));


   real_t ref_velocity = parameters.getParameter<real_t>("ref_velocity");
   real_t ref_length = parameters.getParameter<real_t>("ref_length");
   real_t viscosity = parameters.getParameter<real_t>("viscosity");
   real_t sim_time = parameters.getParameter<real_t>("sim_time");
   real_t mesh_size = parameters.getParameter<real_t>("mesh_size");
   const real_t dx = mesh_size;
   real_t reynolds_number = (ref_velocity * ref_length) / viscosity;
   real_t Cu = ref_velocity / initialVelocity[0];
   real_t Ct = mesh_size / Cu;

   real_t viscosity_lattice = viscosity * Ct / (mesh_size * mesh_size);
   real_t omega = real_c(1.0 / (3.0 * viscosity_lattice + 0.5));
   uint_t timesteps;
   if(sim_time > 0.0)
      timesteps = uint_c(sim_time / Ct);
   else
      timesteps = timestepsFixed;

   real_t rotPerSec = rotationSpeed / (2 * M_PI);
   real_t radPerTimestep = rotationSpeed * Ct;
   real_t rotationAngle = radPerTimestep * real_c(rotationFrequency);


   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   std::string meshFileBase = "CROR_base.obj";
   auto meshBase = make_shared< mesh::TriangleMesh >();
   mesh::readAndBroadcast(meshFileBase, *meshBase);
   auto distanceOctreeMeshBase = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(meshBase));

   std::string meshFileRotor = "CROR_rotor.obj";
   auto meshRotor = make_shared< mesh::TriangleMesh >();
   mesh::readAndBroadcast(meshFileRotor, *meshRotor);
   auto distanceOctreeMeshRotor = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(meshRotor));

   std::string meshFileStator = "CROR_stator.obj";
   auto meshStator = make_shared< mesh::TriangleMesh >();
   mesh::readAndBroadcast(meshFileStator, *meshStator);
   auto distanceOctreeMeshStator = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(meshStator));

   auto aabb = computeAABB(*meshBase);
   aabb.setCenter(aabb.center() - Vector3< real_t >(domainTransforming[0] * aabb.xSize(), domainTransforming[1] * aabb.ySize(), domainTransforming[2] * aabb.zSize()));
   aabb.scale(domainScaling);
   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, Vector3< real_t >(dx), mesh::makeExcludeMeshInterior(distanceOctreeMeshBase, dx));
   bfc.setPeriodicity(periodicity);
   auto blocks = bfc.createStructuredBlockForest(cellsPerBlock);


   WALBERLA_LOG_INFO_ON_ROOT("Simulation Parameter \n"
                             << "Domain Decomposition <" << blocks->getXSize() << "," << blocks->getYSize() << "," << blocks->getZSize() << "> = " << blocks->getXSize() * blocks->getYSize() * blocks->getZSize()  << " Blocks \n"
                             << "Cells per Block " << cellsPerBlock << " \n"
                             << "Timestep Strategy " << timeStepStrategy << " \n"
                             << "Inner Outer Split " << innerOuterSplit << " \n"
                             << "Reynolds_number " << reynolds_number << "\n"
                             << "Inflow velocity " << ref_velocity << " m/s \n"
                             << "Lattice velocity " << initialVelocity[0] << "\n"
                             << "Viscosity " << viscosity << "\n"
                             << "Simulation time " << sim_time << " s \n"
                             << "Timesteps " << timesteps << "\n"
                             << "Mesh_size " << mesh_size << " m \n"
                             << "Omega " << omega << "\n"
                             << "Cu " << Cu << " \n"
                             << "Ct " << Ct << " \n"
                             << "Rotation Speed " << rotationSpeed << " rad/s \n"
                             << "Rotation Angle per Rotation " << rotationAngle * 2 * M_PI  << " ° \n"

   )

   //vtk::writeDomainDecomposition(blocks, "domain_decomposition", "vtk_out", "write_call", true, true, 0);
   //return 0;

   ////////////////////////////////////
   /// PDF Field and Velocity Setup ///
   ////////////////////////////////////

   const StorageSpecification_T StorageSpec = StorageSpecification_T();
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   auto allocator = make_shared< gpu::HostFieldAllocator<real_t> >(); // use pinned memory allocator for faster CPU-GPU memory transfers
#else
   auto allocator = make_shared< field::AllocateAligned< real_t, 64 > >();
#endif
   const BlockDataID pdfFieldId  = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, uint_c(1), field::fzyx, allocator);
   const BlockDataID velocityFieldId = field::addToStorage< VectorField_T >(blocks, "velocity", real_t(0.0), field::fzyx, uint_c(1), allocator);
   const BlockDataID densityFieldId = field::addToStorage< ScalarField_T >(blocks, "density", real_c(0.0), field::fzyx, uint_c(1), allocator);
   const BlockDataID flagFieldId     = field::addFlagFieldToStorage< FlagField_T >(blocks, "flagField");

   const BlockDataID fractionFieldId = field::addToStorage< FracField_T >(blocks, "fractionField", fracSize(0.0), field::fzyx, uint_c(1));
   const BlockDataID objectVelocitiesFieldId = field::addToStorage< VectorField_T >(blocks, "particleVelocitiesField", real_c(0.0), field::fzyx, uint_c(1), allocator);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   // GPU Field for PDFs
   const BlockDataID pdfFieldGPUId = lbm_generated::addGPUPdfFieldToStorage< PdfField_T >(blocks, pdfFieldId, StorageSpec, "pdf field on GPU", true);
   const BlockDataID velocityFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, velocityFieldId, "velocity field on GPU", true);
   const BlockDataID densityFieldGPUId = gpu::addGPUFieldToStorage< ScalarField_T >(blocks, densityFieldId, "density field on GPU", true);
   const BlockDataID fractionFieldGPUId = gpu::addGPUFieldToStorage< FracField_T >(blocks, fractionFieldId, "fraction field on GPU", true);
   const BlockDataID objectVelocitiesFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, objectVelocitiesFieldId, "object velocity field on GPU", true);
#else
   const BlockDataID fractionFieldGPUId;
#endif

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   /////////////////////////
   /// Boundary Handling ///
   /////////////////////////

   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriterBase(meshBase, "meshBase", VTKWriteFrequency);
   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriterRotor(meshRotor, "meshRotor", VTKWriteFrequency);
   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriterStator(meshStator, "meshStator", VTKWriteFrequency);
   const std::function< void() > meshWritingFunc = [&]() { meshWriterBase(); meshWriterRotor(); meshWriterStator(); };
   //meshWritingFunc();

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");

   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldId, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldId, fluidFlagUID);

   //Setting up Object Rotator
   ObjectRotator objectRotatorMeshBase(blocks, meshBase, fractionFieldId, fractionFieldGPUId, objectVelocitiesFieldId, 0, rotationFrequency, rotationAxis, distanceOctreeMeshBase, preProcessFractionFields, false);
   ObjectRotator objectRotatorMeshRotor(blocks, meshRotor, fractionFieldId, fractionFieldGPUId, objectVelocitiesFieldId, rotationAngle, rotationFrequency, rotationAxis, distanceOctreeMeshRotor, preProcessFractionFields, true);
   ObjectRotator objectRotatorMeshStator(blocks, meshStator, fractionFieldId, fractionFieldGPUId, objectVelocitiesFieldId, -rotationAngle, rotationFrequency, rotationAxis,  distanceOctreeMeshStator, preProcessFractionFields, true);

   const std::function< void() > objectRotatorFunc = [&]() {
      objectRotatorMeshBase(timeloop.getCurrentTimeStep());
      objectRotatorMeshRotor(timeloop.getCurrentTimeStep());
      objectRotatorMeshStator(timeloop.getCurrentTimeStep());
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpu::fieldCpy< gpu::GPUField< fracSize >, FracField_T >(blocks, fractionFieldGPUId, fractionFieldId);
#endif
   };

   std::vector<BlockDataID> fractionFieldIds;
   if (preProcessFractionFields) {
      uint_t numFields = uint_c(std::round(2.0 * M_PI / rotationAngle));
      WALBERLA_LOG_INFO_ON_ROOT("Start Preprocessing mesh " << numFields << " times")

      for (uint_t i = 0; i < numFields; ++i) {
         BlockDataID tmpFractionFieldId = field::addToStorage< FracField_T >(blocks, "fractionFieldId_" + std::to_string(i), fracSize(0.0), field::fzyx);
         objectRotatorMeshBase.getFractionFieldFromMesh(tmpFractionFieldId);
         objectRotatorMeshRotor.rotate();
         objectRotatorMeshRotor.getFractionFieldFromMesh(tmpFractionFieldId);
         objectRotatorMeshStator.rotate();
         objectRotatorMeshStator.getFractionFieldFromMesh(tmpFractionFieldId);
         fractionFieldIds.push_back(tmpFractionFieldId);
      }
      WALBERLA_LOG_INFO_ON_ROOT("Finished Preprocessing mesh!")

   }

   const std::function< void() > syncPreprocessedFractionFields = [&]() {
      auto currTimestep = timeloop.getCurrentTimeStep();
      uint_t rotationState = (currTimestep / rotationFrequency) % fractionFieldIds.size();
      for (auto & block : *blocks) {
         FracField_T* realFractionField = block.getData< FracField_T >(fractionFieldId);
         FracField_T* fractionFieldFromVector = block.getData< FracField_T >(fractionFieldIds[rotationState]);
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(realFractionField, realFractionField->get(x,y,z) = fractionFieldFromVector->get(x,y,z);)
      }
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpu::fieldCpy< gpu::GPUField< fracSize >, FracField_T >(blocks, fractionFieldGPUId, fractionFieldId);
#endif
   };


#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   gpu::fieldCpy< gpu::GPUField< real_t >, VectorField_T >(blocks, objectVelocitiesFieldGPUId, objectVelocitiesFieldId);
#endif

   /////////////
   /// Sweep ///
   /////////////

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   SweepCollection_T sweepCollection(blocks, pdfFieldGPUId, densityFieldGPUId, velocityFieldGPUId, 0.0, 0.0, 0.0, omega, innerOuterSplit);
   pystencils::PSMSweep PSMSweep(fractionFieldGPUId, objectVelocitiesFieldGPUId, pdfFieldGPUId, real_t(0), real_t(0.0), real_t(0.0), omega, innerOuterSplit);
   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldGPUId, fluidFlagUID);

#else
   SweepCollection_T sweepCollection(blocks, pdfFieldId, densityFieldId, velocityFieldId, 0.0, 0.0, 0.0, omega, innerOuterSplit);
   pystencils::PSMSweep PSMSweep(fractionFieldId, objectVelocitiesFieldId, pdfFieldId, real_t(0), real_t(0.0), real_t(0.0), omega, innerOuterSplit);
   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldId, fluidFlagUID);
#endif

   for (auto& block : *blocks)
   {
      auto velField = block.getData<VectorField_T>(velocityFieldId);
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(velField, velField->get(x,y,z,0) = initialVelocity[0];)
      sweepCollection.initialise(&block);
   }
   /////////////////
   /// Time Loop ///
   /////////////////


   // Communication
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const bool sendDirectlyFromGPU = false;
   gpu::communication::UniformGPUScheme< Stencil_T > communication(blocks, sendDirectlyFromGPU, false);
   auto packInfo = std::make_shared<lbm_generated::UniformGeneratedGPUPdfPackInfo< GPUPdfField_T >>(pdfFieldGPUId);
   communication.addPackInfo(packInfo);
#else
   blockforest::communication::UniformBufferedScheme< Stencil_T > communication(blocks);
   auto packInfo = std::make_shared<lbm_generated::UniformGeneratedPdfPackInfo< PdfField_T >>(pdfFieldId);
   communication.addPackInfo(packInfo);
#endif



   const auto emptySweep = [](IBlock*) {};
   if (timeStepStrategy == "noOverlap")
   { // Timeloop
      timeloop.add() << BeforeFunction(communication.getCommunicateFunctor(), "Communication")
                     << Sweep(deviceSyncWrapper(boundaryCollection.getSweep(BoundaryCollection_T::ALL)),
                              "Boundary Conditions");
      timeloop.add() << Sweep(deviceSyncWrapper(PSMSweep.getSweep()), "PSMSweep");
   }
   else if(timeStepStrategy == "Overlap") {
      timeloop.add() << BeforeFunction(communication.getStartCommunicateFunctor(), "Start Communication")
                     << Sweep(deviceSyncWrapper(boundaryCollection.getSweep(BoundaryCollection_T::ALL)), "Boundary Conditions");
      timeloop.add() << Sweep(deviceSyncWrapper(PSMSweep.getInnerSweep()), "PSM Sweep Inner Frame");
      timeloop.add() << BeforeFunction(communication.getWaitFunctor(), "Wait for Communication")
                     << Sweep(deviceSyncWrapper(PSMSweep.getOuterSweep()), "PSM Sweep Outer Frame");
   } else {
      WALBERLA_ABORT("timeStepStrategy " << timeStepStrategy << " not supported")
   }

   if( rotationFrequency > 0) {
      if(preProcessFractionFields) {
         timeloop.add() << BeforeFunction(syncPreprocessedFractionFields, "syncPreprocessedFractionFields") << Sweep(emptySweep);
      }
      else {
         timeloop.add() << BeforeFunction(objectRotatorFunc, "ObjectRotator") <<  Sweep(emptySweep);
         timeloop.add() << BeforeFunction(meshWritingFunc, "Meshwriter") <<  Sweep(emptySweep);
      }
   }

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
      vtk::writeDomainDecomposition(blocks, "domain_decompositionDense", "vtk_out", "write_call", true, true, 0);

   }

   lbm_generated::PerformanceEvaluation<FlagField_T> const performance(blocks, flagFieldId, fluidFlagUID);
   WcTimingPool timeloopTiming;
   WcTimer simTimer;

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

   double time = simTimer.max();
   WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
   performance.logResultOnRoot(timesteps, time);

   const auto reducedTimeloopTiming = timeloopTiming.getReduced();
   WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)

   printResidentMemoryStatistics();

   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv) { return walberla::main(argc, argv); }
