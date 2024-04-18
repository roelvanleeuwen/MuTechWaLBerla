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
//! \file PSM_Benchmark_Test.cpp
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
#include "lbm_generated/evaluation/PerformanceEvaluation.h"
#include "lbm_generated/field/AddToStorage.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/communication/UniformGeneratedPdfPackInfo.h"
#include "stencil/D3Q19.h"
#include "timeloop/all.h"
#include <random>


#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#   include "gpu/communication/UniformGPUScheme.h"
#   include "lbm_generated/gpu/AddToStorage.h"
#   include "lbm_generated/gpu/GPUPdfField.h"
#   include "lbm_generated/gpu/UniformGeneratedGPUPdfPackInfo.h"
#endif
#include "../PSM_Complex_Geometry/MovingGeometry.h"
#include "PSM_Benchmark_Test_InfoHeader.h"

namespace walberla
{
///////////////////////
/// Typedef Aliases ///
///////////////////////

using StorageSpecification_T = lbm::PSM_Benchmark_TestStorageSpecification;
using Stencil_T = lbm::PSM_Benchmark_TestStorageSpecification::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;
using PdfField_T = lbm_generated::PdfField< StorageSpecification_T >;
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
using GPUPdfField_T = lbm_generated::GPUPdfField< StorageSpecification_T >;
using gpu::communication::UniformGPUScheme;
#endif

typedef walberla::uint8_t flag_t;
typedef FlagField< flag_t > FlagField_T;
using BoundaryCollection_T = lbm::PSM_Benchmark_TestBoundaryCollection< FlagField_T >;
using SweepCollection_T = lbm::PSM_Benchmark_TestSweepCollection;
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

   const uint_t timesteps = 1001;
   const Vector3< real_t > initialVelocity =  Vector3< real_t >(0.0);
   const real_t omega = 1.4;

   const real_t remainingTimeLoggerFrequency = real_c(5.0);
   const uint_t VTKWriteFrequency = 0;


   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   auto numBlocks = Vector3<uint_t> (1,1,1);
   auto cellsPerBlock = Vector3<uint_t>(256, 256, 256);
   auto domainAABB = AABB(0,0,0,1,1,1);

   auto blocks = walberla::blockforest::createUniformBlockGrid(domainAABB, numBlocks[0], numBlocks[1], numBlocks[2], cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2], true);

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   /////////////////////////
   /// Boundary Handling ///
   /////////////////////////


   const BlockDataID fractionFieldId = field::addToStorage< FracField_T >(blocks, "fractionField", real_t(0.0), field::fzyx, uint_c(1));
   const BlockDataID objectVelocitiesFieldId = field::addToStorage< VectorField_T >(blocks, "particleVelocitiesField", real_c(0.0), field::fzyx, uint_c(1));
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const BlockDataID fractionFieldGPUId = gpu::addGPUFieldToStorage< FracField_T >(blocks, fractionFieldId, "fractionFieldGPU", true);
   const BlockDataID objectVelocitiesFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, objectVelocitiesFieldId, "object velocity field on GPU", true);
   const BlockDataID forceFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, objectVelocitiesFieldId, "force field on GPU", true);

#endif



   /////////////////////////
   /// Fields Creation   ///
   /////////////////////////

   const StorageSpecification_T StorageSpec = StorageSpecification_T();
   const BlockDataID pdfFieldId  = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, uint_c(1), field::fzyx);
   const BlockDataID velocityFieldId = field::addToStorage< VectorField_T >(blocks, "velocity", real_t(0.0), field::fzyx, uint_c(1));
   const BlockDataID densityFieldId = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, uint_c(1));
   const BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flagField");

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const BlockDataID pdfFieldGPUId = lbm_generated::addGPUPdfFieldToStorage< PdfField_T >(blocks, pdfFieldId, StorageSpec, "pdf field on GPU", true);
   const BlockDataID velocityFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, velocityFieldId, "velocity field on GPU", true);
   const BlockDataID densityFieldGPUId = gpu::addGPUFieldToStorage< ScalarField_T >(blocks, densityFieldId, "density field on GPU", true);
#endif


   /////////////
   /// Sweep ///
   /////////////

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   SweepCollection_T sweepCollection(blocks, forceFieldGPUId, fractionFieldGPUId, objectVelocitiesFieldGPUId, pdfFieldGPUId, densityFieldGPUId, velocityFieldGPUId, omega);
   pystencils::PSM_Conditional_Sweep psmConditionalSweep( forceFieldGPUId, fractionFieldGPUId, objectVelocitiesFieldGPUId, pdfFieldGPUId, omega );
#else
   SweepCollection_T sweepCollection(blocks, fractionFieldId, objectVelocitiesFieldId, pdfFieldId, densityFieldId, velocityFieldId, omega);
   pystencils::PSM_Conditional_Sweep psmConditionalSweep( fractionFieldId, objectVelocitiesFieldId, pdfFieldId, omega );
#endif

   for (auto& block : *blocks)
   {
      auto velField = block.getData<VectorField_T>(velocityFieldId);
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(velField, velField->get(x,y,z,0) = initialVelocity[0];)

      auto fractionField = block.getData<FracField_T>(fractionFieldId);
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(fractionField, fractionField->get(x,y,z,0) = ((double) rand() / (RAND_MAX));)

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpu::GPUField<real_t> * dstVel = block.getData<gpu::GPUField<real_t>>( velocityFieldGPUId );
      const VectorField_T * srcVel = block.getData<VectorField_T>( velocityFieldId );
      gpu::fieldCpy( *dstVel, *srcVel );

      gpu::GPUField<real_t> * dstFrac = block.getData<gpu::GPUField<real_t>>( fractionFieldGPUId );
      const FracField_T * srcFrac = block.getData<FracField_T>( fractionFieldId );
      gpu::fieldCpy( *dstFrac, *srcFrac );
#endif
      sweepCollection.initialise(&block, 1);
   }

   /////////////////////
   /// Communication ///
   ////////////////////

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const bool sendDirectlyFromGPU = false;
   UniformGPUScheme< Stencil_T > communication(blocks, sendDirectlyFromGPU);
   auto packInfo = std::make_shared<lbm_generated::UniformGeneratedGPUPdfPackInfo< GPUPdfField_T >>(pdfFieldGPUId);
   communication.addPackInfo(packInfo);
#else
   UniformBufferedScheme< Stencil_T > communication(blocks);
   auto packInfo = std::make_shared<lbm_generated::UniformGeneratedPdfPackInfo< PdfField_T >>(pdfFieldId);
   communication.addPackInfo(packInfo);
#endif

   /////////////////
   /// Time Loop ///
   /////////////////

   const auto emptySweep = [](IBlock*) {};

   //timeloop.add() << BeforeFunction(communication.getCommunicateFunctor(), "Communication")
   //               << Sweep(deviceSyncWrapper(boundaryCollection.getSweep(BoundaryCollection_T::ALL)), "Boundary Conditions");
   timeloop.add() << Sweep(deviceSyncWrapper(psmConditionalSweep), "PSMConditionalSweep");

   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency), "remaining time logger");


   /////////////////
   /// VTK       ///
   /////////////////

   if (VTKWriteFrequency > 0)
   {
      const std::string path = "vtk_out";
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "fields", VTKWriteFrequency, 0, false, path, "simulation_step", false, true, true, false, 0);

      vtkOutput->addBeforeFunction([&]() {
         for (auto& block : *blocks)
            sweepCollection.calculateMacroscopicParameters(&block);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         gpu::fieldCpy< VectorField_T, gpu::GPUField< real_t > >(blocks, velocityFieldId, velocityFieldGPUId);
         gpu::fieldCpy< FracField_T, gpu::GPUField< real_t > >(blocks, fractionFieldId, fractionFieldGPUId);
         gpu::fieldCpy< VectorField_T, gpu::GPUField< real_t > >(blocks, objectVelocitiesFieldId, objectVelocitiesFieldGPUId);

#endif
      });

      auto velWriter = make_shared< field::VTKWriter< VectorField_T > >(velocityFieldId, "Velocity");
      auto fractionFieldWriter = make_shared< field::VTKWriter< FracField_T > >(fractionFieldId, "FractionField");
      auto objVeldWriter = make_shared< field::VTKWriter< VectorField_T > >(objectVelocitiesFieldId, "objectVelocity");

      vtkOutput->addCellDataWriter(velWriter);
      vtkOutput->addCellDataWriter(fractionFieldWriter);
      vtkOutput->addCellDataWriter(objVeldWriter);

      timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
      vtk::writeDomainDecomposition(blocks, "domain_decomposition", "vtk_out", "write_call", true, true, 0);
   }


   for (auto& block : *blocks)
   {
      FlagField_T *flagField = block.getData<FlagField_T>(flagFieldId);
      flagField->registerFlag(fluidFlagUID);
   }

   /////////////////
   /// TIMING    ///
   /////////////////

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

   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv) { return walberla::main(argc, argv); }
