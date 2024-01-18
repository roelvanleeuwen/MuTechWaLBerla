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
//! \file 03_AdvancedLBMCodegen.cpp
//! \author Frederik Hennig <frederik.hennig@fau.de>
//
//======================================================================================================================

#include "blockforest/all.h"

#include "core/all.h"

#   include "gpu/AddGPUFieldToStorage.h"
#   include "gpu/ParallelStreams.h"
#   include "gpu/communication/UniformGPUScheme.h"

#include "domain_decomposition/all.h"

#include "field/all.h"
#include "field/vtk/VTKWriter.h"
#include "lbm_generated/evaluation/PerformanceEvaluation.h"

#include "geometry/all.h"

#include "stencil/D2Q9.h"

#include "timeloop/all.h"

//    Codegen Includes
#include "modified_codeGen/SYCLTestPackInfo.h"
#include "modified_codeGen/SYCLTestSweep.h"
#include "InitialPDFsSetter.h"



namespace walberla
{
///////////////////////
/// Typedef Aliases ///
///////////////////////

// Communication Pack Info
typedef pystencils::SYCLTestPackInfo PackInfo_T;

// LB Method Stencil
typedef stencil::D2Q9 Stencil_T;

// PDF field type
typedef field::GhostLayerField< real_t, Stencil_T::Size > PdfField_T;

// Velocity Field Type
typedef field::GhostLayerField< real_t, Stencil_T::D > VectorField_T;

// Boundary Handling
typedef walberla::uint8_t flag_t;
typedef FlagField< flag_t > FlagField_T;

typedef gpu::GPUField< real_t > GPUField;

//////////////////////////////////////////
/// Shear Flow Velocity Initialization ///
//////////////////////////////////////////

void initShearFlowVelocityField(const shared_ptr< StructuredBlockForest >& blocks, const BlockDataID& velocityFieldId,
                                const Config::BlockHandle& config)
{
   math::RealRandom< real_t > rng(config.getParameter< std::mt19937::result_type >("noiseSeed", 42));

   real_t const velocityMagnitude = config.getParameter< real_t >("velocityMagnitude", real_c(0.08));
   real_t const noiseMagnitude    = config.getParameter< real_t >("noiseMagnitude", real_c(0.1) * velocityMagnitude);

   auto n_y = real_c(blocks->getNumberOfYCells());

   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      auto u = (*blockIt).getData< VectorField_T >(velocityFieldId);

      for (auto cellIt = u->beginWithGhostLayerXYZ(); cellIt != u->end(); ++cellIt)
      {
         Cell globalCell(cellIt.cell());
         blocks->transformBlockLocalToGlobalCell(globalCell, *blockIt);

         auto relative_y = real_c(globalCell.y()) / n_y;

         u->get(cellIt.cell(), 0) = relative_y < 0.3 || relative_y > 0.7 ? velocityMagnitude : -velocityMagnitude;

         u->get(cellIt.cell(), 1) = noiseMagnitude * rng();
      }
   }
}

/////////////////////
/// Main Function ///
/////////////////////

int main(int argc, char** argv)
{
   logging::Logging::instance()->setLogLevel( logging::Logging::INFO );

   walberla::Environment walberlaEnv(argc, argv);

   if (!walberlaEnv.config()) { WALBERLA_ABORT("No configuration file specified!"); }

   ///////////////////////////////////////////////////////
   /// Block Storage Creation and Simulation Parameter ///
   ///////////////////////////////////////////////////////

   auto blocks = blockforest::createUniformBlockGridFromConfig(walberlaEnv.config());

   WALBERLA_LOG_PROGRESS("Create SYCL queue")
   //auto syclQueue = make_shared<sycl::queue> (sycl::default_selector_v);
   auto syclQueue = make_shared<sycl::queue> (sycl::cpu_selector_v);
   WALBERLA_LOG_INFO("Running SYCL on " << (*syclQueue).get_device().get_info<cl::sycl::info::device::name>())
   blocks->setSYCLQueue(syclQueue);

   // read parameters
   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");

   const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(10));
   const real_t omega     = parameters.getParameter< real_t >("omega", real_c(1.8));
   const real_t remainingTimeLoggerFrequency =
      parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(3.0)); // in seconds
   const uint_t VTKwriteFrequency = parameters.getParameter< uint_t >("VTKwriteFrequency", 1000);

   ////////////////////////////////////
   /// PDF Field and Velocity Setup ///
   ////////////////////////////////////

   // Common Fields
   BlockDataID velocityFieldId = field::addToStorage< VectorField_T >(blocks, "velocity", real_c(0.0), field::fzyx);
   BlockDataID const flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");
   BlockDataID pdfFieldId = field::addToStorage< PdfField_T >(blocks, "pdf field", real_c(0.0), field::fzyx);

   // GPU Field for PDFs
   BlockDataID const pdfFieldGPUId = gpu::addGPUFieldToStorage< gpu::GPUField< real_t > >(blocks, "pdf field on GPU", Stencil_T::Size, field::fzyx, uint_t(1), false);
   BlockDataID const pdfFieldTmpGPUId = gpu::addGPUFieldToStorage< gpu::GPUField< real_t > >(blocks, "pdf field tmp on GPU", Stencil_T::Size, field::fzyx, uint_t(1), false);

   BlockDataID velocityFieldIdGPU = gpu::addGPUFieldToStorage< VectorField_T >(blocks, velocityFieldId, "velocity on GPU", false);

   // Velocity field setup
   auto shearFlowSetup = walberlaEnv.config()->getOneBlock("ShearFlowSetup");
   initShearFlowVelocityField(blocks, velocityFieldId, shearFlowSetup);

   real_t const rho = shearFlowSetup.getParameter("rho", real_c(1.0));
   pystencils::InitialPDFsSetter pdfSetter(pdfFieldId, velocityFieldId, rho);
   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      pdfSetter(&(*blockIt));
   }
   // pdfs setup
   gpu::fieldCpy< GPUField, PdfField_T >(blocks, pdfFieldGPUId, pdfFieldId);


   /////////////
   /// Sweep ///
   /////////////

   SYCLTestSweep const SYCLTestSweep(syclQueue, pdfFieldGPUId, pdfFieldTmpGPUId, velocityFieldIdGPU, omega);

   /////////////////////////
   /// Boundary Handling ///
   /////////////////////////

   const FlagUID fluidFlagUID("Fluid");

   //auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");

   //geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldId, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldId, fluidFlagUID);


   /////////////////
   /// Time Loop ///
   /////////////////

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   // Communication

   const bool sendDirectlyFromGPU = false;
   gpu::communication::UniformGPUScheme< Stencil_T > com(blocks, sendDirectlyFromGPU,  false);
   com.addPackInfo(make_shared< PackInfo_T >(syclQueue, pdfFieldGPUId));
   //auto communication = std::function< void() >([&]() { com.communicate(); });


   // Timeloop
   timeloop.add() << BeforeFunction(com, "communication") << Sweep(SYCLTestSweep, "SYCL Sweep");

   // Time logger
   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
                                 "remaining time logger");

   if (VTKwriteFrequency > 0)
   {
      const std::string path = "vtk_out";
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "velocity", VTKwriteFrequency, 0,
                                                      false, path, "simulation_step", false, true, true, false, 0);

      // Copy velocity data to CPU before output
      vtkOutput->addBeforeFunction(
         [&]() { gpu::fieldCpy< VectorField_T, GPUField >(blocks, velocityFieldId, velocityFieldIdGPU); });

      auto velWriter = make_shared< field::VTKWriter< VectorField_T > >(velocityFieldId, "Velocity");
      vtkOutput->addCellDataWriter(velWriter);

      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
      vtk::writeDomainDecomposition(blocks, "domain_decomposition", "vtk_out", "write_call", true, true, 0);
   }

   lbm_generated::PerformanceEvaluation<FlagField_T> const performance(blocks, flagFieldId, fluidFlagUID);
   WcTimingPool timeloopTiming;
   WcTimer simTimer;


   simTimer.start();
   timeloop.run(timeloopTiming);
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
