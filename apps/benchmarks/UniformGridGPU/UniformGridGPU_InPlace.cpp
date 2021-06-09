#include "blockforest/Initialization.h"
#include "blockforest/communication/UniformBufferedScheme.h"

#include "core/Environment.h"
#include "core/logging/Initialization.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/timing/TimingPool.h"

#include "cuda/AddGPUFieldToStorage.h"
#include "cuda/DeviceSelectMPI.h"
#include "cuda/HostFieldAllocator.h"
#include "cuda/ParallelStreams.h"
#include "cuda/communication/GPUPackInfo.h"
#include "cuda/communication/UniformGPUScheme.h"

#include "domain_decomposition/SharedSweep.h"

#include "field/AddToStorage.h"
#include "field/FlagField.h"
#include "field/communication/PackInfo.h"
#include "field/vtk/VTKWriter.h"

#include "geometry/all.h"

#include "gui/Gui.h"

#include "lbm/PerformanceLogger.h"
#include "lbm/inplace_streaming/TimestepTracker.h"

#include "python_coupling/CreateConfig.h"
#include "python_coupling/DictWrapper.h"
#include "python_coupling/PythonCallback.h"

#include "timeloop/all.h"

#include "InitShearVelocity.h"

#ifdef WALBERLA_ENABLE_GUI
#   include "lbm/gui/PdfFieldDisplayAdaptor.h"
#endif

#include <cmath>

#include "UniformGridGPU_InPlace_Defines.h"
#include "UniformGridGPU_InPlace_LbKernel.h"
#include "UniformGridGPU_InPlace_MacroSetter.h"
#include "UniformGridGPU_InPlace_PackInfoEven.h"
#include "UniformGridGPU_InPlace_PackInfoOdd.h"

using namespace walberla;

using CommunicationStencil_T = Stencil_T;
using PdfField_T             = GhostLayerField< real_t, Stencil_T::Q >;
using VelocityField_T        = GhostLayerField< real_t, 3 >;

class AlternatingBeforeFunction
{
 public:
   typedef std::function< void() > BeforeFunction;

   AlternatingBeforeFunction(BeforeFunction evenFunc, BeforeFunction oddFunc,
                             std::shared_ptr< lbm::TimestepTracker >& tracker)
      : tracker_(tracker), funcs_{ evenFunc, oddFunc } {};

   void operator()() { funcs_[tracker_->getCounter()](); }

 private:
   std::shared_ptr< lbm::TimestepTracker > tracker_;
   std::vector< BeforeFunction > funcs_;
};

int main(int argc, char** argv)
{
   mpi::Environment env(argc, argv);
   cuda::selectDeviceBasedOnMpiRank();

   for (auto cfg = python_coupling::configBegin(argc, argv); cfg != python_coupling::configEnd(); ++cfg)
   {
      WALBERLA_MPI_WORLD_BARRIER();

      WALBERLA_CUDA_CHECK(cudaPeekAtLastError());

      auto config = *cfg;
      logging::configureLogging(config);
      auto blocks = blockforest::createUniformBlockGridFromConfig(config);

      Vector3< uint_t > cellsPerBlock =
         config->getBlock("DomainSetup").getParameter< Vector3< uint_t > >("cellsPerBlock");
      // Reading parameters
      auto parameters        = config->getOneBlock("Parameters");
      const real_t omega     = parameters.getParameter< real_t >("omega", real_c(1.4));
      const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(50));

      // Creating fields
      BlockDataID pdfFieldCpuID =
         field::addToStorage< PdfField_T >(blocks, "pdfs cpu", real_t(std::nan("")), field::fzyx);
      BlockDataID velFieldCpuID = field::addToStorage< VelocityField_T >(blocks, "vel", real_t(0), field::fzyx);

      WALBERLA_LOG_INFO_ON_ROOT("Initializing shear flow");
      initShearVelocity(blocks, velFieldCpuID);

      pystencils::UniformGridGPU_InPlace_MacroSetter setterSweep(pdfFieldCpuID, velFieldCpuID);

      for (auto& block : *blocks)
         setterSweep(&block);

      BlockDataID pdfFieldGpuID = cuda::addGPUFieldToStorage< PdfField_T >(blocks, pdfFieldCpuID, "pdfs on GPU", true);
      BlockDataID velFieldGpuID =
         cuda::addGPUFieldToStorage< VelocityField_T >(blocks, velFieldCpuID, "velocity on GPU", true);

      Vector3< int > innerOuterSplit =
         parameters.getParameter< Vector3< int > >("innerOuterSplit", Vector3< int >(1, 1, 1));

      for (uint_t i = 0; i < 3; ++i)
      {
         if (int_c(cellsPerBlock[i]) <= innerOuterSplit[i] * 2)
         {
            WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller or increase cellsPerBlock");
         }
      }

      Cell innerOuterSplitCell(innerOuterSplit[0], innerOuterSplit[1], innerOuterSplit[2]);
      bool cudaEnabledMPI = parameters.getParameter< bool >("cudaEnabledMPI", false);
      Vector3< int32_t > gpuBlockSize =
         parameters.getParameter< Vector3< int32_t > >("gpuBlockSize", Vector3< int32_t >(256, 1, 1));

      int streamHighPriority = 0;
      int streamLowPriority  = 0;
      WALBERLA_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&streamLowPriority, &streamHighPriority));
      WALBERLA_CHECK(gpuBlockSize[2] == 1);

      using LbSweep      = lbm::UniformGridGPU_InPlace_LbKernel;
      using PackInfoEven = lbm::UniformGridGPU_InPlace_PackInfoEven;
      using PackInfoOdd  = lbm::UniformGridGPU_InPlace_PackInfoOdd;
      using cuda::communication::UniformGPUScheme;

      LbSweep lbSweep(pdfFieldGpuID, velFieldGpuID, omega, gpuBlockSize[0], gpuBlockSize[1], innerOuterSplitCell);
      lbSweep.setOuterPriority(streamHighPriority);

      auto evenScheme = make_shared< UniformGPUScheme< Stencil_T > >(blocks, cudaEnabledMPI);
      evenScheme->addPackInfo(make_shared< PackInfoEven >(pdfFieldGpuID));

      auto oddScheme = make_shared< UniformGPUScheme< Stencil_T > >(blocks, cudaEnabledMPI);
      oddScheme->addPackInfo(make_shared< PackInfoOdd >(pdfFieldGpuID));

      auto defaultStream = cuda::StreamRAII::newPriorityStream(streamLowPriority);

      auto tracker = make_shared< lbm::TimestepTracker >(1);

      auto setupPhase = [&]() {
         for (auto& block : *blocks)
            kernelEven(&block);

         pullScheme->communicate();

         for (auto& block : *blocks)
            kernelOdd(&block);
      };

      auto tearDownPhase = [&]() {
         pushScheme->communicate();
         cuda::fieldCpy< PdfField_T, cuda::GPUField< real_t > >(blocks, pdfFieldCpuID, pdfFieldGpuID);
         for (auto& block : *blocks)
            getterSweep(&block);
      };

      auto simpleOverlapTimeStep = [&]() {
         // Even
         pushScheme->startCommunication(defaultStream);
         for (auto& block : *blocks)
            kernelEven.inner(&block, defaultStream);
         pushScheme->wait(defaultStream);
         for (auto& block : *blocks)
            kernelEven.outer(&block, defaultStream);

         // Odd
         pullScheme->startCommunication(defaultStream);
         for (auto& block : *blocks)
            kernelOdd.inner(&block, defaultStream);
         pullScheme->wait(defaultStream);
         for (auto& block : *blocks)
            kernelOdd.outer(&block, defaultStream);
      };

      auto normalTimeStep = [&]() {
         pushScheme->communicate(defaultStream);
         for (auto& block : *blocks)
            kernelEven(&block, defaultStream);

         pullScheme->communicate(defaultStream);
         for (auto& block : *blocks)
            kernelOdd(&block, defaultStream);
      };

      auto kernelOnlyFunc = [&]() {
         for (auto& block : *blocks)
            kernelEven(&block, defaultStream);
         for (auto& block : *blocks)
            kernelOdd(&block, defaultStream);
      };

      SweepTimeloop timeLoop(blocks->getBlockStorage(), timesteps / 2);

      const std::string timeStepStrategy = parameters.getParameter< std::string >("timeStepStrategy", "normal");
      std::function< void() > timeStep;
      if (timeStepStrategy == "noOverlap")
         timeStep = std::function< void() >(normalTimeStep);
      else if (timeStepStrategy == "simpleOverlap")
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
                                      "'complexOverlap', 'simpleOverlap', 'kernelOnly'");
      }

      timeLoop.add() << BeforeFunction(timeStep) << Sweep([](IBlock*) {}, "time step");

      // VTK
      uint_t vtkWriteFrequency = parameters.getParameter< uint_t >("vtkWriteFrequency", 0);
      if (vtkWriteFrequency > 0)
      {
         auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_out",
                                                         "simulation_step", false, true, true, false, 0);
         auto velWriter = make_shared< field::VTKWriter< VelocityField_T > >(velFieldCpuID, "vel");
         vtkOutput->addCellDataWriter(velWriter);
         vtkOutput->addBeforeFunction([&]() {
            tearDownPhase();
            setupPhase();
         });
         timeLoop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
      }

      int warmupSteps     = parameters.getParameter< int >("warmupSteps", 2);
      int outerIterations = parameters.getParameter< int >("outerIterations", 1);
      setupPhase();
      for (int i = 0; i < warmupSteps; ++i)
         timeLoop.singleStep();

      double remainingTimeLoggerFrequency =
         parameters.getParameter< double >("remainingTimeLoggerFrequency", -1.0); // in seconds
      if (remainingTimeLoggerFrequency > 0)
      {
         auto logger = timing::RemainingTimeLogger(timeLoop.getNrOfTimeSteps() * uint_c(outerIterations),
                                                   remainingTimeLoggerFrequency);
         timeLoop.addFuncAfterTimeStep(logger, "remaining time logger");
      }

      bool useGui = parameters.getParameter< bool >("useGui", false);
      if (useGui)
      {
#ifdef WALBERLA_ENABLE_GUI
         cuda::fieldCpy< PdfField_T, cuda::GPUField< real_t > >(blocks, pdfFieldCpuID, pdfFieldGpuID);
         timeLoop.addFuncAfterTimeStep(
            cuda::fieldCpyFunctor< PdfField_T, cuda::GPUField< real_t > >(blocks, pdfFieldCpuID, pdfFieldGpuID),
            "copy to CPU");
         GUI gui(timeLoop, blocks, argc, argv);
         gui.registerDisplayAdaptorCreator(
            [&](const IBlock& block, ConstBlockDataID blockDataID) -> gui::DisplayAdaptor* {
               if (block.isDataOfType< PdfField_T >(blockDataID))
                  return new lbm::PdfFieldDisplayAdaptor< GhostLayerField< real_t, Stencil_T::Q >, Stencil_T >(
                     blockDataID);
               return nullptr;
            });
         gui.run();
#else
         WALBERLA_ABORT_NO_DEBUG_INFO("Application was built without GUI. Set useGui to false or re-compile with GUI.")
#endif
      }
      else
      {
         for (int outerIteration = 0; outerIteration < outerIterations; ++outerIteration)
         {
            WALBERLA_CUDA_CHECK(cudaPeekAtLastError());

            timeLoop.setCurrentTimeStepToZero();
            WcTimer simTimer;
            cudaDeviceSynchronize();
            WALBERLA_CUDA_CHECK(cudaPeekAtLastError());
            WALBERLA_LOG_INFO_ON_ROOT("Starting simulation with " << timesteps << " time steps");
            simTimer.start();
            timeLoop.run();
            cudaDeviceSynchronize();
            simTimer.end();
            WALBERLA_LOG_INFO_ON_ROOT("Simulation finished");
            auto time      = simTimer.last();
            auto nrOfCells = real_c(cellsPerBlock[0] * cellsPerBlock[1] * cellsPerBlock[2]);

            auto mlupsPerProcess = nrOfCells * real_c(timesteps) / time * 1e-6;
            WALBERLA_LOG_RESULT_ON_ROOT("MLUPS per process " << mlupsPerProcess);
            WALBERLA_LOG_RESULT_ON_ROOT("Time per time step " << time / real_c(timesteps));
            WALBERLA_ROOT_SECTION()
            {
               python_coupling::PythonCallback pythonCallbackResults("results_callback");
               if (pythonCallbackResults.isCallable())
               {
                  const char* storagePattern = "InPlace";
                  pythonCallbackResults.data().exposeValue("mlupsPerProcess", mlupsPerProcess);
                  pythonCallbackResults.data().exposeValue("stencil", infoStencil);
                  pythonCallbackResults.data().exposeValue("configName", infoConfigName);
                  pythonCallbackResults.data().exposeValue("storagePattern", storagePattern);
                  pythonCallbackResults.data().exposeValue("cse_global", infoCseGlobal);
                  pythonCallbackResults.data().exposeValue("cse_pdfs", infoCsePdfs);
                  // Call Python function to report results
                  pythonCallbackResults();
               }
            }
         }
      }
   }

   return 0;
}
