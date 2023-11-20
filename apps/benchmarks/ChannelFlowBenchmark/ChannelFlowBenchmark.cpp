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
//! \file RayleighBenardConvection.cpp
//! \author Jonas Plewinski <jonas.plewinski@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"
#include "blockforest/communication/UniformBufferedScheme.h"

#include "core/Environment.h"
#include "core/logging/Initialization.h"
#include "core/math/Constants.h"
#include "core/timing/RemainingTimeLogger.h"

#include "field/AddToStorage.h"
#include "field/FlagField.h"
#include "field/vtk/VTKWriter.h"

#include "geometry/InitBoundaryHandling.h"
#include "geometry/mesh/TriangleMeshIO.h"

#include "lbm/PerformanceEvaluation.h"

#include "python_coupling/CreateConfig.h"
//#include "python_coupling/PythonCallback.h"

#include "timeloop/SweepTimeloop.h"

#include "GenDefines.h"
#include "InitializerFunctions.h"

#include "SQLProperties.h"
#include "sqlite/SQLite.h"

using namespace walberla;

///////////////////////
/// Typedef Aliases ///
///////////////////////
using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

/////////////////////
/// Main Function ///
/////////////////////
int main(int argc, char** argv)
{
   mpi::Environment Env(argc, argv);
   // exportDataStructuresToPython(); //> what does that do?

   for (auto cfg = python_coupling::configBegin(argc, argv); cfg != python_coupling::configEnd(); ++cfg)
   {
      WALBERLA_MPI_WORLD_BARRIER()

      auto config = *cfg;
      logging::configureLogging(config);
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(*config)

      ///////////////////////////
      // ADD DOMAIN PARAMETERS //
      ///////////////////////////

      //uint_t nrOfProcesses = uint_c(MPIManager::instance()->numProcesses());
      //WALBERLA_LOG_DEVEL_VAR_ON_ROOT(nrOfProcesses)

      // get benchmarking parameters from parameter file
      const auto benchmarkParameters       = config->getOneBlock("BenchmarkParameters");
      const bool benchmark                 = benchmarkParameters.getParameter< bool >("benchmark", false);
      const uint_t warmupSteps             = benchmarkParameters.getParameter< uint_t >("warmupSteps", uint_c(2));
      const std::string blockDecomposition = benchmarkParameters.getParameter< std::string >("blockDecomposition", "3D");
      const bool barrierAfterSweep         = benchmarkParameters.getParameter< bool >("barrier_after_sweep", false);
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(benchmark)
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(warmupSteps)
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blockDecomposition)
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(barrierAfterSweep)

      auto domainSetup                = config->getOneBlock("DomainSetup");
      Vector3< uint_t > cellsPerBlock = domainSetup.getParameter< Vector3< uint_t > >("cellsPerBlock");
      Vector3< uint_t > domainSize =
         domainSetup.getParameter< Vector3< uint_t > >("domainSize", Vector3< uint_t >(0, 0, 0));
      const Vector3< bool > periodicity       = domainSetup.getParameter< Vector3< bool > >("periodic");

      /*std::vector< config::Config::Block* > configDomainSetupBlock;
      config->getWritableGlobalBlock().getWritableBlocks("DomainSetup", configDomainSetupBlock, 1, 1);
      Vector3< uint_t > blocksPerDimension;
      Vector3< uint_t > cellsPerBlockDummy;
      blockforest::calculateCellDistribution(cellsPerBlock, nrOfProcesses, blocksPerDimension, cellsPerBlockDummy);

      Vector3< uint_t > domainSize;
      domainSize[0] = blocksPerDimension[0] * cellsPerBlock[0];
      domainSize[1] = blocksPerDimension[1] * cellsPerBlock[1];
      domainSize[2] = blocksPerDimension[2] * cellsPerBlock[2];
      WALBERLA_LOG_INFO_ON_ROOT("cellsPerBlock = (" << cellsPerBlock[0] << ", " << cellsPerBlock[1] << ", " << cellsPerBlock[2] << ")")
      WALBERLA_LOG_INFO_ON_ROOT("blocksPerDimension = (" << blocksPerDimension[0] << ", " << blocksPerDimension[1] << ", " << blocksPerDimension[2] << ")")
      WALBERLA_LOG_INFO_ON_ROOT("domain size = (" << domainSize[0] << ", " << domainSize[1] << ", " << domainSize[2] << ")")

      std::string tmp = "< " + std::to_string(blocksPerDimension[0]) + ", " + std::to_string(blocksPerDimension[1]) + ", " +
                        std::to_string(blocksPerDimension[2]) + " >";
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(tmp)
      configDomainSetupBlock[0]->setOrAddParameter("blocks", tmp);*/

      const uint_t nrOfProcesses = uint_c(MPIManager::instance()->numProcesses());
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(nrOfProcesses)

      if (domainSize[0] == uint_t(0))
      {
         Vector3< uint_t > blocksPerDimension;
         Vector3< uint_t > cellsPerBlockDummy;
         blockforest::calculateCellDistribution(cellsPerBlock, nrOfProcesses, blocksPerDimension, cellsPerBlockDummy);
         WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blocksPerDimension)
         // WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellsPerBlockDummy)

         WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blockDecomposition)
         if (blockDecomposition == "2D")
         {
            // modify resulting block decomposition, s.t. #y-blocks = 1
            if (blocksPerDimension[1] != uint_c(1))
            {
               blocksPerDimension[0] *= blocksPerDimension[1];
               blocksPerDimension[1] = uint_c(1);
            }
            WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blocksPerDimension)
         }
         // define domain size
         domainSize[0] = blocksPerDimension[0] * cellsPerBlock[0]; // domainWidth;
         domainSize[1] = blocksPerDimension[1] * cellsPerBlock[1]; // uint_c(liquidDepth * real_c(2));
         domainSize[2] = blocksPerDimension[2] * cellsPerBlock[2]; // uint_c(1);
      }

      // compute number of blocks as defined by domainSize and cellsPerBlock
      Vector3< uint_t > numBlocks;
      numBlocks[0] = uint_c(std::ceil(real_c(domainSize[0]) / real_c(cellsPerBlock[0])));
      numBlocks[1] = uint_c(std::ceil(real_c(domainSize[1]) / real_c(cellsPerBlock[1])));
      numBlocks[2] = uint_c(std::ceil(real_c(domainSize[2]) / real_c(cellsPerBlock[2])));

      //shared_ptr< StructuredBlockForest > blocks = blockforest::createUniformBlockGridFromConfig(config);
      shared_ptr< StructuredBlockForest > blocks = blockforest::createUniformBlockGrid( numBlocks[0], numBlocks[1], numBlocks[2],
                                                                                            cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2], 1,
                                                                                            0, false, false,
                                                                                            periodicity[0], periodicity[1], periodicity[2], //periodicity
                                                                                            false );
      /*WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blocks->getXSize())
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blocks->getYSize())
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blocks->getZSize())
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blocks->getRootBlockXSize())
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blocks->getRootBlockYSize())
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blocks->getRootBlockZSize())*/

      ///////////////////////////////////////
      // ADD GENERAL SIMULATION PARAMETERS //
      ///////////////////////////////////////
      auto parameters                    = config->getOneBlock("Parameters");
      const std::string timeStepStrategy = parameters.getParameter< std::string >("timeStepStrategy", "normal");
      const uint_t timesteps             = parameters.getParameter< uint_t >("timesteps", uint_c(50));
      const real_t remainingTimeLoggerFrequency =
         parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(3.0));
      // const uint_t scenario = parameters.getParameter< uint_t >("scenario", uint_c(1));
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(timesteps)
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(remainingTimeLoggerFrequency)

      ////////////////////////
      // ADD DATA TO BLOCKS //
      ////////////////////////
      BlockDataID fluid_PDFs_ID =
         field::addToStorage< PdfField_fluid_T >(blocks, "LB PDF field fluid", real_c(0.0), field::fzyx);
      BlockDataID velocity_field_ID =
         field::addToStorage< VelocityField_T >(blocks, "velocity", real_c(0.0), field::fzyx);

      /////////////////////////////
      // ADD PHYSICAL PARAMETERS //
      /////////////////////////////
      auto physical_parameters     = config->getOneBlock("PhysicalParameters");
      const real_t omegaFluid      = physical_parameters.getParameter< real_t >("omegaFluid", real_c(1.95));
      const real_t forceInX         = physical_parameters.getParameter< real_t >("forceInX");
//      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(omegaFluid)

      ///////////////////////////////////
      // ADD INITIALIZATION PARAMETERS //
      ///////////////////////////////////

      ////////////////
      // ADD SWEEPS //
      ////////////////
      pystencils::initialize_fluid_field initializeFluidField(fluid_PDFs_ID, velocity_field_ID,
                                                              forceInX);

      pystencils::fluid_lb_step fluid_lb_step(fluid_PDFs_ID, velocity_field_ID, forceInX,
                                              omegaFluid);

      ///////////////////////
      // ADD COMMUNICATION //
      ///////////////////////
      auto UniformBufferedSchemeVelocityDistributions =
         std::make_shared< blockforest::communication::UniformBufferedScheme< Stencil_fluid_T > >(blocks);
      auto generatedPackInfo_hydro = std::make_shared< walberla::pystencils::PackInfo_hydro >(fluid_PDFs_ID);
      UniformBufferedSchemeVelocityDistributions->addPackInfo(generatedPackInfo_hydro);
      auto Comm_hydro = std::function< void() >([&]() { UniformBufferedSchemeVelocityDistributions->communicate(); });

      ///////////////////////
      // BOUNDARY HANDLING //
      ///////////////////////
      const FlagUID fluidFlagUID("Fluid");
      const FlagUID wallFlagUID("BC_fluid_NoSlip");

      // Boundaries Hydro
      BlockDataID flagFieldHydroID = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field hydro");
      const FlagUID fluidFlagHydroUID("Fluid");
      auto boundariesConfigHydro = config->getBlock("Boundaries_Hydro");
      if (boundariesConfigHydro)
      {
         geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldHydroID, boundariesConfigHydro);
         geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldHydroID, fluidFlagHydroUID);
      }
      lbm::BC_fluid_NoSlip fluid_NoSlip(blocks, fluid_PDFs_ID);
      fluid_NoSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldHydroID, wallFlagUID, fluidFlagUID);

      //------------------------------------- TIME STEP DEFINITIONS ---------------------------------------------------
      /*auto kernelOnlyFuncFluid = [&]() {
         for (auto& block : *blocks)
            fluid_lb_step(&block);
      };

      auto boundaryFluid = [&](IBlock* block) { fluid_NoSlip.run(block); };
      auto channelFlow    = [&]() {
         Comm_hydro();
         for (auto& block : *blocks)
         {
            boundaryFluid(&block);
            fluid_lb_step(&block);
         }
      };*/

      ///////////////
      // TIME LOOP //
      ///////////////
      /*auto benchmark_parameters = config->getOneBlock("BenchmarkParameters");
      std::string scaling_type  = benchmark_parameters.getParameter< std::string >("scalingType");
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(scaling_type)
      WALBERLA_LOG_INFO_ON_ROOT("#blocks = (" << blocks->getXSize() << ", " << blocks->getYSize() << ", " << blocks->getZSize() << ") | total #blocks = " << blocks->getXSize() * blocks->getYSize() * blocks->getZSize())
      WALBERLA_LOG_INFO_ON_ROOT("block size = (" << blocks->getRootBlockXSize() << ", " << blocks->getRootBlockYSize() << ", " << blocks->getRootBlockZSize() << ")")
      WALBERLA_LOG_INFO_ON_ROOT("domain size = (" << domainSize[0] << ", " << domainSize[1] << ", " << domainSize[2] << ")")

      bool weak_scaling         = benchmark_parameters.getParameter< bool >("weakScaling");
      std::function< void() > timeStep;
      if (weak_scaling)
      {
         if (scaling_type == "fluid") {
            timeStep = std::function< void() >(kernelOnlyFuncFluid); }
         else if (scaling_type == "channel")
         {
            timeStep = std::function< void() >(channelFlow);
         }
         else
            WALBERLA_ABORT("Scaling type \"" << scaling_type << "\" not known!")
      }*/

      auto MPIBarrierAfterSweep = std::function< void() >([barrierAfterSweep]() { if(barrierAfterSweep) WALBERLA_MPI_BARRIER() });

      SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

      //{
      if (barrierAfterSweep)
      {
         timeloop.add() << BeforeFunction(Comm_hydro, "Communication of fluid PDFs")
                        << Sweep(fluid_NoSlip, "Fluid NoSlip boundary conditions");
         timeloop.add() << Sweep(fluid_lb_step, "Fluid LB Step")
                        << AfterFunction(MPIBarrierAfterSweep, "MPI Barrier after lbm step");
      }
      else
      {
         timeloop.add() << BeforeFunction(Comm_hydro, "Communication of fluid PDFs")
                        << Sweep(fluid_NoSlip, "Fluid NoSlip boundary conditions");
         timeloop.add() << Sweep(fluid_lb_step, "Fluid LB Step");
      }
         // initialize the two lattice Boltzmann fields
         WALBERLA_LOG_INFO_ON_ROOT("initialization of the distributions")
         for (auto& block : *blocks)
         {
            initializeFluidField(&block);
         }
         WALBERLA_LOG_INFO_ON_ROOT("initialization of the distributions done")
         Comm_hydro();

         if (!benchmark)
         {
            // remaining time logger
            timeloop.addFuncAfterTimeStep(
               timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
               "remaining time logger");

            // write VTK files
            uint_t vtkWriteFrequency = parameters.getParameter< uint_t >("vtkWriteFrequency", 0);
            if (vtkWriteFrequency > 0)
            {
            auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_out",
                                                            "simulation_step", false, true, true, false, 0);

            // add velocity field as VTK output
            auto velWriter = make_shared< field::VTKWriter< VelocityField_T > >(velocity_field_ID, "Velocity");
            vtkOutput->addCellDataWriter(velWriter);

            // add fluid flag field as VTK output
            auto fluidFlagWriter = make_shared< field::VTKWriter< FlagField_T > >(flagFieldHydroID, "FlagFieldFluid");
            vtkOutput->addCellDataWriter(fluidFlagWriter);

            timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
            }
         }

         // Performance evaluation
         lbm::PerformanceEvaluation< FlagField_T > performance(blocks, flagFieldHydroID, fluidFlagUID);
         WcTimingPool timeloopTiming;
         WcTimer simTimer;

         if (benchmark) {
            // execute warmup steps to load relevant data into the Cache
            for (uint_t i = 0; i < warmupSteps; ++i)
            timeloop.singleStep();

            timeloop.setCurrentTimeStepToZero();
         }

         WALBERLA_LOG_INFO_ON_ROOT("Starting simulation with " << timesteps << " time steps")
         WALBERLA_MPI_WORLD_BARRIER()
         simTimer.start();
         timeloop.run(timeloopTiming);
         WALBERLA_MPI_WORLD_BARRIER()
         simTimer.end();

         auto time = real_c(simTimer.max());
         WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
         performance.logResultOnRoot(timesteps, time);

         const auto reducedTimeloopTiming = timeloopTiming.getReduced();
         WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)
         WALBERLA_LOG_INFO_ON_ROOT("Simulation done!")

         WALBERLA_LOG_INFO_ON_ROOT("*** SQL OUTPUT - START ***")
         WALBERLA_ROOT_SECTION()
         {
            std::map< std::string, walberla::int64_t > integerProperties;
            std::map< std::string, double > realProperties;
            std::map< std::string, std::string > stringProperties;

            const auto evaluationParameters      = config->getOneBlock("EvaluationParameters");
            const std::string dbPath             = evaluationParameters.getParameter< std::string >("dbPath");
            const std::string dbFilename         = evaluationParameters.getParameter< std::string >("dbFilename");

            const std::string sqlFile = dbPath + dbFilename;
            WALBERLA_LOG_DEVEL_VAR_ON_ROOT(sqlFile)

            // BENCHMARK QUANTITIES
            realProperties["simulationTime"] = time;

            // GENERAL INFORMATION
            stringProperties["tag"] = "channel_flow";

            // PHYSICS PARAMETERS
            integerProperties["timesteps"] = long(timesteps);

            // BENCHMARK PARAMETERS
            integerProperties["benchmark"]         = benchmark;
            integerProperties["warmupSteps"]       = long(warmupSteps);
            stringProperties["blockDecomposition"] = blockDecomposition;
            integerProperties["barrierAfterSweep"] = barrierAfterSweep;

            addBuildInfoToSQL(integerProperties, realProperties, stringProperties);
            addDomainPropertiesToSQL(blocks, integerProperties, realProperties, stringProperties);
            addSlurmPropertiesToSQL(integerProperties, realProperties, stringProperties);

            const uint_t runId = sqlite::storeRunInSqliteDB(sqlFile, integerProperties, stringProperties, realProperties);
            sqlite::storeTimingPoolInSqliteDB(sqlFile, runId, *reducedTimeloopTiming, "Timeloop");
            WALBERLA_LOG_DEVEL_VAR(runId)
         }
         WALBERLA_LOG_INFO_ON_ROOT("*** SQL OUTPUT - END ***")
      //}
      /*else
      {
         timeloop.add() << BeforeFunction(timeStep) << Sweep([](IBlock*) {}, "time step");

         uint_t benchmarkingIterations = benchmark_parameters.getParameter< uint_t >("benchmarkingIterations", 5);
         //uint_t warmupSteps            = benchmark_parameters.getParameter< uint_t >("warmupSteps", 10);
         for (uint_t i = 0; i < warmupSteps; ++i)
            timeloop.singleStep();

         //std::string performanceStatsFilename = "performance_" + scaling_type + "_" + std::to_string(blocks->getRootBlockXSize()) + ".txt";
         std::string performanceStatsFilename = "performance.txt";

         WALBERLA_LOG_INFO_ON_ROOT("________________________________________________________________________")
         WALBERLA_LOG_INFO_ON_ROOT("------------------------------------------------------------------------")
         WALBERLA_LOG_INFO_ON_ROOT("Start benchmarking!")
         real_t meanMLUPS = real_c(0.0);
         real_t timePerTimestep = real_c(0);
         for (uint_t i = 0; i < benchmarkingIterations; ++i)
         {
            timeloop.setCurrentTimeStepToZero();
            WcTimer simTimer;
            WALBERLA_LOG_INFO_ON_ROOT("Starting simulation with " << timesteps << " time steps")
            simTimer.start();
            timeloop.run();
            simTimer.end();
            WALBERLA_LOG_INFO_ON_ROOT("Simulation finished")
            auto time = real_c(simTimer.last());
            WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
            auto nrOfCells = real_c(cellsPerBlock[0] * cellsPerBlock[1] * cellsPerBlock[2]);

            auto mlupsPerProcess = nrOfCells * real_c(timesteps) / time * 1e-6;
            meanMLUPS += mlupsPerProcess;
            timePerTimestep = real_c(time / real_c(timesteps));
            WALBERLA_LOG_RESULT_ON_ROOT("MLUPS per process " << mlupsPerProcess)
            WALBERLA_LOG_RESULT_ON_ROOT("Time per time step " << time / real_c(timesteps))
         }
         WALBERLA_ROOT_SECTION() {
            meanMLUPS /= real_t(benchmarkingIterations);
            std::ofstream performanceStatsToFile(performanceStatsFilename, std::ios::app);
            performanceStatsToFile << scaling_type << "; " << nrOfProcesses << "; " << blocks->getRootBlockXSize() << "; " << meanMLUPS << "; " << timePerTimestep << "\n";
            performanceStatsToFile.close();
         }
         WALBERLA_LOG_INFO_ON_ROOT("Benchmarking done!")
         WALBERLA_LOG_INFO_ON_ROOT("————————————————————————————————————————————————————————————————————————")
         WALBERLA_LOG_INFO_ON_ROOT("————————————————————————————————————————————————————————————————————————")
      }*/
   }
   return EXIT_SUCCESS;
}