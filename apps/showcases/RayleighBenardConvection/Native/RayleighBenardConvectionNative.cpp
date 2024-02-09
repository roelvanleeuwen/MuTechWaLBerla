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
// #include "blockforest/communication/UniformBufferedScheme.h"
#include "core/Environment.h"
#include "core/logging/Initialization.h"
#include "core/math/Constants.h"
#include "core/timing/RemainingTimeLogger.h"

#include "field/AddToStorage.h"

#include "lbm/blockforest/communication/SimpleCommunication.h"
// #include "field/Gather.h"
#include "field/FlagField.h"
#include "field/vtk/VTKWriter.h"

#include "geometry/InitBoundaryHandling.h"
#include "geometry/mesh/TriangleMeshIO.h"

#include "lbm/PerformanceEvaluation.h"
#include "lbm/boundary/all.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/vtk/Density.h"

#include "python_coupling/CreateConfig.h"
// #include "python_coupling/PythonCallback.h"

#include "timeloop/SweepTimeloop.h"

//#include "InitializerFunctions.h"
#include "RayleighBenardConvectionLatticeModel_fluid.h"
#include "RayleighBenardConvectionLatticeModel_thermal.h"

using namespace walberla;

///////////////////////
/// Typedef Aliases ///
///////////////////////
using ScalarField_T          = GhostLayerField< real_t, 1 >;
using VectorField_T          = GhostLayerField< Vector3< real_t >, 1 >;
using VectorFieldFlattened_T = GhostLayerField< real_t, 3 >;

//using CollisionModel_T = lbm::collision_model::SRT;

//--------------------------- fluid -----------------------------
using LatticeModelFluid_T     = lbm::RayleighBenardConvectionLatticeModel_fluid;
using FluidStencil_T          = LatticeModelFluid_T::Stencil;
using PdfFieldFluid_T         = lbm::PdfField< LatticeModelFluid_T >;
using PdfFluidCommunication_T = blockforest::SimpleCommunication< FluidStencil_T >;

//--------------------------- thermal ---------------------------
using LatticeModelThermal_T     = lbm::RayleighBenardConvectionLatticeModel_thermal;
using ThermalStencil_T          = LatticeModelThermal_T::Stencil;
using PdfFieldThermal_T         = lbm::PdfField< LatticeModelThermal_T >;
using PdfThermalCommunication_T = blockforest::SimpleCommunication< ThermalStencil_T >;

using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

// function describing the initialization profile (in global coordinates)
real_t initializationProfile(real_t x, real_t amplitude, real_t offset, real_t wavelength)
{
   return amplitude * std::cos(x / wavelength * real_c(2) * math::pi + math::pi) + offset;
}

template< typename LatticeModel_T >
void initTemperatureFieldTest(const shared_ptr< StructuredBlockStorage >& blocks,
                              BlockDataID temperature_field_ID,
                              BlockDataID pdf_field_ID,
                              real_t amplitude, Vector3< uint_t > domainSize, real_t temperatureRange)
{
   for (auto& block : *blocks)
   {
      auto temperatureField    = block.getData< ScalarField_T >(temperature_field_ID);
      auto pdfField = block.getData< lbm::PdfField< LatticeModel_T > >(pdf_field_ID);

      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(temperatureField, {
      //WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(pdfField, {
         // cell in block-local coordinates
         Cell globalCell;
         blocks->transformBlockLocalToGlobalCell(globalCell, block, Cell(x, y, z));

         uint_t sampleSize = 100;
         real_t stepSize = real_c(1.) / real_c(sampleSize);
         real_t offset = real_c(domainSize[1] / 2);
         real_t wavelength = real_c(domainSize[0]);

         //--
         Vector3 vel(real_c(0));
         //--
         for (uint_t xSample = uint_c(0); xSample <= sampleSize; ++xSample)
         {
            // value of the sine-function
            const real_t functionValue = initializationProfile(real_c(globalCell[0]) + real_c(xSample) * stepSize, amplitude, offset, wavelength);
            for (uint_t ySample = uint_c(0); ySample <= sampleSize; ++ySample)
            {
               const real_t yPoint = real_c(globalCell[1]) + real_c(ySample) * stepSize;
               //temperatureField->get(x, y, z) = (yPoint < functionValue) ?  temperatureRange/50 : -temperatureRange/50;
               auto temp = (yPoint < functionValue) ?  temperatureRange/50 : -temperatureRange/50;
               pdfField->setDensityAndVelocity(x, y, z, vel, temp);
            }
         }
      }) // WALBERLA_FOR_ALL_CELLS
   }
}

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
      // WALBERLA_LOG_DEVEL_VAR_ON_ROOT(*config)

      ///////////////////////////
      // ADD DOMAIN PARAMETERS //
      ///////////////////////////

      uint_t nrOfProcesses = uint_c(MPIManager::instance()->numProcesses());
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(nrOfProcesses)

      auto domainSetup                = config->getOneBlock("DomainSetup");
      Vector3< uint_t > cellsPerBlock = domainSetup.getParameter< Vector3< uint_t > >("cellsPerBlock");
      std::vector< config::Config::Block* > configDomainSetupBlock;
      config->getWritableGlobalBlock().getWritableBlocks("DomainSetup", configDomainSetupBlock, 1, 1);
      Vector3< uint_t > blocksPerDimension;
      Vector3< uint_t > cellsPerBlockDummy;
      blockforest::calculateCellDistribution(cellsPerBlock, nrOfProcesses, blocksPerDimension, cellsPerBlockDummy);

      Vector3< uint_t > domainSize;
      domainSize[0] = blocksPerDimension[0] * cellsPerBlock[0];
      domainSize[1] = blocksPerDimension[1] * cellsPerBlock[1];
      domainSize[2] = blocksPerDimension[2] * cellsPerBlock[2];
      // WALBERLA_LOG_INFO_ON_ROOT("cellsPerBlock = (" << cellsPerBlock[0] << ", " << cellsPerBlock[1] << ", " <<
      // cellsPerBlock[2] << ")") WALBERLA_LOG_INFO_ON_ROOT("blocksPerDimension = (" << blocksPerDimension[0] << ", " <<
      // blocksPerDimension[1] << ", " << blocksPerDimension[2] << ")") WALBERLA_LOG_INFO_ON_ROOT("domain size = (" <<
      // domainSize[0] << ", " << domainSize[1] << ", " << domainSize[2] << ")")

      std::string tmp = "< " + std::to_string(blocksPerDimension[0]) + ", " + std::to_string(blocksPerDimension[1]) +
                        ", " + std::to_string(blocksPerDimension[2]) + " >";
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(tmp)
      configDomainSetupBlock[0]->setOrAddParameter("blocks", tmp);

      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(*config)

      const shared_ptr< StructuredBlockForest > blockForest = blockforest::createUniformBlockGridFromConfig(config);
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
      //const std::string timeStepStrategy = parameters.getParameter< std::string >("timeStepStrategy", "normal");
      const uint_t timesteps             = parameters.getParameter< uint_t >("timesteps", uint_c(50));
      const real_t remainingTimeLoggerFrequency =
         parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(3.0));
      // const uint_t scenario = parameters.getParameter< uint_t >("scenario", uint_c(1));
      // WALBERLA_LOG_DEVEL_VAR_ON_ROOT(timesteps)
      // WALBERLA_LOG_DEVEL_VAR_ON_ROOT(remainingTimeLoggerFrequency)

      /////////////////////////////
      // ADD PHYSICAL PARAMETERS //
      /////////////////////////////
      auto physical_parameters     = config->getOneBlock("PhysicalParameters");
      const real_t omegaFluid      = physical_parameters.getParameter< real_t >("omegaFluid", real_c(1.95));
      const real_t omegaThermal    = physical_parameters.getParameter< real_t >("omegaThermal");
      const real_t temperatureHot  = physical_parameters.getParameter< real_t >("temperatureHot", real_c(0.5));
      const real_t temperatureCold = physical_parameters.getParameter< real_t >("temperatureCold", real_c(-0.5));
      const real_t gravity         = physical_parameters.getParameter< real_t >("gravitationalAcceleration");
      //      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(omegaFluid)
      //      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(omegaThermal)
      //      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(temperatureHot)
      //      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(temperatureCold)

      ///////////////////////////////////
      // ADD INITIALIZATION PARAMETERS //
      ///////////////////////////////////
      auto initialization_parameters    = config->getOneBlock("InitializationParameters");
      const real_t initAmplitude        = initialization_parameters.getParameter< real_t >("initAmplitude");
      const real_t initTemperatureRange = initialization_parameters.getParameter< real_t >("initTemperatureRange");
      //      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(initAmplitude)
      //      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(initTemperatureRange)

      ////////////////////////
      // ADD DATA TO BLOCKS //
      ////////////////////////
      const BlockDataID temperatureFieldId =
         field::addToStorage< ScalarField_T >(blockForest, "Temperature field", real_c(0), field::fzyx);
      const BlockDataID velocityFieldId = field::addToStorage< VectorFieldFlattened_T >(
         blockForest, "Velocity field", real_c(0), field::fzyx, uint_c(1));
      //const BlockDataID densityFieldID = field::addToStorage< ScalarField_T >(blockForest, "density", real_c(0), field::fzyx);

      // create lattice model
      LatticeModelFluid_T latticeModelFluid =                           //? currently force
         LatticeModelFluid_T(temperatureFieldId, velocityFieldId, gravity, omegaFluid); //? gravity needs a sign????
      LatticeModelThermal_T latticeModelThermal =
         LatticeModelThermal_T(temperatureFieldId, velocityFieldId, omegaThermal);

      // add pdf fields for fluid and thermal parts
      const BlockDataID pdfFieldFluidId = lbm::addPdfFieldToStorage(blockForest, "PDF field fluid", latticeModelFluid,
                                                                    Vector3(real_c(0), real_c(0), real_c(0)),
                                                                    real_t(1), uint_t(1), field::fzyx);
      const BlockDataID pdfFieldThermalId =
         lbm::addPdfFieldToStorage(blockForest, "PDF field thermal", latticeModelThermal, Vector3(real_c(0), real_c(0), real_c(0)),
                                                                    real_t(0), uint_t(1),field::fzyx);
                                                                              //? what initial density makes sense here?
      //initTemperatureField(blockForest, temperatureFieldId, initAmplitude, domainSize, initTemperatureRange);
      initTemperatureFieldTest<LatticeModelThermal_T>(blockForest, temperatureFieldId, pdfFieldThermalId, initAmplitude, domainSize, initTemperatureRange);

      PdfThermalCommunication_T pdfThermalCommunication(blockForest, pdfFieldThermalId);

      ////////////////
      // ADD SWEEPS //
      ////////////////
      auto lbmSweepFluid   = LatticeModelFluid_T::Sweep(pdfFieldFluidId);
      auto lbmSweepThermal = LatticeModelThermal_T::Sweep(pdfFieldThermalId);

      ///////////////////////
      // BOUNDARY HANDLING //
      ///////////////////////
      const FlagUID fluidFlagUID("Fluid");
      const FlagUID thermalFlagUID("Thermal");

      // Boundaries Fluid
      typedef lbm::DefaultBoundaryHandlingFactory< LatticeModelFluid_T, FlagField_T > BHFactoryFluid;

      BlockDataID flagFieldFluidId = field::addFlagFieldToStorage< FlagField_T >(blockForest, "fluid flag field");

      BlockDataID fluidBoundaryHandlingId = BHFactoryFluid::addBoundaryHandlingToStorage(
         blockForest, "Fluid Boundary Handling", flagFieldFluidId, pdfFieldFluidId, fluidFlagUID,
         Vector3< real_t >(real_c(0), real_c(0), real_c(0)), Vector3< real_t >(real_c(0), real_c(0), real_c(0)),
         real_c(1), real_c(1));

      auto fluidBoundariesConfig = config->getBlock("Boundaries_Fluid");
      geometry::initBoundaryHandling< BHFactoryFluid::BoundaryHandling >(*blockForest, fluidBoundaryHandlingId,
                                                                         fluidBoundariesConfig);
      geometry::setNonBoundaryCellsToDomain< BHFactoryFluid::BoundaryHandling >(*blockForest, fluidBoundaryHandlingId);

      // Boundaries Thermal
      typedef lbm::DefaultBoundaryHandlingFactory< LatticeModelThermal_T, FlagField_T > BHFactoryThermal;

      BlockDataID flagFieldThermalId = field::addFlagFieldToStorage< FlagField_T >(blockForest, "Thermal flag field");

      BlockDataID thermalBoundaryHandlingId = BHFactoryThermal::addBoundaryHandlingToStorage(
         blockForest, "Thermal Boundary Handling", flagFieldThermalId, pdfFieldThermalId,
         thermalFlagUID, //? fluidFlagUID, does that work like that?
         Vector3< real_t >(real_c(0), real_c(0), real_c(0)), Vector3< real_t >(real_c(0), real_c(0), real_c(0)),
         temperatureCold, temperatureHot);

      auto thermalBoundariesConfig = config->getBlock("Boundaries_Thermal");
      geometry::initBoundaryHandling< BHFactoryThermal::BoundaryHandling >(*blockForest, thermalBoundaryHandlingId,
                                                                           thermalBoundariesConfig);
      geometry::setNonBoundaryCellsToDomain< BHFactoryThermal::BoundaryHandling >(*blockForest,
                                                                                  thermalBoundaryHandlingId);

      ///////////////
      // TIME LOOP //
      ///////////////
      auto benchmark_parameters = config->getOneBlock("BenchmarkParameters");
      std::string scaling_type  = benchmark_parameters.getParameter< std::string >("scalingType");
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(scaling_type)
      WALBERLA_LOG_DEVEL_VAR_ON_ROOT(nrOfProcesses)
      WALBERLA_LOG_INFO_ON_ROOT("#blocks = ("
                                << blockForest->getXSize() << ", " << blockForest->getYSize() << ", "
                                << blockForest->getZSize() << ") | total #blocks = "
                                << blockForest->getXSize() * blockForest->getYSize() * blockForest->getZSize())
      WALBERLA_LOG_INFO_ON_ROOT("block size = (" << blockForest->getRootBlockXSize() << ", "
                                                 << blockForest->getRootBlockYSize() << ", "
                                                 << blockForest->getRootBlockZSize() << ")")
      WALBERLA_LOG_INFO_ON_ROOT("domain size = (" << domainSize[0] << ", " << domainSize[1] << ", " << domainSize[2]
                                                  << ")")

      bool weak_scaling = benchmark_parameters.getParameter< bool >("weakScaling");

      SweepTimeloop timeloop(blockForest->getBlockStorage(), timesteps);
      if (!weak_scaling)
      {
         timeloop.add() << Sweep(BHFactoryThermal::BoundaryHandling::getBlockSweep(thermalBoundaryHandlingId),
                                 "Thermal boundary conditions")
                        << AfterFunction(PdfThermalCommunication_T(blockForest, pdfFieldThermalId),
                                         "Communication of thermal PDFs");
         timeloop.add() << Sweep(lbmSweepThermal, "Thermal LB Step");

         timeloop.add() << BeforeFunction(PdfFluidCommunication_T(blockForest, pdfFieldFluidId),
                                          "Communication of fluid PDFs")
                        << Sweep(BHFactoryFluid::BoundaryHandling::getBlockSweep(fluidBoundaryHandlingId),
                                 "Fluid NoSlip boundary conditions");
         timeloop.add() << Sweep(lbmSweepFluid, "Fluid LB Step");

         // remaining time logger
         timeloop.addFuncAfterTimeStep(
            RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
            "remaining time logger");

         // write VTK files
         uint_t vtkWriteFrequency = parameters.getParameter< uint_t >("vtkWriteFrequency", 0);
         if (vtkWriteFrequency > 0)
         {
            auto vtkOutput = vtk::createVTKOutput_BlockData(*blockForest, "vtk", vtkWriteFrequency, 1, false, "vtk_out",
                                                            "simulation_step", false, true, true, false, 0);

            vtkOutput->addCellDataWriter( make_shared< lbm::DensityVTKWriter < LatticeModelFluid_T, float > >( pdfFieldFluidId, "DensityFromPDF" ) );
            vtkOutput->addCellDataWriter( make_shared< lbm::DensityVTKWriter < LatticeModelThermal_T, float > >( pdfFieldThermalId, "TemperatureFromPDF" ) );

            // add velocity field as VTK output
            auto velWriter = make_shared< field::VTKWriter< VectorFieldFlattened_T > >(velocityFieldId, "Velocity");
            vtkOutput->addCellDataWriter(velWriter);

            // add temperature field as VTK output
            auto tempWriter = make_shared< field::VTKWriter< ScalarField_T > >(temperatureFieldId, "Temperature");
            vtkOutput->addCellDataWriter(tempWriter);

            // add thermal flag field as VTK output
            auto thermalFlagWriter =
               make_shared< field::VTKWriter< FlagField_T > >(flagFieldThermalId, "FlagFieldThermal");
            vtkOutput->addCellDataWriter(thermalFlagWriter);

            // add fluid flag field as VTK output
            auto fluidFlagWriter = make_shared< field::VTKWriter< FlagField_T > >(flagFieldFluidId, "FlagFieldFluid");
            vtkOutput->addCellDataWriter(fluidFlagWriter);

            timeloop.addFuncBeforeTimeStep(writeFiles(vtkOutput), "VTK Output");
         }

         // Performance evaluation
         lbm::PerformanceEvaluation< FlagField_T > performance(blockForest, flagFieldFluidId, fluidFlagUID);
         WcTimingPool timeloopTiming;
         WcTimer simTimer;

         WALBERLA_LOG_INFO_ON_ROOT("Starting simulation with " << timesteps << " time steps")
         WALBERLA_MPI_WORLD_BARRIER()
         simTimer.start();
         timeloop.run(timeloopTiming);
         WALBERLA_MPI_WORLD_BARRIER()
         simTimer.end();

         auto time = real_c(simTimer.max());
         WALBERLA_MPI_SECTION() { reduceInplace(time, walberla::mpi::MAX); }
         performance.logResultOnRoot(timesteps, time);

         const auto reducedTimeloopTiming = timeloopTiming.getReduced();
         WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)
         WALBERLA_LOG_INFO_ON_ROOT("Simulation done!")
      }
   }
   return EXIT_SUCCESS;
}