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
//! \file ChannelFlowCodeGen.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================
#include "blockforest/all.h"
#include "core/all.h"
#include "domain_decomposition/all.h"
#include "field/all.h"
#include "geometry/all.h"
#include "timeloop/all.h"

#include "python_coupling/CreateConfig.h"
#include "python_coupling/PythonCallback.h"

// CodeGen includes
#include "ChannelFlowCodeGen_InfoHeader.h"
#include "ChannelFlowCodeGen_MacroGetter.h"
#include "ChannelFlowCodeGen_MacroSetter.h"
#include "ChannelFlowCodeGen_NoSlip.h"
#include "ChannelFlowCodeGen_Outflow.h"
#include "ChannelFlowCodeGen_PackInfo.h"
#include "ChannelFlowCodeGen_Sweep.h"
#include "ChannelFlowCodeGen_UBB.h"

typedef pystencils::ChannelFlowCodeGen_PackInfo PackInfo_T;
typedef walberla::uint8_t flag_t;
typedef FlagField< flag_t > FlagField_T;

auto pdfFieldAdder = [](IBlock* const block, StructuredBlockStorage * const storage) {
  return new PdfField_T(storage->getNumberOfXCells(*block),
                        storage->getNumberOfYCells(*block),
                        storage->getNumberOfZCells(*block),
                        uint_t(1),
                        field::fzyx,
                        make_shared<field::AllocateAligned<real_t, 64>>());
};


int main(int argc, char** argv)
{
   walberla::Environment walberlaEnv(argc, argv);

   for( auto cfg = python_coupling::configBegin( argc, argv ); cfg != python_coupling::configEnd(); ++cfg )
   {
      WALBERLA_MPI_WORLD_BARRIER();

      auto config = *cfg;
      logging::configureLogging( config );
      auto blocks = blockforest::createUniformBlockGridFromConfig(config);

      // read parameters
      Vector3< uint_t > cellsPerBlock = config->getBlock("DomainSetup").getParameter< Vector3< uint_t > >("cellsPerBlock");
      auto parameters = config->getOneBlock("Parameters");

      const uint_t timesteps           = parameters.getParameter< uint_t >("timesteps", uint_c(10));
      const real_t omega               = parameters.getParameter< real_t >("omega", real_t(1.9));
      const real_t u_max               = parameters.getParameter< real_t >("u_max", real_t(0.05));
      const real_t reynolds_number     = parameters.getParameter< real_t >("reynolds_number", real_t(1000));

      const double remainingTimeLoggerFrequency =
         parameters.getParameter< double >("remainingTimeLoggerFrequency", 3.0); // in seconds

      // create fields
      BlockDataID pdfFieldID     = blocks->addStructuredBlockData<PdfField_T>(pdfFieldAdder, "PDFs");
      BlockDataID velFieldID     = field::addToStorage< VelocityField_T >(blocks, "velocity", real_t(0), field::fzyx);
      BlockDataID densityFieldID = field::addToStorage< ScalarField_T >(blocks, "density", real_t(0), field::fzyx);

      BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");

      // initialise all PDFs
      pystencils::ChannelFlowCodeGen_MacroSetter setterSweep(pdfFieldID, velFieldID);
      for (auto& block : *blocks)
         setterSweep(&block);

      // setter sweep only initializes interior of domain - for push schemes to work a first communication is required here
      blockforest::communication::UniformBufferedScheme< Stencil_T > initialComm(blocks);
      initialComm.addPackInfo(make_shared< field::communication::PackInfo< PdfField_T > >(pdfFieldID));
      initialComm();

      // create and initialize boundary handling
      const FlagUID fluidFlagUID("Fluid");

      auto boundariesConfig = config->getOneBlock("Boundaries");

      lbm::ChannelFlowCodeGen_UBB ubb(blocks, pdfFieldID, u_max);
      lbm::ChannelFlowCodeGen_NoSlip noSlip(blocks, pdfFieldID);
      lbm::ChannelFlowCodeGen_Outflow outflow(blocks, pdfFieldID);

      geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldId, boundariesConfig);
      geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldId, fluidFlagUID);

      ubb.fillFromFlagField< FlagField_T >(blocks, flagFieldId, FlagUID("UBB"), fluidFlagUID);
      noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldId, FlagUID("NoSlip"), fluidFlagUID);
      outflow.fillFromFlagField< FlagField_T >(blocks, flagFieldId, FlagUID("Outflow"), fluidFlagUID);

      // create time loop
      SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

      // create communication for PdfField
      blockforest::communication::UniformBufferedScheme< Stencil_T > communication(blocks);
      communication.addPackInfo(make_shared< PackInfo_T >(pdfFieldID));

      pystencils::ChannelFlowCodeGen_Sweep LBSweep(pdfFieldID, omega);
      // add LBM sweep and communication to time loop
      timeloop.add() << BeforeFunction(communication, "communication") << Sweep(noSlip, "noSlip boundary");
      timeloop.add() << Sweep(outflow, "outflow boundary");
      timeloop.add() << Sweep(ubb, "ubb boundary");
      timeloop.add() << Sweep(LBSweep, "LB update rule");

      // LBM stability check
      timeloop.addFuncAfterTimeStep(makeSharedFunctor(field::makeStabilityChecker< PdfField_T, FlagField_T >(
                                       config, blocks, pdfFieldID, flagFieldId, fluidFlagUID)),
                                    "LBM stability check");

      // log remaining time
      timeloop.addFuncAfterTimeStep(
         timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
         "remaining time logger");

      // add VTK output to time loop
      pystencils::ChannelFlowCodeGen_MacroGetter getterSweep(densityFieldID, pdfFieldID, velFieldID);
      // VTK
      uint_t vtkWriteFrequency = parameters.getParameter< uint_t >("vtkWriteFrequency", 0);
      if (vtkWriteFrequency > 0)
      {
         auto vtkOutput     = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_out",
                                                         "simulation_step", false, true, true, false, 0);
         auto velWriter     = make_shared< field::VTKWriter< VelocityField_T > >(velFieldID, "velocity");
         auto densityWriter = make_shared< field::VTKWriter< ScalarField_T > >(densityFieldID, "density");

         vtkOutput->addCellDataWriter(velWriter);
         vtkOutput->addCellDataWriter(densityWriter);

         vtkOutput->addBeforeFunction([&]() {
            for (auto& block : *blocks)
               getterSweep(&block);
         });
         timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
      }

      WcTimer simTimer;
      WALBERLA_LOG_INFO_ON_ROOT("Starting simulation with " << timesteps << " time steps and a reynolds number of "
                                                            << reynolds_number)
      simTimer.start();
      timeloop.run();
      simTimer.end();
      WALBERLA_LOG_INFO_ON_ROOT("Simulation finished")
      auto time            = simTimer.last();
      auto nrOfCells       = real_c(cellsPerBlock[0] * cellsPerBlock[1] * cellsPerBlock[2]);
      auto mlupsPerProcess = nrOfCells * real_c(timesteps) / time * 1e-6;
      WALBERLA_LOG_RESULT_ON_ROOT("MLUPS per process " << mlupsPerProcess);
      WALBERLA_LOG_RESULT_ON_ROOT("Time per time step " << time / real_c(timesteps));
   }

   return EXIT_SUCCESS;
}
