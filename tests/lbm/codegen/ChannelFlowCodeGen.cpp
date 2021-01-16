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
#include "lbm/vtk/QCriterion.h"

#include "python_coupling/CreateConfig.h"
#include "python_coupling/PythonCallback.h"

// CodeGen includes
#include "ChannelFlowCodeGen_InfoHeader.h"
// #include "ChannelFlowCodeGen_MacroGetter.h"
#include "ChannelFlowCodeGen_MacroSetter.h"
#include "ChannelFlowCodeGen_NoSlip.h"
#include "ChannelFlowCodeGen_Outflow.h"
#include "ChannelFlowCodeGen_PackInfoEven.h"
#include "ChannelFlowCodeGen_PackInfoOdd.h"
#include "ChannelFlowCodeGen_EvenSweep.h"
#include "ChannelFlowCodeGen_OddSweep.h"
#include "ChannelFlowCodeGen_UBB.h"

typedef lbm::ChannelFlowCodeGen_PackInfoEven PackInfoEven_T;
typedef lbm::ChannelFlowCodeGen_PackInfoOdd PackInfoOdd_T;

typedef walberla::uint8_t flag_t;
typedef FlagField< flag_t > FlagField_T;

using namespace std::placeholders;

auto pdfFieldAdder = [](IBlock* const block, StructuredBlockStorage * const storage) {
  return new PdfField_T(storage->getNumberOfXCells(*block),
                        storage->getNumberOfYCells(*block),
                        storage->getNumberOfZCells(*block),
                        uint_t(1),
                        field::fzyx,
                        make_shared<field::AllocateAligned<real_t, 64>>());
};

auto VelocityCallback = [](const Cell &pos, const shared_ptr<StructuredBlockForest> &SbF, IBlock& block, real_t inflow_velocity)
{
  Cell globalCell;
  CellInterval domain = SbF->getDomainCellBB();
  real_t h_y = domain.yMax() - domain.yMin();
  real_t h_z = domain.zMax() - domain.zMin();
  SbF->transformBlockLocalToGlobalCell(globalCell, block, pos);

  real_t y1 = globalCell[1] - (h_y / 2.0 + 0.5);
  real_t z1 = globalCell[2] - (h_z / 2.0 + 0.5);

  real_t u = (inflow_velocity * 16)/(h_y*h_y*h_z*h_z) * (h_y/2.0 - y1)*(h_y/2 + y1)*(h_z/2 - z1)*(h_z/2 + z1);

  Vector3<real_t> result(u, 0.0, 0.0);
  return result;
};

class TimestepModulusTracker{
private:
   uint_t modulus_;
public:
   TimestepModulusTracker(uint_t initialTimestep) : modulus_(initialTimestep & 1) {};

   void setTimestep(uint_t timestep) { modulus_ = timestep & 1; }

   std::function<void()> advancementFunction() {
      return [this] () {
         this->modulus_ = (this->modulus_ + 1) & 1;
      };
   }

   uint_t modulus() const { return modulus_; }
   bool evenStep() const { return static_cast<bool>(modulus_ ^ 1); }
   bool oddStep() const { return static_cast<bool>(modulus_ & 1); }
};

class AlternatingSweep{
public:
   typedef std::function< void (IBlock *) > SweepFunction;

   AlternatingSweep(SweepFunction evenSweep, SweepFunction oddSweep, std::shared_ptr<TimestepModulusTracker> tracker)
      : tracker_(tracker), sweeps_{ evenSweep, oddSweep } {};

   void operator() (IBlock * block) {
      sweeps_[tracker_->modulus()](block);
   }

private:
   std::shared_ptr<TimestepModulusTracker> tracker_;
   std::vector< SweepFunction > sweeps_;
};

class AlternatingBeforeFunction{
public:
   typedef std::function< void () > BeforeFunction;

   AlternatingBeforeFunction(BeforeFunction evenFunc, BeforeFunction oddFunc, std::shared_ptr<TimestepModulusTracker> &tracker)
      : tracker_(tracker), funcs_{ evenFunc, oddFunc } {};

   void operator() () {
      funcs_[tracker_->modulus()]();
   }

private:
   std::shared_ptr<TimestepModulusTracker> tracker_;
   std::vector< BeforeFunction > funcs_;
};

class Filter {
 public:
   explicit Filter(Vector3<uint_t> numberOfCells) : numberOfCells_(numberOfCells) {}

   void operator()( const IBlock & /*block*/ ){

   }

   bool operator()( const cell_idx_t x, const cell_idx_t y, const cell_idx_t z ) const {
      return x >= -1 && x <= cell_idx_t(numberOfCells_[0]) &&
             y >= -1 && y <= cell_idx_t(numberOfCells_[1]) &&
             z >= -1 && z <= cell_idx_t(numberOfCells_[2]);
   }

 private:
   Vector3<uint_t> numberOfCells_;
};

using FluidFilter_T = Filter;

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

      // Create communication
      blockforest::communication::UniformBufferedScheme< Stencil_T > evenComm(blocks);
      evenComm.addPackInfo(make_shared< PackInfoEven_T >(pdfFieldID));

      blockforest::communication::UniformBufferedScheme< Stencil_T > oddComm(blocks);
      oddComm.addPackInfo(make_shared< PackInfoOdd_T >(pdfFieldID));

      // create and initialize boundary handling
      const FlagUID fluidFlagUID("Fluid");

      auto boundariesConfig = config->getOneBlock("Boundaries");

      std::function<Vector3<real_t>(const Cell &, const shared_ptr<StructuredBlockForest>&, IBlock&)>
         velocity_initialisation = std::bind(VelocityCallback, _1, _2, _3, u_max) ;

      lbm::ChannelFlowCodeGen_UBB ubb(blocks, pdfFieldID, velocity_initialisation);
      lbm::ChannelFlowCodeGen_NoSlip noSlip(blocks, pdfFieldID);
      lbm::ChannelFlowCodeGen_Outflow outflow(blocks, pdfFieldID);

      geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldId, boundariesConfig);
      geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldId, fluidFlagUID);

      ubb.fillFromFlagField< FlagField_T >(blocks, flagFieldId, FlagUID("UBB"), fluidFlagUID);
      noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldId, FlagUID("NoSlip"), fluidFlagUID);
      outflow.fillFromFlagField< FlagField_T >(blocks, flagFieldId, FlagUID("Outflow"), fluidFlagUID);

      // create time loop
      SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

      pystencils::ChannelFlowCodeGen_EvenSweep LBEvenSweep(densityFieldID, pdfFieldID, velFieldID, omega);
      pystencils::ChannelFlowCodeGen_OddSweep LBOddSweep(densityFieldID, pdfFieldID, velFieldID, omega);

      // All the sweeps
      auto tracker = make_shared<TimestepModulusTracker>(0);

      AlternatingSweep LBSweep(LBEvenSweep, LBOddSweep, tracker);
      AlternatingSweep noSlipSweep(noSlip.getEvenSweep(), noSlip.getOddSweep(), tracker);
      AlternatingSweep outflowSweep(outflow.getEvenSweep(), outflow.getOddSweep(), tracker);
      AlternatingSweep ubbSweep(ubb.getEvenSweep(), ubb.getOddSweep(), tracker);
      AlternatingBeforeFunction communication(evenComm, oddComm, tracker);

      // add LBM sweep and communication to time loop
      timeloop.add() << Sweep(noSlipSweep, "noSlip boundary");
      timeloop.add() << Sweep(outflowSweep, "outflow boundary");
      timeloop.add() << Sweep(ubbSweep, "ubb boundary");
      timeloop.add() << BeforeFunction(communication, "communication")
                     << BeforeFunction(tracker->advancementFunction(), "Timestep Advancement")
                     << Sweep(LBSweep, "LB update rule");

      // LBM stability check
      timeloop.addFuncAfterTimeStep(makeSharedFunctor(field::makeStabilityChecker< PdfField_T, FlagField_T >(
                                       config, blocks, pdfFieldID, flagFieldId, fluidFlagUID)),
                                    "LBM stability check");

      // log remaining time
      timeloop.addFuncAfterTimeStep(
         timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
         "remaining time logger");

      // add VTK output to time loop
      // pystencils::ChannelFlowCodeGen_MacroGetter getterSweep(densityFieldID, pdfFieldID, velFieldID);
      // VTK
      uint_t vtkWriteFrequency = parameters.getParameter< uint_t >("vtkWriteFrequency", 0);
      if (vtkWriteFrequency > 0)
      {
         auto vtkOutput     = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_out",
                                                         "simulation_step", false, true, true, false, 0);
         auto velWriter     = make_shared< field::VTKWriter< VelocityField_T > >(velFieldID, "velocity");
         auto densityWriter = make_shared< field::VTKWriter< ScalarField_T > >(densityFieldID, "density");

         FluidFilter_T filter(cellsPerBlock);

         auto QCriterionWriter = make_shared<lbm::QCriterionVTKWriter<VelocityField_T, FluidFilter_T>>(blocks, filter, velFieldID, "QCriterionWriter");

         vtkOutput->addCellDataWriter(velWriter);
         vtkOutput->addCellDataWriter(densityWriter);
         vtkOutput->addCellDataWriter(QCriterionWriter);

         // vtkOutput->addBeforeFunction([&]() {
         //    for (auto& block : *blocks)
         //       getterSweep(&block);
         // });
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
