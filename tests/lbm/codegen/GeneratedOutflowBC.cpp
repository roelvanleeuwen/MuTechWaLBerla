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
//! \file GeneratedOutflowBC.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#include "blockforest/all.h"

#include "core/all.h"

#include "domain_decomposition/all.h"

#include "field/all.h"

#include "geometry/all.h"

#include "gui/all.h"

#include "lbm/vtk/VTKOutput.h"

#include "timeloop/all.h"

#include "GeneratedOutflowBC_Dynamic_UBB.h"
#include "GeneratedOutflowBC_InfoHeader.h"
#include "GeneratedOutflowBC_MacroSetter.h"
#include "GeneratedOutflowBC_NoSlip.h"
#include "GeneratedOutflowBC_Outflow.h"
#include "GeneratedOutflowBC_PackInfo.h"
#include "GeneratedOutflowBC_Static_UBB.h"
#include "GeneratedOutflowBC_Sweep.h"

using namespace walberla;
using namespace std::placeholders;

using PackInfo_T  = lbm::GeneratedOutflowBC_PackInfo;
using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

auto pdfFieldAdder = [](IBlock* const block, StructuredBlockStorage* const storage) {
   return new PdfField_T(storage->getNumberOfXCells(*block), storage->getNumberOfYCells(*block),
                         storage->getNumberOfZCells(*block), uint_t(1), field::fzyx,
                         make_shared< field::AllocateAligned< real_t, 64 > >());
};

auto VelocityCallback = [](const Cell& pos, const shared_ptr< StructuredBlockForest >& SbF, IBlock& block,
                           real_t inflow_velocity) {
   Cell globalCell;
   CellInterval domain = SbF->getDomainCellBB();
   real_t h_y          = domain.yMax() - domain.yMin();
   SbF->transformBlockLocalToGlobalCell(globalCell, block, pos);

   real_t u = inflow_velocity * (globalCell[1] / h_y);

   Vector3< real_t > result(u, 0.0, 0.0);
   return result;
};

int main(int argc, char** argv)
{
   walberla::Environment walberlaEnv(argc, argv);

   auto blocks = blockforest::createUniformBlockGridFromConfig(walberlaEnv.config());

   // read parameters
   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");

   const real_t omega     = parameters.getParameter< real_t >("omega", real_c(1.4));
   const real_t u_max     = parameters.getParameter< real_t >("u_max", real_t(0.05));
   const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(10));

   const double remainingTimeLoggerFrequency =
      parameters.getParameter< double >("remainingTimeLoggerFrequency", 3.0); // in seconds

   // create fields
   BlockDataID pdfFieldID     = blocks->addStructuredBlockData< PdfField_T >(pdfFieldAdder, "PDFs");
   BlockDataID velFieldID     = field::addToStorage< VelocityField_T >(blocks, "velocity", real_t(0), field::fzyx);
   BlockDataID densityFieldID = field::addToStorage< ScalarField_T >(blocks, "density", real_t(0), field::fzyx);

   BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");

   pystencils::GeneratedOutflowBC_MacroSetter setterSweep(pdfFieldID, velFieldID);
   for (auto& block : *blocks)
      setterSweep(&block);

   // create and initialize boundary handling
   const FlagUID fluidFlagUID("Fluid");

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");

   std::function< Vector3< real_t >(const Cell&, const shared_ptr< StructuredBlockForest >&, IBlock&) >
      velocity_initialisation = std::bind(VelocityCallback, _1, _2, _3, u_max);

   lbm::GeneratedOutflowBC_Dynamic_UBB ubb_dynamic(blocks, pdfFieldID, velocity_initialisation);
   lbm::GeneratedOutflowBC_Static_UBB ubb_static(blocks, pdfFieldID, u_max);
   lbm::GeneratedOutflowBC_NoSlip noSlip(blocks, pdfFieldID);
   lbm::GeneratedOutflowBC_Outflow outflow(blocks, pdfFieldID);

   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldId, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldId, fluidFlagUID);

   ubb_dynamic.fillFromFlagField< FlagField_T >(blocks, flagFieldId, FlagUID("UBB_Inflow"), fluidFlagUID);
   ubb_static.fillFromFlagField< FlagField_T >(blocks, flagFieldId, FlagUID("UBB_Wall"), fluidFlagUID);
   noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldId, FlagUID("NoSlip"), fluidFlagUID);
   outflow.fillFromFlagField< FlagField_T >(blocks, flagFieldId, FlagUID("Outflow"), fluidFlagUID);

   // create time loop
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   // create communication for PdfField
   blockforest::communication::UniformBufferedScheme< Stencil_T > communication(blocks);
   communication.addPackInfo(make_shared< PackInfo_T >(pdfFieldID));

   pystencils::GeneratedOutflowBC_Sweep UpdateSweep(densityFieldID, pdfFieldID, velFieldID, omega);

   // add LBM sweep and communication to time loop
   timeloop.add() << BeforeFunction(communication, "communication") << Sweep(noSlip, "noSlip boundary");
   timeloop.add() << Sweep(ubb_dynamic, "ubb inflow");
   timeloop.add() << Sweep(ubb_static, "ubb wall");
   timeloop.add() << Sweep(outflow, "outflow boundary");
   timeloop.add() << Sweep(UpdateSweep, "LB stream & collide");

   // log remaining time
   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
                                 "remaining time logger");

   // VTK Writer
   uint_t vtkWriteFrequency = parameters.getParameter< uint_t >("vtkWriteFrequency", 0);
   if (vtkWriteFrequency > 0)
   {
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "GeneratedOutflowBC_VTK", vtkWriteFrequency, 0, false,
                                                      "vtk_out", "simulation_step", false, true, true, false, 0);

      auto velWriter     = make_shared< field::VTKWriter< VelocityField_T > >(velFieldID, "velocity");
      auto densityWriter = make_shared< field::VTKWriter< ScalarField_T > >(densityFieldID, "density");

      vtkOutput->addCellDataWriter(velWriter);
      vtkOutput->addCellDataWriter(densityWriter);

      timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   }

   timeloop.run();
   WALBERLA_LOG_INFO_ON_ROOT("Simulation finished")

   return EXIT_SUCCESS;
}
