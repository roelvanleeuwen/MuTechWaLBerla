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
//! \file 01_BasicLBM.cpp
//! \author Frederik Hennig <frederik.hennig@fau.de>
//
//======================================================================================================================

#include "blockforest/all.h"

#include "core/all.h"

#include "field/all.h"

#include "geometry/all.h"

#include "timeloop/all.h"

#include "vtk/all.h"

#include "lbm_generated/boundary/D3Q19BoundaryCollection.h"
#include "lbm_generated/field/AddToStorage.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/communication/UniformGeneratedPdfPackInfo.h"
#include "lbm_generated/storage_specification/D3Q19StorageSpecification.h"
#include "lbm_generated/sweep_collection/D3Q19SRT.h"

namespace walberla
{

using namespace lbm_generated;

using LSS_T      = lbm::D3Q19StorageSpecification;
using PdfField_T = PdfField< LSS_T >;

using Stencil    = LSS_T::Stencil;
using CommunicationStencil = LSS_T::CommunicationStencil;

using VectorField_T = field::GhostLayerField< real_t, Stencil::D >;
using ScalarField_T = field::GhostLayerField< real_t, 1 >;
using flag_t        = uint8_t;
using FlagField_T   = field::FlagField< flag_t >;

using SweepCollection_T    = lbm::D3Q19SRT;
using BoundaryCollection_T = lbm::D3Q19BoundaryCollection< FlagField_T >;

using PackInfo_T = UniformGeneratedPdfPackInfo< PdfField_T >;

void main(int argc, char** argv)
{
   walberla::Environment env(argc, argv);

   auto config = env.config();

   auto blocks = blockforest::createUniformBlockGridFromConfig(config);

   // LBM Parameters
   auto parameters        = config->getOneBlock("Parameters");
   const real_t omega     = parameters.getParameter< real_t >("omega", real_c(1.4));
   const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(50));
   const Vector3< real_t > initialVelocity =
      parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >(0.05, 0.0, 0.0));

   auto velInit = [&](VectorField_T* uField, IBlock* const /*block*/) {
      for (auto cIt = uField->beginXYZ(); cIt != uField->end(); ++cIt)
      {
         for (uint_t i = 0; i < Stencil::D; ++i)
         {
            cIt.getF(i) = initialVelocity[i];
         }
      }
   };

   // Field Setup
   const LSS_T storageSpec      = LSS_T();
   const BlockDataID pdfFieldID = addPdfFieldToStorage< LSS_T >(blocks, "pdfs", storageSpec, uint_c(1), field::fzyx);
   const BlockDataID densityFieldId = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx);
   const BlockDataID velFieldId =
      field::addToStorage< VectorField_T >(blocks, "vel", real_c(0.0), field::fzyx, 1, false, velInit);
   const BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "Boundary Flag Field");

   // Boundary setup
   const FlagUID fluidFlagUID("Fluid");
   auto boundariesConfig = config->getBlock("Boundaries");
   if (boundariesConfig)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Setting boundary conditions")
      geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldID, boundariesConfig);
   }
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldID, fluidFlagUID);

   // Simulation Method Setup
   SweepCollection_T lbmSweeps(blocks, pdfFieldID, densityFieldId, velFieldId, omega);
   BoundaryCollection_T boundaryHandling(blocks, flagFieldID, pdfFieldID, fluidFlagUID, 1.0, initialVelocity[0],
                                         initialVelocity[1], initialVelocity[2]);

   // MPI Communication Setup
   auto packInfo = std::make_shared< PackInfo_T >(pdfFieldID);
   blockforest::communication::UniformBufferedScheme< CommunicationStencil > commScheme(blocks);
   commScheme.addPackInfo(packInfo);

   // VTK Output
   const uint_t vtkWriteFrequency = parameters.getParameter< uint_t >("vtkWriteFrequency", 10);

   auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_out",
                                                   "simulation_step", false, true, true, false, 0);
   auto velWriter = make_shared< field::VTKWriter< VectorField_T > >(velFieldId, "vel");
   vtkOutput->addCellDataWriter(velWriter);

   vtkOutput->addBeforeFunction([&]() {
      for (auto& block : *blocks){
         lbmSweeps.calculateMacroscopicParameters(&block);}
   });

   vtkOutput->addCellExclusionFilter(field::FlagFieldCellFilter< FlagField_T >(flagFieldID, FlagUID("NoSlip")));

   // Output meta info
   auto flagFieldOutput = field::createVTKOutput< FlagField_T >(flagFieldID, *blocks, "flagField");
   flagFieldOutput();

   // Time Loop
   SweepTimeloop loop(blocks->getBlockStorage(), timesteps);

   loop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   loop.add() << BeforeFunction(commScheme, "Communication")
              << Sweep(boundaryHandling.getSweep(), "Boundary Handling");
   loop.add() << Sweep(lbmSweeps.streamCollide(), "LBM Stream/Collide");


   // Time Logging
   auto remainingTimeLoggerFrequency =
      parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(-1.0)); // in seconds
   if (remainingTimeLoggerFrequency > 0)
   {
      auto logger = timing::RemainingTimeLogger(loop.getNrOfTimeSteps(), remainingTimeLoggerFrequency);
      loop.addFuncAfterTimeStep(logger, "remaining time logger");
   }

   // Initialization
   for(auto& block: *blocks){
      lbmSweeps.initialise(&block);
   }

   // Simulate
   loop.run();

   // The end.
}
} // namespace walberla

int main(int argc, char** argv)
{
   walberla::main(argc, argv);
   return EXIT_SUCCESS;
}
