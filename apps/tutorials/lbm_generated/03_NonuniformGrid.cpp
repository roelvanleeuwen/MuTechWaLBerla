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
//! \file 03_NonuniformGridApp.cpp
//! \author Frederik Hennig <frederik.hennig@fau.de>
//
//======================================================================================================================

#include "blockforest/all.h"

#include "core/all.h"

#include "field/all.h"

#include "geometry/all.h"

#include "timeloop/all.h"

#include "vtk/all.h"

#include "GeneratedLbmBoundaryCollection.h"
#include "GeneratedLbmStorageSpecification.h"
#include "GeneratedLbmSweepCollection.h"
#include "lbm_generated/communication/NonuniformGeneratedPdfPackInfo.h"
#include "lbm_generated/field/AddToStorage.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/refinement/BasicRecursiveTimeStep.h"

namespace walberla
{

using namespace lbm_generated;

using LSS_T      = lbm::GeneratedLbmStorageSpecification;
using PdfField_T = PdfField< LSS_T >;

using Stencil              = LSS_T::Stencil;
using CommunicationStencil = LSS_T::CommunicationStencil;

using VectorField_T = field::GhostLayerField< real_t, Stencil::D >;
using ScalarField_T = field::GhostLayerField< real_t, 1 >;
using flag_t        = uint8_t;
using FlagField_T   = field::FlagField< flag_t >;

using SweepCollection_T    = lbm::GeneratedLbmSweepCollection;
using BoundaryCollection_T = lbm::GeneratedLbmBoundaryCollection< FlagField_T >;

using blockforest::communication::NonUniformBufferedScheme;

void main(int argc, char** argv)
{
   Environment env(argc, argv);
   mpi::MPIManager::instance()->useWorldComm();

   auto config      = env.config();
   auto domainSetup = config->getOneBlock("DomainSetup");
   const std::string blockForestFilestem =
      domainSetup.getParameter< std::string >("blockForestFilestem", "blockforest");
   Vector3< uint_t > cellsPerBlock = domainSetup.getParameter< Vector3< uint_t > >("cellsPerBlock");

   // Load structured block forest from file
   std::ostringstream oss;
   oss << blockForestFilestem << ".bfs";
   const std::string setupBlockForestFilepath = oss.str();

   WALBERLA_LOG_INFO_ON_ROOT("Creating structured block forest...")
   auto bfs    = std::make_shared< BlockForest >(uint_c(MPIManager::instance()->worldRank()),
                                              setupBlockForestFilepath.c_str(), false);
   auto blocks = std::make_shared< StructuredBlockForest >(bfs, cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2]);
   blocks->createCellBoundingBoxes();

   auto parameters        = config->getOneBlock("Parameters");
   const real_t omega     = parameters.getParameter< real_t >("omega", real_c(1.4));
   const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(50));

   const Vector3< real_t > initialVelocity =
      parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >(0.05, 0.0, 0.0));
   const Vector3< real_t > ubbVelocity =
      parameters.getParameter< Vector3< real_t > >("ubbVelocity", Vector3< real_t >(0.05, 0.0, 0.0));

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
   const LSS_T storageSpec = LSS_T();
   const BlockDataID pdfFieldID =
      lbm_generated::addPdfFieldToStorage(blocks, "pdfs", storageSpec, uint_c(2), field::fzyx);
   const BlockDataID velFieldId =
      field::addToStorage< VectorField_T >(blocks, "vel", real_c(0.0), field::fzyx, uint_c(2));
   const BlockDataID densityFieldId =
      field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, uint_c(2));
   const BlockDataID flagFieldID =
      field::addFlagFieldToStorage< FlagField_T >(blocks, "Boundary Flag Field", uint_c(3));

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
   BoundaryCollection_T boundaryHandling(blocks, flagFieldID, pdfFieldID, fluidFlagUID, ubbVelocity[0], ubbVelocity[1],
                                         ubbVelocity[2]);

   // Communication
   auto communication = std::make_shared< NonUniformBufferedScheme< CommunicationStencil > >(blocks);
   auto packInfo      = lbm_generated::setupNonuniformPdfCommunication< PdfField_T >(blocks, pdfFieldID);
   communication->addPackInfo(packInfo);

   // VTK Output
   const uint_t vtkWriteFrequency = parameters.getParameter< uint_t >("vtkWriteFrequency", 10);

   auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_out",
                                                   "simulation_step", false, true, true, false, 0);
   auto velWriter = make_shared< field::VTKWriter< VectorField_T > >(velFieldId, "vel");
   vtkOutput->addCellDataWriter(velWriter);

   vtkOutput->addBeforeFunction([&]() {
      for (auto& block : *blocks)
      {
         lbmSweeps.calculateMacroscopicParameters(&block);
      }
   });

   vtkOutput->addCellExclusionFilter(field::FlagFieldCellFilter< FlagField_T >(flagFieldID, FlagUID("NoSlip")));
   vtkOutput->addCellExclusionFilter(field::FlagFieldCellFilter< FlagField_T >(flagFieldID, FlagUID("UBB")));
   vtkOutput->addCellExclusionFilter(field::FlagFieldCellFilter< FlagField_T >(flagFieldID, FlagUID("Outflow")));

   // Output meta info
   auto flagFieldOutput = field::createVTKOutput< FlagField_T >(flagFieldID, *blocks, "flagField");
   flagFieldOutput();

   // Time Loop
   SweepTimeloop loop(blocks->getBlockStorage(), timesteps);

   lbm_generated::BasicRecursiveTimeStep< PdfField_T, SweepCollection_T, BoundaryCollection_T > LBMMeshRefinement(
      blocks, pdfFieldID, lbmSweeps, boundaryHandling, communication, packInfo);
   LBMMeshRefinement.addRefinementToTimeLoop(loop);

   loop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput));

   // Time Logging
   auto remainingTimeLoggerFrequency =
      parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(-1.0)); // in seconds
   if (remainingTimeLoggerFrequency > 0)
   {
      auto logger = timing::RemainingTimeLogger(loop.getNrOfTimeSteps(), remainingTimeLoggerFrequency);
      loop.addFuncAfterTimeStep(logger, "remaining time logger");
   }

   // Initialization
   for (auto& block : *blocks)
   {
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
