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
//! \file PSM_Moving_Geometry.cpp
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

#include "mesh_common/MeshOperations.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#   include "gpu/communication/UniformGPUScheme.h"
#   include "lbm_generated/gpu/AddToStorage.h"
#   include "lbm_generated/gpu/GPUPdfField.h"
#   include "lbm_generated/gpu/UniformGeneratedGPUPdfPackInfo.h"
#endif
#include <fstream>
#include <iostream>

#include "lbm/geometry/moving_geometry/PredefinedMovingGeometry.h"


namespace walberla
{
///////////////////////
/// Typedef Aliases ///
///////////////////////

typedef field::GhostLayerField< real_t, 3 > VectorField_T;
typedef field::GhostLayerField< real_t, 1 > FracField_T;

/////////////////////
/// Main Function ///
/////////////////////


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
   auto domainParameters = walberlaEnv.config()->getOneBlock("DomainSetup");
   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");

   const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(10));

   const real_t remainingTimeLoggerFrequency = parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(5.0));
   const uint_t VTKWriteFrequency = parameters.getParameter< uint_t >("VTKwriteFrequency", uint_c(10));
   const uint_t maxSuperSamplingDepth = parameters.getParameter< uint_t >("maxSuperSamplingDepth", uint_c(1));

   const std::string meshFile = domainParameters.getParameter< std::string >("meshFile");

   const uint_t objectResolution = parameters.getParameter< uint_t >("objectResolution", uint_c(20));

   const Vector3<real_t> rotationVector = parameters.getParameter< Vector3<real_t> >("rotationVector", Vector3< real_t >(0.0));
   const Vector3<real_t> objectVelocity = parameters.getParameter< Vector3<real_t> >("objectVelocity", Vector3< real_t >(0.0));


   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   auto mesh = make_shared< mesh::TriangleMesh >();
   mesh::readAndBroadcast(meshFile, *mesh);
   auto meshCenterPoint = mesh::computeCentroid(*mesh);
   WALBERLA_LOG_INFO_ON_ROOT("meshCenterPoint is " << meshCenterPoint)

   auto distanceOctreeMesh = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh));
   auto aabb = computeAABB(*mesh);
   real_t dx = aabb.xSize() / real_t(objectResolution);
   real_t dt = dx*dx;
   WALBERLA_LOG_INFO_ON_ROOT("dx is " << dx)
   WALBERLA_LOG_INFO_ON_ROOT("aabb size is <" << aabb.xSize() << "," << aabb.ySize() << "," << aabb.zSize() << ">")

   real_t analyticVolume = mesh::computeVolume(*mesh);

   auto domainScaling = Vector3<real_t>(3.5, 1.6, 1.6);
   aabb.scale(domainScaling);

   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, Vector3< real_t >(dx), mesh::makeExcludeMeshInterior(distanceOctreeMesh, dx));

   auto cellsPerBlock = Vector3<uint_t>(32);
   auto blocks = bfc.createStructuredBlockForest(cellsPerBlock);
   WALBERLA_LOG_INFO_ON_ROOT("<" << blocks->getNumberOfXCells() << "," << blocks->getNumberOfYCells() << "," << blocks->getNumberOfZCells() << ">" )
   auto domainAABB = blocks->getDomain();

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);


   /////////////////////////
   /// Fields Creation   ///
   /////////////////////////


   const BlockDataID fractionFieldId = field::addToStorage< FracField_T >(blocks, "fractionField", real_t(0.0), field::fzyx, uint_c(1));
   const BlockDataID objectVelocitiesFieldId = field::addToStorage< VectorField_T >(blocks, "particleVelocitiesField", real_c(0.0), field::fzyx, uint_c(1));
   const BlockDataID forceFieldId = field::addToStorage< VectorField_T >(blocks, "forceField", real_c(0.0), field::fzyx, uint_c(1));

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const BlockDataID fractionFieldGPUId = gpu::addGPUFieldToStorage< FracField_T >(blocks, fractionFieldId, "fractionFieldGPU", true);
   const BlockDataID objectVelocitiesFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, objectVelocitiesFieldId, "object velocity field on GPU", true);
   const BlockDataID forceFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, forceFieldId, "force field on GPU", true);
#endif


   /////////////////////////
   /// Rotation Calls    ///
   /////////////////////////

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   auto objectMover = make_shared<PredefinedMovingGeometry<FracField_T, VectorField_T>> (blocks, mesh, fractionFieldGPUId, objectVelocitiesFieldGPUId, forceFieldGPUId,
                                                                                                     distanceOctreeMesh, "geometry", maxSuperSamplingDepth, false, 0.0, dt, domainAABB, objectVelocity * dx / dt, rotationVector / dt, 0);
#else
   auto objectMover = make_shared<PredefinedMovingGeometry<FracField_T, VectorField_T>> (blocks, mesh, fractionFieldId, objectVelocitiesFieldId, forceFieldId,
                                                                                                     distanceOctreeMesh, "geometry", maxSuperSamplingDepth, false, 0.0, dt, domainAABB,
                                                                                                     objectVelocity * dx / dt, rotationVector / dt, 0);
#endif

   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriter(mesh, "meshBase", VTKWriteFrequency);
   const std::function< void() > meshWritingFunc = [&]() {
      objectMover->moveTriangleMesh(timeloop.getCurrentTimeStep(), VTKWriteFrequency);
      meshWriter();
   };

   std::vector<real_t> errorVector;
   const std::function< void() > objectRotatorFunc = [&]() {
      objectMover->resetFractionField();
      (*objectMover)();
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
#endif
      real_t fractionFieldVolume = objectMover->getVolumeFromFractionField();
      fractionFieldVolume *= pow(dx,3.);
      real_t l2error = pow(fractionFieldVolume - analyticVolume,2.) / pow(analyticVolume,2.);
      errorVector.push_back(l2error);
      //WALBERLA_LOG_INFO_ON_ROOT("Mesh Volume is " << analyticVolume <<  ", fraction Field volume is " << fractionFieldVolume << ", L2 error is " << l2error)
   };

   /////////////////
   /// Time Loop ///
   /////////////////


   const auto emptySweep = [](IBlock*) {};
   if( VTKWriteFrequency > 0) {
      timeloop.add() << BeforeFunction(meshWritingFunc, "Meshwriter") <<  Sweep(emptySweep);
   }
   timeloop.add() << BeforeFunction(objectRotatorFunc, "ObjectRotator") <<  Sweep(emptySweep);
   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency), "remaining time logger");



   /////////////////
   /// VTK       ///
   /////////////////

   if (VTKWriteFrequency > 0)
   {
      const std::string path = "vtk_out";
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "fields", VTKWriteFrequency, 0, false, path, "simulation_step", false, true, true, false, 0);

      vtkOutput->addBeforeFunction([&]() {
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         gpu::fieldCpy< FracField_T, gpu::GPUField< real_t > >(blocks, fractionFieldId, fractionFieldGPUId);
#endif
      });

      auto fractionFieldWriter = make_shared< field::VTKWriter< FracField_T > >(fractionFieldId, "FractionField");

      vtkOutput->addCellDataWriter(fractionFieldWriter);

      timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
      vtk::writeDomainDecomposition(blocks, "domain_decomposition", "vtk_out", "write_call", true, true, 0);
   }


   /////////////////
   /// TIMING    ///
   /////////////////

   WcTimingPool timeloopTiming;
   WcTimer simTimer;

   simTimer.start();
   timeloop.run(timeloopTiming, true);
   simTimer.end();

   real_t averageError = std::reduce(errorVector.begin(), errorVector.end()) / real_t(errorVector.size());
   WALBERLA_LOG_INFO_ON_ROOT("Average over error of all time steps is " << averageError)

   double time = simTimer.max();
   WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }

   const auto reducedTimeloopTiming = timeloopTiming.getReduced();
   WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)

   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv) { return walberla::main(argc, argv); }
