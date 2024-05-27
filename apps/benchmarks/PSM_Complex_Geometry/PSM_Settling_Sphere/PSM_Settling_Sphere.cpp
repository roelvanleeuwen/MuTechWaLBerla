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
//! \file PSM_Settling_Sphere.cpp
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

#include "lbm/geometry/moving_geometry/FreeMovingGeometry.h"


#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#   include "gpu/communication/UniformGPUScheme.h"
#   include "lbm_generated/gpu/AddToStorage.h"
#   include "lbm_generated/gpu/GPUPdfField.h"
#   include "lbm_generated/gpu/UniformGeneratedGPUPdfPackInfo.h"
#endif
#include <fstream>
#include <iostream>

#include "PSM_Settling_Sphere_InfoHeader.h"

namespace walberla
{
///////////////////////
/// Typedef Aliases ///
///////////////////////

using StorageSpecification_T = lbm::PSM_Settling_SphereStorageSpecification;
using Stencil_T = lbm::PSM_Settling_SphereStorageSpecification::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;
using PdfField_T = lbm_generated::PdfField< StorageSpecification_T >;
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
using GPUPdfField_T = lbm_generated::GPUPdfField< StorageSpecification_T >;
using gpu::communication::UniformGPUScheme;
#endif

typedef walberla::uint8_t flag_t;
typedef FlagField< flag_t > FlagField_T;
using BoundaryCollection_T = lbm::PSM_Settling_SphereBoundaryCollection< FlagField_T >;
using SweepCollection_T = lbm::PSM_Settling_SphereSweepCollection;
const FlagUID fluidFlagUID("Fluid");
using blockforest::communication::UniformBufferedScheme;


typedef field::GhostLayerField< real_t, 3 > VectorField_T;
typedef field::GhostLayerField< real_t, 1 > FracField_T;



/////////////////////
/// Main Function ///
/////////////////////

auto deviceSyncWrapper = [](std::function< void(IBlock*) > sweep) {
   return [sweep](IBlock* b) {
      sweep(b);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      cudaDeviceSynchronize();
#endif
   };
};






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


   const Vector3< bool > periodicity = domainParameters.getParameter< Vector3< bool > >("periodic", Vector3< bool >(true));
   const std::string meshFile = domainParameters.getParameter< std::string >("meshFile");


   const Vector3<real_t> rotationVector = parameters.getParameter< Vector3<real_t> >("rotationVector", Vector3< real_t >(0.0));
   const Vector3<real_t> objectVelocity = parameters.getParameter< Vector3<real_t> >("objectVelocity", Vector3< real_t >(0.0));

   const uint_t fluidType = parameters.getParameter< uint_t >("fluidType", uint_c(1));


   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   auto mesh = make_shared< mesh::TriangleMesh >();
   mesh::readAndBroadcast(meshFile, *mesh);
   mesh::scale(*mesh, Vector3<real_t> (0.007633691));
   mesh::translate(*mesh, Vector3<real_t> (0.05, 0.05, 0.123));
   auto meshAABB = mesh::computeAABB(*mesh);
   WALBERLA_LOG_INFO_ON_ROOT("MeshAABB is " << meshAABB << " size is <" << meshAABB.xSize() << "," << meshAABB.ySize() << "," << meshAABB.zSize())

   auto distanceOctreeMesh = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh));

   real_t dx = 0.000740741;

   auto domainAABB = AABB(0.0, 0.0, 0.0, 0.1, 0.1, 0.16);
   const Vector3<uint_t> numCells(135, 135, 216);

   auto numBlocks = Vector3<uint_t> (1,1,2);
   auto cellsPerBlock = Vector3<uint_t>(numCells[0] / numBlocks[0], numCells[1] / numBlocks[1], numCells[2] / numBlocks[2]);

   auto blocks = walberla::blockforest::createUniformBlockGrid(domainAABB, numBlocks[0], numBlocks[1], numBlocks[2], cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2], true, periodicity[0], periodicity[1], periodicity[2], false);
   domainAABB = blocks->getDomain();
   WALBERLA_LOG_INFO("DomainAABB is " << domainAABB << " dx is " << dx )
   WALBERLA_LOG_INFO_ON_ROOT("<" << blocks->getNumberOfXCells() << "," << blocks->getNumberOfYCells() << "," << blocks->getNumberOfZCells() << ">" )

   const real_t diameter_SI      = real_t(15e-3);
   const real_t densitySphere_SI = real_t(1120);

   real_t densityFluid_SI, dynamicViscosityFluid_SI;
   real_t expectedSettlingVelocity_SI;
   switch (fluidType)
   {
   case 1:
      // Re_p around 1.5
      densityFluid_SI             = real_t(970);
      dynamicViscosityFluid_SI    = real_t(373e-3);
      expectedSettlingVelocity_SI = real_t(0.035986);
      break;
   case 2:
      // Re_p around 4.1
      densityFluid_SI             = real_t(965);
      dynamicViscosityFluid_SI    = real_t(212e-3);
      expectedSettlingVelocity_SI = real_t(0.05718);
      break;
   case 3:
      // Re_p around 11.6
      densityFluid_SI             = real_t(962);
      dynamicViscosityFluid_SI    = real_t(113e-3);
      expectedSettlingVelocity_SI = real_t(0.087269);
      break;
   case 4:
      // Re_p around 31.9
      densityFluid_SI             = real_t(960);
      dynamicViscosityFluid_SI    = real_t(58e-3);
      expectedSettlingVelocity_SI = real_t(0.12224);
      break;
   default:
      WALBERLA_ABORT("Only four different fluids are supported! Choose type between 1 and 4.");
   }
   const real_t kinematicViscosityFluid_SI = dynamicViscosityFluid_SI / densityFluid_SI;

   const real_t expectedSettlingVelocity = real_t(0.01);
   const real_t dt                    = expectedSettlingVelocity / expectedSettlingVelocity_SI * dx;

   const real_t viscosity      = kinematicViscosityFluid_SI * dt / (dx * dx);
   const real_t omega = lbm::collision_model::omegaFromViscosity(viscosity);

   const real_t gravitationalAcceleration_SI = real_t(9.81);
   const real_t sphereVolume = mesh::computeVolume(*mesh);
   Vector3< real_t > gravitationalForce(real_t(0), real_t(0),
                                        -(densitySphere_SI - densityFluid_SI) * gravitationalAcceleration_SI * sphereVolume);

   WALBERLA_LOG_INFO_ON_ROOT("Simulation Parameter: \n"
                             << "Domain Decomposition " << numBlocks << " \n" //<< "<" << setupForest->getXSize() << "," << setupForest->getYSize() << "," << setupForest->getZSize() << "> = " << setupForest->getXSize() * setupForest->getYSize() * setupForest->getZSize()  << " root Blocks \n"
                             << "Cells per Block " << numCells << " \n"
                             << "Number of cells "  << numCells << " \n"
                             << "Timesteps " << timesteps << "\n"
                             << "Omega " << omega << "\n"
                             << "rotationVector " << rotationVector << "\n"
                             << "objectVelocity " << objectVelocity << "\n"
                             << "dx " << dx << "\n"
                             << "dt " << dt << "\n"
   )


   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);


   /////////////////////////
   /// Fields Creation   ///
   /////////////////////////


   const StorageSpecification_T StorageSpec = StorageSpecification_T();
   const BlockDataID pdfFieldId  = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, uint_c(1), field::fzyx);
   const BlockDataID velocityFieldId = field::addToStorage< VectorField_T >(blocks, "velocity", real_t(0.0), field::fzyx, uint_c(1));
   const BlockDataID densityFieldId = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, uint_c(1));
   const BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flagField");

   const BlockDataID fractionFieldId = field::addToStorage< FracField_T >(blocks, "fractionField", real_t(0.0), field::fzyx, uint_c(1));
   const BlockDataID objectVelocitiesFieldId = field::addToStorage< VectorField_T >(blocks, "particleVelocitiesField", real_c(0.0), field::fzyx, uint_c(1));
   const BlockDataID forceFieldId = field::addToStorage< VectorField_T >(blocks, "forceField", real_c(0.0), field::fzyx, uint_c(1));

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const BlockDataID pdfFieldGPUId = lbm_generated::addGPUPdfFieldToStorage< PdfField_T >(blocks, pdfFieldId, StorageSpec, "pdf field on GPU", true);
   const BlockDataID velocityFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, velocityFieldId, "velocity field on GPU", true);
   const BlockDataID densityFieldGPUId = gpu::addGPUFieldToStorage< ScalarField_T >(blocks, densityFieldId, "density field on GPU", true);

   const BlockDataID fractionFieldGPUId = gpu::addGPUFieldToStorage< FracField_T >(blocks, fractionFieldId, "fractionFieldGPU", true);
   const BlockDataID objectVelocitiesFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, objectVelocitiesFieldId, "object velocity field on GPU", true);
   const BlockDataID forceFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, forceFieldId, "force field on GPU", true);
#endif



   /////////////////////////
   /// Rotation Calls    ///
   /////////////////////////

   WALBERLA_LOG_INFO_ON_ROOT("Setting up objectMover")
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   auto objectMover = make_shared<FreeMovingGeometry<FracField_T, VectorField_T>> (blocks, mesh, fractionFieldGPUId,
                                                                                      objectVelocitiesFieldGPUId, forceFieldGPUId,
                                                                                      distanceOctreeMesh, "geometry",
                                                                                      maxSuperSamplingDepth, true,
                                                                                      omega, dt, domainAABB, objectVelocity,
                                                                                      rotationVector, densityFluid_SI, densitySphere_SI,
                                                                                      gravitationalForce);
#else
   auto objectMover = make_shared<FreeMovingGeometry<FracField_T, VectorField_T>> (blocks, mesh, fractionFieldId,
                                                                                      objectVelocitiesFieldId, forceFieldId,
                                                                                      distanceOctreeMesh, "geometry",
                                                                                      maxSuperSamplingDepth, true,
                                                                                      omega, dt, domainAABB, objectVelocity,
                                                                                      rotationVector, densityFluid_SI, densitySphere_SI,
                                                                                      gravitationalForce);
#endif



   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriter(mesh, "meshBase", VTKWriteFrequency);
   const std::function< void() > meshWritingFunc = [&]() {
      objectMover->moveTriangleMesh(timeloop.getCurrentTimeStep(), VTKWriteFrequency);
      meshWriter();
   };


   std::string outputFile = "output_fluid_" + std::to_string(fluidType) + ".txt";
   std::filesystem::remove(outputFile);
   real_t maxSettlingVel = 0;
   const std::function< void() > objectRotatorFunc = [&]() {
      objectMover->resetFractionField();
      (*objectMover)();
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
#endif
      WALBERLA_ROOT_SECTION(){
         Vector3<real_t> velocity = objectMover->getLinearVelocity();
         if(abs(velocity[2]) > maxSettlingVel)
            maxSettlingVel = abs(velocity[2]);
         std::ofstream myFile(outputFile, std::ios::app);
         myFile << real_c(timeloop.getCurrentTimeStep())*dt << " " << velocity[0] << " " << velocity[1] << " " << velocity[2] << "\n";
         myFile.close();
      }
   };


   for (auto& block : *blocks)
   {
      FlagField_T *flagField = block.getData<FlagField_T>(flagFieldId);
      flagField->registerFlag(fluidFlagUID);
      flagField->registerFlag(FlagUID("NoSlip"));
   }

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");
   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldId, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldId, fluidFlagUID);


   /////////////
   /// Sweep ///
   /////////////
   WALBERLA_LOG_INFO_ON_ROOT("Setting up Sweeps")

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   SweepCollection_T sweepCollection(blocks, forceFieldGPUId, fractionFieldGPUId, objectVelocitiesFieldGPUId, pdfFieldGPUId, densityFieldGPUId, velocityFieldGPUId, omega);
   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldGPUId, fluidFlagUID);
   pystencils::PSM_Conditional_Sweep psmConditionalSweep( forceFieldGPUId, fractionFieldGPUId, objectVelocitiesFieldGPUId, pdfFieldGPUId, omega );
#else
   SweepCollection_T sweepCollection(blocks, forceFieldId, fractionFieldId, objectVelocitiesFieldId, pdfFieldId, densityFieldId, velocityFieldId, omega);
   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldId, fluidFlagUID);
   pystencils::PSM_Conditional_Sweep psmConditionalSweep( forceFieldId, fractionFieldId, objectVelocitiesFieldId, pdfFieldId, omega );
#endif

   WALBERLA_LOG_INFO_ON_ROOT("Initialize Velocity and PDF fields")
   Vector3<real_t> initialVel(0, 0,0);
   for (auto& block : *blocks)
   {
      auto velField = block.getData<VectorField_T>(velocityFieldId);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpu::fieldCpy< FracField_T, gpu::GPUField< real_t > >(blocks, fractionFieldId, fractionFieldGPUId);
      gpu::fieldCpy< VectorField_T, gpu::GPUField< real_t > >(blocks, objectVelocitiesFieldId, objectVelocitiesFieldGPUId);
#endif
      auto fracField = block.getData<FracField_T>(fractionFieldId);
      auto objVelField = block.getData<VectorField_T>(objectVelocitiesFieldId);




      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(velField,
                                                       real_t fracValue = fracField->get(x,y,z,0);
                                                       for (int d = 0; d < 3; ++d)
                                                         velField->get(x,y,z,d) = (1.0 - fracValue) * initialVel[d] + fracValue * objVelField->get(x,y,z,d);
                                                       )
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpu::GPUField<real_t> * dst = block.getData<gpu::GPUField<real_t>>( velocityFieldGPUId );
      const VectorField_T * src = block.getData<VectorField_T>( velocityFieldId );
      gpu::fieldCpy( *dst, *src );
#endif
      sweepCollection.initialise(&block, 1);
   }
   WALBERLA_LOG_INFO_ON_ROOT("Finish initialize Velocity and PDF fields")


   /////////////////////
   /// Communication ///
   ////////////////////

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const bool sendDirectlyFromGPU = false;
   UniformGPUScheme< Stencil_T > communication(blocks, sendDirectlyFromGPU);
   auto packInfo = std::make_shared<lbm_generated::UniformGeneratedGPUPdfPackInfo< GPUPdfField_T >>(pdfFieldGPUId);
   communication.addPackInfo(packInfo);
#else
   UniformBufferedScheme< Stencil_T > communication(blocks);
   auto packInfo = std::make_shared<lbm_generated::UniformGeneratedPdfPackInfo< PdfField_T >>(pdfFieldId);
   communication.addPackInfo(packInfo);
#endif

   /////////////////
   /// Time Loop ///
   /////////////////

   WALBERLA_LOG_INFO_ON_ROOT("Starting Timeloop")

   const auto emptySweep = [](IBlock*) {};
   if( VTKWriteFrequency > 0) {
      timeloop.add() << BeforeFunction(meshWritingFunc, "Meshwriter") <<  Sweep(emptySweep);
   }
   timeloop.add() << BeforeFunction(objectRotatorFunc, "ObjectRotator") <<  Sweep(emptySweep);

   timeloop.add() << BeforeFunction(communication.getCommunicateFunctor(), "Communication")
                  << Sweep(deviceSyncWrapper(boundaryCollection.getSweep(BoundaryCollection_T::ALL)), "Boundary Conditions");
   timeloop.add() << Sweep(deviceSyncWrapper(psmConditionalSweep), "PSMConditionalSweep");

   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency), "remaining time logger");


   /////////////////
   /// VTK       ///
   /////////////////

   if (VTKWriteFrequency > 0)
   {
      const std::string path = "vtk_out";
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "fields", VTKWriteFrequency, 0, false, path, "simulation_step", false, true, true, false, 0);

      vtkOutput->addBeforeFunction([&]() {
         for (auto& block : *blocks)
            sweepCollection.calculateMacroscopicParameters(&block);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         gpu::fieldCpy< VectorField_T, gpu::GPUField< real_t > >(blocks, velocityFieldId, velocityFieldGPUId);
         gpu::fieldCpy< FracField_T, gpu::GPUField< real_t > >(blocks, fractionFieldId, fractionFieldGPUId);
         gpu::fieldCpy< VectorField_T, gpu::GPUField< real_t > >(blocks, objectVelocitiesFieldId, objectVelocitiesFieldGPUId);
         gpu::fieldCpy< VectorField_T, gpu::GPUField< real_t > >(blocks, forceFieldId, forceFieldGPUId);

#endif
      });

      auto velWriter = make_shared< field::VTKWriter< VectorField_T > >(velocityFieldId, "Velocity");
      auto fractionFieldWriter = make_shared< field::VTKWriter< FracField_T > >(fractionFieldId, "FractionField");
      auto objVeldWriter = make_shared< field::VTKWriter< VectorField_T > >(objectVelocitiesFieldId, "objectVelocity");
      auto forceFieldWriter = make_shared< field::VTKWriter< VectorField_T > >(forceFieldId, "ForceField");

      vtkOutput->addCellDataWriter(velWriter);
      vtkOutput->addCellDataWriter(fractionFieldWriter);
      vtkOutput->addCellDataWriter(objVeldWriter);
      vtkOutput->addCellDataWriter(forceFieldWriter);

      timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
      vtk::writeDomainDecomposition(blocks, "domain_decomposition", "vtk_out", "write_call", true, true, 0);
   }


   /////////////////
   /// TIMING    ///
   /////////////////

   lbm_generated::PerformanceEvaluation<FlagField_T> const performance(blocks, flagFieldId, fluidFlagUID);
   WcTimingPool timeloopTiming;
   WcTimer simTimer;

   simTimer.start();
   timeloop.run(timeloopTiming, true);
   simTimer.end();


   double time = simTimer.max();
   WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
   performance.logResultOnRoot(timesteps, time);

   const auto reducedTimeloopTiming = timeloopTiming.getReduced();
   WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)

   real_t velError = (maxSettlingVel - expectedSettlingVelocity_SI) / expectedSettlingVelocity_SI;
   WALBERLA_LOG_INFO_ON_ROOT("Max settling velocity is " << maxSettlingVel << " , expected is " << expectedSettlingVelocity_SI << ", error is " << velError)


   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv) { return walberla::main(argc, argv); }
