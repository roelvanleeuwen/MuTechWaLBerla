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


#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#   include "gpu/communication/UniformGPUScheme.h"
#   include "lbm_generated/gpu/AddToStorage.h"
#   include "lbm_generated/gpu/GPUPdfField.h"
#   include "lbm_generated/gpu/UniformGeneratedGPUPdfPackInfo.h"
#endif
#include "../PSM_Complex_Geometry/MovingGeometry.h"
#include "PSM_Moving_Geometry_InfoHeader.h"

namespace walberla
{
///////////////////////
/// Typedef Aliases ///
///////////////////////

using StorageSpecification_T = lbm::PSM_Moving_GeometryStorageSpecification;
using Stencil_T = lbm::PSM_Moving_GeometryStorageSpecification::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;
using PdfField_T = lbm_generated::PdfField< StorageSpecification_T >;
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
using GPUPdfField_T = lbm_generated::GPUPdfField< StorageSpecification_T >;
using gpu::communication::UniformGPUScheme;
#endif

typedef walberla::uint8_t flag_t;
typedef FlagField< flag_t > FlagField_T;
using BoundaryCollection_T = lbm::PSM_Moving_GeometryBoundaryCollection< FlagField_T >;
using SweepCollection_T = lbm::PSM_Moving_GeometrySweepCollection;
const FlagUID fluidFlagUID("Fluid");
using blockforest::communication::UniformBufferedScheme;

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


class GeometryMovementFunction {
 public:
   GeometryMovementFunction(AABB domainAABB, AABB meshAABB, Vector3< mesh::TriangleMesh::Scalar > rotationAxis, real_t rotationAngle, Vector3<real_t> translationVector)
      : domainAABB_(domainAABB), meshAABB_(meshAABB), rotationAxis_(rotationAxis), rotationAngle_(rotationAngle)  {};

   GeometryMovementStruct operator() (uint_t timestep) {
      GeometryMovementStruct geoMovement;
      geoMovement.rotationAxis = rotationAxis_;
      geoMovement.rotationAngle =  0.25 * math::pi * cos(real_t(timestep) / (1000.0 * math::half_pi)) ;
      geoMovement.translationVector = Vector3<real_t> (0,//15 * cos(real_t(timestep) / (1000.0 * math::half_pi)),
                                                       0,
                                                        7 * sin(real_t(timestep) / (1000.0 * math::half_pi) ));
      geoMovement.movementBoundingBox = AABB(domainAABB_.xMin(), domainAABB_.yMin(), domainAABB_.zMin(),
                                               domainAABB_.xMax(), domainAABB_.yMax(), domainAABB_.zMax());
      geoMovement.timeDependentMovement = true;

      return geoMovement;
   }

 private:
   AABB domainAABB_;
   AABB meshAABB_;
   Vector3< mesh::TriangleMesh::Scalar > rotationAxis_;
   real_t rotationAngle_;
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
   const Vector3< real_t > initialVelocity = parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >(0.0));
   const real_t dx = parameters.getParameter<real_t>("dx");
   const real_t omega = parameters.getParameter<real_t>("omega");

   const real_t remainingTimeLoggerFrequency = parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(5.0));
   const uint_t VTKWriteFrequency = parameters.getParameter< uint_t >("VTKwriteFrequency", uint_c(10));
   const uint_t maxSuperSamplingDepth = parameters.getParameter< uint_t >("maxSuperSamplingDepth", uint_c(1));


   const Vector3< real_t > domainScaling = domainParameters.getParameter< Vector3< real_t > >("domainScaling", Vector3< real_t >(1.0));
   const Vector3< real_t > domainTransforming = domainParameters.getParameter< Vector3< real_t > >("domainTransforming", Vector3< real_t >(0.0));
   const Vector3< bool > periodicity = domainParameters.getParameter< Vector3< bool > >("periodic", Vector3< bool >(true));
   const Vector3< uint_t > cellsPerBlock = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
   const std::string meshFile = domainParameters.getParameter< std::string >("meshFile");


   const Vector3< int > rotationAxis = domainParameters.getParameter< Vector3< int > >("rotationAxis", Vector3< int >(1,0,0));
   const real_t rotationPerTimestep = domainParameters.getParameter< real_t >("rotationPerTimestep");
   const Vector3<real_t> translationPerTimestep = domainParameters.getParameter< Vector3<real_t> >("translationPerTimestep");

   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   auto mesh = make_shared< mesh::TriangleMesh >();
   mesh::readAndBroadcast(meshFile, *mesh);
   auto meshCenter = computeCentroid(*mesh);
   mesh::rotate(*mesh, Vector3<mesh::TriangleMesh::Scalar> (0,1,0), - 0.5 * math::pi, Vector3<mesh::TriangleMesh::Scalar> (meshCenter[0], meshCenter[1], meshCenter[2]));
   mesh::rotate(*mesh, Vector3<mesh::TriangleMesh::Scalar> (1,0,0), + 0.5 * math::pi, Vector3<mesh::TriangleMesh::Scalar> (meshCenter[0], meshCenter[1], meshCenter[2]));


   auto distanceOctreeMesh = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh));



   auto meshAABB = computeAABB(*mesh);
   auto aabb = meshAABB;
   aabb.setCenter(aabb.center() - Vector3< real_t >(domainTransforming[0] * aabb.xSize(), domainTransforming[1] * aabb.ySize(), domainTransforming[2] * aabb.zSize()));
   aabb.scale(domainScaling);

   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, Vector3<real_t>(dx));//, mesh::makeExcludeMeshInterior(distanceOctreeMeshBase, dx), mesh::makeExcludeMeshInteriorRefinement(distanceOctreeMeshBase, dx));
   bfc.setPeriodicity(periodicity);
   auto setupForest = bfc.createSetupBlockForest( cellsPerBlock, 1 );

   const uint_t numCells = setupForest->getXSize() * setupForest->getYSize() * setupForest->getZSize() * cellsPerBlock[0] * cellsPerBlock[1] * cellsPerBlock[2];

   const real_t latticeViscosity =  1.0/3.0 * (1.0 / omega - 0.5);
   const real_t ReynoldsNumber = (meshAABB.xSize() / dx) * initialVelocity[0] / latticeViscosity;


   WALBERLA_LOG_INFO_ON_ROOT("Simulation Parameter: \n"
                          << "Domain Decomposition <" << setupForest->getXSize() << "," << setupForest->getYSize() << "," << setupForest->getZSize() << "> = " << setupForest->getXSize() * setupForest->getYSize() * setupForest->getZSize()  << " root Blocks \n"
                          << "Cells per Block " << cellsPerBlock << " \n"
                          << "Number of cells "  << numCells << " \n"
                          << "Mesh_size " << dx << " m \n"
                          << "initialVelocity " << initialVelocity[0] << "\n"
                          << "Timesteps " << timesteps << "\n"
                          << "Omega " << omega << "\n"
                          << "rotationPerTimestep " << rotationPerTimestep << "\n"
                          << "translationPerTimestep " << translationPerTimestep << "\n"
                          << "Reynolds Number " << ReynoldsNumber << "\n"

   )

   auto blocks = bfc.createStructuredBlockForest(cellsPerBlock);
   auto domainAABB = blocks->getDomain();

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   /////////////////////////
   /// Boundary Handling ///
   /////////////////////////


   const BlockDataID fractionFieldId = field::addToStorage< FracField_T >(blocks, "fractionField", real_t(0.0), field::fzyx, uint_c(1));
   const BlockDataID objectVelocitiesFieldId = field::addToStorage< VectorField_T >(blocks, "particleVelocitiesField", real_c(0.0), field::fzyx, uint_c(1));
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const BlockDataID fractionFieldGPUId = gpu::addGPUFieldToStorage< FracField_T >(blocks, fractionFieldId, "fractionFieldGPU", true);
   const BlockDataID objectVelocitiesFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, objectVelocitiesFieldId, "object velocity field on GPU", true);
#endif
   //Setting up Object Rotator
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   auto objectMover = make_shared<MovingGeometry> (blocks, mesh, fractionFieldGPUId, objectVelocitiesFieldGPUId,
                                                    GeometryMovementFunction(domainAABB, meshAABB, rotationAxis, rotationPerTimestep, translationPerTimestep),
                                                    distanceOctreeMesh, "geometry", maxSuperSamplingDepth, 1, MovingGeometry::TRANSLATING);
#else
   auto objectMover = make_shared<MovingGeometry> (blocks, mesh, fractionFieldId, objectVelocitiesFieldId,
                                                    GeometryMovementFunction(domainAABB, meshAABB, rotationAxis, rotationPerTimestep, translationPerTimestep),
                                                    distanceOctreeMesh, "geometry", maxSuperSamplingDepth, 1, MovingGeometry::TRANSLATING);
#endif

   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriter(mesh, "meshBase", VTKWriteFrequency);

   const std::function< void() > meshWritingFunc = [&]() {
      objectMover->moveTriangleMesh(timeloop.getCurrentTimeStep(), VTKWriteFrequency);
      meshWriter();
   };

   /////////////////////////
   /// Fields Creation   ///
   /////////////////////////

   const StorageSpecification_T StorageSpec = StorageSpecification_T();
   const BlockDataID pdfFieldId  = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, uint_c(1), field::fzyx);
   const BlockDataID velocityFieldId = field::addToStorage< VectorField_T >(blocks, "velocity", real_t(0.0), field::fzyx, uint_c(1));
   const BlockDataID densityFieldId = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, uint_c(1));
   const BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flagField");

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const BlockDataID pdfFieldGPUId = lbm_generated::addGPUPdfFieldToStorage< PdfField_T >(blocks, pdfFieldId, StorageSpec, "pdf field on GPU", true);
   const BlockDataID velocityFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, velocityFieldId, "velocity field on GPU", true);
   const BlockDataID densityFieldGPUId = gpu::addGPUFieldToStorage< ScalarField_T >(blocks, densityFieldId, "density field on GPU", true);
#endif

   /////////////////////////
   /// Rotation Calls    ///
   /////////////////////////

   const std::function< void() > objectRotatorFunc = [&]() {
      objectMover->resetFractionField();
      (*objectMover)(timeloop.getCurrentTimeStep());
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
#endif
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

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   SweepCollection_T sweepCollection(blocks, fractionFieldGPUId, objectVelocitiesFieldGPUId, pdfFieldGPUId, densityFieldGPUId, velocityFieldGPUId, omega);
   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldGPUId, fluidFlagUID);
   pystencils::PSM_Conditional_Sweep psmConditionalSweep( fractionFieldGPUId, objectVelocitiesFieldGPUId, pdfFieldGPUId, omega );
#else
   SweepCollection_T sweepCollection(blocks, fractionFieldId, objectVelocitiesFieldId, pdfFieldId, densityFieldId, velocityFieldId, omega);
   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldId, fluidFlagUID);
   pystencils::PSM_Conditional_Sweep psmConditionalSweep( fractionFieldId, objectVelocitiesFieldId, pdfFieldId, omega );
#endif

   for (auto& block : *blocks)
   {
      auto velField = block.getData<VectorField_T>(velocityFieldId);
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(velField, velField->get(x,y,z,0) = initialVelocity[0];)
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpu::GPUField<real_t> * dst = block.getData<gpu::GPUField<real_t>>( velocityFieldGPUId );
      const VectorField_T * src = block.getData<VectorField_T>( velocityFieldId );
      gpu::fieldCpy( *dst, *src );
#endif
      sweepCollection.initialise(&block, 1);
   }

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

#endif
      });

      auto velWriter = make_shared< field::VTKWriter< VectorField_T > >(velocityFieldId, "Velocity");
      auto fractionFieldWriter = make_shared< field::VTKWriter< FracField_T > >(fractionFieldId, "FractionField");
      auto objVeldWriter = make_shared< field::VTKWriter< VectorField_T > >(objectVelocitiesFieldId, "objectVelocity");

      vtkOutput->addCellDataWriter(velWriter);
      vtkOutput->addCellDataWriter(fractionFieldWriter);
      vtkOutput->addCellDataWriter(objVeldWriter);

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

   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv) { return walberla::main(argc, argv); }
