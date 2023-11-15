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
//! \file CROR.cpp
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"
#include "blockforest/SetupBlockForest.h"
#include "core/all.h"
#include "core/MemoryUsage.h"
#include "domain_decomposition/all.h"
#include "field/all.h"
#include "field/vtk/VTKWriter.h"
#include "geometry/all.h"
#include "stencil/D3Q19.h"
#include "timeloop/all.h"

#include "lbm_generated/communication/NonuniformGeneratedPdfPackInfo.h"
#include "lbm_generated/evaluation/PerformanceEvaluation.h"
#include "lbm_generated/field/AddToStorage.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/refinement/BasicRecursiveTimeStep.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#include "lbm_generated/gpu/AddToStorage.h"
#include "lbm_generated/gpu/GPUPdfField.h"
#include "lbm_generated/gpu/BasicRecursiveTimeStepGPU.h"
#include "lbm_generated/gpu/NonuniformGeneratedGPUPdfPackInfo.h"
#include "gpu/communication/NonUniformGPUScheme.h"
#endif

#include "../MovingGeometry.h"
#include "CROR_InfoHeader.h"

namespace walberla
{
///////////////////////
/// Typedef Aliases ///
///////////////////////

using StorageSpecification_T = lbm::CRORStorageSpecification;
using Stencil_T = lbm::CRORStorageSpecification::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;
using PdfField_T = lbm_generated::PdfField< StorageSpecification_T >;
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
using GPUPdfField_T = lbm_generated::GPUPdfField< StorageSpecification_T >;
using gpu::communication::NonUniformGPUScheme;
#endif

typedef walberla::uint8_t flag_t;
typedef FlagField< flag_t > FlagField_T;
using BoundaryCollection_T = lbm::CRORBoundaryCollection< FlagField_T >;
using SweepCollection_T = lbm::CRORSweepCollection;
const FlagUID fluidFlagUID("Fluid");
const uint_t ghostLayers = 2;


class LDCRefinement
{


 public:
   explicit LDCRefinement(const uint_t depth, AABB rotorMeshAABB, AABB statorMeshAABB, shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>>& distOctreeBase) : refinementDepth_(depth), rotorMeshAABB_(rotorMeshAABB), statorMeshAABB_(statorMeshAABB), distOctreeBase_(distOctreeBase){};

   void operator()(SetupBlockForest& forest) const
   {
      const auto distFunct = make_shared<MeshDistanceFunction<mesh::DistanceOctree<mesh::TriangleMesh>>>( distOctreeBase_ );

      for(auto & block : forest) {
         auto blockAABB = block.getAABB();

         if (rotorMeshAABB_.intersects(blockAABB) || statorMeshAABB_.intersects(blockAABB)) {
            if( block.getLevel() < refinementDepth_)
               block.setMarker( true );
         }
         auto blockCenter = blockAABB.center();
         const real_t sqSignedDistance = (*distFunct)(blockCenter);
         const real_t circumRadius   = blockAABB.sizes().length() * real_t(0.5);
         const real_t sqCircumRadius = circumRadius * circumRadius;
         if (! (sqSignedDistance > sqCircumRadius)) {
            if( block.getLevel() < refinementDepth_)
               block.setMarker( true );
         }
      }
   }

 private:
   const uint_t refinementDepth_;
   const AABB rotorMeshAABB_;
   const AABB statorMeshAABB_;
   shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>>& distOctreeBase_;
};


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


   const uint_t rotationFrequency = domainParameters.getParameter< uint_t >("rotationFrequency", uint_t(1));
   const Vector3< int > rotationAxis = domainParameters.getParameter< Vector3< int > >("rotationAxis", Vector3< int >(1,0,0));
   const real_t rotationSpeed = domainParameters.getParameter< real_t >("rotationSpeed");


   const Vector3< real_t > domainScaling = domainParameters.getParameter< Vector3< real_t > >("domainScaling", Vector3< real_t >(1.0));
   const Vector3< real_t > domainTransforming = domainParameters.getParameter< Vector3< real_t > >("domainTransforming", Vector3< real_t >(0.0));
   const Vector3< bool > periodicity = domainParameters.getParameter< Vector3< bool > >("periodic", Vector3< bool >(true));
   const Vector3< uint_t > cellsPerBlock = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
   const uint_t refinementDepth = domainParameters.getParameter< uint_t >("refinementDepth");
   const uint_t maxSuperSamplingDepth = parameters.getParameter< uint_t >("maxSuperSamplingDepth", uint_c(1));

   const Vector3< real_t > initialVelocity = parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >(0.0));
   const uint_t timestepsFixed = parameters.getParameter< uint_t >("timesteps", uint_c(10));
   const uint_t VTKWriteFrequency = parameters.getParameter< uint_t >("VTKwriteFrequency", uint_c(10));
   const real_t remainingTimeLoggerFrequency = parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(3.0));
   const bool writeDomainDecompositionAndReturn = parameters.getParameter< bool >("writeDomainDecompositionAndReturn", false);

   const std::string timeStepStrategy = parameters.getParameter<std::string>("timeStepStrategy", "noOverlap");
   const Cell innerOuterSplit = Cell(parameters.getParameter< Vector3<cell_idx_t> >("innerOuterSplit", Vector3<cell_idx_t>(1, 1, 1)));

   const real_t ref_lattice_velocity = initialVelocity[0];
   const real_t ref_velocity = parameters.getParameter<real_t>("ref_velocity");
   const real_t ref_length = parameters.getParameter<real_t>("ref_length");
   const real_t viscosity = parameters.getParameter<real_t>("viscosity");
   real_t sim_time = parameters.getParameter<real_t>("sim_time");
   const real_t mesh_size = parameters.getParameter<real_t>("mesh_size");
   const real_t dx = mesh_size;
   const real_t reynolds_number = (ref_velocity * ref_length) / viscosity;
   const real_t Cu = ref_velocity / ref_lattice_velocity;
   const real_t Ct = mesh_size / Cu;

   const real_t viscosity_lattice = viscosity * Ct / (mesh_size * mesh_size);
   const real_t omega = real_c(1.0 / (3.0 * viscosity_lattice + 0.5));
   uint_t timesteps;
   if(sim_time > 0.0)
      timesteps = uint_c(sim_time / Ct);
   else {
      timesteps = timestepsFixed;
      sim_time = Ct * real_t(timesteps);
   }

   const real_t rotPerSec = rotationSpeed * (2 * M_PI) / 360;

   const real_t radPerTimestep = rotationSpeed * Ct;
   const real_t rotationAngle = radPerTimestep * real_c(rotationFrequency);


   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   const std::string meshFileBase = "../Meshfiles/CROR_base_downScaled100.obj";
   auto meshBase = make_shared< mesh::TriangleMesh >();
   mesh::readAndBroadcast(meshFileBase, *meshBase);
   auto distanceOctreeMeshBase = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(meshBase));

   const std::string meshFileRotor = "../Meshfiles/CROR_rotor_downScaled100.obj";
   auto meshRotor = make_shared< mesh::TriangleMesh >();
   mesh::readAndBroadcast(meshFileRotor, *meshRotor);
   auto distanceOctreeMeshRotor = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(meshRotor));

   const std::string meshFileStator = "../Meshfiles/CROR_stator_downScaled100.obj";
   auto meshStator = make_shared< mesh::TriangleMesh >();
   mesh::readAndBroadcast(meshFileStator, *meshStator);
   auto distanceOctreeMeshStator = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(meshStator));


   auto aabbBase = computeAABB(*meshBase);
   AABB aabb = aabbBase;
   aabb.setCenter(aabb.center() - Vector3< real_t >(domainTransforming[0] * aabb.xSize(), domainTransforming[1] * aabb.ySize(), domainTransforming[2] * aabb.zSize()));
   aabb.scale(domainScaling);

   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, Vector3< real_t >(dx), mesh::makeExcludeMeshInterior(distanceOctreeMeshBase, dx), mesh::makeExcludeMeshInteriorRefinement(distanceOctreeMeshBase, dx));
   bfc.setPeriodicity(periodicity);

   bfc.setRefinementSelectionFunction(LDCRefinement(refinementDepth, computeAABB(*meshRotor), computeAABB(*meshStator), distanceOctreeMeshBase));

   auto setupForest = bfc.createSetupBlockForest( cellsPerBlock, 1 );

   const uint_t numCells = setupForest->getNumberOfBlocks() * cellsPerBlock[0] * cellsPerBlock[1] * cellsPerBlock[2];
   const AABB domainAABB = setupForest->getDomain();
   const real_t fullRefinedMeshSize = mesh_size / pow(2, real_c(refinementDepth));
   const uint_t numFullRefinedCell = uint_c((domainAABB.xSize() / fullRefinedMeshSize) * (domainAABB.ySize() / fullRefinedMeshSize) * (domainAABB.zSize() / fullRefinedMeshSize));

   AABB rotorAABB =  computeAABB(*meshRotor);
   Vector3<uint_t> cellsForGeometryMesh = rotorAABB.sizes() / (fullRefinedMeshSize / pow(2, real_t(maxSuperSamplingDepth)));
   real_t geoFieldSize = real_t(cellsForGeometryMesh[0] * cellsForGeometryMesh[1] * cellsForGeometryMesh[2]) / (1000 * 1000);
   real_t memoryPerBlock = real_t(cellsPerBlock[0]) * real_t(cellsPerBlock[1]) * real_t(cellsPerBlock[2]) * 8 * (19 * 2 + 3 + 3 + 1 + 1 + 1) / (1000 * 1000);

   WALBERLA_LOG_INFO_ON_ROOT("Simulation Parameter \n"
                             << "Domain Decomposition <" << setupForest->getXSize() << "," << setupForest->getYSize() << "," << setupForest->getZSize() << "> = " << setupForest->getXSize() * setupForest->getYSize() * setupForest->getZSize()  << " root Blocks \n"
                             << "Number of blocks is " << setupForest->getNumberOfBlocks() << " \n"
                             << "Cells per Block " << cellsPerBlock << " \n"
                             << "Number of cells "  << numCells << ", number of potential Cells (full refined) " << numFullRefinedCell <<  ", Saved computation " << (1.0 - (real_c(numCells) / real_c(numFullRefinedCell))) * 100 << "% \n"
                             << "Mesh_size " << mesh_size << " m \n"
                             << "Refined Mesh size " << fullRefinedMeshSize << " m \n"
                             << "Reynolds_number " << reynolds_number << "\n"
                             << "Inflow velocity " << ref_velocity << " m/s \n"
                             << "Lattice velocity " << initialVelocity[0] << "\n"
                             << "Viscosity " << viscosity << "\n"
                             << "Simulation time " << sim_time << " s \n"
                             << "Timesteps " << timesteps << "\n"
                             << "Omega " << omega << "\n"
                             << "Cu " << Cu << " \n"
                             << "Ct " << Ct << " \n"
                             << "Rotation Speed " << rotationSpeed << " rad/s \n"
                             << "Rotations per second " << rotPerSec << " 1/s \n"
                             << "Rotation Angle per Rotation " << rotationAngle / (2 * M_PI) * 360  << " ° \n"
                             << "Size of geometry Field is " << geoFieldSize << " MB per process \n"
                             << "Memory per block " << memoryPerBlock << " MB \n"
                             << "Memory per process " << memoryPerBlock *  real_t(setupForest->getNumberOfBlocks()) / mpi::MPIManager::instance()->numProcesses() << " MB \n"


   )

   if(writeDomainDecompositionAndReturn) {
      WALBERLA_ROOT_SECTION() { setupForest->writeVTKOutput("SetupBlockForest"); }
      return EXIT_SUCCESS;
   }
   auto blocks = bfc.createStructuredBlockForest(cellsPerBlock);

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   /////////////////////////
   /// Boundary Handling ///
   /////////////////////////

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   auto allocator = make_shared< gpu::HostFieldAllocator<real_t> >(); // use pinned memory allocator for faster CPU-GPU memory transfers
#else
   auto allocator = make_shared< field::AllocateAligned< real_t, 64 > >();
#endif

   const BlockDataID fractionFieldId = field::addToStorage< FracField_T >(blocks, "fractionField", real_t(0.0), field::fzyx, ghostLayers);
   const BlockDataID objectVelocitiesFieldId = field::addToStorage< VectorField_T >(blocks, "particleVelocitiesField", real_c(0.0), field::fzyx, ghostLayers, allocator);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const BlockDataID fractionFieldGPUId = gpu::addGPUFieldToStorage< FracField_T >(blocks, fractionFieldId, "fractionFieldGPU", true);
#endif

   //Setting up Object Rotator
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   auto objectRotatorBase = make_shared<MovingGeometry> (blocks, meshBase, fractionFieldGPUId, objectVelocitiesFieldId,
                                                          Vector3<real_t>(0,0,0), 0,
                                                          rotationAxis,  distanceOctreeMeshBase, "rotor", maxSuperSamplingDepth, false);
   auto objectRotatorRotor = make_shared<MovingGeometry> (blocks, meshRotor, fractionFieldGPUId, objectVelocitiesFieldId,
                                                           Vector3<real_t>(0,0,0), rotationAngle,
                                                           rotationAxis,  distanceOctreeMeshRotor, "rotor", maxSuperSamplingDepth, true);
   auto objectRotatorStator = make_shared<MovingGeometry> (blocks, meshStator, fractionFieldGPUId, objectVelocitiesFieldId,
                                                            Vector3<real_t>(0,0,0), -rotationAngle,
                                                            rotationAxis,  distanceOctreeMeshStator, "rotor", maxSuperSamplingDepth, true);

#else
   auto objectRotatorBase = make_shared<MovingGeometry> (blocks, meshBase, fractionFieldId, objectVelocitiesFieldId,
                                                           Vector3<real_t>(0,0,0), 0,
                                                           rotationAxis,  distanceOctreeMeshBase, "rotor", maxSuperSamplingDepth, false);
   auto objectRotatorRotor = make_shared<MovingGeometry> (blocks, meshRotor, fractionFieldId, objectVelocitiesFieldId,
                                                           Vector3<real_t>(0,0,0), rotationAngle,
                                                           rotationAxis,  distanceOctreeMeshRotor, "rotor", maxSuperSamplingDepth, true);
   auto objectRotatorStator = make_shared<MovingGeometry> (blocks, meshStator, fractionFieldId, objectVelocitiesFieldId,
                                                           Vector3<real_t>(0,0,0), -rotationAngle,
                                                           rotationAxis,  distanceOctreeMeshStator, "rotor", maxSuperSamplingDepth, true);


#endif

   //VTK Mesh
   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriterBase(meshBase, "meshBase", VTKWriteFrequency);
   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriterRotor(meshRotor, "meshRotor", VTKWriteFrequency);
   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriterStator(meshStator, "meshStator", VTKWriteFrequency);
   const std::function< void() > meshWritingFunc = [&]() {
      meshWriterBase();
      meshWriterRotor();
      objectRotatorRotor->moveTriangleMesh(timeloop.getCurrentTimeStep(), VTKWriteFrequency);
      meshWriterStator();
      objectRotatorStator->moveTriangleMesh(timeloop.getCurrentTimeStep(), VTKWriteFrequency);
   };

   /////////////////////////
   /// Fields Creation   ///
   /////////////////////////

   const StorageSpecification_T StorageSpec = StorageSpecification_T();
   const BlockDataID pdfFieldId  = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, ghostLayers, field::fzyx, allocator);
   const BlockDataID velocityFieldId = field::addToStorage< VectorField_T >(blocks, "velocity", real_t(0.0), field::fzyx, ghostLayers, allocator);
   const BlockDataID densityFieldId = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, ghostLayers, allocator);
   const BlockDataID flagFieldId     = field::addFlagFieldToStorage< FlagField_T >(blocks, "flagField", ghostLayers);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const BlockDataID pdfFieldGPUId = lbm_generated::addGPUPdfFieldToStorage< PdfField_T >(blocks, pdfFieldId, StorageSpec, "pdf field on GPU", true);
   const BlockDataID velocityFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, velocityFieldId, "velocity field on GPU", true);
   const BlockDataID densityFieldGPUId = gpu::addGPUFieldToStorage< ScalarField_T >(blocks, densityFieldId, "density field on GPU", true);
   const BlockDataID objectVelocitiesFieldGPUId = gpu::addGPUFieldToStorage< VectorField_T >(blocks, objectVelocitiesFieldId, "object velocity field on GPU", true);
#endif


   /////////////////////////
   /// Rotation Calls    ///
   /////////////////////////

   const std::function< void() > objectRotatorFunc = [&]() {
      objectRotatorBase->resetFractionField();
      (*objectRotatorBase)(timeloop.getCurrentTimeStep());
      (*objectRotatorRotor)(timeloop.getCurrentTimeStep());
      (*objectRotatorStator)(timeloop.getCurrentTimeStep());
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
#endif
   };


#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   gpu::fieldCpy< gpu::GPUField< real_t >, VectorField_T >(blocks, objectVelocitiesFieldGPUId, objectVelocitiesFieldId);
#endif

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");
   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldId, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldId, fluidFlagUID);


   /////////////
   /// Sweep ///
   /////////////

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   SweepCollection_T sweepCollection(blocks, fractionFieldGPUId, objectVelocitiesFieldGPUId, pdfFieldGPUId, densityFieldGPUId, velocityFieldGPUId, omega, innerOuterSplit);
   //pystencils::PSMSweep PSMSweep(fractionFieldGPUId, objectVelocitiesFieldGPUId, pdfFieldGPUId, omega, /*real_t(0.0),*/ innerOuterSplit);
   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldGPUId, fluidFlagUID);

#else
   SweepCollection_T sweepCollection(blocks, fractionFieldId, objectVelocitiesFieldId, pdfFieldId, densityFieldId, velocityFieldId, omega, innerOuterSplit);
   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldId, fluidFlagUID, initialVelocity[0]);
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
      sweepCollection.initialise(&block, ghostLayers);
   }


   // Communication
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const bool sendDirectlyFromGPU = false;
   auto communication = std::make_shared< NonUniformGPUScheme <CommunicationStencil_T>> (blocks, sendDirectlyFromGPU);
   auto packInfo = lbm_generated::setupNonuniformGPUPdfCommunication<GPUPdfField_T>(blocks, pdfFieldGPUId);
   communication->addPackInfo(packInfo);
   lbm_generated::BasicRecursiveTimeStepGPU< GPUPdfField_T, SweepCollection_T, BoundaryCollection_T > LBMMeshRefinement(blocks, pdfFieldGPUId, sweepCollection, boundaryCollection, communication, packInfo);
#else
   auto communication = std::make_shared< NonUniformBufferedScheme< CommunicationStencil_T > >(blocks);
   auto packInfo      = lbm_generated::setupNonuniformPdfCommunication< PdfField_T >(blocks, pdfFieldId);
   //blockforest::communication::UniformBufferedScheme< Stencil_T > communication(blocks);
   //auto packInfo = std::make_shared<lbm_generated::UniformGeneratedPdfPackInfo< PdfField_T >>(pdfFieldId);
   communication->addPackInfo(packInfo);
   lbm_generated::BasicRecursiveTimeStep< PdfField_T, SweepCollection_T, BoundaryCollection_T > LBMMeshRefinement(blocks, pdfFieldId, sweepCollection, boundaryCollection, communication, packInfo);
#endif


   /////////////////
   /// Time Loop ///
   /////////////////

   const auto emptySweep = [](IBlock*) {};
   if( rotationFrequency > 0) {
      timeloop.add() << BeforeFunction(meshWritingFunc, "Meshwriter") <<  Sweep(emptySweep);
      timeloop.add() << BeforeFunction(objectRotatorFunc, "ObjectRotator") <<  Sweep(emptySweep);
   }

   // Time logger
   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
                                 "remaining time logger");

   //timeloop.addFuncAfterTimeStep( makeSharedFunctor( field::makeStabilityChecker< PdfField_T >( blocks, pdfFieldId, VTKWriteFrequency ) ), "LBM stability check" );

   if (VTKWriteFrequency > 0)
   {
      const std::string path = "vtk_out";
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "fields", VTKWriteFrequency, 0,
                                                              false, path, "simulation_step", false, true, true, false, 0);

      vtkOutput->addBeforeFunction([&]() {
         for (auto& block : *blocks)
            sweepCollection.calculateMacroscopicParameters(&block);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         gpu::fieldCpy< VectorField_T, gpu::GPUField< real_t > >(blocks, velocityFieldId, velocityFieldGPUId);
         gpu::fieldCpy< FracField_T, gpu::GPUField< real_t > >(blocks, fractionFieldId, fractionFieldGPUId);
#endif
      });

      auto velWriter = make_shared< field::VTKWriter< VectorField_T > >(velocityFieldId, "Velocity");
      //auto flagWriter = make_shared< field::VTKWriter< FlagField_T > >(flagFieldId, "Flag");
      auto fractionFieldWriter = make_shared< field::VTKWriter< FracField_T > >(fractionFieldId, "FractionField");
      //auto objVeldWriter = make_shared< field::VTKWriter< VectorField_T > >(objectVelocitiesFieldId, "objectVelocity");

      vtkOutput->addCellDataWriter(velWriter);
      //vtkOutput->addCellDataWriter(flagWriter);
      vtkOutput->addCellDataWriter(fractionFieldWriter);
      //vtkOutput->addCellDataWriter(objVeldWriter);
      //vtkOutput->setSamplingResolution(0.05);


      const AABB sliceAABB(real_c(domainAABB.xMin() + domainAABB.xSize() * 0.18), real_c(domainAABB.yMin() + domainAABB.ySize() * 0.2), real_c(domainAABB.zMin() + domainAABB.zSize() * 0.2),
                     real_c(domainAABB.xMin() + domainAABB.xSize() * 0.6), real_c(domainAABB.yMin() + domainAABB.ySize() * 0.8), real_c(domainAABB.zMin() + domainAABB.zSize() * 0.8));

      vtkOutput->addCellInclusionFilter(vtk::AABBCellFilter(sliceAABB));

      //timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");

      vtk::writeDomainDecomposition(blocks, "domain_decomposition", "vtk_out", "write_call", true, true, 0);
   }


   if(timeStepStrategy == "refinement")  {
      LBMMeshRefinement.addRefinementToTimeLoop(timeloop);
   }
   else {
      WALBERLA_ABORT("timeStepStrategy " << timeStepStrategy << " not supported")
   }

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

   printResidentMemoryStatistics();

   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv) { return walberla::main(argc, argv); }
