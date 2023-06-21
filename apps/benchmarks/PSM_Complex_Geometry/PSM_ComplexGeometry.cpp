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
//! \file PSM_Test.cpp
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================



#include "core/all.h"
#include "core/MemoryUsage.h"

#include "domain_decomposition/all.h"
#include "field/all.h"
#include "field/vtk/VTKWriter.h"
#include "geometry/all.h"

#include "lbm_generated/communication/UniformGeneratedPdfPackInfo.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/field/AddToStorage.h"

#include "lbm_generated/evaluation/PerformanceEvaluation.h"

#include "stencil/D3Q19.h"
#include "timeloop/all.h"

#include "ObjectRotator.h"
#include "PSM_InfoHeader.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#include "lbm_generated/gpu/AddToStorage.h"
#include "lbm_generated/gpu/GPUPdfField.h"
#include "lbm_generated/gpu/UniformGeneratedGPUPdfPackInfo.h"
#endif

namespace walberla
{
///////////////////////
/// Typedef Aliases ///
///////////////////////

using StorageSpecification_T = lbm::PSMStorageSpecification;
using Stencil_T = lbm::PSMStorageSpecification::Stencil;
using PdfField_T = lbm_generated::PdfField< StorageSpecification_T >;
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
using GPUPdfField_T = lbm_generated::GPUPdfField< StorageSpecification_T >;
#endif

typedef walberla::uint8_t flag_t;
typedef FlagField< flag_t > FlagField_T;
using BoundaryCollection_T = lbm::PSMBoundaryCollection< FlagField_T >;

using SweepCollection_T = lbm::PSMSweepCollection;

const FlagUID fluidFlagUID("Fluid");
const FlagUID noSlipFlagUID("NoSlip");

template< typename MeshType >
void vertexToFaceColor(MeshType& mesh, const typename MeshType::Color& defaultColor)
{
   WALBERLA_CHECK(mesh.has_vertex_colors())
   mesh.request_face_colors();

   for (auto faceIt = mesh.faces_begin(); faceIt != mesh.faces_end(); ++faceIt)
   {
      typename MeshType::Color vertexColor;

      bool useVertexColor = true;

      auto vertexIt = mesh.fv_iter(*faceIt);
      WALBERLA_ASSERT(vertexIt.is_valid())

      vertexColor = mesh.color(*vertexIt);

      ++vertexIt;
      while (vertexIt.is_valid() && useVertexColor)
      {
         if (vertexColor != mesh.color(*vertexIt)) useVertexColor = false;
         ++vertexIt;
      }

      mesh.set_color(*faceIt, useVertexColor ? vertexColor : defaultColor);
   }
}

auto deviceSyncWrapper = [](std::function< void(IBlock*) > sweep) {
   return [sweep](IBlock* b) {
      sweep(b);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpuDeviceSynchronize();
#endif
   };
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



   mpi::MPIManager::instance()->useWorldComm();

   ///////////////////////
   /// PARAMETER INPUT ///
   ///////////////////////

   // read general simulation parameters
   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");

   real_t omega = parameters.getParameter< real_t >("omega", real_c(1.4));
   const Vector3< real_t > initialVelocity = parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >(0.0));
   const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(10));
   const uint_t VTKWriteFrequency = parameters.getParameter< uint_t >("VTKwriteFrequency", uint_c(10));
   const real_t remainingTimeLoggerFrequency = parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(3.0)); // in seconds

   auto domainParameters = walberlaEnv.config()->getOneBlock("DomainSetup");
   std::string meshFile = domainParameters.getParameter< std::string >("meshFile");

   const uint_t rotationFrequency = domainParameters.getParameter< uint_t >("rotationFrequency", uint_t(1));
   const real_t rotationAngle = domainParameters.getParameter< real_t >("rotationAngle", real_t(0.017453292519943));

   const real_t dx = domainParameters.getParameter< real_t >("dx", real_t(1));
   Vector3< uint_t > domainScaling = domainParameters.getParameter< Vector3< uint_t > >("domainScaling", Vector3< uint_t >(1));
   const Vector3< bool > periodicity = domainParameters.getParameter< Vector3< bool > >("periodic", Vector3< bool >(true));
   const Vector3< uint_t > cellsPerBlock = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");

   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   WALBERLA_LOG_INFO_ON_ROOT("Using mesh from " << meshFile << ".")

   auto mesh = make_shared< mesh::TriangleMesh >();
   mesh->request_vertex_colors();
   mesh::readAndBroadcast(meshFile, *mesh);

   vertexToFaceColor(*mesh, mesh::TriangleMesh::Color(255, 255, 255));

   auto triDist = make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh);
   auto distanceOctree = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(triDist);
   WALBERLA_ROOT_SECTION()
   {
      distanceOctree->writeVTKOutput("distanceOctree");
   }

   auto aabb = computeAABB(*mesh);
   aabb.scale(domainScaling);
   aabb.setCenter(aabb.center() + 0.3 * Vector3< real_t >(0, aabb.ySize(), 0));

   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, Vector3< real_t >(dx), mesh::makeExcludeMeshInterior(distanceOctree, dx));
   bfc.setPeriodicity(periodicity);
   auto blocks = bfc.createStructuredBlockForest(cellsPerBlock);

   ////////////////////////////////////
   /// PDF Field and Velocity Setup ///
   ////////////////////////////////////

   const StorageSpecification_T StorageSpec = StorageSpecification_T();
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   auto allocator = make_shared< gpu::HostFieldAllocator<real_t> >(); // use pinned memory allocator for faster CPU-GPU memory transfers
#else
   auto allocator = make_shared< field::AllocateAligned< real_t, 64 > >();
#endif
   const BlockDataID pdfFieldId  = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, uint_c(1), field::fzyx, allocator);
   const BlockDataID velocityFieldId = field::addToStorage< VectorField_T >(blocks, "velocity", real_c(0.0), field::fzyx, uint_c(1), allocator);
   const BlockDataID densityFieldId = field::addToStorage< ScalarField_T >(blocks, "density", real_c(0.0), field::fzyx, uint_c(1), allocator);
   const BlockDataID flagFieldId     = field::addFlagFieldToStorage< FlagField_T >(blocks, "flagField");

   const BlockDataID fractionFieldId = field::addToStorage< FracField_T >(blocks, "fractionField", fracSize(0.0), field::fzyx, uint_c(1));
   const BlockDataID objectVelocitiesFieldId = field::addToStorage< VectorField_T >(blocks, "particleVelocitiesField", real_c(0.0), field::fzyx, uint_c(1), allocator);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   // GPU Field for PDFs
   const BlockDataID pdfFieldGPUId = lbm_generated::addGPUPdfFieldToStorage< PdfField_T >(blocks, pdfFieldId, StorageSpec, "pdf field on GPU", true);
   const BlockDataID fractionFieldGPUId = gpu::addGPUFieldToStorage< FracField_T >(blocks, fractionFieldId, "fraction field on GPU", true);
   const BlockDataID objectVelocitiesFieldGPUId = gpu::addGPUFieldToStorage< gpu::GPUField< real_t > >(blocks, objectVelocitiesFieldId, "object velocity field on GPU", true);
#endif

   /////////////////////////
   /// Boundary Handling ///
   /////////////////////////

   const BoundaryUID wallFlagUID("NoSlip");
   mesh::ColorToBoundaryMapper< mesh::TriangleMesh > colorToBoundaryMapper((mesh::BoundaryInfo(wallFlagUID)));
   colorToBoundaryMapper.set(mesh::TriangleMesh::Color(255, 255, 255), mesh::BoundaryInfo(wallFlagUID));

   auto boundaryLocations = colorToBoundaryMapper.addBoundaryInfoToMesh(*mesh);

   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriter(mesh, "meshBoundaries", VTKWriteFrequency);
   meshWriter.addDataSource(make_shared< mesh::BoundaryUIDFaceDataSource< mesh::TriangleMesh > >(boundaryLocations));
   meshWriter.addDataSource(make_shared< mesh::ColorFaceDataSource< mesh::TriangleMesh > >());
   meshWriter.addDataSource(make_shared< mesh::ColorVertexDataSource< mesh::TriangleMesh > >());
   meshWriter();
   const std::function< void() > meshWritingFunc = [&]() { meshWriter(); };

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");

   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldId, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldId, fluidFlagUID);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   BoundaryCollection_T boundaryCollection(blocks, flagFieldGPUId, pdfFieldGPUId, fluidFlagUID);
#else
   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldId, fluidFlagUID);
#endif

   //Setting up Object Rotator
   const bool preProcessFractionFields = true;
   ObjectRotator objectRotator(blocks, mesh, fractionFieldId,
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
                               fractionFieldGPUId,
#endif
                               objectVelocitiesFieldId, rotationAngle, rotationFrequency, makeMeshDistanceFunction(distanceOctree), preProcessFractionFields);

   const std::function< void() > objectRotatorFunc = [&]() { objectRotator(); };
   objectRotator.getFractionFieldFromMesh(fractionFieldId);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   gpu::fieldCpy< gpu::GPUField< real_t >, VectorField_T >(blocks, objectVelocitiesFieldGPUId, objectVelocitiesFieldId);
#endif

   /////////////
   /// Sweep ///
   /////////////

   const Cell innerOuterSplit = Cell(parameters.getParameter< Vector3<cell_idx_t> >("innerOuterSplit", Vector3<cell_idx_t>(1, 1, 1)));

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   SweepCollection_T sweepCollection(blocks, pdfFieldGPUId, densityFieldGPUId, velocityFieldGPUId, 0.0, 0.0, 0.0, omega, innerOuterSplit);
   pystencils::PSMSweep PSMSweep(fractionFieldGPUId, objectVelocitiesFieldGPUId, pdfFieldGPUId, real_t(0), real_t(0.0), real_t(0.0), omega);
#else
   SweepCollection_T sweepCollection(blocks, pdfFieldId, densityFieldId, velocityFieldId, 0.0, 0.0, 0.0, omega, innerOuterSplit);
   pystencils::PSMSweep PSMSweep(fractionFieldId, objectVelocitiesFieldId, pdfFieldId, real_t(0), real_t(0.0), real_t(0.0), omega);
#endif

   for (auto& block : *blocks)
   {
      sweepCollection.initialise(&block);
   }
   /////////////////
   /// Time Loop ///
   /////////////////

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   // Communication
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const bool sendDirectlyFromGPU = true;
   gpu::communication::UniformGPUScheme< Stencil_T > com(blocks, sendDirectlyFromGPU, false);
   auto packInfo = std::make_shared<lbm_generated::UniformGeneratedGPUPdfPackInfo< GPUPdfField_T >>(pdfFieldGpuID);
   communication.addPackInfo(packInfo);
   auto communication = std::function< void() >([&]() { com.communicate(nullptr); });
#else
   blockforest::communication::UniformBufferedScheme< Stencil_T > communication(blocks);
   auto packInfo = std::make_shared<lbm_generated::UniformGeneratedPdfPackInfo< PdfField_T >>(pdfFieldId);
   communication.addPackInfo(packInfo);
#endif

   const auto emptySweep = [](IBlock*) {};


   // Timeloop
   timeloop.add() << BeforeFunction(communication, "Communication")
                  << Sweep(deviceSyncWrapper(boundaryCollection.getSweep(BoundaryCollection_T::ALL)), "Boundary Conditions");
   timeloop.add() << Sweep(deviceSyncWrapper(PSMSweep), "PSMSweep");
   if(rotationAngle > 0.0 && rotationFrequency > 0) {
      timeloop.add() << BeforeFunction(objectRotatorFunc, "ObjectRotator") <<  Sweep(emptySweep);
   }


   // Time logger
   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
                                 "remaining time logger");

   if (VTKWriteFrequency > 0)
   {
      const std::string path = "vtk_out";
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "fields", VTKWriteFrequency, 0,
                                                              false, path, "simulation_step", false, true, true, false, 0);


      vtkOutput->addBeforeFunction([&]() {
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         gpu::fieldCpy< PdfField_T, gpu::GPUField< real_t > >(blocks, pdfFieldId, pdfFieldGPUId);
#endif
         for (auto& block : *blocks)
            sweepCollection.calculateMacroscopicParameters(&block);
      });

      auto velWriter = make_shared< field::VTKWriter< VectorField_T > >(velocityFieldId, "Velocity");
      //auto flagWriter = make_shared< field::VTKWriter< FlagField_T > >(flagFieldId, "Flag");
      auto fractionFieldWriter = make_shared< field::VTKWriter< FracField_T > >(fractionFieldId, "FractionField");
      //auto objVeldWriter = make_shared< field::VTKWriter< VectorField_T > >(objectVelocitiesFieldId, "objectVelocity");

      vtkOutput->addCellDataWriter(velWriter);
      //vtkOutput->addCellDataWriter(flagWriter);
      vtkOutput->addCellDataWriter(fractionFieldWriter);
      //vtkOutput->addCellDataWriter(objVeldWriter);

      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
      vtk::writeDomainDecomposition(blocks, "domain_decompositionDense", "vtk_out", "write_call", true, true, 0);
   }

   lbm_generated::PerformanceEvaluation<FlagField_T> const performance(blocks, flagFieldId, fluidFlagUID);
   WcTimingPool timeloopTiming;
   WcTimer simTimer;

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
   gpuDeviceSynchronize();
#endif


   simTimer.start();
   timeloop.run(timeloopTiming);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
   gpuDeviceSynchronize();
#endif


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
