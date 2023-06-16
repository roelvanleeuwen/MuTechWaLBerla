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

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#   include "gpu/AddGPUFieldToStorage.h"
#   include "gpu/ParallelStreams.h"
#   include "gpu/communication/UniformGPUScheme.h"
#endif

#include "domain_decomposition/all.h"
#include "field/all.h"
#include "field/vtk/VTKWriter.h"
#include "geometry/all.h"
#include "lbm_generated/evaluation/PerformanceEvaluation.h"
#include "stencil/D3Q19.h"
#include "timeloop/all.h"

//    Codegen Includes
#include "LBMSweep.h"
#include "PSMSweep.h"
#include "NoSlip.h"
#include "UBB.h"
#include "FixedDensity.h"
#include "PackInfo.h"
#include "MacroSetter.h"
#include "MacroGetter.h"

#include "ObjectRotator.h"
//#include "lbm_mesapd_coupling/partially_saturated_cells_method/cuda/PSMSweepCollectionGPU.h"


namespace walberla
{
///////////////////////
/// Typedef Aliases ///
///////////////////////

// Communication Pack Info
typedef pystencils::PackInfo PackInfo_T;
// LB Method Stencil
typedef stencil::D3Q19 Stencil_T;
// PDF field type
typedef field::GhostLayerField< real_t, Stencil_T::Size > PdfField_T;
// Velocity Field Type
typedef field::GhostLayerField< real_t, Stencil_T::D > VectorField_T;

typedef field::GhostLayerField< real_t, 1 > ScalarField_T;
// Boundary Handling
typedef walberla::uint8_t flag_t;
typedef FlagField< flag_t > FlagField_T;
const FlagUID fluidFlagUID("Fluid");
const FlagUID noSlipFlagUID("NoSlip");
const FlagUID PSMFlagUID("PSM");



#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
typedef gpu::GPUField< real_t > GPUField;
#endif




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




/////////////////////
/// Main Function ///
/////////////////////

int main(int argc, char** argv)
{
   walberla::Environment walberlaEnv(argc, argv);

   mpi::MPIManager::instance()->useWorldComm();

   ///////////////////////
   /// PARAMETER INPUT ///
   ///////////////////////

   // read general simulation parameters
   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");

   real_t omega = parameters.getParameter< real_t >("omega", real_c(1.4));
   const Vector3< real_t > initialVelocity =
      parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >());
   const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(10));
   const uint_t VTKWriteFrequency = parameters.getParameter< uint_t >("VTKwriteFrequency", uint_c(10));

   const real_t remainingTimeLoggerFrequency =
      parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(3.0)); // in seconds

   //! [parseDomainParameters]
   // read domain parameters
   auto domainParameters = walberlaEnv.config()->getOneBlock("DomainSetup");

   std::string meshFile = domainParameters.getParameter< std::string >("meshFile");
   //! [parseDomainParameters]

   Vector3< uint_t > domainScaling =
      domainParameters.getParameter< Vector3< uint_t > >("domainScaling", Vector3< uint_t >(1));

   const uint_t rotationFrequency = domainParameters.getParameter< uint_t >("rotationFrequency", uint_t(1));
   const real_t rotationAngle = domainParameters.getParameter< real_t >("rotationAngle", real_t(0.017453292519943));


   const real_t dx = domainParameters.getParameter< real_t >("dx", real_t(1));
   const Vector3< bool > periodicity =
      domainParameters.getParameter< Vector3< bool > >("periodic", Vector3< bool >(true));
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
   WALBERLA_LOG_INFO_ON_ROOT("Octree has height " << distanceOctree->height())

   WALBERLA_ROOT_SECTION()
   {
      distanceOctree->writeVTKOutput("distanceOctree");
   }

   auto aabb = computeAABB(*mesh);
   aabb.scale(domainScaling);
   //aabb.setCenter(aabb.center() + 0.2 * Vector3< real_t >(aabb.xSize(), 0, 0));

   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, Vector3< real_t >(dx), mesh::makeExcludeMeshInterior(distanceOctree, dx));
   bfc.setPeriodicity(periodicity);
   auto blocks = bfc.createStructuredBlockForest(cellsPerBlock);
   //! [blockForest]


   ////////////////////////////////////
   /// PDF Field and Velocity Setup ///
   ////////////////////////////////////

   // Common Fields
   BlockDataID velocityFieldId = field::addToStorage< VectorField_T >(blocks, "velocity", real_c(0.0), field::fzyx);
   BlockDataID densityFieldId = field::addToStorage< ScalarField_T >(blocks, "density", real_c(0.0), field::fzyx);
   BlockDataID const flagFieldId     = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   // GPU Field for PDFs
   BlockDataID const pdfFieldId = gpu::addGPUFieldToStorage< gpu::GPUField< real_t > >(
      blocks, "pdf field on GPU", Stencil_T::Size, field::fzyx, uint_t(1));

   // GPU Velocity Field
   BlockDataID velocityFieldIdGPU =
      gpu::addGPUFieldToStorage< VectorField_T >(blocks, velocityFieldId, "velocity on GPU", true);
#else
   // CPU Field for PDFs
   BlockDataID pdfFieldId = field::addToStorage< PdfField_T >(blocks, "pdf field", real_c(0.0), field::fzyx);
#endif

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

   /////////////
   /// Sweep ///
   /////////////

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   pystencils::CumulantMRTSweep const CumulantMRTSweep(pdfFieldId, 0,0,0, omega);
#else
   pystencils::LBMSweep const lbmSweep(pdfFieldId, 0,0,0, omega);
#endif

   /////////////////////////
   /// Boundary Handling ///
   /////////////////////////


   lbm::NoSlip noSlip(blocks, pdfFieldId);
   lbm::UBB ubb(blocks, pdfFieldId);
   lbm::FixedDensity fixedDensity(blocks, pdfFieldId);



   noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldId, noSlipFlagUID, fluidFlagUID);
   ubb.fillFromFlagField< FlagField_T >(blocks, flagFieldId, FlagUID("UBB"), fluidFlagUID);
   fixedDensity.fillFromFlagField< FlagField_T >(blocks, flagFieldId, FlagUID("FixedDensity"), fluidFlagUID);

   BlockDataID BFieldId = field::addToStorage< ScalarField_T >(blocks, "BFieldID", real_c(0.0), field::fzyx);
   BlockDataID objectVelocitiesFieldId = field::addToStorage< VectorField_T >(blocks, "particleVelocitiesField", real_c(0.0), field::fzyx);

   ObjectRotator objectRotator(blocks, mesh, BFieldId, objectVelocitiesFieldId, rotationAngle, rotationFrequency, makeMeshDistanceFunction(distanceOctree));
   const std::function< void() > objectRotatorFunc = [&]() { objectRotator(); };

   objectRotator.getFractionFieldFromMesh();


   pystencils::PSMSweep PSMSweep(BFieldId, objectVelocitiesFieldId, pdfFieldId, real_t(0), real_t(0.0), real_t(0.0), omega);

   pystencils::MacroGetter getterSweep(densityFieldId, pdfFieldId, velocityFieldId, real_t(0.0), real_t(0.0), real_t(0.0));

   pystencils::MacroSetter setterSweep( pdfFieldId, velocityFieldId, 0.0, 0.0, 0.0, 1.0 );
   for(auto &block : *blocks) {
      setterSweep(&block);
   }

   /////////////////
   /// Time Loop ///
   /////////////////

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   // Communication
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const bool sendDirectlyFromGPU = false;
   gpu::communication::UniformGPUScheme< Stencil_T > com(blocks, sendDirectlyFromGPU);
   com.addPackInfo(make_shared< PackInfo_T >(pdfFieldId));
   auto communication = std::function< void() >([&]() { com.communicate(nullptr); });
#else
   blockforest::communication::UniformBufferedScheme< Stencil_T > communication(blocks);
   communication.addPackInfo(make_shared< PackInfo_T >(pdfFieldId));
#endif

   const auto emptySweep = [](IBlock*) {};


   // Timeloop
   timeloop.add() << BeforeFunction(communication, "Communication")
                  << Sweep(ubb, "UBB");
   timeloop.add() << Sweep(noSlip, "NoSlip");
   timeloop.add() << Sweep(fixedDensity, "FixedDensity");
   //timeloop.add() << Sweep(lbmSweep, "LBM Sweep");
   timeloop.add() << Sweep(PSMSweep, "PSMSweep");
   if(rotationAngle > 0.0 && rotationFrequency > 0) {
      timeloop.add() << BeforeFunction(objectRotatorFunc, "ObjectRotator") << BeforeFunction(meshWritingFunc, "meshWriter")<<  Sweep(emptySweep);
   }


   // Time logger
   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
                                 "remaining time logger");

   if (VTKWriteFrequency > 0)
   {
      const std::string path = "vtk_out";
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "fields", VTKWriteFrequency, 0,
                                                              false, path, "simulation_step", false, true, true, false, 0);

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      // Copy velocity data to CPU before output
      vtkOutput->addBeforeFunction(
         [&]() { gpu::fieldCpy< VectorField_T, GPUField >(blocks, velocityFieldId, velocityFieldIdGPU); });
#endif

      vtkOutput->addBeforeFunction([&]() {
         for (auto& block : *blocks)
            getterSweep(&block);
      });

      auto velWriter = make_shared< field::VTKWriter< VectorField_T > >(velocityFieldId, "Velocity");
      auto flagWriter = make_shared< field::VTKWriter< FlagField_T > >(flagFieldId, "Flag");
      auto fractionFieldWriter = make_shared< field::VTKWriter< ScalarField_T > >(BFieldId, "FractionField");
      auto objVeldWriter = make_shared< field::VTKWriter< VectorField_T > >(objectVelocitiesFieldId, "objectVelocity");

      vtkOutput->addCellDataWriter(velWriter);
      vtkOutput->addCellDataWriter(flagWriter);
      vtkOutput->addCellDataWriter(fractionFieldWriter);
      vtkOutput->addCellDataWriter(objVeldWriter);


      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   }

   lbm_generated::PerformanceEvaluation<FlagField_T> const performance(blocks, flagFieldId, fluidFlagUID);
   WcTimingPool timeloopTiming;
   WcTimer simTimer;

   simTimer.start();
   timeloop.run(timeloopTiming);
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
