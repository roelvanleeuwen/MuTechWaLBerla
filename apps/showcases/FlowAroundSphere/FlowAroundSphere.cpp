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
//! \file FlowAroundSphere.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#include "blockforest/communication/UniformBufferedScheme.h"
#include "blockforest/communication/NonUniformBufferedScheme.h"
#include "blockforest/Initialization.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/logging/Initialization.h"
#include "core/SharedFunctor.h"
#include "core/math/Vector3.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/MemoryUsage.h"

#include "field/AddToStorage.h"
#include "field/StabilityChecker.h"
#include "field/vtk/VTKWriter.h"

#include "geometry/InitBoundaryHandling.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/FieldCopy.h"
#include "gpu/ErrorChecking.h"
#include "gpu/HostFieldAllocator.h"
#include "gpu/ParallelStreams.h"
#include "gpu/communication/UniformGPUScheme.h"
#include "gpu/communication/NonUniformGPUScheme.h"
#endif

#include "lbm_generated/communication/NonuniformGeneratedPdfPackInfo.h"
#include "lbm_generated/communication/UniformGeneratedPdfPackInfo.h"
#include "lbm_generated/field/AddToStorage.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/refinement/BasicRecursiveTimeStep.h"
#include "lbm_generated/evaluation/PerformanceEvaluation.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#include "lbm_generated/gpu/UniformGeneratedGPUPdfPackInfo.h"
#include "lbm_generated/gpu/NonuniformGeneratedGPUPdfPackInfo.h"
#include "lbm_generated/gpu/GPUPdfField.h"
#include "lbm_generated/gpu/AddToStorage.h"
#include "lbm_generated/gpu/BasicRecursiveTimeStepGPU.h"
#endif

#include "mesh/blockforest/BlockExclusion.h"
#include "mesh/blockforest/BlockForestInitialization.h"
#include "mesh/blockforest/RefinementSelection.h"
#include "mesh/boundary/BoundaryInfo.h"
#include "mesh/boundary/BoundaryLocation.h"
#include "mesh/boundary/BoundaryLocationFunction.h"
#include "mesh/boundary/BoundarySetup.h"
#include "mesh/boundary/BoundaryUIDFaceDataSource.h"
#include "mesh/boundary/ColorToBoundaryMapper.h"
#include "mesh/blockforest/BlockWorkloadMemory.h"

#include "timeloop/SweepTimeloop.h"

#include "mesh_common/DistanceComputations.h"
#include "mesh_common/DistanceFunction.h"
#include "mesh_common/MeshIO.h"
#include "mesh_common/MeshOperations.h"
#include "mesh_common/TriangleMeshes.h"
#include "mesh_common/distance_octree/DistanceOctree.h"
#include "mesh_common/vtk/CommonDataSources.h"
#include "mesh_common/vtk/VTKMeshWriter.h"

#include "FlowAroundSphereInfoHeader.h"

using namespace walberla;

using StorageSpecification_T = lbm::FlowAroundSphereStorageSpecification;
using Stencil_T              = StorageSpecification_T::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;

using PdfField_T           = lbm_generated::PdfField< StorageSpecification_T >;
using FlagField_T          = FlagField< uint8_t >;
using BoundaryCollection_T = lbm::FlowAroundSphereBoundaryCollection< FlagField_T >;

using SweepCollection_T = lbm::FlowAroundSphereSweepCollection;

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
using GPUPdfField_T           = lbm_generated::GPUPdfField< StorageSpecification_T >;
using gpu::communication::UniformGPUScheme;
using gpu::communication::NonUniformGPUScheme;

using lbm_generated::UniformGeneratedGPUPdfPackInfo;
using lbm_generated::NonuniformGeneratedGPUPdfPackInfo;
#else
using PdfField_T           = lbm_generated::PdfField< StorageSpecification_T >;
using blockforest::communication::UniformBufferedScheme;
using blockforest::communication::NonUniformBufferedScheme;

using lbm_generated::UniformGeneratedPdfPackInfo;
using lbm_generated::NonuniformGeneratedPdfPackInfo;
#endif

using RefinementSelectionFunctor = SetupBlockForest::RefinementSelectionFunction;


template<typename MeshType>
void vertexToFaceColor(MeshType &mesh, const typename MeshType::Color &defaultColor) {
   WALBERLA_CHECK(mesh.has_vertex_colors())
   mesh.request_face_colors();

   for (auto faceIt = mesh.faces_begin(); faceIt != mesh.faces_end(); ++faceIt) {
      typename MeshType::Color vertexColor;

      bool useVertexColor = true;

      auto vertexIt = mesh.fv_iter(*faceIt);
      WALBERLA_ASSERT(vertexIt.is_valid())

      vertexColor = mesh.color(*vertexIt);

      ++vertexIt;
      while (vertexIt.is_valid() && useVertexColor) {
         if (vertexColor != mesh.color(*vertexIt)) useVertexColor = false;
         ++vertexIt;
      }

      mesh.set_color(*faceIt, useVertexColor ? vertexColor : defaultColor);
   }
}

class wallDistance
{
 public:

   wallDistance( const std::shared_ptr<mesh::TriangleMesh>& mesh ) : mesh_( mesh )
   {
      auto n_of_faces = mesh->n_faces();
      if(n_of_faces == 0)
         WALBERLA_LOG_INFO_ON_ROOT("The mesh contains no triangles!")

      size_t index_tr = 0;
      for(auto it_f = mesh->faces_begin(); it_f != mesh->faces_end(); ++it_f){
         auto vertexIt = mesh->fv_iter(*it_f);
         Vector3<real_t> vert(0.0);
         triangles_.emplace_back();
         while(vertexIt.is_valid()){
            auto v = mesh->point(*vertexIt);
            for(size_t i = 0; i!= 3; ++i){vert[i] = v[i];}
            triangles_[index_tr].push_back(vert);
            ++vertexIt;
         }
         ++index_tr;
      }
      if(triangles_.size() != n_of_faces){
         WALBERLA_LOG_INFO_ON_ROOT("Wrong number of found triangles!")
         WALBERLA_LOG_INFO_ON_ROOT("Return an empty triangles vector!")
         triangles_.clear();
      }
   }

   real_t operator()( const Cell& fluidCell, const Cell& boundaryCell, const shared_ptr< StructuredBlockForest >& SbF, IBlock& block ) const;
   bool computePointToMeshDistance(const Vector3<real_t> pf, const Vector3<real_t> ps, const std::vector< Vector3<real_t> >& triangle, real_t& q) const;
   Vector3<real_t> cell2GlobalCCPosition(const shared_ptr<StructuredBlockStorage>& blocks, const Cell loc, IBlock& block) const;

 private:

   const std::shared_ptr<mesh::TriangleMesh> mesh_;
   std::vector< std::vector< Vector3<real_t> > > triangles_;
}; // class wallDistance

real_t wallDistance::operator()( const Cell& fluidCell, const Cell& boundaryCell, const shared_ptr< StructuredBlockForest >& SbF, IBlock& block ) const
{

   real_t q = 0.0;

   const Vector3<real_t> pf = cell2GlobalCCPosition(SbF, fluidCell, block);
   const Vector3<real_t> ps = cell2GlobalCCPosition(SbF, boundaryCell, block);

   WALBERLA_CHECK_GREATER( triangles_.size(), std::size_t(0) )

   for (std::size_t x = 0; x != triangles_.size(); ++x){
      const bool intersects = computePointToMeshDistance(pf, ps, triangles_[x], q);
      if(intersects && q > -1.0){
         break;
      }
   }

   WALBERLA_CHECK_GREATER_EQUAL( q, real_t(0) )
   WALBERLA_CHECK_LESS_EQUAL( q, real_t(1) )
   return q;
}

bool wallDistance::computePointToMeshDistance(const Vector3<real_t> pf, const Vector3<real_t> ps, const std::vector< Vector3<real_t> >& triangle, real_t& q) const
{
   Vector3<real_t> v0;
   Vector3<real_t> e0;
   Vector3<real_t> e1;

   Vector3<real_t> normal;
   Vector3<real_t> dir;
   Vector3<real_t> intersection;
   Vector3<real_t> tmp;

   real_t a[2][2];
   real_t b[2];
   real_t num;
   real_t den;
   real_t t;
   real_t u;
   real_t v;
   real_t det;
   real_t upv;
   real_t norm;
   const real_t eps = 100.0 * std::numeric_limits<real_t>::epsilon();

   v0 = triangle[0];
   e0 = triangle[1] - v0; //triangle edge from triangle[1] to triangle[0]
   e1 = triangle[2] - v0; //triangle edge from triangle[2] to triangle[0]
   normal = e0 % e1;
   norm = std::sqrt(normal*normal);
   if (std::fabs(norm) < eps){
      q = -1;
      return false;
   }
   normal /= norm;
   dir = ps - pf;
   num = v0 * normal - pf * normal;
   den = dir * normal;
   t = num / den;
   //
   if(std::fabs(t) < eps || std::fabs(t-1) < eps){
      v0 = v0 + 2.0*eps*normal;
      num = v0 * normal - pf*normal;
      t = num / den;
   }
   //
   if (std::fabs(den) < eps){
      return std::fabs(num) < eps;
   }

   if(t < 0.0 || t > 1.0){
      q = -1.0;
      return false;
   }
   intersection = pf + dir*t;

   a[0][0] = e0 * e0;
   a[0][1] = e0 * e1;
   a[1][0] = a[0][1];
   a[1][1] = e1 * e1;
   tmp[0]  = intersection[0] - v0[0];
   tmp[1]  = intersection[1] - v0[1];
   tmp[2]  = intersection[2] - v0[2];
   b[0]    = tmp * e0;
   b[1]    = tmp * e1;
   det     = a[0][0] * a[1][1] - a[0][1] * a[1][0];
   u       = (a[1][1] * b[0] - a[0][1] * b[1]) / det;
   v       = (a[0][0] * b[1] - a[1][0] * b[0]) / det;
   upv     = u + v;
   const bool ueq0 = std::fabs(u) < eps;
   const bool ueq1 = std::fabs(u-1.0) < eps;
   const bool veq0 = std::fabs(v) < eps;
   const bool veq1 = std::fabs(v-1.0) < eps;
   const bool upveq1 = std::fabs(upv - 1) < eps;
   if( (u < 0.0 && !ueq0) || (u > 1.0 && !ueq1) || (v < 0.0 && !veq0) || (v>1.0 && !veq1)  || (upv > 1 && !upveq1)){
      q = -1;
      return false;
   }
   else{
      q = std::fabs(t);
      return true;
   }
   return true;
}

Vector3<real_t> wallDistance::cell2GlobalCCPosition(const shared_ptr<StructuredBlockStorage>& blocks, const Cell loc, IBlock& block) const{
   CellInterval globalCell(loc.x(),loc.y(),loc.z(),loc.x(),loc.y(),loc.z()); //At this level globalCell is a CellInterval which contains ONLY the cell provided in input. At the moment everything is local.
   blocks->transformBlockLocalToGlobalCellInterval(globalCell, block); //Now globalCell contains the global cell interval
   math::GenericAABB<real_t> const cellAABB = blocks->getAABBFromCellBB(globalCell,blocks->getLevel(block)); //cellAABB is the AABB around the cell (x_loc,y_loc,z_loc) in global coordinates!
   Vector3<real_t> p = cellAABB.center(); // this Vector contains the center of the precedent AABB, corresponding to the global c.c. of the cell (x_loc,y_loc,z_loc)
   blocks->mapToPeriodicDomain(p);
   return p;
}//end cell2GlobalCC_Position


//////////////////////
// Parameter Struct //
//////////////////////

struct Setup
{
   real_t Re;

   real_t viscosity; // on the coarsest grid
   real_t omega; // on the coarsest grid
   real_t inletVelocity;
   uint_t refinementLevels;
   uint_t timeSteps;

   void logSetup() const
   {
      WALBERLA_LOG_INFO_ON_ROOT( "FlowAroundSphere simulation properties:"
                                "\n   + Reynolds number:   " << Re <<
                                "\n   + kin. viscosity:    " << viscosity << " (on the coarsest grid)" <<
                                "\n   + relaxation rate:   " << std::setprecision(16) << omega << " (on the coarsest grid)" <<
                                "\n   + inlet velocity:    " << inletVelocity << " (in lattice units)" <<
                                "\n   + refinement Levels: " << refinementLevels <<
                                "\n   + #time steps:       " << timeSteps << " (on the coarsest grid)" )

   }
};


int main(int argc, char **argv) {
   walberla::Environment walberlaEnv(argc, argv);
   mpi::MPIManager::instance()->useWorldComm();
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   gpu::selectDeviceBasedOnMpiRank();
#endif

   logging::configureLogging(walberlaEnv.config());

   ///////////////////////
   /// PARAMETER INPUT ///
   ///////////////////////

   // read general simulation parameters
   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");

   const real_t referenceVelocity = parameters.getParameter<real_t>("referenceVelocity");
   const real_t maximumLatticeVelocity = parameters.getParameter<real_t>("maximumLatticeVelocity");
   const real_t referenceLength = parameters.getParameter<real_t>("referenceLength");
   const real_t viscosity = parameters.getParameter<real_t>("viscosity");
   const real_t simulationTime = parameters.getParameter<real_t>("simulationTime");
   const uint_t timeStepsForBenchmarking = parameters.getParameter<uint_t>("timeStepsForBenchmarking", 0);
   const real_t coarseMeshSize = parameters.getParameter<real_t>("coarseMeshSize");
   auto resolution = Vector3<real_t>(coarseMeshSize);
   const real_t relaxationRateOutlet = parameters.getParameter<real_t>("relaxationRateOutlet");
   const cell_idx_t spongeZoneStart = parameters.getParameter<cell_idx_t>("SpongeZoneStart");


   // read domain parameters
   auto domainParameters = walberlaEnv.config()->getOneBlock("DomainSetup");

   const std::string meshFile = domainParameters.getParameter<std::string>("meshFile");
   const Vector3<uint_t> blockSize = domainParameters.getParameter<Vector3<uint_t>>("cellsPerBlock");
   const Vector3<real_t> domainSize = domainParameters.getParameter<Vector3<real_t> >("domainSize");
   const Vector3<bool> periodicity = domainParameters.getParameter<Vector3<bool> >("periodic", Vector3<bool>(false));
   const bool weakScaling = domainParameters.getParameter<bool>("weakScaling", false);

   if(weakScaling)
   {
      auto blocks = math::getFactors3D( uint_c( MPIManager::instance()->numProcesses() ) );
      resolution[0] = domainSize[0] / real_c(blocks[0] * blockSize[0]);
      resolution[1] = domainSize[1] / real_c(blocks[1] * blockSize[1]);
      resolution[2] = domainSize[2] / real_c(blocks[2] * blockSize[2]);
      WALBERLA_LOG_INFO_ON_ROOT("Setting up a weak scaling benchmark with a resolution of: " << resolution)
   }

   const real_t reynoldsNumber = (referenceVelocity * referenceLength) / viscosity;
   const real_t Cu = referenceVelocity / maximumLatticeVelocity;
   const real_t Ct = resolution.min() / Cu;

   const real_t inletVelocity = referenceVelocity / Cu;
   const real_t viscosityLattice = viscosity * Ct / (resolution.min() * resolution.min());
   const real_t omega = real_c(1.0 / (3.0 * viscosityLattice + 0.5));
   uint_t timesteps = uint_c(simulationTime / Ct);
   if (timeStepsForBenchmarking > 0) {timesteps = timeStepsForBenchmarking;}

   const uint_t refinementLevels = domainParameters.getParameter< uint_t >( "refinementLevels");
   bool uniformGrid = true;
   auto numGhostLayers = uint_c(2);
   if (refinementLevels > 0) {uniformGrid = false; numGhostLayers = 2;}

   const uint_t numProcesses = domainParameters.getParameter< uint_t >( "numberProcesses");

   auto loggingParameters = walberlaEnv.config()->getOneBlock("Logging");
   const bool writeSetupForestAndReturn = loggingParameters.getParameter<bool>("writeSetupForestAndReturn", false);
   const bool writeDistanceOctree = loggingParameters.getParameter<bool>("writeDistanceOctree", false);
   const bool writeMeshBoundaries = loggingParameters.getParameter<bool>("writeMeshBoundaries", false);

   const Setup setup{reynoldsNumber, viscosity, omega, inletVelocity, refinementLevels, uint_c(simulationTime / Ct)};
   const real_t fineMeshSize = resolution.min() / real_c(std::pow(2, refinementLevels));

   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   WALBERLA_LOG_INFO_ON_ROOT("Using mesh from " << meshFile << ".")

   // read in mesh with vertex colors on a single process and broadcast it
   auto mesh = make_shared<mesh::TriangleMesh>();
   mesh->request_vertex_colors();
   mesh::readAndBroadcast(meshFile, *mesh);
   vertexToFaceColor(*mesh, mesh::TriangleMesh::Color(255, 255, 255));

   // building distanceOctree
   auto triDist = make_shared<mesh::TriangleDistance<mesh::TriangleMesh> >(mesh);
   auto distanceOctree = make_shared<mesh::DistanceOctree<mesh::TriangleMesh> >(triDist);

   // write distance octree to file
   if (writeDistanceOctree)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Writing distanceOctree to VTK file")
      distanceOctree->writeVTKOutput("distanceOctree");
   }

   ///////////////////////////
   /// CREATE BLOCK FOREST ///
   ///////////////////////////

   auto aabb = computeAABB(*mesh);
   auto domainScaling = Vector3<real_t> (domainSize[0] / aabb.xSize(), domainSize[1] / aabb.ySize(), domainSize[2] / aabb.zSize());
   aabb.scale( domainScaling );
   aabb.setCenter(Vector3<real_t >(5.0,0.0,0.0));

   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, resolution);

   bfc.setRootBlockExclusionFunction(mesh::makeExcludeMeshInterior(distanceOctree, resolution.min()));
   bfc.setBlockExclusionFunction(mesh::makeExcludeMeshInteriorRefinement(distanceOctree, fineMeshSize));

   auto meshWorkloadMemory = mesh::makeMeshWorkloadMemory( distanceOctree, resolution.min() );
   meshWorkloadMemory.setInsideCellWorkload(1);
   meshWorkloadMemory.setOutsideCellWorkload(1);
   bfc.setWorkloadMemorySUIDAssignmentFunction( meshWorkloadMemory );
   bfc.setPeriodicity(periodicity);

   if( !uniformGrid ) {
      WALBERLA_LOG_INFO_ON_ROOT("Using " << refinementLevels << " refinement levels. The resolution around the object is " << fineMeshSize << " m")
      bfc.setRefinementSelectionFunction(mesh::RefinementSelection(distanceOctree, refinementLevels, real_c(0.0), resolution.min()));
   }
   else {
      WALBERLA_LOG_INFO_ON_ROOT("Using a uniform Grid. The resolution around the object is " << fineMeshSize << " m")
   }

   if (writeSetupForestAndReturn)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Creating SetupBlockForest for " << numProcesses << " processes")
      auto setupForest = bfc.createSetupBlockForest( blockSize, numProcesses );
      WALBERLA_LOG_INFO_ON_ROOT("Writing SetupBlockForest to VTK file")
      WALBERLA_ROOT_SECTION() { setupForest->writeVTKOutput("SetupBlockForest"); }

      auto finalDomain = setupForest->getDomain();
      WALBERLA_LOG_INFO_ON_ROOT("Wind tunnel size in x-direction: " << finalDomain.xSize() << " m")
      WALBERLA_LOG_INFO_ON_ROOT("Wind tunnel size in y-direction: " << finalDomain.ySize() << " m")
      WALBERLA_LOG_INFO_ON_ROOT("Wind tunnel size in z-direction: " << finalDomain.zSize() << " m")

      WALBERLA_LOG_INFO_ON_ROOT("Blocks created: " << setupForest->getNumberOfBlocks())
      double totalCellUpdates( 0.0 );
      for (uint_t level = 0; level <= refinementLevels; level++)
      {
         const uint_t numberOfBlocks = setupForest->getNumberOfBlocks(level);
         const uint_t numberOfCells = numberOfBlocks * blockSize[0] * blockSize[1] * blockSize[2];
         totalCellUpdates += double_c( setup.timeSteps * math::uintPow2(level) ) * double_c( numberOfCells );
         WALBERLA_LOG_INFO_ON_ROOT("Level " << level << " Blocks: " << numberOfBlocks)
      }

      const uint_t totalNumberCells = setupForest->getNumberOfBlocks() * blockSize[0] * blockSize[1] * blockSize[2];

      const uint_t valuesPerCell = (Stencil_T::Q + VelocityField_T::F_SIZE + uint_c(2) * ScalarField_T::F_SIZE);
      const uint_t sizePerValue = sizeof(PdfField_T::value_type);
      const uint_t conversionFactor = uint_c(1024) * uint_c(1024) * uint_c(1024);
      const uint_t expectedMemory = (totalNumberCells * valuesPerCell * sizePerValue) / conversionFactor;

      WALBERLA_LOG_INFO_ON_ROOT( "Total number of cells will be " << totalNumberCells << " fluid cells (in total on all levels)")
      WALBERLA_LOG_INFO_ON_ROOT( "Expected total memory demand will be " << expectedMemory << " GB")
      WALBERLA_LOG_INFO_ON_ROOT( "The total cell updates after " << setup.timeSteps << " timesteps (on the coarse level) will be " << totalCellUpdates)

      setup.logSetup();

      WALBERLA_LOG_INFO_ON_ROOT("Ending program")
      return EXIT_SUCCESS;
   }
   auto blocks = bfc.createStructuredBlockForest(blockSize);

   auto finalDomain = blocks->getDomain();
   WALBERLA_LOG_INFO_ON_ROOT("Wind tunnel size in x-direction: " << finalDomain.xSize() << " m")
   WALBERLA_LOG_INFO_ON_ROOT("Wind tunnel size in y-direction: " << finalDomain.ySize() << " m")
   WALBERLA_LOG_INFO_ON_ROOT("Wind tunnel size in z-direction: " << finalDomain.zSize() << " m")

   WALBERLA_LOG_INFO_ON_ROOT("Blocks created: " << blocks->getNumberOfBlocks())
   for (uint_t level = 0; level <= refinementLevels; level++)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Level " << level << " Blocks: " << blocks->getNumberOfBlocks(level))
   }

   ////////////////////////////////////
   /// CREATE AND INITIALIZE FIELDS ///
   ////////////////////////////////////

   // create fields
   const StorageSpecification_T StorageSpec = StorageSpecification_T();

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   auto allocator = make_shared< gpu::HostFieldAllocator<real_t> >();
   const BlockDataID pdfFieldID = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, numGhostLayers, field::fzyx);
   const BlockDataID velFieldID = field::addToStorage< VelocityField_T >(blocks, "velocity", real_c(0.0), field::fzyx, numGhostLayers);
   const BlockDataID densityFieldID = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, numGhostLayers);
   const BlockDataID omegaFieldID = field::addToStorage<ScalarField_T>(blocks, "omega", omega, field::fzyx, numGhostLayers);
   const BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "Boundary Flag Field", uint_c(2));

   const BlockDataID pdfFieldGPUID = lbm_generated::addGPUPdfFieldToStorage< PdfField_T >(blocks, pdfFieldID, StorageSpec, "pdfs on GPU", true);
   const BlockDataID velFieldGPUID = gpu::addGPUFieldToStorage< VelocityField_T >(blocks, velFieldID, "velocity on GPU", true);
   const BlockDataID densityFieldGPUID = gpu::addGPUFieldToStorage< ScalarField_T >(blocks, densityFieldID, "density on GPU", true);
   const BlockDataID omegaFieldGPUID = gpu::addGPUFieldToStorage< ScalarField_T >(blocks, omegaFieldID, "omega on GPU", true);

   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#else
   const BlockDataID pdfFieldID = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, numGhostLayers, field::fzyx);
   const BlockDataID velFieldID = field::addToStorage< VelocityField_T >(blocks, "vel", real_c(0.0), field::fzyx, numGhostLayers);
   const BlockDataID densityFieldID = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, numGhostLayers);
   const BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "Boundary Flag Field", uint_c(2));
   const BlockDataID omegaFieldID = field::addToStorage<ScalarField_T>(blocks, "omega", omega, field::fzyx, numGhostLayers);
#endif

   WALBERLA_MPI_BARRIER()

   const Cell innerOuterSplit = Cell(parameters.getParameter< Vector3< cell_idx_t > >("innerOuterSplit", Vector3< cell_idx_t >(1, 1, 1)));
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const Vector3<int64_t> gpuBlockSize = parameters.getParameter<Vector3<int64_t>>("gpuBlockSize");
   SweepCollection_T sweepCollection(blocks, omegaFieldGPUID, pdfFieldGPUID, densityFieldGPUID, velFieldGPUID, gpuBlockSize[0], gpuBlockSize[1], gpuBlockSize[2], innerOuterSplit);
   for (auto& block : *blocks)
   {
      sweepCollection.initialise(&block, cell_idx_c(numGhostLayers - uint_c(1)));
   }
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
#else
   SweepCollection_T sweepCollection(blocks, omegaFieldID, pdfFieldID, densityFieldID, velFieldID, innerOuterSplit);
   for (auto& block : *blocks)
   {
      sweepCollection.initialise(&block, cell_idx_c(numGhostLayers));
   }
#endif
   WALBERLA_MPI_BARRIER()
   const real_t omega_diff = omega - relaxationRateOutlet;
   const real_t offset = std::abs(finalDomain.xMin());
   const real_t sizeX = finalDomain.xSize();
   const real_t spongeZoneLength = sizeX - real_c(spongeZoneStart);

   for (auto &iBlock: *blocks) {
      auto & block = dynamic_cast< blockforest::Block & >( iBlock );
      const uint_t level = block.getLevel();

      auto omegaField = block.getData< ScalarField_T >(omegaFieldID);
      WALBERLA_FOR_ALL_CELLS_XYZ(
         omegaField, Cell globalCell;
         blocks->transformBlockLocalToGlobalCell(globalCell, block, Cell(x, y, z));
         Vector3<real_t> cellCenter = blocks->getCellCenter(globalCell, level);
         cellCenter[0] += offset;
         // Adaption due to sponge layer
         if(cellCenter[0] > spongeZoneStart)
         {
            omegaField->get(x, y, z) = omega - omega_diff * ((real_c(cellCenter[0]) - real_c(spongeZoneStart)) / spongeZoneLength);
         }
         // Adaption due to level scaling
         const real_t level_scale_factor = real_c(uint_t(1) << level);
         const real_t one                = real_c(1.0);
         const real_t half               = real_c(0.5);
         const real_t oldOmega = omegaField->get(x, y, z);

         omegaField->get(x, y, z) = real_c(oldOmega / (level_scale_factor * (-oldOmega * half + one) + oldOmega * half));
      )
   }

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   gpu::fieldCpy< gpu::GPUField< real_t >, ScalarField_T >(blocks, omegaFieldGPUID, omegaFieldID);
   WALBERLA_LOG_INFO_ON_ROOT("Setting up communication")
   std::shared_ptr<UniformGPUScheme< CommunicationStencil_T >> uniformCommunication;
   std::shared_ptr<UniformGeneratedGPUPdfPackInfo< GPUPdfField_T >> uniformPackInfo;

   std::shared_ptr<NonUniformGPUScheme< CommunicationStencil_T >> nonUniformCommunication;
   std::shared_ptr<NonuniformGeneratedGPUPdfPackInfo<GPUPdfField_T >> nonUniformPackInfo;
   if (uniformGrid){
      uniformCommunication = std::make_shared< UniformGPUScheme< CommunicationStencil_T > >(blocks, false);
      uniformPackInfo = std::make_shared<lbm_generated::UniformGeneratedGPUPdfPackInfo< GPUPdfField_T >>(pdfFieldGPUID);
      uniformCommunication->addPackInfo(uniformPackInfo);
   } else {
      nonUniformCommunication = std::make_shared< NonUniformGPUScheme< CommunicationStencil_T > >(blocks);
      nonUniformPackInfo = lbm_generated::setupNonuniformGPUPdfCommunication< GPUPdfField_T >(blocks, pdfFieldGPUID);
      nonUniformCommunication->addPackInfo(nonUniformPackInfo);
   }
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#else
   WALBERLA_LOG_INFO_ON_ROOT("Setting up communication...")
   std::shared_ptr<UniformBufferedScheme< CommunicationStencil_T >> uniformCommunication;
   std::shared_ptr<UniformGeneratedPdfPackInfo< PdfField_T >> uniformPackInfo;

   std::shared_ptr<NonUniformBufferedScheme< CommunicationStencil_T >> nonUniformCommunication;
   std::shared_ptr<NonuniformGeneratedPdfPackInfo<PdfField_T >> nonUniformPackInfo;
   if (uniformGrid){
      uniformCommunication = std::make_shared< UniformBufferedScheme< CommunicationStencil_T > >(blocks);
      uniformPackInfo= std::make_shared<lbm_generated::UniformGeneratedPdfPackInfo< PdfField_T >>(pdfFieldID);
      uniformCommunication->addPackInfo(uniformPackInfo);
   } else {
      nonUniformCommunication = std::make_shared< NonUniformBufferedScheme< CommunicationStencil_T > >(blocks);
      nonUniformPackInfo = lbm_generated::setupNonuniformPdfCommunication< PdfField_T >(blocks, pdfFieldID);
      nonUniformCommunication->addPackInfo(nonUniformPackInfo);
   }
#endif
   WALBERLA_MPI_BARRIER()
   WALBERLA_LOG_INFO_ON_ROOT("Setting up communication done")


   /////////////////////////
   /// BOUNDARY HANDLING ///
   /////////////////////////
   WALBERLA_LOG_INFO_ON_ROOT("Start BOUNDARY HANDLING")
   // create and initialize boundary handling
   const FlagUID fluidFlagUID("Fluid");
   static const walberla::BoundaryUID wallFlagUID("NoSlipBouzidi");
   for (auto &block: *blocks) {
      auto flagField = block.getData<FlagField_T>( flagFieldID );
      flagField->registerFlag(FlagUID("NoSlipBouzidi"));
   }

   // write mesh info to file
   if (writeMeshBoundaries) {
      // set NoSlip UID to boundaries that we colored
      mesh::ColorToBoundaryMapper<mesh::TriangleMesh> colorToBoundaryMapper((mesh::BoundaryInfo(wallFlagUID)));
      colorToBoundaryMapper.set(mesh::TriangleMesh::Color(255, 255, 255), mesh::BoundaryInfo(wallFlagUID));
      // mark boundaries
      auto boundaryLocations = colorToBoundaryMapper.addBoundaryInfoToMesh(*mesh);

      mesh::VTKMeshWriter<mesh::TriangleMesh> meshWriter(mesh, "meshBoundaries", 1);
      meshWriter.addDataSource(make_shared<mesh::BoundaryUIDFaceDataSource<mesh::TriangleMesh> >(boundaryLocations));
      meshWriter.addDataSource(make_shared<mesh::ColorFaceDataSource<mesh::TriangleMesh> >());
      meshWriter.addDataSource(make_shared<mesh::ColorVertexDataSource<mesh::TriangleMesh> >());
      meshWriter();
   }

   WALBERLA_LOG_INFO_ON_ROOT("Voxelize mesh")
   mesh::BoundarySetup boundarySetup(blocks, makeMeshDistanceFunction(distanceOctree), numGhostLayers, false);
   WALBERLA_LOG_INFO_ON_ROOT("Voxelize mesh done")

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");
   geometry::initBoundaryHandling<FlagField_T>(*blocks, flagFieldID, boundariesConfig);
   boundarySetup.setFlag<FlagField_T>(flagFieldID, FlagUID("NoSlipBouzidi"), mesh::BoundarySetup::INSIDE);
   geometry::setNonBoundaryCellsToDomain<FlagField_T>(*blocks, flagFieldID, fluidFlagUID, cell_idx_c(numGhostLayers));

   const wallDistance wallDistanceCallback{mesh};
   std::function<real_t(const Cell&, const Cell&, const shared_ptr< StructuredBlockForest >&, IBlock&) > wallDistanceFunctor = wallDistanceCallback;

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   BoundaryCollection_T boundaryCollection(blocks, flagFieldID, pdfFieldGPUID, fluidFlagUID, inletVelocity, wallDistanceFunctor, pdfFieldID);
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#else
   BoundaryCollection_T boundaryCollection(blocks, flagFieldID, pdfFieldID, fluidFlagUID, inletVelocity, wallDistanceFunctor);
#endif
   WALBERLA_MPI_BARRIER()
   WALBERLA_LOG_INFO_ON_ROOT("BOUNDARY HANDLING done")

   //////////////////////////////////
   /// SET UP SWEEPS AND TIMELOOP ///
   //////////////////////////////////
   WALBERLA_LOG_INFO_ON_ROOT("Start SWEEPS AND TIMELOOP")

   // create time loop
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);
   std::shared_ptr<lbm_generated::BasicRecursiveTimeStepGPU<GPUPdfField_T, SweepCollection_T, BoundaryCollection_T>> LBMRefinement;

   if (!uniformGrid) {
      LBMRefinement = std::make_shared<lbm_generated::BasicRecursiveTimeStepGPU<GPUPdfField_T , SweepCollection_T, BoundaryCollection_T> > (
         blocks, pdfFieldGPUID, sweepCollection, boundaryCollection, nonUniformCommunication,
         nonUniformPackInfo);
      LBMRefinement->addRefinementToTimeLoop(timeloop);

   }
   else {
      WALBERLA_LOG_INFO_ON_ROOT("Using uniform Grid")
      timeloop.add() << BeforeFunction(uniformCommunication->getCommunicateFunctor(), "Communication")
                     << Sweep(boundaryCollection.getSweep(BoundaryCollection_T::ALL), "Boundary Conditions");
      timeloop.add() << Sweep(sweepCollection.streamCollide(SweepCollection_T::ALL), "LBM StreamCollide");
   }
#else
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);
   std::shared_ptr<lbm_generated::BasicRecursiveTimeStep<PdfField_T, SweepCollection_T, BoundaryCollection_T>> LBMRefinement;

   if (!uniformGrid) {
      LBMRefinement = std::make_shared<lbm_generated::BasicRecursiveTimeStep<PdfField_T, SweepCollection_T, BoundaryCollection_T> > (
         blocks, pdfFieldID, sweepCollection, boundaryCollection, nonUniformCommunication,
         nonUniformPackInfo);
      LBMRefinement->addRefinementToTimeLoop(timeloop);

   }
   else {
      WALBERLA_LOG_INFO_ON_ROOT("Using uniform Grid")
      timeloop.add() << BeforeFunction(uniformCommunication->getCommunicateFunctor(), "Communication")
                     << Sweep(boundaryCollection.getSweep(BoundaryCollection_T::ALL), "Boundary Conditions");
      timeloop.add() << Sweep(sweepCollection.streamCollide(SweepCollection_T::ALL), "LBM StreamCollide");
   }
#endif
   //////////////////
   /// VTK OUTPUT ///
   //////////////////
   WALBERLA_LOG_INFO_ON_ROOT("SWEEPS AND TIMELOOP done")

   const uint_t vtkWriteFrequency = parameters.getParameter<uint_t>("vtkWriteFrequency", 0);

   auto VTKWriter = walberlaEnv.config()->getOneBlock("VTKWriter");
   const bool writeVelocity = VTKWriter.getParameter<bool>("velocity");
   const bool writeDensity = VTKWriter.getParameter<bool>("density");
   const bool writeOmega = VTKWriter.getParameter<bool>("omega");
   const bool writeFlag = VTKWriter.getParameter<bool>("flag");
   const bool writeOnlySlice = VTKWriter.getParameter<bool>("writeOnlySlice", true);
   if (vtkWriteFrequency > 0)
   {
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_FlowAroundSphere",
                                                      "simulation_step", false, true, true, false, 0);

      vtkOutput->addBeforeFunction([&]() {
         for (auto& block : *blocks)
         {
            sweepCollection.calculateMacroscopicParameters(&block);
         }

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         gpu::fieldCpy< VelocityField_T, gpu::GPUField< real_t > >(blocks, velFieldID, velFieldGPUID);
         gpu::fieldCpy< ScalarField_T , gpu::GPUField< real_t > >(blocks, densityFieldID, densityFieldGPUID);
#endif
      });


      if (writeOnlySlice)
      {
         const AABB sliceAABB(finalDomain.xMin(), finalDomain.yMin(), finalDomain.center()[2] - resolution.min(), finalDomain.xMax(), finalDomain.yMax(), finalDomain.center()[2] + resolution.min());
         vtkOutput->addCellInclusionFilter(vtk::AABBCellFilter(sliceAABB));
      }

      if (writeVelocity)
      {
         auto velWriter = make_shared<field::VTKWriter<VelocityField_T> >(velFieldID, "velocity");
         vtkOutput->addCellDataWriter(velWriter);
      }
      if (writeDensity)
      {
         auto densityWriter = make_shared<field::VTKWriter<ScalarField_T> >(densityFieldID, "density");
         vtkOutput->addCellDataWriter(densityWriter);
      }
      if (writeOmega)
      {
         auto omegaWriter = make_shared<field::VTKWriter<ScalarField_T> >(omegaFieldID, "omega");
         vtkOutput->addCellDataWriter(omegaWriter);
      }
      if (writeFlag)
      {
         auto flagWriter = make_shared<field::VTKWriter<FlagField_T> >(flagFieldID, "flag");
         vtkOutput->addCellDataWriter(flagWriter);
      }
      timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   }

   // log remaining time
   const real_t remainingTimeLoggerFrequency = parameters.getParameter<real_t>("remainingTimeLoggerFrequency", 3.0); // in seconds
   if (uint_c(remainingTimeLoggerFrequency) > 0)
   {
      timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency), "remaining time logger");
   }

   // LBM stability check
   auto CheckerParameters = walberlaEnv.config()->getOneBlock("StabilityChecker");
   const uint_t checkFrequency = CheckerParameters.getParameter<uint_t>("checkFrequency", uint_t(0));
   if (checkFrequency > 0) {
      auto checkFunction = [](PdfField_T::value_type value) {return value < math::abs(PdfField_T::value_type(10));};
      timeloop.addFuncAfterTimeStep( makeSharedFunctor( field::makeStabilityChecker< PdfField_T, FlagField_T >( walberlaEnv.config(), blocks, pdfFieldID, flagFieldID, fluidFlagUID, checkFunction) ),"Stability check" );
   }


   WALBERLA_LOG_INFO_ON_ROOT("Execute single timestep to fully complete the preprocessing")
   // Do a single timestep to make sure all setups are completed before benchmarking
   timeloop.singleStep();
   timeloop.setCurrentTimeStepToZero();

   WALBERLA_MPI_BARRIER()
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif
   WALBERLA_LOG_INFO_ON_ROOT("Execute single timestep to fully complete the preprocessing done")


   //////////////////////
   /// RUN SIMULATION ///
   //////////////////////
   const lbm_generated::PerformanceEvaluation<FlagField_T> performance(blocks, flagFieldID, fluidFlagUID);
   field::CellCounter< FlagField_T > fluidCells( blocks, flagFieldID, fluidFlagUID );
   fluidCells();

   WALBERLA_LOG_INFO_ON_ROOT( "Simulating FlowAroundSphere with " << fluidCells.numberOfCells() << " fluid cells (in total on all levels)")
   setup.logSetup();
   WcTimingPool timeloopTiming;
   WcTimer simTimer;

   WALBERLA_MPI_BARRIER()
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif

   simTimer.start();
   timeloop.run(timeloopTiming);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif
   WALBERLA_MPI_BARRIER()
   simTimer.end();

   WALBERLA_LOG_INFO_ON_ROOT("Simulation finished")
   real_t time = simTimer.max();
   WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
   performance.logResultOnRoot(timesteps, time);

   const auto reducedTimeloopTiming = timeloopTiming.getReduced();
   WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)

   printResidentMemoryStatistics();

   return EXIT_SUCCESS;
}
