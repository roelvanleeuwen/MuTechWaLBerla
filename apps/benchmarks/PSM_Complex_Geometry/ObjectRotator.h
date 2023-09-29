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
//! \file ObjectRotator.h
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================
#pragma once

#include "core/math/Constants.h"
#include "blockforest/all.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/FieldCopy.h"
#include "gpu/GPUWrapper.h"
#include "gpu/HostFieldAllocator.h"
#include "gpu/ParallelStreams.h"
#include "gpu/communication/UniformGPUScheme.h"
#endif

#include "geometry/containment_octree/ContainmentOctree.h"

#include "mesh_common/DistanceComputations.h"
#include "mesh_common/DistanceFunction.h"
#include "mesh_common/MatrixVectorOperations.h"
#include "mesh_common/MeshIO.h"
#include "mesh_common/MeshOperations.h"
#include "mesh_common/TriangleMeshes.h"
#include "mesh_common/distance_octree/DistanceOctree.h"
#include "mesh_common/vtk/CommonDataSources.h"
#include "mesh_common/vtk/VTKMeshWriter.h"
#include "mesh/blockforest/BlockExclusion.h"
#include "mesh/blockforest/BlockForestInitialization.h"
#include "mesh/boundary/BoundaryInfo.h"
#include "mesh/boundary/BoundarySetup.h"
#include "mesh/boundary/BoundaryUIDFaceDataSource.h"
#include "mesh/boundary/ColorToBoundaryMapper.h"

#include "GeometryOctree.h"

#define CROSS(dest,v1,v2) \
         dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
         dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
         dest[2]=v1[0]*v2[1]-v1[1]*v2[0];

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2) \
         dest[0]=v1[0]-v2[0]; \
         dest[1]=v1[1]-v2[1]; \
         dest[2]=v1[2]-v2[2];

#define ADD(dest,v1,v2) \
         dest[0]=v1[0]+v2[0]; \
         dest[1]=v1[1]+v2[1]; \
         dest[2]=v1[2]+v2[2];


namespace walberla
{
typedef field::GhostLayerField< real_t, 1 > ScalarField_T;
using fracSize = real_t;
typedef field::GhostLayerField< fracSize, 1 > FracField_T;




class ObjectRotator
{
   typedef std::function< real_t(const Vector3< real_t >&) > DistanceFunction;
   typedef field::GhostLayerField< real_t, 3 > VectorField_T;

 public:
   ObjectRotator(shared_ptr< StructuredBlockForest >& blocks, shared_ptr< mesh::TriangleMesh >& mesh, const BlockDataID objectVelocityId,
                 const real_t rotationAngle, const uint_t frequency, Vector3<uint_t> rotationAxis, shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>>& distOctree,
                 std::string meshName, uint_t maxSuperSamplingDepth, const bool rotate = true)
      : blocks_(blocks), mesh_(mesh), objectVelocityId_(objectVelocityId),
        rotationAngle_(rotationAngle), frequency_(frequency), rotationAxis_(rotationAxis), distOctree_(distOctree),
        meshName_(meshName), maxSuperSamplingDepth_(maxSuperSamplingDepth), rotate_(rotate)
   {
      fractionFieldId_ = field::addToStorage< FracField_T >(blocks, "fractionField_" + meshName_, fracSize(0.0), field::fzyx, uint_c(1));
      currentAngle = 0.0;
      readFile();
      meshCenter = computeCentroid(*mesh_);
      initObjectVelocityField();

      auto aabbMesh = computeAABB(*mesh_);
      WcTimer octreeTimer;
      octreeTimer.start();
      geometryOctree_ = make_shared<GeometryOctreeNode>(aabbMesh, 0, 1, triangles_, numTriangles_, vertices_);
      octreeTimer.end();
      WALBERLA_LOG_INFO_ON_ROOT("Built GeometryOctree in " << octreeTimer.max() << "s")
      WALBERLA_LOG_INFO("Geometry aaBB is " << geometryOctree_->getBoxAABB() << ", it has " << geometryOctree_->getContainedTriangles().size() << " triangles and " << geometryOctree_->getChildNodes().size() << " child nodes")
      WALBERLA_LOG_INFO("Num triangles is " << numTriangles_)


      WcTimer simTimer;
      WALBERLA_LOG_INFO_ON_ROOT("Start Voxelization")
      simTimer.start();
      //voxelizeRayTracing();
      voxelizeRayTracingWithGeometryOctree();
      simTimer.end();
      WALBERLA_LOG_INFO_ON_ROOT("Finished Voxelization in " << simTimer.max() << "s")
   }

   void operator()(uint_t timestep) {
      if (timestep % frequency_ == 0)
      {
         if(rotate_) {
            //rotate();
            resetFractionField();
            //getFractionFieldFromMesh();
            //voxelizeRayTracing();
            voxelizeRayTracingWithGeometryOctree();
            currentAngle += rotationAngle_;


         }
      }
   }

   void rotate()
   {
      const Vector3< mesh::TriangleMesh::Scalar > axis_foot(meshCenter[0], meshCenter[1], meshCenter[2]);
      mesh::rotate(*mesh_, rotationAxis_, rotationAngle_, axis_foot);

      distOctree_ = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(
         make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh_));
   }

   void resetFractionField()
   {
      for (auto& block : *blocks_)
      {
         auto fractionField = block.getData< FracField_T >(fractionFieldId_);
         CellInterval blockCi = fractionField->xyzSizeWithGhostLayer();
         std::fill(fractionField->beginSliceXYZ(blockCi), fractionField->end(), 0.0);
      }
   }

   void initObjectVelocityField() {
      for (auto& block : *blocks_)
      {
         auto aabbMesh = computeAABB(*mesh_);
         auto level = blocks_->getLevel(block);
         auto cellBB = blocks_->getCellBBFromAABB( aabbMesh, level );

         auto objVelField = block.getData< VectorField_T >(objectVelocityId_);
         Vector3< real_t > angularVel;


         if (rotate_ == false || frequency_ == 0 || rotationAngle_ <= 0.0) {
            angularVel = Vector3< real_t >(0,0,0);
         }
         else
            angularVel = Vector3< real_t > (rotationAxis_[0] * rotationAngle_ / real_c(frequency_), rotationAxis_[1] * rotationAngle_ / real_c(frequency_), rotationAxis_[2] * rotationAngle_ / real_c(frequency_));

         const real_t dx = blocks_->dx(level);
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(objVelField,
            Cell cell(x,y,z);
            blocks_->transformBlockLocalToGlobalCell(cell, block);

            // Set velocity only in aabb of mesh for x dir
            if(cell[0] < cellBB.xMin() || cell[0] > cellBB.xMax())
               continue;

            Vector3< real_t > cellCenter = blocks_->getCellCenter(cell, level);
            Vector3< real_t > distance((cellCenter[0] - meshCenter[0]) / dx, (cellCenter[1] - meshCenter[1]) / dx, (cellCenter[2] - meshCenter[2]) / dx);
            real_t velX = angularVel[1] * distance[2] - angularVel[2] * distance[1];
            real_t velY = angularVel[2] * distance[0] - angularVel[0] * distance[2];
            real_t velZ = angularVel[0] * distance[1] - angularVel[1] * distance[0];

            blocks_->transformGlobalToBlockLocalCell(cell, block);
            objVelField->get(cell, 0) = velX;
            objVelField->get(cell, 1) = velY;
            objVelField->get(cell, 2) = velZ;

         )
      }
   }


   fracSize recursiveSuperSampling(const shared_ptr<MeshDistanceFunction<mesh::DistanceOctree<mesh::TriangleMesh>>> &distFunct, Vector3<real_t> cellCenter, real_t dx, uint_t depth) {
      //if only one cell left, split cell into 8 cell centers to get these cellCenter distances
      fracSize fraction = 0.0;
      const fracSize fracValue = fracSize(1.0 / pow(8, real_c(depth)));
      const real_t offsetMod = real_c(1.0 / pow(2, real_c(depth+2)));

      if(depth == maxSuperSamplingDepth_) {
         if((*distFunct)( cellCenter )  < real_t(0))
            fraction = fracValue;
      }
      else {
         std::vector<int> xOffset{-1,-1,-1,-1, 1, 1, 1, 1};
         std::vector<int> yOffset{-1,-1, 1, 1,-1,-1, 1, 1};
         std::vector<int> zOffset{-1, 1,-1, 1,-1, 1,-1, 1};
         Vector3<real_t> octreeCenter;
         for (uint_t i = 0; i < 8; ++i) {
            octreeCenter = Vector3<real_t>(cellCenter[0] + xOffset[i] * dx * offsetMod, cellCenter[1] + yOffset[i] * dx * offsetMod, cellCenter[2] + zOffset[i] * dx * offsetMod);
            fraction += recursiveSuperSampling(distFunct, octreeCenter, dx, depth+1);
         }
      }
      return fraction;
   }

   void getFractionFieldFromMesh()
   {
      auto aabbMesh = computeAABB(*mesh_);
      const auto distFunct = make_shared<MeshDistanceFunction<mesh::DistanceOctree<mesh::TriangleMesh>>>( distOctree_ );

      for (auto& block : *blocks_)
      {
         FracField_T* fractionField = block.getData< FracField_T >(fractionFieldId_);
         auto level = blocks_->getLevel(block);
         const real_t dx = blocks_->dx(level);
         auto cellBBMesh = blocks_->getCellBBFromAABB( aabbMesh, level );

         CellInterval blockCi = fractionField->xyzSizeWithGhostLayer();
         blocks_->transformBlockLocalToGlobalCellInterval(blockCi, block);
         cellBBMesh.intersect(blockCi);

         std::queue< CellInterval > ciQueue;
         ciQueue.push(cellBBMesh);

         while (!ciQueue.empty())
         {
            const CellInterval& curCi = ciQueue.front();

            WALBERLA_ASSERT(!curCi.empty(), "Cell Interval: " << curCi);

            const AABB curAABB = blocks_->getAABBFromCellBB(curCi, level);

            WALBERLA_ASSERT(!curAABB.empty(), "AABB: " << curAABB);

            Vector3< real_t > cellCenter = curAABB.center();
            blocks_->mapToPeriodicDomain(cellCenter);
            const real_t sqSignedDistance = (*distFunct)(cellCenter);

            if (curCi.numCells() == uint_t(1))
            {
               fracSize fraction = recursiveSuperSampling(distFunct, cellCenter, dx, 0);
               Cell localCell;
               blocks_->transformGlobalToBlockLocalCell(localCell, block, curCi.min());
               fractionField->get(localCell) = fraction;

               ciQueue.pop();
               continue;
            }

            const real_t circumRadius   = curAABB.sizes().length() * real_t(0.5);
            const real_t sqCircumRadius = circumRadius * circumRadius;

            // the cell interval is fully covered by the mesh
            if (sqSignedDistance < -sqCircumRadius)
            {
               CellInterval localCi;
               blocks_->transformGlobalToBlockLocalCellInterval(localCi, block, curCi);
               std::fill(fractionField->beginSliceXYZ(localCi), fractionField->end(), 1.0);

               ciQueue.pop();
               continue;
            }
            // the cell interval is fully outside of mesh
            if (sqSignedDistance > sqCircumRadius)
            {
               ciQueue.pop();
               continue;
            }

            WALBERLA_ASSERT_GREATER(curCi.numCells(), uint_t(1));
            divideAndPushCellInterval(curCi, ciQueue);

            ciQueue.pop();
         }
      }
   }


   void divideAndPushCellInterval(const CellInterval& ci, std::queue< CellInterval >& outputQueue)
   {
      WALBERLA_ASSERT(!ci.empty());

      Cell newMax(ci.xMin() + std::max(cell_idx_c(ci.xSize()) / cell_idx_t(2) - cell_idx_t(1), cell_idx_t(0)),
                  ci.yMin() + std::max(cell_idx_c(ci.ySize()) / cell_idx_t(2) - cell_idx_t(1), cell_idx_t(0)),
                  ci.zMin() + std::max(cell_idx_c(ci.zSize()) / cell_idx_t(2) - cell_idx_t(1), cell_idx_t(0)));

      WALBERLA_ASSERT(ci.contains(newMax));

      Cell newMin(newMax[0] + cell_idx_c(1), newMax[1] + cell_idx_c(1), newMax[2] + cell_idx_c(1));

      outputQueue.push(CellInterval(ci.xMin(), ci.yMin(), ci.zMin(), newMax[0], newMax[1], newMax[2]));
      if (newMin[2] <= ci.zMax())
         outputQueue.push(CellInterval(ci.xMin(), ci.yMin(), newMin[2], newMax[0], newMax[1], ci.zMax()));
      if (newMin[1] <= ci.yMax())
      {
         outputQueue.push(CellInterval(ci.xMin(), newMin[1], ci.zMin(), newMax[0], ci.yMax(), newMax[2]));
         if (newMin[2] <= ci.zMax())
            outputQueue.push(CellInterval(ci.xMin(), newMin[1], newMin[2], newMax[0], ci.yMax(), ci.zMax()));
      }
      if (newMin[0] <= ci.xMax())
      {
         outputQueue.push(CellInterval(newMin[0], ci.yMin(), ci.zMin(), ci.xMax(), newMax[1], newMax[2]));
         if (newMin[2] <= ci.zMax())
            outputQueue.push(CellInterval(newMin[0], ci.yMin(), newMin[2], ci.xMax(), newMax[1], ci.zMax()));
         if (newMin[1] <= ci.yMax())
         {
            outputQueue.push(CellInterval(newMin[0], newMin[1], ci.zMin(), ci.xMax(), ci.yMax(), newMax[2]));
            if (newMin[2] <= ci.zMax())
               outputQueue.push(CellInterval(newMin[0], newMin[1], newMin[2], ci.xMax(), ci.yMax(), ci.zMax()));
         }
      }
   }

   BlockDataID getObjectFractionFieldID() {
      return fractionFieldId_;
   }

   void readFile() {
      std::ifstream meshFile (meshName_ + ".obj");
      std::string line;
      std::stringstream stream;
      std::string keyword;

      std::vector<Vector3<real_t>> vertices;
      std::vector<Vector3<uint>> triangles;

      Vector3<real_t> vertex;
      Vector3<uint_t> triangle;

      if (meshFile.is_open())
      {
         while ( getline (meshFile,line) )
         {
            stream.str(line);
            stream.clear();

            stream >> keyword;
            if (keyword == "v") {
               stream >> vertex[0] >> vertex[1] >> vertex[2];
               vertices.push_back(vertex);
            }
            else if(keyword == "f") {
               std::string vertexSrt;
               size_t found;
               for (uint i = 0; i < 3; ++i) {
                  stream >> vertexSrt;
                  found = vertexSrt.find("/");
                  triangle[i] = uint(stoi(vertexSrt.substr(0,found))-1);
               }
               triangles.push_back(triangle);
            }
         }
         meshFile.close();
         numVertices_ = uint_t(vertices.size());
         vertices_ = (real_t *) std::malloc( sizeof(real_t) * 3 * numVertices_);
         for ( uint_t i = 0; i < numVertices_; ++i) {
            for ( uint_t j = 0; j < 3; ++j)
            {
               vertices_[i * 3 + j] = vertices[i][j];
               vertices_[i * 3 + j] = vertices[i][j];
               vertices_[i * 3 + j] = vertices[i][j];
            }
         }
         numTriangles_ = uint_t(triangles.size());
         triangles_ = (uint_t *) std::malloc( sizeof(uint_t) * 3 * numTriangles_);
         for ( uint_t i = 0; i < numTriangles_; ++i) {
            for ( uint_t j = 0; j < 3; ++j)
            {
               triangles_[i * 3 + j] = triangles[i][j];
               triangles_[i * 3 + j] = triangles[i][j];
               triangles_[i * 3 + j] = triangles[i][j];
            }
         }
      }
      else
         WALBERLA_ABORT("Couldn't open mesh file")
   }

   bool RayIntersectsTriangle(real_t rayOrigin[3], real_t rayVector[3], real_t inTriangle[3][3])
   {
      const real_t EPSILON = 0.00000000001f;
      real_t vertex0[3]    = { inTriangle[0][0], inTriangle[0][1], inTriangle[0][2] };
      real_t vertex1[3]    = { inTriangle[1][0], inTriangle[1][1], inTriangle[1][2] };
      real_t vertex2[3]    = { inTriangle[2][0], inTriangle[2][1], inTriangle[2][2] };
      real_t edge1[3], edge2[3], h[3], s[3], q[3];
      real_t a, f, u, v;
      SUB(edge1, vertex1, vertex0)
      SUB(edge2, vertex2, vertex0)
      CROSS(h, rayVector, edge2)
      a = DOT(edge1, h);
      f = 1.0f / a;
      SUB(s, rayOrigin, vertex0)
      u = f * DOT(s, h);
      if (u < 0.0 || u > 1.0) return false;
      CROSS(q, s, edge1)
      v = f * DOT(rayVector, q);
      if (v < 0.0 || u + v > 1.0) return false;
      real_t t = f * DOT(edge2, q);
      if (t > EPSILON) // ray intersection
         return true;
      else // This means that there is a line intersection but not a ray intersection.
         return false;
   }

   #define RIGHT	0
   #define LEFT	1
   #define MIDDLE	2

   bool RayBoxIntersection(real_t minB[3], real_t maxB[3], real_t origin[3], real_t rayDir[3]) {
      uint_t quadrant[3];
      uint_t whichPlane;
      real_t candidatePlane[3];
      bool inside = true;
      real_t coord[3];
      real_t maxT[3];

      for (uint_t i=0; i<3; i++)
         if(origin[i] < minB[i]) {
            quadrant[i] = LEFT;
            candidatePlane[i] = minB[i];
            inside = false;
         }else if (origin[i] > maxB[i]) {
            quadrant[i] = RIGHT;
            candidatePlane[i] = maxB[i];
            inside = false;
         }else	{
            quadrant[i] = MIDDLE;
            candidatePlane[i] = 0.0;
         }

      /* Ray origin inside bounding box */
      if(inside)
         return true;

      /* Calculate T distances to candidate planes */
      for (uint_t i = 0; i < 3; i++)
         if (quadrant[i] != MIDDLE && (rayDir[i] < 0.0 || rayDir[i] > 0.0))
            maxT[i] = (candidatePlane[i]-origin[i]) / rayDir[i];
         else
            maxT[i] = -1.;

      /* Get largest of the maxT's for final choice of intersection */
      whichPlane = 0;
      for (uint_t i = 1; i < 3; i++)
         if (maxT[whichPlane] < maxT[i])
            whichPlane = i;

      /* Check final candidate actually inside box */
      if (maxT[whichPlane] < 0.)
         return false;
      for (uint_t i = 0; i < 3; i++)
         if (whichPlane != i) {
            coord[i] = origin[i] + maxT[whichPlane] * rayDir[i];
            if (coord[i] < minB[i] || coord[i] > maxB[i])
               return false;
         } else {
            coord[i] = candidatePlane[i];
         }
      return true;
   }


   void voxelizeRayTracing() {
      Matrix3< real_t > rotationMat(rotationAxis_, currentAngle);
      WALBERLA_LOG_INFO(currentAngle)
      WALBERLA_LOG_INFO("Mesh Center " << meshCenter)

      for (auto& block : *blocks_)
      {
         auto fractionField = block.getData< FracField_T >(fractionFieldId_);
         auto level         = blocks_->getLevel(block);
         real_t cellCenter[3], rotatedCellCenter[3];
         real_t triangle[3][3];
         real_t rayDirection[3];
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(fractionField,
            cell::Cell curCell = Cell(x,y,z);
            //WALBERLA_LOG_INFO(curCell)

            auto cellAABB = blocks_->getAABBFromCellBB(CellInterval( curCell, curCell), level);
            cellCenter[0] = real_t(cellAABB.center()[0]);
            cellCenter[1] = real_t(cellAABB.center()[1]);
            cellCenter[2] = real_t(cellAABB.center()[2]);

            SUB(cellCenter, cellCenter, meshCenter);

            rotatedCellCenter[0] = rotationMat[0] * cellCenter[0] + rotationMat[1] * cellCenter[1] + rotationMat[2] * cellCenter[2];
            rotatedCellCenter[1] = rotationMat[3] * cellCenter[0] + rotationMat[4] * cellCenter[1] + rotationMat[5] * cellCenter[2];
            rotatedCellCenter[2] = rotationMat[6] * cellCenter[0] + rotationMat[7] * cellCenter[1] + rotationMat[8] * cellCenter[2];

            ADD(rotatedCellCenter, rotatedCellCenter, meshCenter);

            uint intersections = 0;
            //TODO Shoot multiple rays

            rayDirection[0] = real_t(rand()) /  real_t(RAND_MAX);
            rayDirection[1] = real_t(rand()) /  real_t(RAND_MAX);
            rayDirection[2] = real_t(rand()) /  real_t(RAND_MAX);

            for (uint_t i = 0; i < numTriangles_; ++i) {
               for(uint_t ty = 0; ty < 3; ++ty) {
                  for(uint_t tx = 0; tx < 3; ++tx) {
                     triangle[ty][tx] = vertices_[3 * triangles_[3*i + ty] + tx];
                  }
               }
               if(RayIntersectsTriangle(rotatedCellCenter, rayDirection, triangle))
                  intersections ++;
            }
            if (intersections % 2 == 1) {
               fractionField->get(x,y,z) = 1.0;
            }
         )
      }
   }



   void voxelizeRayTracingWithGeometryOctree() {
      Matrix3< real_t > rotationMat(rotationAxis_, currentAngle);

      for (auto& block : *blocks_)
      {
         auto fractionField = block.getData< FracField_T >(fractionFieldId_);
         auto level         = blocks_->getLevel(block);
         auto dx            = real_t(blocks_->dx(level));

         real_t cellCenter[3], rotatedCellCenter[3];

         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(fractionField,

            //get global cell center in
            cell::Cell curCell = Cell(x,y,z);
            WALBERLA_LOG_INFO(curCell)

            auto cellAABB = blocks_->getAABBFromCellBB(CellInterval( curCell, curCell), level);
            cellCenter[0] = real_t(cellAABB.center()[0]);
            cellCenter[1] = real_t(cellAABB.center()[1]);
            cellCenter[2] = real_t(cellAABB.center()[2]);

            //rotate cell center (instead of rotating the mesh)
            SUB(cellCenter, cellCenter, meshCenter);
            rotatedCellCenter[0] = rotationMat[0] * cellCenter[0] + rotationMat[1] * cellCenter[1] + rotationMat[2] * cellCenter[2];
            rotatedCellCenter[1] = rotationMat[3] * cellCenter[0] + rotationMat[4] * cellCenter[1] + rotationMat[5] * cellCenter[2];
            rotatedCellCenter[2] = rotationMat[6] * cellCenter[0] + rotationMat[7] * cellCenter[1] + rotationMat[8] * cellCenter[2];
            ADD(rotatedCellCenter, rotatedCellCenter, meshCenter);

            //get fraction
            fracSize fraction = recursiveSuperSampling(geometryOctree_, rotatedCellCenter, dx, 0);
            fractionField->get(x,y,z) = fraction;
         )
      }
   }

   fracSize recursiveSuperSampling(shared_ptr<GeometryOctreeNode> &geometryOctreeNode, real_t cellCenter[3], real_t dx, uint_t supersamplingDepth)
   {
      // if only one cell left, split cell into 8 cell centers to get these cellCenter distances

      fracSize fraction        = 0.0;
      const fracSize fracValue = fracSize(1.0 / pow(8, real_t(supersamplingDepth)));
      const real_t offsetMod    = real_t(1.0 / pow(2, real_t(supersamplingDepth + 2)));

      if (supersamplingDepth == maxSuperSamplingDepth_)
      {
         uint_t numRays = 1;
         uint_t isInside = 0;
         for(uint_t rays = 0; rays < numRays; ++rays) {
            std::vector<uint_t> hitTriangles;
            real_t rayDirection[3] = {real_t(rand()) /  real_t(RAND_MAX) - 0.5f, real_t(rand()) /  real_t(RAND_MAX) - 0.5f, real_t(rand()) /  real_t(RAND_MAX) - 0.5f};
            recursiveIntersection(geometryOctreeNode, cellCenter, rayDirection, 0, hitTriangles);
            std::sort(hitTriangles.begin(), hitTriangles.end());
            hitTriangles.erase(std::unique(hitTriangles.begin(), hitTriangles.end()), hitTriangles.end()) ;
            if (hitTriangles.size() % 2 == 1)
               isInside++;
         }
         if (isInside == numRays )
            fraction = fracValue;
      }
      else
      {
         for (uint_t i = 0; i < 8; ++i)
         {
            real_t octreeCenter[3] = { cellCenter[0] + xOffset[i] * dx * offsetMod, cellCenter[1] + yOffset[i] * dx * offsetMod,
                             cellCenter[2] + zOffset[i] * dx * offsetMod };
            fraction += recursiveSuperSampling(geometryOctreeNode, octreeCenter, dx, supersamplingDepth + 1);
         }
      }
      return fraction;
   }


   void recursiveIntersection(shared_ptr<GeometryOctreeNode> &geometryOctreeNode, real_t cellCenter[3], real_t rayDirection[3], uint_t depth, std::vector<uint_t> &hitTriangles) {
      AABB boxAABB = geometryOctreeNode->getBoxAABB();
      real_t aabbMin[3] = {real_t(boxAABB.xMin()), real_t(boxAABB.yMin()), real_t(boxAABB.zMin())};
      real_t aabbMax[3] = {real_t(boxAABB.xMax()), real_t(boxAABB.yMax()), real_t(boxAABB.zMax())};

      if(!RayBoxIntersection(aabbMin, aabbMax, cellCenter, rayDirection)) {
         return; // = return 0
      }

      if(depth == geometryOctreeNode->getMaxDepth()) {
         auto containedTriangles = geometryOctreeNode->getContainedTriangles();
         for (auto tris : containedTriangles) {
            real_t triangle[3][3];
            for(uint_t y = 0; y < 3; ++y) {
               for(uint_t x = 0; x < 3; ++x) {
                  triangle[y][x] = vertices_[3 * triangles_[3 * tris + y] + x];
               }
            }
            if(RayIntersectsTriangle(cellCenter, rayDirection, triangle))
               hitTriangles.push_back(tris);
         }
      }
      else{
         for( auto node : geometryOctreeNode->getChildNodes()) {
            recursiveIntersection(node, cellCenter, rayDirection, depth+1, hitTriangles);
         }
      }
   }



 private:
   shared_ptr< StructuredBlockForest > blocks_;
   shared_ptr< mesh::TriangleMesh > mesh_;

   real_t * vertices_;
   uint_t * triangles_;

   uint_t numVertices_;
   uint_t numTriangles_;

   BlockDataID fractionFieldId_;
   const BlockDataID objectVelocityId_;
   const real_t rotationAngle_;
   const uint_t frequency_;
   Vector3< mesh::TriangleMesh::Scalar > rotationAxis_;
   shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>> distOctree_;
   std::string meshName_;
   uint_t maxSuperSamplingDepth_;
   const bool rotate_;
   mesh::TriangleMesh::Point meshCenter;
   real_t currentAngle;

   shared_ptr<GeometryOctreeNode> geometryOctree_;

   real_t xOffset[8]{ -1, -1, -1, -1, 1, 1, 1, 1 };
   real_t yOffset[8]{ -1, -1, 1, 1, -1, -1, 1, 1 };
   real_t zOffset[8]{ -1, 1, -1, 1, -1, 1, -1, 1 };


};

void fuseFractionFields(shared_ptr< StructuredBlockForest >& blocks, BlockDataID dstFracFieldID, std::vector<BlockDataID> srcFracFieldIDs) {
   for (auto &block : *blocks) {
      FracField_T* dstFractionField = block.getData< FracField_T >(dstFracFieldID);
      std::vector<FracField_T*> srcFracFields;
      for (auto srcFracFieldID : srcFracFieldIDs) {
         srcFracFields.push_back(block.getData< FracField_T >(srcFracFieldID));
      }
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(dstFractionField,
          dstFractionField->get(x,y,z) = 0;
          for (auto srcFracField : srcFracFields) {
             dstFractionField->get(x,y,z) = std::max(srcFracField->get(x,y,z), dstFractionField->get(x,y,z));
          }
      )
   }
}

}//namespace waLBerla