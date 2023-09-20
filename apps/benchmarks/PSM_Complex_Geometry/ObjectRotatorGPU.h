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
//! \file ObjectRotatorGPU.h
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================
#pragma once

#include "core/math/Constants.h"
#include "blockforest/all.h"
#include "field/all.h"
#include "gpu/GPUField.h"

#include <iostream>
#include <fstream>
#include <string>
#include "lbm_generated/gpu/GPUPdfField.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

//#include "BoxTriangleIntersection.h"

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
#include "mesh/boundary/BoundaryLocation.h"
#include "mesh/boundary/BoundaryLocationFunction.h"
#include "mesh/boundary/BoundarySetup.h"


#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif



namespace walberla
{
typedef field::GhostLayerField< real_t, 1 > ScalarField_T;
using fracSize = float;
typedef field::GhostLayerField< fracSize, 1 > FracField_T;


class ObjectRotatorGPU
{
   typedef field::GhostLayerField< real_t, 3 > VectorField_T;

 public:
   ObjectRotatorGPU(shared_ptr< StructuredBlockForest >& blocks, const BlockDataID fractionFieldGPUId, shared_ptr< mesh::TriangleMesh >& mesh, const BlockDataID objectVelocityId,
                 const real_t rotationAngle, const uint_t frequency, Vector3<int> rotationAxis,
                 std::string meshName, uint_t maxSuperSamplingDepth, const bool rotate = true)
      : blocks_(blocks), fractionFieldGPUId_(fractionFieldGPUId), mesh_(mesh), objectVelocityId_(objectVelocityId),
        rotationAngle_(rotationAngle), frequency_(frequency), rotationAxis_(rotationAxis),
        meshName_(meshName), maxSuperSamplingDepth_(maxSuperSamplingDepth), rotate_(rotate)
   {
      //fractionFieldId_ = field::addToStorage< FracField_T >(blocks, "fractionField_" + meshName_, fracSize(0.0), field::fzyx, uint_c(1));
      readFile();
      Vector3<uint_t> cellsPerBlock(blocks_->getNumberOfXCellsPerBlock() + 2, blocks_->getNumberOfYCellsPerBlock() + 2, blocks_->getNumberOfZCellsPerBlock() + 2);

      cudaMalloc(&dev_curand_states, (cellsPerBlock[0] + 2) * (cellsPerBlock[1] + 2) * (cellsPerBlock[2] + 2) * sizeof(curandState));
      cudaMalloc(&verticesGPU_, numVertices_ * 3 * sizeof(float));
      cudaMalloc(&trianglesGPU_, numTriangles_ * 3 * sizeof(float));
      cudaMemcpy(verticesGPU_, vertices_,  numVertices_ * 3 * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(trianglesGPU_, triangles_,  numTriangles_ * 3 * sizeof(float), cudaMemcpyHostToDevice);

      meshCenter = computeCentroid(*mesh_);
      initObjectVelocityField();
      WcTimer simTimer;
      WALBERLA_LOG_INFO_ON_ROOT("Start voxelizeBoxTriangleIntersection")
      simTimer.start();
      voxelizeRayTracingGPUCall();
      WALBERLA_GPU_CHECK( gpuDeviceSynchronize() )
      simTimer.end();
      WALBERLA_LOG_INFO_ON_ROOT("Finished voxelizeBoxTriangleIntersection in " << simTimer.max() << "s")
   }


   void operator()(uint_t timestep) {
      if (timestep % frequency_ == 0)
      {
         if(rotate_) {
            //rotate();
            resetFractionFieldGPUCall();
            //voxelizeRayTracingGPUCall();
         }
      }
   }


   void readFile() {
      std::ifstream meshFile (meshName_ + ".obj");
      std::string line;
      std::stringstream stream;
      std::string keyword;

      std::vector<Vector3<float>> vertices;
      std::vector<Vector3<uint>> triangles;

      Vector3<float> vertex;
      Vector3<int> triangle;

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
         numVertices_ = int(vertices.size());
         vertices_ = (float *) std::malloc( sizeof(float) * 3 * numVertices_);
         for ( size_t i = 0; i < numVertices_; ++i) {
            for ( size_t j = 0; j < 3; ++j)
            {
               vertices_[i * 3 + j] = vertices[i][j];
               vertices_[i * 3 + j] = vertices[i][j];
               vertices_[i * 3 + j] = vertices[i][j];
            }
         }
         numTriangles_ = int(triangles.size());
         triangles_ = (int *) std::malloc( sizeof(int) * 3 * numTriangles_);
         for ( int i = 0; i < numTriangles_; ++i) {
            for ( int j = 0; j < 3; ++j)
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

   BlockDataID getObjectFractionFieldID() {
      return fractionFieldGPUId_;
   }


   void rotate()
   {
      const Vector3< mesh::TriangleMesh::Scalar > axis_foot(meshCenter[0], meshCenter[1], meshCenter[2]);
      mesh::rotate(*mesh_, rotationAxis_, rotationAngle_, axis_foot);
   }


   void initObjectVelocityField() {
      for (auto& block : *blocks_)
      {
         auto aabbMesh = computeAABB(*mesh_);
         auto level = blocks_->getLevel(block);
         auto cellBB = blocks_->getCellBBFromAABB( aabbMesh, level );

         auto objVelField = block.getData< VectorField_T >(objectVelocityId_);
         Vector3< real_t > angularVel;


         if (!rotate_ || frequency_ == 0 || rotationAngle_ <= 0.0) {
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

   /*
   math::GenericAABB< float > computeAABBFromTriangle( float triverts[3][3] )
   {
      float min[3], max[3];
      min[0] = max[0] = triverts[0][0];
      min[1] = max[1] = triverts[0][1];
      min[2] = max[2] = triverts[0][2];

      for( uint i = 1; i < 3; ++i )
      {
         min[0] = std::min( min[0], triverts[i][0] );
         min[1] = std::min( min[1], triverts[i][1] );
         min[2] = std::min( min[2], triverts[i][2] );

         max[0] = std::max( max[0], triverts[i][0] );
         max[1] = std::max( max[1], triverts[i][1] );
         max[2] = std::max( max[2], triverts[i][2] );
      }

      return math::GenericAABB< float >::createFromMinMaxCorner( Vector3<float>(min[0], min[1], min[2]), Vector3<float>(max[0], max[1], max[2]) );
   }


   void voxelizeBoxTriangleIntersection() {
      for (auto& block : *blocks_)
      {
         auto fractionField   = block.getData< FracField_T >(fractionFieldId_);
         auto level           = blocks_->getLevel(block);
         const float dx       = float(blocks_->dx(level));
         for (uint i = 0; i < numTriangles_; ++i) {
            float triangle[3][3] = {{vertices_[3 * triangles_[i*3]], vertices_[3 * triangles_[i*3] + 1], vertices_[3 * triangles_[i*3] + 2]},
                                     {vertices_[3 * triangles_[i*3+1]], vertices_[3 * triangles_[i*3+1] + 1], vertices_[3 * triangles_[i*3+1] + 2]},
                                    {vertices_[3 * triangles_[i*3+2]], vertices_[3 * triangles_[i*3+2] + 1], vertices_[3 * triangles_[i*3+2] + 2]}};
            //WALBERLA_LOG_INFO_ON_ROOT("Triangle " << triangles_[i].x << "," << triangles_[i].y << "," << triangles_[i].z << " with vertex " << vertices_[triangles_[i].x].x << "," <<  vertices_[triangles_[i].x].y << "," << vertices_[triangles_[i].x].z << " " << vertices_[triangles_[i].y].x << "," <<  vertices_[triangles_[i].y].y << "," << vertices_[triangles_[i].y].z << " " << vertices_[triangles_[i].z].x << "," <<  vertices_[triangles_[i].z].y << "," << vertices_[triangles_[i].z].z)
            auto faceAABB = computeAABBFromTriangle( triangle );
            auto cellBB = blocks_->getCellBBFromAABB( faceAABB, level );
            faceAABB.intersect(block.getAABB());
            if(faceAABB.empty()) continue;

            for (auto cellIt = cellBB.begin(); cellIt != cellBB.end(); ++cellIt) {
               auto cellAABB = blocks_->getAABBFromCellBB(CellInterval( *cellIt, *cellIt), level);
               cellAABB.intersect(block.getAABB());
               if(cellAABB.empty()) continue;
               float cellCenter[3] = {float(cellAABB.center()[0]), float(cellAABB.center()[1]), float(cellAABB.center()[2])};
               float boxHalfSize[3] = {float(dx*0.5), float(dx*0.5), float(dx*0.5)};

               const bool intersection = triBoxOverlap(cellCenter, boxHalfSize , triangle);
               if (intersection) {
                  Cell localCell;
                  blocks_->transformGlobalToBlockLocalCell(localCell, block, *cellIt);
                  fractionField->get(localCell) += 1.0f;
               }
            }
         }
      }
   }


   void voxelizeRayTracing() {
      for (auto& block : *blocks_)
      {
         auto fractionField = block.getData< FracField_T >(fractionFieldId_);
         auto level         = blocks_->getLevel(block);
         float cellCenter[3];
         float triangle[3][3];
         float rayDirection[3];
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(fractionField,
            cell::Cell curCell = Cell(x,y,z);

            auto cellAABB = blocks_->getAABBFromCellBB(CellInterval( curCell, curCell), level);
            cellCenter[0] = float(cellAABB.center()[0]);
            cellCenter[1] = float(cellAABB.center()[1]);
            cellCenter[2] = float(cellAABB.center()[2]);
            uint intersections = 0;
            //TODO Shoot multiple rays
            rayDirection[0] = float(rand()) /  float(RAND_MAX); rayDirection[1] = float(rand()) /  float(RAND_MAX); rayDirection[2] = float(rand()) /  float(RAND_MAX);
            for (int i = 0; i < numTriangles_; ++i) {
               for(int y = 0; y < 3; ++y) {
                  for(int x = 0; x < 3; ++x) {
                     triangle[y][x] = vertices_[3 * triangles_[3*i + y] + x];
                  }
               }
               if(RayIntersectsTriangle(cellCenter, rayDirection, triangle))
                  intersections ++;
            }
            if (intersections % 2 == 1) {
               fractionField->get(x,y,z) = 1.0;
            }
         )
      }
   }
*/


   void voxelizeRayTracingGPUCall();
   void resetFractionFieldGPUCall();




 private:
   shared_ptr< StructuredBlockForest > blocks_;
   shared_ptr< mesh::TriangleMesh > mesh_;

   float * vertices_;
   int * triangles_;

   float * verticesGPU_;
   int * trianglesGPU_;

   int numVertices_;
   int numTriangles_;

   BlockDataID fractionFieldGPUId_;
   const BlockDataID objectVelocityId_;
   const real_t rotationAngle_;
   const uint_t frequency_;
   Vector3< mesh::TriangleMesh::Scalar > rotationAxis_;
   std::string meshName_;
   uint_t maxSuperSamplingDepth_;
   const bool rotate_;
   mesh::TriangleMesh::Point meshCenter;

   curandState* dev_curand_states;
};


}//namespace waLBerla