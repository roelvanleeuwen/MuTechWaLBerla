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
#include <iostream>
#include <fstream>
#include <string>

#include "BoxTriangleIntersection.h"



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
   ObjectRotatorGPU(shared_ptr< StructuredBlockForest >& blocks, shared_ptr< mesh::TriangleMesh >& mesh, const BlockDataID objectVelocityId,
                 const real_t rotationAngle, const uint_t frequency, Vector3<int> rotationAxis,
                 std::string meshName, uint_t maxSuperSamplingDepth, const bool rotate = true)
      : blocks_(blocks), mesh_(mesh), objectVelocityId_(objectVelocityId),
        rotationAngle_(rotationAngle), frequency_(frequency), rotationAxis_(rotationAxis),
        meshName_(meshName), maxSuperSamplingDepth_(maxSuperSamplingDepth), rotate_(rotate)
   {
      fractionFieldId_ = field::addToStorage< FracField_T >(blocks, "fractionField_" + meshName_, fracSize(0.0), field::fzyx, uint_c(1));
      readFile();
      meshCenter = computeCentroid(*mesh_);
      initObjectVelocityField();
      WALBERLA_LOG_INFO_ON_ROOT("Start voxelizeBoxTriangleIntersection")
      voxelizeBoxTriangleIntersection();
      WALBERLA_LOG_INFO_ON_ROOT("Finished voxelizeBoxTriangleIntersection")
   }


   void operator()(uint_t timestep) {
      if (timestep % frequency_ == 0)
      {
         if(rotate_) {
            //rotate();
            //resetFractionField();
            //getFractionFieldFromMesh();
            voxelizeBoxTriangleIntersection();
         }
      }
   }


   void readFile() {
      std::ifstream meshFile (meshName_ + ".obj");
      std::string line;
      std::stringstream stream;
      std::string keyword;

      float3 vertex;
      uint triangle[3];

      if (meshFile.is_open())
      {
         while ( getline (meshFile,line) )
         {
            stream.str(line);
            stream.clear();

            //WALBERLA_LOG_INFO_ON_ROOT("Keyword is " << keyword << " Line is " << line << " stingstream is " << stream.str())
            stream >> keyword;
            if (keyword == "v") {
               stream >> vertex.x >> vertex.y >> vertex.z;
               vertices_.push_back(vertex);
            }
            else if(keyword == "f") {
               std::string vertexSrt;
               size_t found;
               for (uint i = 0; i < 3; ++i) {
                  stream >> vertexSrt;
                  found = vertexSrt.find("/");
                  triangle[i] = uint(stoi(vertexSrt.substr(0,found))-1);
               }
               triangles_.push_back(make_uint3(triangle[0], triangle[1], triangle[2]));
            }
         }
         meshFile.close();
      }
      else
      WALBERLA_LOG_INFO_ON_ROOT("Num Tringles is " << triangles_.size()  << " and vertices is " << vertices_.size())
   }

   BlockDataID getObjectFractionFieldID() {
      return fractionFieldId_;
   }


   void rotate()
   {
      const Vector3< mesh::TriangleMesh::Scalar > axis_foot(meshCenter[0], meshCenter[1], meshCenter[2]);
      mesh::rotate(*mesh_, rotationAxis_, rotationAngle_, axis_foot);
   }


   void resetFractionField()
   {
      auto aabbMesh = computeAABB(*mesh_);
      for (auto& block : *blocks_)
      {
         auto fractionField = block.getData< FracField_T >(fractionFieldId_);
         auto level = blocks_->getLevel(block);
         auto cellBBMesh = blocks_->getCellBBFromAABB( aabbMesh, level );
         CellInterval blockCi = fractionField->xyzSizeWithGhostLayer();
         blocks_->transformBlockLocalToGlobalCellInterval(blockCi, block);
         cellBBMesh.intersect(blockCi);
         blocks_->transformGlobalToBlockLocalCellInterval(cellBBMesh, block);
         std::fill(fractionField->beginSliceXYZ(cellBBMesh), fractionField->end(), 0.0);
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
         for (uint i = 0; i < triangles_.size(); ++i) {
            float triangle[3][3] = {{vertices_[triangles_[i].x].x, vertices_[triangles_[i].x].y, vertices_[triangles_[i].x].z},
                                    {vertices_[triangles_[i].y].x, vertices_[triangles_[i].y].y, vertices_[triangles_[i].y].z},
                                    {vertices_[triangles_[i].z].x, vertices_[triangles_[i].z].y, vertices_[triangles_[i].z].z}};
            //WALBERLA_LOG_INFO_ON_ROOT("Triangle " << triangles_[i].x << "," << triangles_[i].y << "," << triangles_[i].z << " with vertex " << vertices_[triangles_[i].x].x << "," <<  vertices_[triangles_[i].x].y << "," << vertices_[triangles_[i].x].z << " " << vertices_[triangles_[i].y].x << "," <<  vertices_[triangles_[i].y].y << "," << vertices_[triangles_[i].y].z << " " << vertices_[triangles_[i].z].x << "," <<  vertices_[triangles_[i].z].y << "," << vertices_[triangles_[i].z].z)
            auto faceAABB = computeAABBFromTriangle( triangle );
            auto cellBB = blocks_->getCellBBFromAABB( faceAABB, level );
            faceAABB.intersect(block.getAABB());
            if(faceAABB.empty()) continue;

            float normal[3],e0[3],e1[3],e2[3];
            SUB(e0,triangle[1],triangle[0]);      /* tri edge 0 */
            SUB(e1,triangle[2],triangle[1]);      /* tri edge 1 */
            CROSS(normal,e0,e1);

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
                  float flag;
                  if (normal[0] <= 0) {   //front side triangle
                     flag = 2.0;
                     if (fractionField->get(localCell) == 3.0) flag = 4.0; //cell triangles with front and back faces
                  } else if (normal[0] > 0) {
                     flag = 3.0;
                     if (fractionField->get(localCell) == 2.0) flag = 4.0; //cell triangles with front and back faces
                  }
                  fractionField->get(localCell) = flag;
               }
            }
         }
      }
   }




 private:
   shared_ptr< StructuredBlockForest > blocks_;
   shared_ptr< mesh::TriangleMesh > mesh_;
   std::vector<float3> vertices_;
   std::vector<uint3> triangles_;
   BlockDataID fractionFieldId_;
   const BlockDataID objectVelocityId_;
   const real_t rotationAngle_;
   const uint_t frequency_;
   Vector3< mesh::TriangleMesh::Scalar > rotationAxis_;
   std::string meshName_;
   uint_t maxSuperSamplingDepth_;
   const bool rotate_;
   mesh::TriangleMesh::Point meshCenter;
};


}//namespace waLBerla