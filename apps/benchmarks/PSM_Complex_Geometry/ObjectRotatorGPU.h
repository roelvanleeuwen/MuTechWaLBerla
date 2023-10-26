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

#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

#include "core/math/Constants.h"
#include "blockforest/all.h"
#include "field/all.h"
#include "gpu/GPUField.h"
#include "mesh_common/DistanceComputations.h"
#include "lbm_generated/gpu/GPUPdfField.h"
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
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/FieldCopy.h"
#include "gpu/GPUWrapper.h"
#include "gpu/HostFieldAllocator.h"
#include "gpu/ParallelStreams.h"
#include "gpu/communication/UniformGPUScheme.h"
#endif

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
using fracSize = real_t;
typedef field::GhostLayerField< fracSize, 1 > FracField_T;



struct DistancePropertiesGPU
{
   double2 e0, e1, e2;
   double2 e1_normal, e2_normal;
   double2 e1_normalized, e2_normalized, e0_normalized;
   double e0l, e1l, e2l;

   double3 translation;
   double rotation[9];

   double3 region_normal[7];
};



class ObjectRotatorGPU
{
   typedef field::GhostLayerField< real_t, 3 > VectorField_T;

 public:
   ObjectRotatorGPU(shared_ptr< StructuredBlockForest >& blocks, const BlockDataID fractionFieldGPUId,
                    const BlockDataID fractionFieldCPUId, shared_ptr< mesh::TriangleMesh >& mesh,
                    shared_ptr< mesh::TriangleDistance<mesh::TriangleMesh> >& triangleDistance,
                    const BlockDataID objectVelocityId, const real_t rotationAngle, const uint_t frequency,
                    Vector3<int> rotationAxis,  std::string meshName, uint_t maxSuperSamplingDepth,
                    const bool rotate = true)
      : blocks_(blocks), fractionFieldGPUId_(fractionFieldGPUId), mesh_(mesh), triangleDistance_(triangleDistance), objectVelocityId_(objectVelocityId),
        rotationAngle_(rotationAngle), frequency_(frequency), rotationAxis_(rotationAxis), meshName_(meshName), rotate_(rotate)
   {
      tmpFracFieldGPUId = gpu::addGPUFieldToStorage< FracField_T >(blocks_, fractionFieldCPUId, meshName_ + "_fracFieldGPU", true );
      tmpFracFieldGPUOldId = gpu::addGPUFieldToStorage< FracField_T >(blocks_, fractionFieldCPUId, meshName_ + "_fracFieldOldGPU", true );

      meshCenter = computeCentroid(*mesh_);
      convertDistancePropertiesToGPU();

      initObjectVelocityField();
      voxelizeGPUCall(0);
      if(!rotate) {
         swapFractionFields();
      }
   }


   void operator()(uint_t timestep) {
      if(rotate_) {
         if (timestep % frequency_ == 0)
         {
            swapFractionFields();
            voxelizeGPUCall(timestep + frequency_);
         }
         interpolateFractionFieldsCall(timestep);
      }
      else{
         interpolateFractionFieldsCall(0);
      }
   }

   void swapFractionFields() {
      for (auto& block : *blocks_)
      {
         auto tmpFractionFieldGPU = block.getData< gpu::GPUField<fracSize> >(tmpFracFieldGPUId);
         auto tmpFractionFieldGPUOld = block.getData< gpu::GPUField<fracSize> >(tmpFracFieldGPUOldId);
         tmpFractionFieldGPU->swapDataPointers(tmpFractionFieldGPUOld);
      }
   }

   void convertDistancePropertiesToGPU() {
      auto distanceProperties = triangleDistance_->getDistanceProperties();
      for(auto f_it = mesh_->faces_begin(); f_it != mesh_->faces_end(); ++f_it) {
         mesh::DistanceProperties<mesh::TriangleMesh> dp = distanceProperties[*f_it];
         DistancePropertiesGPU dpGPU;

         dpGPU.e0 = {double(dp.e0[0]), double(dp.e0[1])};
         dpGPU.e1 = {double(dp.e1[0]), double(dp.e1[1])};
         dpGPU.e2 = {double(dp.e2[0]), double(dp.e2[1])};

         dpGPU.e1_normal = {double(dp.e1_normal[0]), double(dp.e1_normal[1])};
         dpGPU.e2_normal = {double(dp.e2_normal[0]), double(dp.e2_normal[1])};

         dpGPU.e0_normalized = {double(dp.e0_normalized[0]), double(dp.e0_normalized[1])};
         dpGPU.e1_normalized = {double(dp.e1_normalized[0]), double(dp.e1_normalized[1])};
         dpGPU.e2_normalized = {double(dp.e2_normalized[0]), double(dp.e2_normalized[1])};

         dpGPU.e0l = double(dp.e0l);
         dpGPU.e1l = double(dp.e1l);
         dpGPU.e2l = double(dp.e2l);

         dpGPU.translation = {double(dp.translation[0]), double(dp.translation[1]), double(dp.translation[2])};

         for (int i = 0; i < 9; ++i)
            dpGPU.rotation[i] = double(dp.rotation[i]);

         //also save normals of faces, edges and vertices to compute sign (Voronoi areas)
         auto normal = mesh_->normal( *f_it );
         dpGPU.region_normal[0] = {double(normal[0]), double(normal[1]), double(normal[2])};
         normal = mesh_->normal( getVertexHandle( *mesh_, *f_it, 0U ) );
         dpGPU.region_normal[1] = {double(normal[0]), double(normal[1]), double(normal[2])};
         normal = mesh_->normal( getVertexHandle( *mesh_, *f_it, 1U ) );
         dpGPU.region_normal[2] = {double(normal[0]), double(normal[1]), double(normal[2])};
         normal = mesh_->normal( getVertexHandle( *mesh_, *f_it, 2U ) );
         dpGPU.region_normal[3] = {double(normal[0]), double(normal[1]), double(normal[2])};
         normal = mesh_->normal( getHalfedgeHandle( *mesh_, *f_it, 0U, 1U ) );
         dpGPU.region_normal[4] = {double(normal[0]), double(normal[1]), double(normal[2])};
         normal = mesh_->normal( getHalfedgeHandle( *mesh_, *f_it, 0U, 2U ) );
         dpGPU.region_normal[5] = {double(normal[0]), double(normal[1]), double(normal[2])};
         normal = mesh_->normal( getHalfedgeHandle( *mesh_, *f_it, 1U, 2U ) );
         dpGPU.region_normal[6] = {double(normal[0]), double(normal[1]), double(normal[2])};

         distancePropertiesCPUPtr.push_back(dpGPU);
      }

      numFaces_ = uint(distancePropertiesCPUPtr.size());
      cudaMalloc((void **)&distancePropertiesGPUPtr, numFaces_ * sizeof(DistancePropertiesGPU));
      cudaMemcpy(distancePropertiesGPUPtr, &distancePropertiesCPUPtr[0],  numFaces_ * sizeof(DistancePropertiesGPU), cudaMemcpyHostToDevice);
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

   void voxelizeGPUCall(uint_t timestep);

   void interpolateFractionFieldsCall(uint_t timestep);


 private:
   shared_ptr< StructuredBlockForest > blocks_;
   BlockDataID fractionFieldGPUId_;
   BlockDataID tmpFracFieldGPUId;
   BlockDataID tmpFracFieldGPUOldId;

   shared_ptr< mesh::TriangleMesh > mesh_;
   shared_ptr< mesh::TriangleDistance<mesh::TriangleMesh> >& triangleDistance_;
   std::vector<DistancePropertiesGPU> distancePropertiesCPUPtr;
   DistancePropertiesGPU * distancePropertiesGPUPtr;
   uint numFaces_;

   const BlockDataID objectVelocityId_;
   const real_t rotationAngle_;
   const uint_t frequency_;
   Vector3< mesh::TriangleMesh::Scalar > rotationAxis_;
   std::string meshName_;
   const bool rotate_;
   mesh::TriangleMesh::Point meshCenter;
};

void resetFractionFieldGPUCall(shared_ptr< StructuredBlockForest >& blocks, BlockDataID fractionFieldGPUId);


}//namespace waLBerla