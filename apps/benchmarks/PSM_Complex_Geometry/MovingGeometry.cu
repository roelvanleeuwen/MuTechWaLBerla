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
//! \file ObjectRotatorGPU.cu
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================
#include "MovingGeometry.h"

#define SUB(dest,v1,v2) \
         dest.x=v1.x-v2.x; \
         dest.y=v1.y-v2.y; \
         dest.z=v1.z-v2.z;

#define ADD(dest,v1,v2) \
         dest.x=v1.x+v2.x; \
         dest.y=v1.y+v2.y; \
         dest.z=v1.z+v2.z;

namespace walberla
{

__global__ void resetFractionFieldGPUKernel( real_t * RESTRICT const fractionFieldData, int3 fieldSize, int3 stride) {
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < fieldSize.x  && y < fieldSize.y  && z < fieldSize.z )
   {
      const int idx = (x) + (y) * stride.y + (z) * stride.z;

      fractionFieldData[idx] = 0;
   }
}

void MovingGeometry::resetFractionField() {
   for (auto& block : *blocks_) {
      auto fractionFieldGPU = block.getData< gpu::GPUField<real_t> >(fractionFieldId_);
      real_t * RESTRICT const _data_FractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);

      int3 size = {int(fractionFieldGPU->xSizeWithGhostLayer()), int(fractionFieldGPU->ySizeWithGhostLayer()), int(fractionFieldGPU->zSizeWithGhostLayer()) };
      int3 stride_frac_field = {int(fractionFieldGPU->xStride()), int(fractionFieldGPU->yStride()), int(fractionFieldGPU->zStride())};

      dim3 _block(uint64_c(((16 < size.x - 2) ? 16 : size.x - 2)), uint64_c(((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))), uint64_c(((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))));
      dim3 _grid(uint64_c(( (size.x - 2) % (((16 < size.x - 2) ? 16 : size.x - 2)) == 0 ? (int64_t)(size.x - 2) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)) : ( (int64_t)(size.x - 2) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)) ) +1 )), uint64_c(( (size.y - 2) % (((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))) == 0 ? (int64_t)(size.y - 2) / (int64_t)(((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))) : ( (int64_t)(size.y - 2) / (int64_t)(((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))) ) +1 )), uint64_c(( (size.z - 2) % (((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))) == 0 ? (int64_t)(size.z - 2) / (int64_t)(((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))) : ( (int64_t)(size.z - 2) / (int64_t)(((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))) ) +1 )));

      resetFractionFieldGPUKernel<<<_grid, _block>>>(_data_FractionFieldGPU, size, stride_frac_field);
   }
}

__global__ void getFractionFieldFromGeometryMeshKernel(real_t * RESTRICT const _data_fractionFieldGPU, real_t * RESTRICT const _data_geometryFieldGPU, int3 field_size, int3 field_stride, int3 geometry_field_size, int3 geometry_field_stride, double3 blockAABBMin, double3 meshAABBMin, double dx, double3 meshCenter, double3 rotationMatrixX, double3 rotationMatrixY, double3 rotationMatrixZ, double3 translation) {
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < field_size.x  && y < field_size.y  && z < field_size.z ) {
      const int idx = (x) + (y) * field_stride.y + (z) * field_stride.z;

      double dxHalf = 0.5 * dx;
      double3 cellCenter = { blockAABBMin.x + double(x) * dx + dxHalf, blockAABBMin.y + double(y) * dx + dxHalf, blockAABBMin.z + double(z) * dx + dxHalf };
      double3 rotatedCellCenter;
      double fraction;

      //translation
      SUB(cellCenter, cellCenter, translation)

      //rotation
      SUB(cellCenter, cellCenter, meshCenter)
      rotatedCellCenter.x = cellCenter.x * rotationMatrixX.x + cellCenter.y * rotationMatrixX.y + cellCenter.z * rotationMatrixX.z;
      rotatedCellCenter.y = cellCenter.x * rotationMatrixY.x + cellCenter.y * rotationMatrixY.y + cellCenter.z * rotationMatrixY.z;
      rotatedCellCenter.z = cellCenter.x * rotationMatrixZ.x + cellCenter.y * rotationMatrixZ.y + cellCenter.z * rotationMatrixZ.z;
      ADD(rotatedCellCenter, rotatedCellCenter, meshCenter)

      //get cell of geometry field
      int3 cellInGeometrySpace;
      cellInGeometrySpace.x = int(round((rotatedCellCenter.x - meshAABBMin.x - dxHalf) / dx));
      cellInGeometrySpace.y = int(round((rotatedCellCenter.y - meshAABBMin.y - dxHalf) / dx));
      cellInGeometrySpace.z = int(round((rotatedCellCenter.z - meshAABBMin.z - dxHalf) / dx));

      if (cellInGeometrySpace.x < 0 || cellInGeometrySpace.x >= geometry_field_size.x ||
          cellInGeometrySpace.y < 0 || cellInGeometrySpace.y >= geometry_field_size.y ||
          cellInGeometrySpace.z < 0 || cellInGeometrySpace.z >= geometry_field_size.z )
      {
         fraction = 0.0;
      }
      else {
         const int idx_geo = cellInGeometrySpace.x + cellInGeometrySpace.y * geometry_field_stride.y + cellInGeometrySpace.z * geometry_field_stride.z;
         fraction = _data_geometryFieldGPU[idx_geo];
      }
      _data_fractionFieldGPU[idx] = min(1.0, _data_fractionFieldGPU[idx] + fraction);
   }
}

void MovingGeometry::getFractionFieldFromGeometryMesh(uint_t timestep)  {

   Matrix3< real_t > rotationMat(rotationAxis_, real_t(timestep) * -rotationAngle_);
   double3 rotationMatrixX = {rotationMat[0], rotationMat[1], rotationMat[2]};
   double3 rotationMatrixY = {rotationMat[3], rotationMat[4], rotationMat[5]};
   double3 rotationMatrixZ = {rotationMat[6], rotationMat[7], rotationMat[8]};
   double3 translation = {translation_[0] * real_t(timestep), translation_[1] * real_t(timestep), translation_[2] * real_t(timestep)};

   for (auto& block : *blocks_) {
      auto fractionFieldGPU = block.getData< gpu::GPUField<real_t> >(fractionFieldId_);
      real_t * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);

      auto geometryFieldGPU = block.getData< gpu::GPUField<real_t> >(geometryFieldGPUId_);
      real_t * RESTRICT const _data_geometryFieldGPU = geometryFieldGPU->dataAt(0, 0, 0, 0);

      auto level         = blocks_->getLevel(block);
      auto dx     = double(blocks_->dx(level));
      double3 meshCenterGPU = {meshCenter[0], meshCenter[1], meshCenter[2]};
      auto blockAABB = block.getAABB();
      double3 blockAABBmin = {blockAABB.minCorner()[0], blockAABB.minCorner()[1], blockAABB.minCorner()[2]};
      double3 meshAABBmin = {meshAABB_.minCorner()[0], meshAABB_.minCorner()[1], meshAABB_.minCorner()[2]};

      int3 field_size = {int(fractionFieldGPU->xSizeWithGhostLayer()), int(fractionFieldGPU->ySizeWithGhostLayer()), int(fractionFieldGPU->zSizeWithGhostLayer()) };
      int3 field_stride = {int(fractionFieldGPU->xStride()), int(fractionFieldGPU->yStride()), int(fractionFieldGPU->zStride())};

      int3 geometry_field_size = {int(geometryFieldGPU->xSize()), int(geometryFieldGPU->ySize()), int(geometryFieldGPU->zSize()) };
      int3 geometry_field_stride = {int(geometryFieldGPU->xStride()), int(geometryFieldGPU->yStride()), int(geometryFieldGPU->zStride())};

      dim3 _block(uint64_c(((16 < field_size.x - 2) ? 16 : field_size.x - 2)), uint64_c(((1024 < ((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))) ? 1024 : ((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))))), uint64_c(((64 < ((field_size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))))) ? field_size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))))))) ? 64 : ((field_size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))))) ? field_size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))))))));
      dim3 _grid(uint64_c(( (field_size.x - 2) % (((16 < field_size.x - 2) ? 16 : field_size.x - 2)) == 0 ? (int64_t)(field_size.x - 2) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)) : ( (int64_t)(field_size.x - 2) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)) ) +1 )), uint64_c(( (field_size.y - 2) % (((1024 < ((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))) ? 1024 : ((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))))) == 0 ? (int64_t)(field_size.y - 2) / (int64_t)(((1024 < ((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))) ? 1024 : ((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))))) : ( (int64_t)(field_size.y - 2) / (int64_t)(((1024 < ((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))) ? 1024 : ((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))))) ) +1 )), uint64_c(( (field_size.z - 2) % (((64 < ((field_size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))))) ? field_size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))))))) ? 64 : ((field_size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))))) ? field_size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))))))) == 0 ? (int64_t)(field_size.z - 2) / (int64_t)(((64 < ((field_size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))))) ? field_size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))))))) ? 64 : ((field_size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))))) ? field_size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))))))) : ( (int64_t)(field_size.z - 2) / (int64_t)(((64 < ((field_size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))))) ? field_size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))))))) ? 64 : ((field_size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))))) ? field_size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)*((field_size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2)))) ? field_size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2) ? 16 : field_size.x - 2))))))))) ) +1 )));

      getFractionFieldFromGeometryMeshKernel<<<_grid, _block>>>(_data_fractionFieldGPU, _data_geometryFieldGPU, field_size, field_stride, geometry_field_size, geometry_field_stride, blockAABBmin, meshAABBmin, dx, meshCenterGPU, rotationMatrixX, rotationMatrixY, rotationMatrixZ, translation);
   }
}

__global__ void addStaticGeometryToFractionFieldKernel( real_t * RESTRICT const fractionFieldData, real_t * RESTRICT const staticFractionFieldData, int3 fieldSize, int3 stride) {
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < fieldSize.x  && y < fieldSize.y  && z < fieldSize.z )
   {
      const int idx = x + y * stride.y + z * stride.z;

      fractionFieldData[idx] = staticFractionFieldData[idx];
   }
}

void MovingGeometry::addStaticGeometryToFractionField() {
   for (auto& block : *blocks_) {
      auto fractionFieldGPU = block.getData< gpu::GPUField<real_t> >(fractionFieldId_);
      real_t * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);

      auto staticFractionFieldGPU = block.getData< gpu::GPUField<real_t> >(staticFractionFieldGPUId_);
      real_t * RESTRICT const _data_staticFractionFieldGPU = staticFractionFieldGPU->dataAt(0, 0, 0, 0);


      int3 size = {int(fractionFieldGPU->xSizeWithGhostLayer()), int(fractionFieldGPU->ySizeWithGhostLayer()), int(fractionFieldGPU->zSizeWithGhostLayer()) };
      int3 stride_frac_field = {int(fractionFieldGPU->xStride()), int(fractionFieldGPU->yStride()), int(fractionFieldGPU->zStride())};

      dim3 _block(uint64_c(((16 < size.x - 2) ? 16 : size.x - 2)), uint64_c(((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))), uint64_c(((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))));
      dim3 _grid(uint64_c(( (size.x - 2) % (((16 < size.x - 2) ? 16 : size.x - 2)) == 0 ? (int64_t)(size.x - 2) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)) : ( (int64_t)(size.x - 2) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)) ) +1 )), uint64_c(( (size.y - 2) % (((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))) == 0 ? (int64_t)(size.y - 2) / (int64_t)(((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))) : ( (int64_t)(size.y - 2) / (int64_t)(((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))) ) +1 )), uint64_c(( (size.z - 2) % (((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))) == 0 ? (int64_t)(size.z - 2) / (int64_t)(((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))) : ( (int64_t)(size.z - 2) / (int64_t)(((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))) ) +1 )));

      addStaticGeometryToFractionFieldKernel<<<_grid, _block>>>(_data_fractionFieldGPU, _data_staticFractionFieldGPU, size, stride_frac_field);
   }
}
} //namespace walberla