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

#define ADDS1(dest,v1,s1) \
         dest.x=v1.x+s1; \
         dest.y=v1.y+s1; \
         dest.z=v1.z+s1;

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

      dim3 _block(uint64_c(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)), uint64_c(((1024 < ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))) ? 1024 : ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))), uint64_c(((64 < ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))))) ? 64 : ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))))));
      dim3 _grid(uint64_c(( (size.x - 2 * ghostLayers_) % (((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)) == 0 ? (int64_t)(size.x - 2 * ghostLayers_) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)) : ( (int64_t)(size.x - 2 * ghostLayers_) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)) ) +1 )), uint64_c(( (size.y - 2 * ghostLayers_) % (((1024 < ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))) ? 1024 : ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))) == 0 ? (int64_t)(size.y - 2 * ghostLayers_) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))) ? 1024 : ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))) : ( (int64_t)(size.y - 2 * ghostLayers_) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))) ? 1024 : ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))) ) +1 )), uint64_c(( (size.z - 2 * ghostLayers_) % (((64 < ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))))) ? 64 : ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))))) == 0 ? (int64_t)(size.z - 2 * ghostLayers_) / (int64_t)(((64 < ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))))) ? 64 : ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))))) : ( (int64_t)(size.z - 2 * ghostLayers_) / (int64_t)(((64 < ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))))) ? 64 : ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))))) ) +1 )));

      resetFractionFieldGPUKernel<<<_grid, _block>>>(_data_FractionFieldGPU, size, stride_frac_field);
   }
}

__global__ void getFractionFieldFromGeometryMeshKernel(real_t * RESTRICT const _data_fractionFieldGPU, geoSize * RESTRICT const _data_geometryFieldGPU, int3 field_size, int3 field_stride, int3 geometry_field_size, int3 geometry_field_stride, double3 blockAABBMin, double3 meshAABBMin, float3 dxyz, int superSamplingDepth, int interpolationStencilSize, double oneOverInterpolArea, float3 dxyzSS, double3 meshCenter, double3 rotationMatrixX, double3 rotationMatrixY, double3 rotationMatrixZ, double3 translation) {
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < field_size.x  && y < field_size.y  && z < field_size.z ) {
      const int idx = (x) + (y) * field_stride.y + (z) * field_stride.z;

      double3 vecDxSSHalf = {0.5 * dxyzSS.x, 0.5 * dxyzSS.y, 0.5 * dxyzSS.z};
      double3 cellCenter = { blockAABBMin.x + double(x) * dxyz.x + 0.5 * dxyz.x, blockAABBMin.y + double(y) * dxyz.y +  0.5 * dxyz.y, blockAABBMin.z + double(z) * dxyz.z +  0.5 * dxyz.z };
      double3 rotatedCellCenter;

      //translation
      SUB(cellCenter, cellCenter, translation)

      //rotation
      SUB(cellCenter, cellCenter, meshCenter)
      rotatedCellCenter.x = cellCenter.x * rotationMatrixX.x + cellCenter.y * rotationMatrixX.y + cellCenter.z * rotationMatrixX.z;
      rotatedCellCenter.y = cellCenter.x * rotationMatrixY.x + cellCenter.y * rotationMatrixY.y + cellCenter.z * rotationMatrixY.z;
      rotatedCellCenter.z = cellCenter.x * rotationMatrixZ.x + cellCenter.y * rotationMatrixZ.y + cellCenter.z * rotationMatrixZ.z;
      ADD(rotatedCellCenter, rotatedCellCenter, meshCenter)

      double3 pointInGeometrySpace;
      SUB(pointInGeometrySpace, rotatedCellCenter, meshAABBMin);
      SUB(pointInGeometrySpace, pointInGeometrySpace, vecDxSSHalf);

      //get cell of geometry field
      int3 cellInGeometrySpace;
      cellInGeometrySpace.x = int((pointInGeometrySpace.x) / dxyzSS.x);
      cellInGeometrySpace.y = int((pointInGeometrySpace.y) / dxyzSS.y);
      cellInGeometrySpace.z = int((pointInGeometrySpace.z) / dxyzSS.z);

      double fraction = 0.0;

      if (cellInGeometrySpace.x < 0 || cellInGeometrySpace.x >= geometry_field_size.x ||
          cellInGeometrySpace.y < 0 || cellInGeometrySpace.y >= geometry_field_size.y ||
          cellInGeometrySpace.z < 0 || cellInGeometrySpace.z >= geometry_field_size.z )
      {
         fraction = 0.0;
      }
      else if (superSamplingDepth == 0){

         double3 cellCenterInGeometrySpace = {double(cellInGeometrySpace.x) * dxyzSS.x, double(cellInGeometrySpace.y) * dxyzSS.y, double(cellInGeometrySpace.z) * dxyzSS.z};
         double3 distanceToCellCenter;
         SUB(distanceToCellCenter, pointInGeometrySpace, cellCenterInGeometrySpace)
         int3 offset = {int(distanceToCellCenter.x / abs(distanceToCellCenter.x)), int(distanceToCellCenter.y / abs(distanceToCellCenter.y)), int(distanceToCellCenter.z / abs(distanceToCellCenter.z))};
         int3 iterationStart = {((offset.x < 0) ? -1 : 0), ((offset.y < 0) ? -1 : 0), ((offset.z < 0) ? -1 : 0)};
         int3 iterationEnd;
         ADDS1(iterationEnd, iterationStart, interpolationStencilSize)
         for (int zOff = iterationStart.z; zOff < iterationEnd.z; ++zOff) {
            for (int yOff = iterationStart.y; yOff < iterationEnd.y; ++yOff) {
               for (int xOff = iterationStart.x; xOff < iterationEnd.x; ++xOff) {
                  int idx_geo = (cellInGeometrySpace.x + xOff) + (cellInGeometrySpace.y + yOff) * geometry_field_stride.y + (cellInGeometrySpace.z + zOff) * geometry_field_stride.z;
                  fraction += _data_geometryFieldGPU[idx_geo];
               }
            }
         }
         fraction *= oneOverInterpolArea;
      }
      else {
         int halfInterpolationStencilSize = int(real_t(interpolationStencilSize) * 0.5);
         for (int zOff = -halfInterpolationStencilSize; zOff <= halfInterpolationStencilSize; ++zOff) {
            for (int yOff = -halfInterpolationStencilSize; yOff <= halfInterpolationStencilSize; ++yOff) {
               for (int xOff = -halfInterpolationStencilSize; xOff <= halfInterpolationStencilSize; ++xOff) {
                  int idx_geo = (cellInGeometrySpace.x + xOff) + (cellInGeometrySpace.y + yOff) * geometry_field_stride.y + (cellInGeometrySpace.z + zOff) * geometry_field_stride.z;
                  fraction += _data_geometryFieldGPU[idx_geo];
               }
            }
         }
         fraction *= oneOverInterpolArea;
      }
      _data_fractionFieldGPU[idx] = min(1.0, _data_fractionFieldGPU[idx] + fraction);
   }
}

void MovingGeometry::getFractionFieldFromGeometryMesh(uint_t timestep)  {

   auto geometryMovement = movementFunction_(timestep);
   Matrix3<real_t>rotationMat(geometryMovement.rotationAxis, -geometryMovement.rotationAngle);
   double3 rotationMatrixX = {rotationMat[0], rotationMat[1], rotationMat[2]};
   double3 rotationMatrixY = {rotationMat[3], rotationMat[4], rotationMat[5]};
   double3 rotationMatrixZ = {rotationMat[6], rotationMat[7], rotationMat[8]};
   double3 translation = {geometryMovement.translationVector[0], geometryMovement.translationVector[1], geometryMovement.translationVector[2]};

   for (auto& block : *blocks_) {
      if(!geometryMovement.movementBoundingBox.intersects(block.getAABB()) )
         continue;
      uint_t level = blocks_->getLevel(block);
      auto fractionFieldGPU = block.getData< gpu::GPUField<real_t> >(fractionFieldId_);
      real_t * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);
      geoSize * RESTRICT const _data_geometryFieldGPU = geometryFieldGPU_->dataAt(0, 0, 0, 0);

      float3 dxyz = {float(blocks_->dx(level)), float(blocks_->dy(level)), float(blocks_->dz(level))};
      double3 meshCenterGPU = {meshCenter[0], meshCenter[1], meshCenter[2]};
      auto blockAABB = block.getAABB();
      double3 blockAABBmin = {blockAABB.minCorner()[0], blockAABB.minCorner()[1], blockAABB.minCorner()[2]};
      double3 meshAABBmin = {meshAABB_.minCorner()[0], meshAABB_.minCorner()[1], meshAABB_.minCorner()[2]};

      uint_t interpolationStencilSize = uint_t( pow(2, real_t(superSamplingDepth_)) + 1);
      auto oneOverInterpolArea = 1.0 / real_t( interpolationStencilSize * interpolationStencilSize * interpolationStencilSize);
      Vector3<real_t> dxyzSSreal_t = maxRefinementDxyz_ / pow(2, real_t(superSamplingDepth_));
      float3 dxyzSS = {float(dxyzSSreal_t[0]), float(dxyzSSreal_t[1]), float(dxyzSSreal_t[2])};

      int3 field_size = {int(fractionFieldGPU->xSizeWithGhostLayer()), int(fractionFieldGPU->ySizeWithGhostLayer()), int(fractionFieldGPU->zSizeWithGhostLayer()) };
      int3 field_stride = {int(fractionFieldGPU->xStride()), int(fractionFieldGPU->yStride()), int(fractionFieldGPU->zStride())};

      int3 geometry_field_size = {int(geometryFieldGPU_->xSize()), int(geometryFieldGPU_->ySize()), int(geometryFieldGPU_->zSize()) };
      int3 geometry_field_stride = {int(geometryFieldGPU_->xStride()), int(geometryFieldGPU_->yStride()), int(geometryFieldGPU_->zStride())};

      dim3 _block(uint64_c(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)), uint64_c(((1024 < ((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))) ? 1024 : ((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))))), uint64_c(((64 < ((field_size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))))) ? field_size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))))))) ? 64 : ((field_size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))))) ? field_size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))))))));
      dim3 _grid(uint64_c(( (field_size.x - 2 * ghostLayers_) % (((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)) == 0 ? (int64_t)(field_size.x - 2 * ghostLayers_) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)) : ( (int64_t)(field_size.x - 2 * ghostLayers_) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)) ) +1 )), uint64_c(( (field_size.y - 2 * ghostLayers_) % (((1024 < ((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))) ? 1024 : ((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))))) == 0 ? (int64_t)(field_size.y - 2 * ghostLayers_) / (int64_t)(((1024 < ((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))) ? 1024 : ((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))))) : ( (int64_t)(field_size.y - 2 * ghostLayers_) / (int64_t)(((1024 < ((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))) ? 1024 : ((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))))) ) +1 )), uint64_c(( (field_size.z - 2 * ghostLayers_) % (((64 < ((field_size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))))) ? field_size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))))))) ? 64 : ((field_size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))))) ? field_size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))))))) == 0 ? (int64_t)(field_size.z - 2 * ghostLayers_) / (int64_t)(((64 < ((field_size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))))) ? field_size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))))))) ? 64 : ((field_size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))))) ? field_size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))))))) : ( (int64_t)(field_size.z - 2 * ghostLayers_) / (int64_t)(((64 < ((field_size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))))) ? field_size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))))))) ? 64 : ((field_size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))))) ? field_size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)*((field_size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_)))) ? field_size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers_) ? 16 : field_size.x - 2 * ghostLayers_))))))))) ) +1 )));

      getFractionFieldFromGeometryMeshKernel<<<_grid, _block>>>(_data_fractionFieldGPU, _data_geometryFieldGPU, field_size, field_stride,
                                                                      geometry_field_size, geometry_field_stride, blockAABBmin, meshAABBmin,
                                                                      dxyz, superSamplingDepth_, interpolationStencilSize, oneOverInterpolArea, dxyzSS, meshCenterGPU,
                                                                      rotationMatrixX, rotationMatrixY, rotationMatrixZ, translation);
   }
}

__global__ void addStaticGeometryToFractionFieldKernel( real_t * RESTRICT const fractionFieldData, real_t * RESTRICT const staticFractionFieldData, int3 fieldSize, int3 stride) {
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < fieldSize.x  && y < fieldSize.y  && z < fieldSize.z )
   {
      const int idx = x + y * stride.y + z * stride.z;

      fractionFieldData[idx] = min(1.0, fractionFieldData[idx] + staticFractionFieldData[idx]);
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

      dim3 _block(uint64_c(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)), uint64_c(((1024 < ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))) ? 1024 : ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))), uint64_c(((64 < ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))))) ? 64 : ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))))));
      dim3 _grid(uint64_c(( (size.x - 2 * ghostLayers_) % (((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)) == 0 ? (int64_t)(size.x - 2 * ghostLayers_) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)) : ( (int64_t)(size.x - 2 * ghostLayers_) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)) ) +1 )), uint64_c(( (size.y - 2 * ghostLayers_) % (((1024 < ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))) ? 1024 : ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))) == 0 ? (int64_t)(size.y - 2 * ghostLayers_) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))) ? 1024 : ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))) : ( (int64_t)(size.y - 2 * ghostLayers_) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))) ? 1024 : ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))) ) +1 )), uint64_c(( (size.z - 2 * ghostLayers_) % (((64 < ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))))) ? 64 : ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))))) == 0 ? (int64_t)(size.z - 2 * ghostLayers_) / (int64_t)(((64 < ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))))) ? 64 : ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))))) : ( (int64_t)(size.z - 2 * ghostLayers_) / (int64_t)(((64 < ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))))) ? 64 : ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))))) ) +1 )));

      addStaticGeometryToFractionFieldKernel<<<_grid, _block>>>(_data_fractionFieldGPU, _data_staticFractionFieldGPU, size, stride_frac_field);
   }
}

__global__ void updateObjectVelocityFieldKernel(real_t * RESTRICT const _data_objectVelocityFieldGPU, real_t * RESTRICT const _data_fractionFieldGPU, int3 field_size, int3 field_stride, int fStride, double3 blockAABBMin, double3 dxyz, double3 meshCenter, double3 angularVel, double3 translationSpeed, bool timeDependentMovement, double3 movementBoundingBoxMin, double3 movementBoundingBoxMax) {

   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < field_size.x  && y < field_size.y  && z < field_size.z )
   {
      const int idx = (x) + (y) *field_stride.y + (z) *field_stride.z;

      double3 cellCenter = { blockAABBMin.x + double(x) * dxyz.x + 0.5 * dxyz.x, blockAABBMin.y + double(y) * dxyz.y + 0.5 * dxyz.y,
                             blockAABBMin.z + double(z) * dxyz.z + 0.5 * dxyz.z };

      if(timeDependentMovement)
      {
         if (_data_fractionFieldGPU[idx] <= 0.0)
         {
            _data_objectVelocityFieldGPU[idx + 0 * fStride] = 0.0;
            _data_objectVelocityFieldGPU[idx + 1 * fStride] = 0.0;
            _data_objectVelocityFieldGPU[idx + 2 * fStride] = 0.0;
            return;
         }
      }
      else {
         if (cellCenter.x + 0.5*dxyz.x < movementBoundingBoxMin.x || cellCenter.y + 0.5*dxyz.y < movementBoundingBoxMin.y || cellCenter.z + 0.5*dxyz.z < movementBoundingBoxMin.z
             || cellCenter.x - 0.5*dxyz.x  > movementBoundingBoxMax.x || cellCenter.y - 0.5*dxyz.y > movementBoundingBoxMax.y || cellCenter.z - 0.5*dxyz.z > movementBoundingBoxMax.z)
            return;
      }

      double3 distance = { (cellCenter.x - meshCenter.x) / dxyz.x, (cellCenter.y - meshCenter.y) / dxyz.y,
                           (cellCenter.z - meshCenter.z) / dxyz.z };

      double velX = angularVel.y * distance.z - angularVel.z * distance.y;
      double velY = angularVel.z * distance.x - angularVel.x * distance.z;
      double velZ = angularVel.x * distance.y - angularVel.y * distance.x;

      _data_objectVelocityFieldGPU[idx + 0 * fStride] = velX + translationSpeed.x / dxyz.x;
      _data_objectVelocityFieldGPU[idx + 1 * fStride] = velY + translationSpeed.y / dxyz.y;
      _data_objectVelocityFieldGPU[idx + 2 * fStride] = velZ + translationSpeed.z / dxyz.z;
   }
}


void MovingGeometry::updateObjectVelocityField(uint_t timestep) {
   auto geometryMovement = movementFunction_(timestep+1);
   auto geometryMovementLastTimestep = movementFunction_(timestep);
   const Vector3<real_t> dxyz_root = Vector3<real_t>(blocks_->dx(0), blocks_->dy(0), blocks_->dz(0));
   geometryMovement.movementBoundingBox.extend(dxyz_root);

   //update object velocity field only on 0th timestep for time independent movement
   if(!geometryMovement.timeDependentMovement && timestep > 0)
      return;

   auto rotationSpeed = geometryMovement.rotationAngle - geometryMovementLastTimestep.rotationAngle;
   auto translationSpeed = geometryMovement.translationVector - geometryMovementLastTimestep.translationVector;
   double3 translationSpeedGPU = {translationSpeed[0], translationSpeed[1], translationSpeed[2]};
   double3 angularVel = {geometryMovement.rotationAxis[0] * rotationSpeed, geometryMovement.rotationAxis[1] * rotationSpeed, geometryMovement.rotationAxis[2] * rotationSpeed};
   double3 meshCenterGPU = {meshCenter[0], meshCenter[1], meshCenter[2]};
   double3 movementBoundingBoxMin = {geometryMovement.movementBoundingBox.xMin(), geometryMovement.movementBoundingBox.yMin(), geometryMovement.movementBoundingBox.zMin()};
   double3 movementBoundingBoxMax = {geometryMovement.movementBoundingBox.xMax(), geometryMovement.movementBoundingBox.yMax(), geometryMovement.movementBoundingBox.zMax()};
   for (auto& block : *blocks_)
   {
      if(!geometryMovement.movementBoundingBox.intersects(block.getAABB()) )
         continue;

      auto level = blocks_->getLevel(block);
      double3 dxyz = {double(blocks_->dx(level)), double(blocks_->dy(level)), double(blocks_->dz(level))};

      auto blockAABB = block.getAABB();
      double3 blockAABBMin = {blockAABB.minCorner()[0], blockAABB.minCorner()[1], blockAABB.minCorner()[2]};

      auto fractionFieldGPU = block.getData< gpu::GPUField<real_t> >(fractionFieldId_);
      real_t * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);

      auto objectVelocityFieldGPU = block.getData< gpu::GPUField<real_t> >(objectVelocityId_);
      real_t * RESTRICT const _data_objectVelocityFieldGPU = objectVelocityFieldGPU->dataAt(0, 0, 0, 0);


      int3 size = {int(objectVelocityFieldGPU->xSizeWithGhostLayer()), int(objectVelocityFieldGPU->ySizeWithGhostLayer()), int(objectVelocityFieldGPU->zSizeWithGhostLayer()) };
      int3 stride_frac_field = {int(objectVelocityFieldGPU->xStride()), int(objectVelocityFieldGPU->yStride()), int(objectVelocityFieldGPU->zStride())};
      int fStride = objectVelocityFieldGPU->fStride();

      dim3 _block(uint64_c(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)), uint64_c(((1024 < ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))) ? 1024 : ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))), uint64_c(((64 < ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))))) ? 64 : ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))))));
      dim3 _grid(uint64_c(( (size.x - 2 * ghostLayers_) % (((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)) == 0 ? (int64_t)(size.x - 2 * ghostLayers_) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)) : ( (int64_t)(size.x - 2 * ghostLayers_) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)) ) +1 )), uint64_c(( (size.y - 2 * ghostLayers_) % (((1024 < ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))) ? 1024 : ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))) == 0 ? (int64_t)(size.y - 2 * ghostLayers_) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))) ? 1024 : ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))) : ( (int64_t)(size.y - 2 * ghostLayers_) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))) ? 1024 : ((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))) ) +1 )), uint64_c(( (size.z - 2 * ghostLayers_) % (((64 < ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))))) ? 64 : ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))))) == 0 ? (int64_t)(size.z - 2 * ghostLayers_) / (int64_t)(((64 < ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))))) ? 64 : ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))))) : ( (int64_t)(size.z - 2 * ghostLayers_) / (int64_t)(((64 < ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))))))) ? 64 : ((size.z - 2 * ghostLayers_ < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))) ? size.z - 2 * ghostLayers_ : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)*((size.y - 2 * ghostLayers_ < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_)))) ? size.y - 2 * ghostLayers_ : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers_) ? 16 : size.x - 2 * ghostLayers_))))))))) ) +1 )));

      updateObjectVelocityFieldKernel<<<_grid, _block>>>(_data_objectVelocityFieldGPU, _data_fractionFieldGPU, size, stride_frac_field, fStride, blockAABBMin, dxyz, meshCenterGPU, angularVel, translationSpeedGPU, geometryMovement.timeDependentMovement, movementBoundingBoxMin, movementBoundingBoxMax);
   }
}

















} //namespace walberla