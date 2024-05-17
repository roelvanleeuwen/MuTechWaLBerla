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

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#ifdef WALBERLA_DOUBLE_ACCURACY
using real_t3 = double3;
#else
using real_t3 = float3;
#endif


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

__device__ double atomicAddCAS(double* address, double val)
{
   unsigned long long int* address_as_ull = (unsigned long long int*)address;
   unsigned long long int old = *address_as_ull, assumed;

   do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
   } while (assumed != old);

   return __longlong_as_double(old);
}



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

template < typename FractionField_T, typename VectorField_T >
void MovingGeometry<FractionField_T, VectorField_T>::resetFractionField() {
   for (auto& block : *blocks_) {
      auto fractionFieldGPU = block.getData< gpu::GPUField<real_t> >(fractionFieldId_);
      real_t * RESTRICT const _data_FractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);
      
      uint_t ghostLayers = fractionFieldGPU->nrOfGhostLayers();
      int3 size = {int(fractionFieldGPU->xSizeWithGhostLayer()), int(fractionFieldGPU->ySizeWithGhostLayer()), int(fractionFieldGPU->zSizeWithGhostLayer()) };
      int3 stride_frac_field = {int(fractionFieldGPU->xStride()), int(fractionFieldGPU->yStride()), int(fractionFieldGPU->zStride())};

      dim3 _block(uint64_c(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)), uint64_c(((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))), uint64_c(((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))));
      dim3 _grid(uint64_c(( (size.x - 2 * ghostLayers) % (((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)) == 0 ? (int64_t)(size.x - 2 * ghostLayers) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)) : ( (int64_t)(size.x - 2 * ghostLayers) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)) ) +1 )), uint64_c(( (size.y - 2 * ghostLayers) % (((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))) == 0 ? (int64_t)(size.y - 2 * ghostLayers) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))) : ( (int64_t)(size.y - 2 * ghostLayers) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))) ) +1 )), uint64_c(( (size.z - 2 * ghostLayers) % (((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))) == 0 ? (int64_t)(size.z - 2 * ghostLayers) / (int64_t)(((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))) : ( (int64_t)(size.z - 2 * ghostLayers) / (int64_t)(((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))) ) +1 )));

      resetFractionFieldGPUKernel<<<_grid, _block>>>(_data_FractionFieldGPU, size, stride_frac_field);
   }
}

template < typename FractionField_T, typename VectorField_T >
__global__ void getFractionFieldFromGeometryMeshKernel(real_t * RESTRICT const _data_fractionFieldGPU, bool * RESTRICT const _data_geometryFieldGPU, int3 field_size, int3 field_stride, int3 geometry_field_size, int3 geometry_field_stride, real_t3 blockAABBMin, real_t3 meshAABBMin, real_t3 dxyz, int superSamplingDepth, int interpolationStencilSize, real_t oneOverInterpolArea, real_t3 dxyzSS, real_t3 meshCenter, real_t3 rotationMatrixX, real_t3 rotationMatrixY, real_t3 rotationMatrixZ, real_t3 translation, real_t tau_, bool useTauInFractionField_) {
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < field_size.x  && y < field_size.y  && z < field_size.z ) {
      const int idx = (x) + (y) * field_stride.y + (z) * field_stride.z;

      real_t3 vecDxSSHalf = {0.5 * dxyzSS.x, 0.5 * dxyzSS.y, 0.5 * dxyzSS.z};
      real_t3 cellCenter = { blockAABBMin.x + real_t(x) * dxyz.x + 0.5 * dxyz.x, blockAABBMin.y + real_t(y) * dxyz.y +  0.5 * dxyz.y, blockAABBMin.z + real_t(z) * dxyz.z +  0.5 * dxyz.z };
      real_t3 rotatedCellCenter;

      //translation
      SUB(cellCenter, cellCenter, translation)

      //rotation
      SUB(cellCenter, cellCenter, meshCenter)
      rotatedCellCenter.x = cellCenter.x * rotationMatrixX.x + cellCenter.y * rotationMatrixX.y + cellCenter.z * rotationMatrixX.z;
      rotatedCellCenter.y = cellCenter.x * rotationMatrixY.x + cellCenter.y * rotationMatrixY.y + cellCenter.z * rotationMatrixY.z;
      rotatedCellCenter.z = cellCenter.x * rotationMatrixZ.x + cellCenter.y * rotationMatrixZ.y + cellCenter.z * rotationMatrixZ.z;
      ADD(rotatedCellCenter, rotatedCellCenter, meshCenter)

      real_t3 pointInGeometrySpace;
      SUB(pointInGeometrySpace, rotatedCellCenter, meshAABBMin);
      SUB(pointInGeometrySpace, pointInGeometrySpace, vecDxSSHalf);

      //get cell of geometry field
      int3 cellInGeometrySpace;
      cellInGeometrySpace.x = int((pointInGeometrySpace.x) / dxyzSS.x);
      cellInGeometrySpace.y = int((pointInGeometrySpace.y) / dxyzSS.y);
      cellInGeometrySpace.z = int((pointInGeometrySpace.z) / dxyzSS.z);

      real_t fraction = 0.0;

      if (cellInGeometrySpace.x < 0 || cellInGeometrySpace.x >= geometry_field_size.x ||
          cellInGeometrySpace.y < 0 || cellInGeometrySpace.y >= geometry_field_size.y ||
          cellInGeometrySpace.z < 0 || cellInGeometrySpace.z >= geometry_field_size.z )
      {
         fraction = 0.0;
      }
      else if (superSamplingDepth == 0){

         real_t3 cellCenterInGeometrySpace = {real_t(cellInGeometrySpace.x) * dxyzSS.x, real_t(cellInGeometrySpace.y) * dxyzSS.y, real_t(cellInGeometrySpace.z) * dxyzSS.z};
         real_t3 distanceToCellCenter;
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
      //B2 from "A comparative study of fluid-particle coupling methods for fully resolved lattice Boltzmann simulations" from Rettinger et al
      if (useTauInFractionField_)
         fraction = fraction * (tau_ - 0.5) / ((1.0 - fraction) + (tau_ - 0.5));

      _data_fractionFieldGPU[idx] = min(1.0, _data_fractionFieldGPU[idx] + fraction);
   }
}

template < typename FractionField_T, typename VectorField_T >
void MovingGeometry<FractionField_T, VectorField_T>::getFractionFieldFromGeometryMesh()  {


   Matrix3<real_t> rotationMat = particleAccessor_->getRotation(0).getMatrix();
   real_t3 rotationMatrixX = {rotationMat[0], rotationMat[1], rotationMat[2]};
   real_t3 rotationMatrixY = {rotationMat[3], rotationMat[4], rotationMat[5]};
   real_t3 rotationMatrixZ = {rotationMat[6], rotationMat[7], rotationMat[8]};

   auto translationVector = particleAccessor_->getPosition(0) - meshCenter_;
   real_t3 translation = {translationVector[0], translationVector[1], translationVector[2]};
   const Vector3<real_t> dxyz_coarse = Vector3<real_t>(blocks_->dx(0), blocks_->dy(0), blocks_->dz(0));

   for (auto& block : *blocks_) {
      if(!movementBoundingBox_.intersects(block.getAABB()) )
         continue;
      uint_t level = blocks_->getLevel(block);

      auto fractionFieldGPU = block.getData< gpu::GPUField<real_t> >(fractionFieldId_);
      real_t * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);

      bool * RESTRICT const _data_geometryFieldGPU = geometryFieldGPU_->dataAt(0, 0, 0, 0);

      real_t3 dxyz = {real_t(blocks_->dx(level)), real_t(blocks_->dy(level)), real_t(blocks_->dz(level))};
      real_t3 meshCenterGPU = {meshCenter_[0], meshCenter_[1], meshCenter_[2]};

      auto blockAABB = block.getAABB();
      real_t3 blockAABBmin = {blockAABB.minCorner()[0], blockAABB.minCorner()[1], blockAABB.minCorner()[2]};
      real_t3 meshAABBmin = {meshAABB_.minCorner()[0], meshAABB_.minCorner()[1], meshAABB_.minCorner()[2]};

      uint_t interpolationStencilSize = uint_t( pow(2, real_t(superSamplingDepth_)) + 1);
      auto oneOverInterpolArea = 1.0 / real_t( interpolationStencilSize * interpolationStencilSize * interpolationStencilSize);
      Vector3<real_t> dxyzSSreal_t = maxRefinementDxyz_ / pow(2, real_t(superSamplingDepth_));
      real_t3 dxyzSS = {real_t(dxyzSSreal_t[0]), real_t(dxyzSSreal_t[1]), real_t(dxyzSSreal_t[2])};

      int3 field_size = {int(fractionFieldGPU->xSizeWithGhostLayer()), int(fractionFieldGPU->ySizeWithGhostLayer()), int(fractionFieldGPU->zSizeWithGhostLayer()) };
      int3 field_stride = {int(fractionFieldGPU->xStride()), int(fractionFieldGPU->yStride()), int(fractionFieldGPU->zStride())};

      int3 geometry_field_size = {int(geometryFieldGPU_->xSize()), int(geometryFieldGPU_->ySize()), int(geometryFieldGPU_->zSize()) };
      int3 geometry_field_stride = {int(geometryFieldGPU_->xStride()), int(geometryFieldGPU_->yStride()), int(geometryFieldGPU_->zStride())};

      uint_t ghostLayers = fractionFieldGPU->nrOfGhostLayers();
      dim3 _block(uint64_c(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)), uint64_c(((1024 < ((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))) ? 1024 : ((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))))), uint64_c(((64 < ((field_size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))))) ? field_size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))))))) ? 64 : ((field_size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))))) ? field_size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))))))));
      dim3 _grid(uint64_c(( (field_size.x - 2 * ghostLayers) % (((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)) == 0 ? (int64_t)(field_size.x - 2 * ghostLayers) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)) : ( (int64_t)(field_size.x - 2 * ghostLayers) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)) ) +1 )), uint64_c(( (field_size.y - 2 * ghostLayers) % (((1024 < ((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))) ? 1024 : ((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))))) == 0 ? (int64_t)(field_size.y - 2 * ghostLayers) / (int64_t)(((1024 < ((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))) ? 1024 : ((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))))) : ( (int64_t)(field_size.y - 2 * ghostLayers) / (int64_t)(((1024 < ((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))) ? 1024 : ((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))))) ) +1 )), uint64_c(( (field_size.z - 2 * ghostLayers) % (((64 < ((field_size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))))) ? field_size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))))))) ? 64 : ((field_size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))))) ? field_size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))))))) == 0 ? (int64_t)(field_size.z - 2 * ghostLayers) / (int64_t)(((64 < ((field_size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))))) ? field_size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))))))) ? 64 : ((field_size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))))) ? field_size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))))))) : ( (int64_t)(field_size.z - 2 * ghostLayers) / (int64_t)(((64 < ((field_size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))))) ? field_size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))))))) ? 64 : ((field_size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))))) ? field_size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)*((field_size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers)))) ? field_size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < field_size.x - 2 * ghostLayers) ? 16 : field_size.x - 2 * ghostLayers))))))))) ) +1 )));

      getFractionFieldFromGeometryMeshKernel<FractionField_T, VectorField_T> <<<_grid, _block>>>(_data_fractionFieldGPU, _data_geometryFieldGPU, field_size, field_stride,
                                                                      geometry_field_size, geometry_field_stride, blockAABBmin, meshAABBmin,
                                                                      dxyz, superSamplingDepth_, interpolationStencilSize, oneOverInterpolArea, dxyzSS, meshCenterGPU,
                                                                      rotationMatrixX, rotationMatrixY, rotationMatrixZ, translation, tau_, useTauInFractionField_);
   }
}


__global__ void updateObjectVelocityFieldKernel(real_t * RESTRICT const _data_objectVelocityFieldGPU, real_t * RESTRICT const _data_fractionFieldGPU, int3 field_size, int3 field_stride, int fStride, real_t3 blockAABBMin, real_t3 dxyz, real_t3 dxyz_root, real_t3 objectPositionGPU, real_t3 angularVel, real_t3 translationSpeed, real_t3 movementBoundingBoxMin, real_t3 movementBoundingBoxMax, real_t dt) {

   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < field_size.x  && y < field_size.y  && z < field_size.z )
   {
      const int idx = (x) + (y) *field_stride.y + (z) *field_stride.z;

      real_t3 cellCenter = { blockAABBMin.x + real_t(x) * dxyz.x + 0.5 * dxyz.x, blockAABBMin.y + real_t(y) * dxyz.y + 0.5 * dxyz.y,
                             blockAABBMin.z + real_t(z) * dxyz.z + 0.5 * dxyz.z };


      if (_data_fractionFieldGPU[idx] <= 0.0)
      {
         _data_objectVelocityFieldGPU[idx + 0 * fStride] = 0.0;
         _data_objectVelocityFieldGPU[idx + 1 * fStride] = 0.0;
         _data_objectVelocityFieldGPU[idx + 2 * fStride] = 0.0;
         return;
      }


      real_t3 distance = { (cellCenter.x - objectPositionGPU.x), (cellCenter.y - objectPositionGPU.y), (cellCenter.z - objectPositionGPU.z)};

      real_t linearVelX = angularVel.y * distance.z - angularVel.z * distance.y;
      real_t linearVelY = angularVel.z * distance.x - angularVel.x * distance.z;
      real_t linearVelZ = angularVel.x * distance.y - angularVel.y * distance.x;

      _data_objectVelocityFieldGPU[idx + 0 * fStride] = (linearVelX + translationSpeed.x) * dt / dxyz_root.x;
      _data_objectVelocityFieldGPU[idx + 1 * fStride] = (linearVelY + translationSpeed.y) * dt / dxyz_root.x;
      _data_objectVelocityFieldGPU[idx + 2 * fStride] = (linearVelZ + translationSpeed.z) * dt / dxyz_root.x;
   }
}


template < typename FractionField_T, typename VectorField_T >
void MovingGeometry<FractionField_T, VectorField_T>::updateObjectVelocityField() {

   auto objectPosition = particleAccessor_->getPosition(0);
   auto objectLinearVelocity = particleAccessor_->getLinearVelocity(0);
   auto objectAngularVelocity = particleAccessor_->getAngularVelocity(0);

   real_t3 translationSpeedGPU = {objectLinearVelocity[0], objectLinearVelocity[1], objectLinearVelocity[2]};
   real_t3 angularVel = {objectAngularVelocity[0], objectAngularVelocity[1], objectAngularVelocity[2]};
   real_t3 objectPositionGPU = {objectPosition[0], objectPosition[1], objectPosition[2]};
   real_t3 movementBoundingBoxMin = {movementBoundingBox_.xMin(), movementBoundingBox_.yMin(), movementBoundingBox_.zMin()};
   real_t3 movementBoundingBoxMax = {movementBoundingBox_.xMax(), movementBoundingBox_.yMax(), movementBoundingBox_.zMax()};
   real_t3 dxyz_root = {real_t(blocks_->dx(0)), real_t(blocks_->dy(0)), real_t(blocks_->dz(0))};

   for (auto& block : *blocks_)
   {
      if(!movementBoundingBox_.intersects(block.getAABB()) )
         continue;

      auto level = blocks_->getLevel(block);
      real_t3 dxyz = {real_t(blocks_->dx(level)), real_t(blocks_->dy(level)), real_t(blocks_->dz(level))};

      auto blockAABB = block.getAABB();
      real_t3 blockAABBMin = {blockAABB.minCorner()[0], blockAABB.minCorner()[1], blockAABB.minCorner()[2]};

      auto fractionFieldGPU = block.getData< gpu::GPUField<real_t> >(fractionFieldId_);
      real_t * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);

      auto objectVelocityFieldGPU = block.getData< gpu::GPUField<real_t> >(objectVelocityId_);
      real_t * RESTRICT const _data_objectVelocityFieldGPU = objectVelocityFieldGPU->dataAt(0, 0, 0, 0);


      int3 size = {int(objectVelocityFieldGPU->xSizeWithGhostLayer()), int(objectVelocityFieldGPU->ySizeWithGhostLayer()), int(objectVelocityFieldGPU->zSizeWithGhostLayer()) };
      int3 stride_frac_field = {int(objectVelocityFieldGPU->xStride()), int(objectVelocityFieldGPU->yStride()), int(objectVelocityFieldGPU->zStride())};
      int fStride = objectVelocityFieldGPU->fStride();

      uint_t ghostLayers = fractionFieldGPU->nrOfGhostLayers();
      dim3 _block(uint64_c(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)), uint64_c(((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))), uint64_c(((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))));
      dim3 _grid(uint64_c(( (size.x - 2 * ghostLayers) % (((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)) == 0 ? (int64_t)(size.x - 2 * ghostLayers) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)) : ( (int64_t)(size.x - 2 * ghostLayers) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)) ) +1 )), uint64_c(( (size.y - 2 * ghostLayers) % (((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))) == 0 ? (int64_t)(size.y - 2 * ghostLayers) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))) : ( (int64_t)(size.y - 2 * ghostLayers) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))) ) +1 )), uint64_c(( (size.z - 2 * ghostLayers) % (((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))) == 0 ? (int64_t)(size.z - 2 * ghostLayers) / (int64_t)(((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))) : ( (int64_t)(size.z - 2 * ghostLayers) / (int64_t)(((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))) ) +1 )));

      updateObjectVelocityFieldKernel<<<_grid, _block>>>(_data_objectVelocityFieldGPU, _data_fractionFieldGPU, size, stride_frac_field, fStride, blockAABBMin, dxyz, dxyz_root, objectPositionGPU, angularVel, translationSpeedGPU, movementBoundingBoxMin, movementBoundingBoxMax, dt_);
   }
}

__global__ void calculateForcesOnBodyKernel(real_t * RESTRICT const _data_forceFieldGPU, real_t * RESTRICT const _data_fractionFieldGPU, int3 field_size, int3 stride, int fStride, real_t* RESTRICT const hydrodynamicForce, real_t* __restrict__ const hydrodynamicTorque) {
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < field_size.x  && y < field_size.y  && z < field_size.z )
   {
      const int idx = (x) + (y) * stride.y + (z) * stride.z;

      if(_data_fractionFieldGPU[idx] > 0) {
         gpuAtomicAdd( &(hydrodynamicForce[0]), _data_forceFieldGPU[idx + 0 * fStride ]);
         gpuAtomicAdd( &(hydrodynamicForce[1]), _data_forceFieldGPU[idx + 1 * fStride ]);
         gpuAtomicAdd( &(hydrodynamicForce[2]), _data_forceFieldGPU[idx + 2 * fStride ]);
      }
   }
}

template < typename FractionField_T, typename VectorField_T >
void MovingGeometry<FractionField_T, VectorField_T>::calculateForcesOnBody() {

   Vector3<real_t> summedForceOnObject;

   real_t * forceOnDevice;
   gpuMalloc((void **) &forceOnDevice, 3 * sizeof(real_t));
   cudaMemset(forceOnDevice, 0, 3 * sizeof(real_t));

   real_t * torqueOnDevice;
   gpuMalloc((void **) &torqueOnDevice, 3 * sizeof(real_t));
   cudaMemset(torqueOnDevice, 0, 3 * sizeof(real_t));

   for (auto &block : *blocks_) {
      auto forceFieldGPU = block.getData< gpu::GPUField<real_t> >(forceFieldId_);
      real_t * RESTRICT const _data_forceFieldGPU = forceFieldGPU->dataAt(0, 0, 0, 0);

      auto fractionFieldGPU = block.getData< gpu::GPUField<real_t> >(fractionFieldId_);
      real_t * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);

      int3 size = {int(forceFieldGPU->xSizeWithGhostLayer()), int(forceFieldGPU->ySizeWithGhostLayer()), int(forceFieldGPU->zSizeWithGhostLayer()) };
      int3 stride_frac_field = {int(forceFieldGPU->xStride()), int(forceFieldGPU->yStride()), int(forceFieldGPU->zStride())};
      int fStride = forceFieldGPU->fStride();

      uint_t ghostLayers = fractionFieldGPU->nrOfGhostLayers();
      dim3 _block(uint64_c(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)), uint64_c(((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))), uint64_c(((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))));
      dim3 _grid(uint64_c(( (size.x - 2 * ghostLayers) % (((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)) == 0 ? (int64_t)(size.x - 2 * ghostLayers) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)) : ( (int64_t)(size.x - 2 * ghostLayers) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)) ) +1 )), uint64_c(( (size.y - 2 * ghostLayers) % (((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))) == 0 ? (int64_t)(size.y - 2 * ghostLayers) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))) : ( (int64_t)(size.y - 2 * ghostLayers) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))) ) +1 )), uint64_c(( (size.z - 2 * ghostLayers) % (((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))) == 0 ? (int64_t)(size.z - 2 * ghostLayers) / (int64_t)(((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))) : ( (int64_t)(size.z - 2 * ghostLayers) / (int64_t)(((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))) ) +1 )));

      calculateForcesOnBodyKernel<<<_grid, _block>>> (_data_forceFieldGPU, _data_fractionFieldGPU, size, stride_frac_field, fStride, forceOnDevice, torqueOnDevice);
   }

   gpuDeviceSynchronize();
   real_t forceOnHost[3];
   gpuMemcpy(forceOnHost, forceOnDevice, 3 * sizeof(real_t), gpuMemcpyDeviceToHost);
   summedForceOnObject = Vector3<real_t> (forceOnHost[0], forceOnHost[1], forceOnHost[2]);
   WALBERLA_MPI_SECTION() {
      walberla::mpi::reduceInplace(summedForceOnObject, walberla::mpi::SUM);
   }
   real_t forceFactor = fluidDensity_ * pow(blocks_->dx(0),4) / (dt_ * dt_); //(kg / m³ -> kg m / s²)
   Vector3<real_t> forceSI = summedForceOnObject * forceFactor;
   particleAccessor_->setForce(0, forceSI);
}


__global__ void getVolumeFromFractionFieldKernel(real_t * RESTRICT const _data_fractionFieldGPU, int3 field_size, int3 stride, real_t* __restrict__ const summedFraction) {
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < field_size.x  && y < field_size.y  && z < field_size.z )
   {
      const int idx = (x) + (y) * stride.y + (z) * stride.z;

      if(_data_fractionFieldGPU[idx] > 0) {
         gpuAtomicAdd( &(summedFraction[0]), _data_fractionFieldGPU[idx]);
      }
   }
}



template < typename FractionField_T, typename VectorField_T>
real_t MovingGeometry<FractionField_T, VectorField_T>::getVolumeFromFractionField() {
   real_t * summedFraction_d;
   gpuMalloc((void **) &summedFraction_d, 1 * sizeof(real_t));
   cudaMemset(summedFraction_d, 0, 1 * sizeof(real_t));


   for (auto &block : *blocks_) {

      auto fractionFieldGPU = block.getData< gpu::GPUField<real_t> >(fractionFieldId_);
      real_t * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);

      int3 size = {int(fractionFieldGPU->xSizeWithGhostLayer()), int(fractionFieldGPU->ySizeWithGhostLayer()), int(fractionFieldGPU->zSizeWithGhostLayer()) };
      int3 stride_frac_field = {int(fractionFieldGPU->xStride()), int(fractionFieldGPU->yStride()), int(fractionFieldGPU->zStride())};

      uint_t ghostLayers = fractionFieldGPU->nrOfGhostLayers();
      dim3 _block(uint64_c(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)), uint64_c(((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))), uint64_c(((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))));
      dim3 _grid(uint64_c(( (size.x - 2 * ghostLayers) % (((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)) == 0 ? (int64_t)(size.x - 2 * ghostLayers) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)) : ( (int64_t)(size.x - 2 * ghostLayers) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)) ) +1 )), uint64_c(( (size.y - 2 * ghostLayers) % (((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))) == 0 ? (int64_t)(size.y - 2 * ghostLayers) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))) : ( (int64_t)(size.y - 2 * ghostLayers) / (int64_t)(((1024 < ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))) ? 1024 : ((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))) ) +1 )), uint64_c(( (size.z - 2 * ghostLayers) % (((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))) == 0 ? (int64_t)(size.z - 2 * ghostLayers) / (int64_t)(((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))) : ( (int64_t)(size.z - 2 * ghostLayers) / (int64_t)(((64 < ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))))))) ? 64 : ((size.z - 2 * ghostLayers < ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))) ? size.z - 2 * ghostLayers : ((int64_t)(256) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)*((size.y - 2 * ghostLayers < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers)))) ? size.y - 2 * ghostLayers : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2 * ghostLayers) ? 16 : size.x - 2 * ghostLayers))))))))) ) +1 )));

      getVolumeFromFractionFieldKernel<<<_grid, _block>>> (_data_fractionFieldGPU, size, stride_frac_field, summedFraction_d);
   }

   gpuDeviceSynchronize();
   real_t summedFraction_h[1];
   gpuMemcpy(summedFraction_h, summedFraction_d, 1 * sizeof(real_t), gpuMemcpyDeviceToHost);
   real_t summedFraction = summedFraction_h[0];
   WALBERLA_MPI_SECTION() {
      walberla::mpi::reduceInplace(summedFraction, walberla::mpi::SUM);
   }
   return summedFraction;
}



template class MovingGeometry<field::GhostLayerField< real_t, 1 >, field::GhostLayerField< real_t, 3 >>;


} //namespace walberla