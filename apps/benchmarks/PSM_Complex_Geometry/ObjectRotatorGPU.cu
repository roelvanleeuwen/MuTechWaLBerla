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
#include "ObjectRotatorGPU.h"

#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wshadow"
#   pragma GCC diagnostic ignored "-Wconversion"
#   pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#if ( defined WALBERLA_CXX_COMPILER_IS_INTEL )
#pragma warning push
#pragma warning( disable :  1599 )
#endif


#define DOT(v1,v2) (v1.x*v2.x+v1.y*v2.y+v1.z*v2.z)

#define DOT2(v1,v2) (v1.x*v2.x+v1.y*v2.y)

#define SQNORM2(v1) (v1.x*v1.x + v1.y*v1.y)

#define SUB(dest,v1,v2) \
         dest.x=v1.x-v2.x; \
         dest.y=v1.y-v2.y; \
         dest.z=v1.z-v2.z;

#define SUB2(dest,v1,v2) \
         dest.x=v1.x-v2.x; \
         dest.y=v1.y-v2.y;

#define ADD(dest,v1,v2) \
         dest.x=v1.x+v2.x; \
         dest.y=v1.y+v2.y; \
         dest.z=v1.z+v2.z;

#define MATVECMUL(dest, m1, v2) \
         dest.x=m1[0]*v2.x+m1[1]*v2.y+m1[2]*v2.z; \
         dest.y=m1[3]*v2.x+m1[4]*v2.y+m1[5]*v2.z; \
         dest.z=m1[6]*v2.x+m1[7]*v2.y+m1[8]*v2.z;

#define TRANSPOSE(dest, m1) \
         dest[0] = m1[0]; dest[1] = m1[3]; dest[2] = m1[6]; \
         dest[3] = m1[1]; dest[4] = m1[4]; dest[5] = m1[7]; \
         dest[6] = m1[2]; dest[7] = m1[5]; dest[8] = m1[8];

namespace walberla
{


__global__ void resetFractionFieldGPU( fracSize * RESTRICT const fractionFieldData, int3 fieldSize, int3 stride) {
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < fieldSize.x  && y < fieldSize.y  && z < fieldSize.z )
   {
      const int idx = (x) + (y) * stride.y + (z) * stride.z;

      fractionFieldData[idx] = 0;
   }
}

void resetFractionFieldGPUCall(shared_ptr< StructuredBlockForest >& blocks, BlockDataID fractionFieldGPUId) {
   for (auto& block : *blocks) {
      auto fractionFieldGPU = block.getData< gpu::GPUField<fracSize> >(fractionFieldGPUId);
      fracSize * RESTRICT const _data_FractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);
      int3 size = {int(fractionFieldGPU->xSizeWithGhostLayer()), int(fractionFieldGPU->ySizeWithGhostLayer()), int(fractionFieldGPU->zSizeWithGhostLayer()) };
      int3 stride_frac_field = {int(fractionFieldGPU->xStride()), int(fractionFieldGPU->yStride()), int(fractionFieldGPU->zStride())};

      dim3 _block(uint64_c(((16 < size.x - 2) ? 16 : size.x - 2)), uint64_c(((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))), uint64_c(((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))));
      dim3 _grid(uint64_c(( (size.x - 2) % (((16 < size.x - 2) ? 16 : size.x - 2)) == 0 ? (int64_t)(size.x - 2) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)) : ( (int64_t)(size.x - 2) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)) ) +1 )), uint64_c(( (size.y - 2) % (((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))) == 0 ? (int64_t)(size.y - 2) / (int64_t)(((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))) : ( (int64_t)(size.y - 2) / (int64_t)(((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))) ) +1 )), uint64_c(( (size.z - 2) % (((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))) == 0 ? (int64_t)(size.z - 2) / (int64_t)(((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))) : ( (int64_t)(size.z - 2) / (int64_t)(((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))) ) +1 )));

      resetFractionFieldGPU<<<_grid, _block>>>(_data_FractionFieldGPU, size, stride_frac_field);
   }
}


__global__ void interpolateFractionFields(fracSize * RESTRICT const fractionFieldData, fracSize * RESTRICT const tmpFractionFieldData, fracSize * RESTRICT const tmpFractionFieldOldData, int3 fieldSize, int3 stride_frac_field, double interpolFactor) {

   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z;
   if (x < fieldSize.x && y < fieldSize.y && z < fieldSize.z )
   {
      const int idx = x  + y * stride_frac_field.y + z  * stride_frac_field.z;
      fractionFieldData[idx] += (interpolFactor * tmpFractionFieldOldData[idx] + (1.0 - interpolFactor) * tmpFractionFieldData[idx]);
   }
}

void ObjectRotatorGPU::interpolateFractionFieldsCall(uint_t timestep) {

   double interpolFactor = double(frequency_ - (timestep % frequency_)) / double(frequency_);

   for (auto& block : *blocks_)
   {
      auto tmpFractionFieldGPU = block.getData< gpu::GPUField<fracSize> >(tmpFracFieldGPUId);
      fracSize * RESTRICT const _data_tmpFractionFieldGPU = tmpFractionFieldGPU->dataAt(0, 0, 0, 0);

      auto tmpFractionFieldGPUOld = block.getData< gpu::GPUField<fracSize> >(tmpFracFieldGPUOldId);
      fracSize * RESTRICT const _data_tmpFractionFieldGPUOld = tmpFractionFieldGPUOld->dataAt(0, 0, 0, 0);

      auto fractionFieldGPU = block.getData< gpu::GPUField<fracSize> >(fractionFieldGPUId_);
      fracSize * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);

      int3 stride_frac_field = {int(tmpFractionFieldGPU->xStride()), int(tmpFractionFieldGPU->yStride()), int(tmpFractionFieldGPU->zStride())};
      int3 size = {int(tmpFractionFieldGPU->xSizeWithGhostLayer()), int(tmpFractionFieldGPU->ySizeWithGhostLayer()), int(tmpFractionFieldGPU->zSizeWithGhostLayer()) };

      dim3 _block(uint64_c(((16 < size.x - 2) ? 16 : size.x - 2)), uint64_c(((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))), uint64_c(((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))));
      dim3 _grid(uint64_c(( (size.x - 2) % (((16 < size.x - 2) ? 16 : size.x - 2)) == 0 ? (int64_t)(size.x - 2) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)) : ( (int64_t)(size.x - 2) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)) ) +1 )), uint64_c(( (size.y - 2) % (((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))) == 0 ? (int64_t)(size.y - 2) / (int64_t)(((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))) : ( (int64_t)(size.y - 2) / (int64_t)(((1024 < ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))) ? 1024 : ((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))) ) +1 )), uint64_c(( (size.z - 2) % (((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))) == 0 ? (int64_t)(size.z - 2) / (int64_t)(((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))) : ( (int64_t)(size.z - 2) / (int64_t)(((64 < ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))))))) ? 64 : ((size.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))) ? size.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)*((size.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2)))) ? size.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size.x - 2) ? 16 : size.x - 2))))))))) ) +1 )));

      interpolateFractionFields<<<_grid, _block>>>(_data_fractionFieldGPU, _data_tmpFractionFieldGPU, _data_tmpFractionFieldGPUOld, size, stride_frac_field, interpolFactor);
   }
}


__device__ double getSqSignedDistance(DistancePropertiesGPU * distancePropertiesGpu, int numFaces, double3 cellCenter) {
   // get sqDistance
   double final_sqDistance = INFINITY;
   double3 final_closestPoint;
   double3 final_normal;

   for (int i = 0; i < numFaces; ++i) {
      DistancePropertiesGPU dp = distancePropertiesGpu[i];
      double3 closestPoint;
      int region;
      double3 temp;
      double2 temp2;

      double3 pt3;
      ADD(temp, cellCenter, dp.translation)
      MATVECMUL(pt3, dp.rotation, temp)

      closestPoint.z = 0.0;

      double sqDistance = pt3.z * pt3.z;

      double2 pt = {pt3.x, pt3.y};

      double e0p = DOT2(dp.e0_normalized, pt);
      double e1p = DOT2(dp.e1_normalized, pt);
      SUB2(temp2, pt, dp.e0)
      double e2p = DOT2(dp.e2_normalized, temp2);

      temp2 = {-1,0};
      double e0d = DOT2(temp2, pt);
      double e1d = DOT2(dp.e1_normal, pt);
      SUB2(temp2, pt, dp.e0)
      double e2d = DOT2(dp.e2_normal, temp2);

      if( e0p <= 0 && e1p <= 0  )
      {
         // Voronoi area of vertex 0
         region = 1;
         sqDistance += SQNORM2(pt); // distance from v0
         closestPoint.x = closestPoint.y = 0;
      }
      else if( e0p >= dp.e0l && e2p <= 0 )
      {
         // Voronoi area of vertex 1
         region = 2;
         SUB2(temp2, pt, dp.e0)
         sqDistance += SQNORM2(temp2); // distance from v1
         closestPoint.x = dp.e0.x;
         closestPoint.y = dp.e0.y;
      }
      else if( e1p >= dp.e1l && e2p >= dp.e2l )
      {
         // Voronoi area of vertex 2
         region = 3;
         SUB2(temp2, pt, dp.e1)
         sqDistance += SQNORM2(temp2); // distance from v2
         closestPoint.x = dp.e1.x;
         closestPoint.y = dp.e1.y;
      }
      else if( e0d <= 0 && e1d <= 0 && e2d <= 0 )
      {
         // Voronoi area of face
         region = 0;
         // result += 0;
         closestPoint.x = pt.x;
         closestPoint.y = pt.y;
      }
      else if( e0d >= 0 && e0p > 0 && e0p < dp.e0l )
      {
         // Voronoi area of edge 0
         region = 4;
         sqDistance += pt.x * pt.x;
         closestPoint.x = 0;
         closestPoint.y = pt.y;
      }
      else if( e1d >= 0 && e1p > 0 && e1p < dp.e1l )
      {
         // Voronoi area of edge 1
         region = 5;
         sqDistance += e1d * e1d;
         closestPoint.x = e1p * dp.e1_normalized.x;
         closestPoint.y = e1p * dp.e1_normalized.y;
      }
      else if( e2d >= 0 && e2p > 0 && e2p < dp.e2l )
      {
         // Voronoi area of edge 2
         region = 6;
         sqDistance += e2d * e2d;
         closestPoint.x = dp.e0.x + e2p * dp.e2_normalized.x;
         closestPoint.y = dp.e0.y + e2p * dp.e2_normalized.y;
      }
      else
      {
         //TODO ERROR MESSAGE AND ABORT
         printf("Error in sqDist calculation\n");
         return 0.0;
      }

      if(sqDistance <= final_sqDistance) {
         final_sqDistance = sqDistance;
         double transpose[9];
         TRANSPOSE(transpose, dp.rotation)
         MATVECMUL(temp, transpose, closestPoint)
         SUB(final_closestPoint, temp, dp.translation)
         final_normal = dp.region_normal[region];
      }
   }
   //printf("final_sqDistance is %f\n", final_sqDistance);
   double3 temp;
   SUB(temp, cellCenter, final_closestPoint)
   double dot = DOT(temp, final_normal);
   return dot >= 0.0 ? final_sqDistance : -final_sqDistance;
}


__global__ void voxelizeGPU(DistancePropertiesGPU * distancePropertiesGpu, fracSize * RESTRICT const fractionFieldData,
                            double3 minAABB, int3 cellBBSize, int3 cellBBLocalMin, int3 stride_frac_field, double dx,
                            int numFaces, double3 rotationMatrixX, double3 rotationMatrixY, double3 rotationMatrixZ,
                            double3 meshCenter)
{
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < cellBBSize.x  && y < cellBBSize.y  && z < cellBBSize.z )
   {
      const int idx = (x + cellBBLocalMin.x) + (y + cellBBLocalMin.y) * stride_frac_field.y + (z + cellBBLocalMin.z) * stride_frac_field.z;

      double dxHalf = 0.5 * dx;    //0.5f * dx;
      real_t sqDx = dx * dx;
      real_t sqDxHalf = (0.5 * dx) * (0.5 * dx);
      double3 cellCenter = { minAABB.x + double(x) * dx + dxHalf, minAABB.y + double(y) * dx + dxHalf,
                            minAABB.z + double(z) * dx + dxHalf };

      //rotate cell center instead of mesh
      double3 newCellCenter;
      SUB(cellCenter, cellCenter, meshCenter)
      newCellCenter.x = cellCenter.x * rotationMatrixX.x + cellCenter.y * rotationMatrixX.y + cellCenter.z * rotationMatrixX.z;
      newCellCenter.y = cellCenter.x * rotationMatrixY.x + cellCenter.y * rotationMatrixY.y + cellCenter.z * rotationMatrixY.z;
      newCellCenter.z = cellCenter.x * rotationMatrixZ.x + cellCenter.y * rotationMatrixZ.y + cellCenter.z * rotationMatrixZ.z;
      ADD(newCellCenter, newCellCenter, meshCenter)

      double sqSignedDistance = getSqSignedDistance(distancePropertiesGpu, numFaces, newCellCenter);

      fracSize fraction;
      fraction = max(0.0, min(1.0, (sqDx - (sqSignedDistance + sqDxHalf) ) / sqDx));
      fractionFieldData[idx] = fraction;
   }
}



void ObjectRotatorGPU::voxelizeGPUCall(uint_t timestep) {
   double3 rotationMatrixX;
   double3 rotationMatrixY;
   double3 rotationMatrixZ;
   Matrix3< real_t > rotationMat(rotationAxis_, (timestep / frequency_) * -rotationAngle_);
   rotationMatrixX = {rotationMat[0], rotationMat[1], rotationMat[2]};
   rotationMatrixY = {rotationMat[3], rotationMat[4], rotationMat[5]};
   rotationMatrixZ = {rotationMat[6], rotationMat[7], rotationMat[8]};
   double interpolFactor = double(frequency_ - (timestep % frequency_)) / double(frequency_);

   for (auto& block : *blocks_)
   {
      auto tmpFractionFieldGPU = block.getData< gpu::GPUField<fracSize> >(tmpFracFieldGPUId);
      fracSize * RESTRICT const _data_tmpFractionFieldGPU = tmpFractionFieldGPU->dataAt(0, 0, 0, 0);

      auto tmpFractionFieldGPUOld = block.getData< gpu::GPUField<fracSize> >(tmpFracFieldGPUOldId);
      fracSize * RESTRICT const _data_tmpFractionFieldGPUOld = tmpFractionFieldGPUOld->dataAt(0, 0, 0, 0);

      auto fractionFieldGPU = block.getData< gpu::GPUField<fracSize> >(fractionFieldGPUId_);
      fracSize * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);

      auto level         = blocks_->getLevel(block);
      auto dx     = double(blocks_->dx(level));
      auto blockAABB = block.getAABB();

      if(blockAABB.empty())
         continue;

      double3 meshCenterGPU = {meshCenter[0], meshCenter[1], meshCenter[2]};
      CellInterval cellBB = blocks_->getCellBBFromAABB(blockAABB);
      int3 cellBBSize = {int(cellBB.xSize() + 2), int(cellBB.ySize() + 2), int(cellBB.zSize() + 2)}; //TODO +2 ??
      Cell cellBBGlobalMin = cellBB.min();
      blocks_->transformGlobalToBlockLocalCell(cellBBGlobalMin, block);
      int3 cellBBLocalMin = {int(cellBBGlobalMin[0]), int(cellBBGlobalMin[1]), int(cellBBGlobalMin[2])};
      double3 minAABB = {double(blockAABB.minCorner()[0]), double(blockAABB.minCorner()[1]), double(blockAABB.minCorner()[2])};

      int3 stride_frac_field = {int(tmpFractionFieldGPU->xStride()), int(tmpFractionFieldGPU->yStride()), int(tmpFractionFieldGPU->zStride())};

      dim3 _block(uint64_c(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)), uint64_c(((1024 < ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))) ? 1024 : ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))), uint64_c(((64 < ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))))) ? 64 : ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))))));
      dim3 _grid(uint64_c(( (cellBBSize.x - 2) % (((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)) == 0 ? (int64_t)(cellBBSize.x - 2) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)) : ( (int64_t)(cellBBSize.x - 2) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)) ) +1 )), uint64_c(( (cellBBSize.y - 2) % (((1024 < ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))) ? 1024 : ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))) == 0 ? (int64_t)(cellBBSize.y - 2) / (int64_t)(((1024 < ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))) ? 1024 : ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))) : ( (int64_t)(cellBBSize.y - 2) / (int64_t)(((1024 < ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))) ? 1024 : ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))) ) +1 )), uint64_c(( (cellBBSize.z - 2) % (((64 < ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))))) ? 64 : ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))))) == 0 ? (int64_t)(cellBBSize.z - 2) / (int64_t)(((64 < ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))))) ? 64 : ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))))) : ( (int64_t)(cellBBSize.z - 2) / (int64_t)(((64 < ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))))) ? 64 : ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))))) ) +1 )));

      voxelizeGPU<<<_grid, _block>>>(distancePropertiesGPUPtr, _data_tmpFractionFieldGPU, minAABB, cellBBSize, cellBBLocalMin, stride_frac_field, dx, numFaces_, rotationMatrixX, rotationMatrixY, rotationMatrixZ, meshCenterGPU);
   }
}

} //namespace walberla