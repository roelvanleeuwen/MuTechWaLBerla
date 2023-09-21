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

#   pragma GCC diagnostic ignored "-Wconversion"


#define CROSS(dest,v1,v2) \
         dest.x=v1.y*v2.z-v1.z*v2.y; \
         dest.y=v1.z*v2.x-v1.x*v2.z; \
         dest.z=v1.x*v2.y-v1.y*v2.x;

#define DOT(v1,v2) (v1.x*v2.x+v1.y*v2.y+v1.z*v2.z)

#define SUB(dest,v1,v2) \
         dest.x=v1.x-v2.x; \
         dest.y=v1.y-v2.y; \
         dest.z=v1.z-v2.z;


namespace walberla
{

__global__ static void resetFractionFieldGPU(fracSize * RESTRICT const fractionFieldData, int3 size_frac_fieldWithGL) {

   const int idx   = threadIdx.x + blockIdx.x * blockDim.x;
   if (idx < size_frac_fieldWithGL.x * size_frac_fieldWithGL.y * size_frac_fieldWithGL.z)
   {
      fractionFieldData[idx] = idx;
   }
}

//TODO only run over aabb of mesh
void ObjectRotatorGPU::resetFractionFieldGPUCall() {
   for (auto& block : *blocks_)
   {
      auto fractionFieldGPU = block.getData< gpu::GPUField<fracSize> >(fractionFieldGPUId_);
      fracSize * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(-1, -1, -1, 0);
      int3 size_frac_field = {int(fractionFieldGPU->xSizeWithGhostLayer()), int(fractionFieldGPU->ySizeWithGhostLayer()), int(fractionFieldGPU->zSizeWithGhostLayer())};

      const uint threadsPerBlock(512);
      const uint numBlocks((size_frac_field.x * size_frac_field.y * size_frac_field.z / threadsPerBlock) + 1);

      resetFractionFieldGPU<<<numBlocks, threadsPerBlock>>>(_data_fractionFieldGPU, size_frac_field);
   }
}







__device__ bool RayIntersectsTriangleGPU(float3 rayOrigin, float3 rayVector, float3 vertex0, float3 vertex1, float3 vertex2)
{
   const float EPSILON = 0.00000001f;
   float3 edge1, edge2, h, s, q;
   float a, f, u, v;
   SUB(edge1, vertex1, vertex0)
   SUB(edge2, vertex2, vertex0)
   CROSS(h, rayVector, edge2)
   a = DOT(edge1, h);
   if (a > -EPSILON && a < EPSILON) return false; // This ray is parallel to this triangle.
   f = 1.0f / a;
   SUB(s, rayOrigin, vertex0)
   u = f * DOT(s, h);
   if (u < 0.0 || u > 1.0) return false;
   CROSS(q, s, edge1)
   v = f * DOT(rayVector, q);
   if (v < 0.0 || u + v > 1.0) return false;
   // At this stage we can compute t to find out where the intersection point is on the line.
   float t = f * DOT(edge2, q);
   if (t > EPSILON)
      return true;
   else
      return false;
}




__device__ fracSize recursiveSuperSampling(int* triangles, float* vertices, int numTriangles, float3 cellCenter, float dx, int depth, int maxDepth, curandState state)
{
   // if only one cell left, split cell into 8 cell centers to get these cellCenter distances

   fracSize fraction        = 0.0;
   const fracSize fracValue = float(1.0 / pow(8, float(depth)));
   const float offsetMod    = float(1.0 / pow(2, float(depth + 2)));

   if (depth == maxDepth)
   {
      int raySamples = 3;
      int cellInside = 0;
      for (int r = 0; r < raySamples; ++r) {
         float3 rayDirection = { float(curand_uniform(&state)), float(curand_uniform(&state)),
                                 float(curand_uniform(&state)) };
         int intersections   = 0;
         // TODO Shoot multiple rays
         for (int i = 0; i < numTriangles; ++i)
         {
            float3 vertex0 = { vertices[3 * triangles[i * 3]], vertices[3 * triangles[i * 3] + 1],
                               vertices[3 * triangles[i * 3] + 2] };
            float3 vertex1 = { vertices[3 * triangles[i * 3 + 1]], vertices[3 * triangles[i * 3 + 1] + 1],
                               vertices[3 * triangles[i * 3 + 1] + 2] };
            float3 vertex2 = { vertices[3 * triangles[i * 3 + 2]], vertices[3 * triangles[i * 3 + 2] + 1],
                               vertices[3 * triangles[i * 3 + 2] + 2] };
            if (RayIntersectsTriangleGPU(cellCenter, rayDirection, vertex0, vertex1, vertex2)) intersections++;
         }
         if (intersections % 2 == 1) {
            cellInside++;
         }
      }
      if (cellInside > (raySamples / 2))
      {
         fraction = fracValue;
      }

   }
   else
   {
      float xOffset[8]{ -1, -1, -1, -1, 1, 1, 1, 1 };
      float yOffset[8]{ -1, -1, 1, 1, -1, -1, 1, 1 };
      float zOffset[8]{ -1, 1, -1, 1, -1, 1, -1, 1 };
      float3 octreeCenter;
      for (uint_t i = 0; i < 8; ++i)
      {
         octreeCenter = { cellCenter.x + xOffset[i] * dx * offsetMod, cellCenter.y + yOffset[i] * dx * offsetMod,
                          cellCenter.z + zOffset[i] * dx * offsetMod };
         fraction +=
            recursiveSuperSampling(triangles, vertices, numTriangles, octreeCenter, dx, depth + 1, maxDepth, state);
      }
   }
   return fraction;
}


__global__ void voxelizeRayTracingGPUSuperSampling(fracSize * RESTRICT const fractionFieldData, float3 minAABB, int3 size_frac_fieldWithGL, int3 stride_frac_field,
                                         float dx, int* triangles, float* vertices, int numTriangles, curandState* state)
{
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x + 1;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y + 1;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z + 1;
   if (x < size_frac_fieldWithGL.x - 1 && y < size_frac_fieldWithGL.y - 1 && z < size_frac_fieldWithGL.z - 1)
   {
      //printf("threadX %d threadY %d threadZ %d blockDimX %d blockDimY %d blockDimZ %d blockIdxX %d blockIdxY %d blockIdxZ %d \n",
      //threadIdx.x,threadIdx.y,threadIdx.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x, blockIdx.y, blockIdx.z);
      //printf("%d %d %d \n", x,y,z);
      const int idx = x + y * stride_frac_field.y + z * stride_frac_field.z;
      const int xyz = x+y+z;
      curand_init(1234, xyz, 7, &state[xyz]);

      float dxHalf        = -0.5f * dx;
      float3 cellCenter = { minAABB.x + float(x) * dx + dxHalf, minAABB.y + float(y) * dx + dxHalf,
                            minAABB.z + float(z) * dx + dxHalf };
      fracSize fraction;
      int maxDepth = 1;
      fraction = recursiveSuperSampling(triangles, vertices, numTriangles, cellCenter, dx, 0, maxDepth, state[xyz]);
      fractionFieldData[idx] = fraction;

   }
}





__global__ void voxelizeRayTracingGPU(fracSize * RESTRICT const fractionFieldData, float3 minAABB, int3 size_frac_fieldWithGL, int3 stride_frac_field,
                                             float dx, int* triangles, float* vertices, int numTriangles, curandState* state)
{
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x + 1;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y + 1;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z + 1;
   if (x < size_frac_fieldWithGL.x - 1 && y < size_frac_fieldWithGL.y - 1 && z < size_frac_fieldWithGL.z - 1)
   {
      //printf("threadX %d threadY %d threadZ %d blockDimX %d blockDimY %d blockDimZ %d blockIdxX %d blockIdxY %d blockIdxZ %d \n",
             //threadIdx.x,threadIdx.y,threadIdx.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x, blockIdx.y, blockIdx.z);
      //printf("%d %d %d \n", x,y,z);
      const int idx = x + y * stride_frac_field.y + z * stride_frac_field.z;
      const int xyz = x+y+z;
      curand_init(1234, xyz, 7, &state[xyz]);

      float dxHalf        = -0.5f * dx;
      float3 cellCenter = { minAABB.x + float(x) * dx + dxHalf, minAABB.y + float(y) * dx + dxHalf,
                              minAABB.z + float(z) * dx + dxHalf };
      int raySamples = 3;
      int cellInside = 0;
      for (int r = 0; r < raySamples; ++r) {
         float3 rayDirection = {float(curand_uniform(&state[xyz])), float(curand_uniform(&state[xyz])), float(curand_uniform(&state[xyz]))};
         int intersections = 0;
         // TODO Shoot multiple rays
         for (int i = 0; i < numTriangles; ++i)
         {
            float3 vertex0 = { vertices[3 * triangles[i * 3]], vertices[3 * triangles[i * 3] + 1], vertices[3 * triangles[i * 3] + 2] };
            float3 vertex1 = { vertices[3 * triangles[i * 3 + 1]], vertices[3 * triangles[i * 3 + 1] + 1], vertices[3 * triangles[i * 3 + 1] + 2] };
            float3 vertex2 = { vertices[3 * triangles[i * 3 + 2]], vertices[3 * triangles[i * 3 + 2] + 1], vertices[3 * triangles[i * 3 + 2] + 2] };
            if (RayIntersectsTriangleGPU(cellCenter, rayDirection, vertex0, vertex1, vertex2))
               intersections++;
         }
         if (intersections % 2 == 1) {
            cellInside++;
         }
      }
      if (cellInside > (raySamples / 2))
      {
         fractionFieldData[idx] = 1.0;
      }
   }
}

void ObjectRotatorGPU::voxelizeRayTracingGPUCall() {
   for (auto& block : *blocks_)
   {
      auto fractionFieldGPU = block.getData< gpu::GPUField<fracSize> >(fractionFieldGPUId_);
      fracSize * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(-1, -1, -1, 0);

      auto level         = blocks_->getLevel(block);
      auto dx     = float(blocks_->dx(level));
      auto blockAABB = block.getAABB();
      float3 minAABB = {float(blockAABB.minCorner()[0]), float(blockAABB.minCorner()[1]), float(blockAABB.minCorner()[2])};
      int3 size_frac_field = {int(fractionFieldGPU->xSizeWithGhostLayer()), int(fractionFieldGPU->ySizeWithGhostLayer()), int(fractionFieldGPU->zSizeWithGhostLayer())};
      int3 stride_frac_field = {int(fractionFieldGPU->xStride()), int(fractionFieldGPU->yStride()), int(fractionFieldGPU->zStride())};
      WALBERLA_LOG_INFO("stride_frac_field is (" <<  stride_frac_field.x << "," << stride_frac_field.y << "," << stride_frac_field.z << ")" << " fstride " << fractionFieldGPU->fStride())

      dim3 _block(uint64_c(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)), uint64_c(((1024 < ((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))) ? 1024 : ((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))))), uint64_c(((64 < ((size_frac_field.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))))) ? size_frac_field.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))))))) ? 64 : ((size_frac_field.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))))) ? size_frac_field.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))))))));
      dim3 _grid(uint64_c(( (size_frac_field.x - 2) % (((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)) == 0 ? (int64_t)(size_frac_field.x - 2) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)) : ( (int64_t)(size_frac_field.x - 2) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)) ) +1 )), uint64_c(( (size_frac_field.y - 2) % (((1024 < ((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))) ? 1024 : ((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))))) == 0 ? (int64_t)(size_frac_field.y - 2) / (int64_t)(((1024 < ((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))) ? 1024 : ((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))))) : ( (int64_t)(size_frac_field.y - 2) / (int64_t)(((1024 < ((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))) ? 1024 : ((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))))) ) +1 )), uint64_c(( (size_frac_field.z - 2) % (((64 < ((size_frac_field.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))))) ? size_frac_field.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))))))) ? 64 : ((size_frac_field.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))))) ? size_frac_field.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))))))) == 0 ? (int64_t)(size_frac_field.z - 2) / (int64_t)(((64 < ((size_frac_field.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))))) ? size_frac_field.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))))))) ? 64 : ((size_frac_field.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))))) ? size_frac_field.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))))))) : ( (int64_t)(size_frac_field.z - 2) / (int64_t)(((64 < ((size_frac_field.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))))) ? size_frac_field.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))))))) ? 64 : ((size_frac_field.z - 2 < ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))))) ? size_frac_field.z - 2 : ((int64_t)(256) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)*((size_frac_field.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2)))) ? size_frac_field.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < size_frac_field.x - 2) ? 16 : size_frac_field.x - 2))))))))) ) +1 )));
      WALBERLA_LOG_INFO("Grid is (" <<  _grid.x << "," << _grid.y << "," << _grid.z << ") and Block is (" << _block.x << "," << _block.y << "," << _block.z << ")")
      //voxelizeRayTracingGPU<<<_grid, _block>>>(_data_fractionFieldGPU, minAABB, size_frac_field, stride_frac_field, dx, trianglesGPU_, verticesGPU_, numTriangles_, dev_curand_states);
      voxelizeRayTracingGPUSuperSampling<<<_grid, _block>>>(_data_fractionFieldGPU, minAABB, size_frac_field, stride_frac_field, dx, trianglesGPU_, verticesGPU_, numTriangles_, dev_curand_states);
   }
}


} //namespace walberla