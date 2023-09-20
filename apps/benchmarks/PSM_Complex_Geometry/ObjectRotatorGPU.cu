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
         dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
         dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
         dest[2]=v1[0]*v2[1]-v1[1]*v2[0];

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2) \
         dest[0]=v1[0]-v2[0]; \
         dest[1]=v1[1]-v2[1]; \
         dest[2]=v1[2]-v2[2];


namespace walberla
{

__global__ static void resetFractionFieldGPU(fracSize* RESTRICT const fractionFieldData, int3 size_frac_fieldWithGL) {

   const int idx   = threadIdx.x + blockIdx.x * blockDim.x;
   if (idx < size_frac_fieldWithGL.x * size_frac_fieldWithGL.y * size_frac_fieldWithGL.z)
   {
      fractionFieldData[0] = 0.0;
   }
}






void ObjectRotatorGPU::resetFractionFieldGPUCall() {
   for (auto& block : *blocks_)
   {
      auto fractionField = block.getData< gpu::GPUField< fracSize > >(fractionFieldGPUId_);
      //fracSize * RESTRICT const fractionFieldData = fractionField->dataAt(-1, -1, -1, 0);
      fracSize * RESTRICT const fractionFieldData = fractionField->dataAt(-1, -1, -1, 0);
      int3 size_frac_field = {int(fractionField->xSizeWithGhostLayer()), int(fractionField->ySizeWithGhostLayer()), int(fractionField->zSizeWithGhostLayer())};

      const uint threadsPerBlock(512);
      const uint numBlocks((size_frac_field.x * size_frac_field.y * size_frac_field.z / threadsPerBlock) + 1);

      resetFractionFieldGPU<<<numBlocks, threadsPerBlock>>>(fractionFieldData, size_frac_field);
   }
}







__device__ bool RayIntersectsTriangleGPU(float rayOrigin[3], float rayVector[3], float inTriangle[3][3])
{
   const float EPSILON = 0.00000001f;
   float edge1[3], edge2[3], h[3], s[3], q[3];
   float a, f, u, v;
   SUB(edge1, inTriangle[1], inTriangle[0])
   SUB(edge2, inTriangle[2], inTriangle[0])
   CROSS(h, rayVector, edge2)
   a = DOT(edge1, h);
   if (a > -EPSILON && a < EPSILON) return false; // This ray is parallel to this triangle.
   f = 1.0f / a;
   SUB(s, rayOrigin, inTriangle[0])
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

__global__ void voxelizeRayTracingGPU(fracSize* RESTRICT const fractionFieldData, float3 minAABB, int3 size_frac_fieldWithGL,
                                             float dx, int* triangles, float* vertices, int numTriangles,
                                             curandState* state)
{
   const int x   = threadIdx.x;
   const int y   = blockIdx.x;
   const int z   = blockIdx.y;


   if (x < size_frac_fieldWithGL.x &&
       y < size_frac_fieldWithGL.y &&
       z < size_frac_fieldWithGL.z)
   {
      //printf("threadX %d threadY %d threadZ %d blockDimX %d blockDimY %d blockDimZ %d blockIdxX %d blockIdxY %d blockIdxZ %d \n",
             //threadIdx.x,threadIdx.y,threadIdx.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x, blockIdx.y, blockIdx.z);
      //printf("%d %d %d \n", x,y,z);
      const int idx = x + y * size_frac_fieldWithGL.y + z * size_frac_fieldWithGL.x * size_frac_fieldWithGL.y;
      curand_init(1234, idx, 0, &state[idx]);

      float dxHalf        = 0.5 * dx;
      float cellCenter[3] = { minAABB.x + x * dx + dxHalf, minAABB.y + y * dx + dxHalf,
                              minAABB.z + z * dx + dxHalf };

      float rayDirection[3];
      rayDirection[0] = 1.0f; //float(curand_uniform(state + idx));
      rayDirection[1] = 2.0f; //float(curand_uniform(state + idx + 1));
      rayDirection[2] = 3.0f; //float(curand_uniform(state + idx + 2));

      int intersections = 0;
      // TODO Shoot multiple rays
      for (int i = 0; i < numTriangles; ++i)
      {
         float triangle[3][3] = { { vertices[3 * triangles[i * 3]], vertices[3 * triangles[i * 3] + 1],
                                    vertices[3 * triangles[i * 3] + 2] },
                                  { vertices[3 * triangles[i * 3 + 1]], vertices[3 * triangles[i * 3 + 1] + 1],
                                    vertices[3 * triangles[i * 3 + 1] + 2] },
                                  { vertices[3 * triangles[i * 3 + 2]], vertices[3 * triangles[i * 3 + 2] + 1],
                                    vertices[3 * triangles[i * 3 + 2] + 2] } };
         if (RayIntersectsTriangleGPU(cellCenter, rayDirection, triangle))
            intersections++;
         if (intersections % 2 == 1) {
            fracSize tmp = fractionFieldData[0];
            printf(" %f", tmp);
            //fractionFieldData[idx] = 1.0;
            return;
         }
      }
   }
}

void ObjectRotatorGPU::voxelizeRayTracingGPUCall() {
   for (auto& block : *blocks_)
   {
      auto fractionField = block.getData< gpu::GPUField< fracSize > >(fractionFieldGPUId_);
      //fracSize * RESTRICT const fractionFieldData = fractionField->dataAt(-1, -1, -1, 0);
      fracSize * RESTRICT const fractionFieldData = fractionField->dataAt(0, 0, 0, 0);

      auto level         = blocks_->getLevel(block);
      auto dx     = float(blocks_->dx(level));
      auto blockAABB = block.getAABB();
      float3 minAABB = {float(blockAABB.minCorner()[0]), float(blockAABB.minCorner()[1]), float(blockAABB.minCorner()[2])};
      int3 size_frac_field = {int(fractionField->xSizeWithGhostLayer()), int(fractionField->ySizeWithGhostLayer()), int(fractionField->zSizeWithGhostLayer())};

      const dim3 threadsPerBlock(size_frac_field.x);
      const dim3 numBlocks(size_frac_field.y, size_frac_field.z);

      voxelizeRayTracingGPU<<<numBlocks, threadsPerBlock>>>(fractionFieldData, minAABB, size_frac_field, dx, triangles_, vertices_, numTriangles_, dev_curand_states);
   }
}


} //namespace walberla