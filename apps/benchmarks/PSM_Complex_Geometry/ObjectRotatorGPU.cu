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


namespace walberla
{
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

__global__ static void voxelizeRayTracingGPU(fracSize* fractionFieldData, float minAABB[3], int size_frac_field[3],
                                             float dx, int* triangles, float* vertices, float numTriangles,
                                             curandState* state)
{
   if (threadIdx.x < size_frac_field[0] &&
       blockDim.y * blockIdx.y  < size_frac_field[1] &&
       blockDim.z * blockIdx.z  < size_frac_field[2])
   {
      //TODO is -1 correct?
      const int x   = threadIdx.x - 1;
      const int y   = blockDim.y * blockIdx.y - 1;
      const int z   = blockDim.z * blockIdx.z - 1;
      const int idx = x + y * size_frac_field[1] + y * size_frac_field[1] * z * y * size_frac_field[2];
      curand_init(1234, idx, 0, &state[idx]);

      float dxHalf        = 0.5 * dx;
      float cellCenter[3] = { minAABB[0] + x * dx + dxHalf, minAABB[1] + y * dx + dxHalf,
                              minAABB[2] + z * dx + dxHalf };

      float rayDirection[3];
      rayDirection[0] = float(curand_uniform(state + idx));
      rayDirection[1] = float(curand_uniform(state + idx + 1));
      rayDirection[2] = float(curand_uniform(state + idx + 2));

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
         if (RayIntersectsTriangleGPU(cellCenter, rayDirection, triangle)) intersections++;
         if (intersections % 2 == 1) { fractionFieldData[idx] = 1.0; }
      }
   }
}

void ObjectRotatorGPU::voxelizeRayTracingGPUCall() {
   for (auto& block : *blocks_)
   {
      auto fractionField = block.getData< gpu::GPUField< fracSize > >(fractionFieldId_);
      fracSize * RESTRICT const fractionFieldData = fractionField->dataAt(-1, -1, -1, 0);
      auto level         = blocks_->getLevel(block);
      auto dx     = float(blocks_->dx(level));
      auto blockAABB = block.getAABB();
      float minAABB[3] = {float(blockAABB.minCorner()[0]), float(blockAABB.minCorner()[1]), float(blockAABB.minCorner()[2])};
      int size_frac_field[3] = {int(fractionField->xSizeWithGhostLayer()), int(fractionField->ySizeWithGhostLayer()), int(fractionField->zSizeWithGhostLayer())};

      const dim3 numBlocks(size_frac_field[1], size_frac_field[2]);
      const dim3 threadsPerBlock(size_frac_field[0]);

      voxelizeRayTracingGPU<<<numBlocks, threadsPerBlock>>>(fractionFieldData, minAABB, size_frac_field, dx, triangles_, vertices_, numTriangles_, dev_curand_states);
   }
};


} //namespace walberla