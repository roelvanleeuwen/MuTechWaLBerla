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
#   pragma GCC diagnostic ignored "-Wfloat-equal"
#   pragma GCC diagnostic ignored "-Wshadow"
#   pragma GCC diagnostic ignored "-Wconversion"
#   pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#if ( defined WALBERLA_CXX_COMPILER_IS_INTEL )
#pragma warning push
#pragma warning( disable :  1599 )
#endif

#define CROSS(dest,v1,v2) \
         dest.x=v1.y*v2.z-v1.z*v2.y; \
         dest.y=v1.z*v2.x-v1.x*v2.z; \
         dest.z=v1.x*v2.y-v1.y*v2.x;

#define DOT(v1,v2) (v1.x*v2.x+v1.y*v2.y+v1.z*v2.z)

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

__global__ static void rotateGPU(float * vertices, int numVertices, float3 rotationMatrixX, float3 rotationMatrixY, float3 rotationMatrixZ, float3 axis_foot) {

   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x;
   if (x < numVertices) {
      float3 vertex = {vertices[x * 3], vertices[x * 3 + 1], vertices[x * 3 + 2]};
      float3 newVertex;

      SUB(vertex, vertex, axis_foot)

      newVertex.x = rotationMatrixX.x * vertex.x + rotationMatrixX.y * vertex.y + rotationMatrixX.z * vertex.z;
      newVertex.y = rotationMatrixY.x * vertex.x + rotationMatrixY.y * vertex.y + rotationMatrixY.z * vertex.z;
      newVertex.z = rotationMatrixZ.x * vertex.x + rotationMatrixZ.y * vertex.y + rotationMatrixZ.z * vertex.z;

      ADD(newVertex, newVertex, axis_foot)

      vertices[x * 3] = newVertex.x;
      vertices[x * 3 + 1] = newVertex.y;
      vertices[x * 3 + 2] = newVertex.z;
   }
}


void ObjectRotatorGPU::rotateGPUCall()
{
   float3 axis_foot = {float(meshCenter[0]), float(meshCenter[1]), float(meshCenter[2])};
   const float sina( std::sin(rotationAngle_) );
   const float cosa( std::cos(rotationAngle_) );
   const float tmp( 1.f - cosa );
   float3 rotationAxis = {float(rotationAxis_[0]), float(rotationAxis_[1]), float(rotationAxis_[2])};
   float3 rotationMatrixX;
   float3 rotationMatrixY;
   float3 rotationMatrixZ;

   rotationMatrixX.x = cosa + rotationAxis.x*rotationAxis.x*tmp;
   rotationMatrixX.y = rotationAxis.x*rotationAxis.y*tmp - rotationAxis.z*sina;
   rotationMatrixX.z = rotationAxis.x*rotationAxis.z*tmp + rotationAxis.y*sina;

   rotationMatrixY.x = rotationAxis.y*rotationAxis.x*tmp + rotationAxis.z*sina;
   rotationMatrixY.y = cosa + rotationAxis.y*rotationAxis.y*tmp;
   rotationMatrixY.z = rotationAxis.y*rotationAxis.z*tmp - rotationAxis.x*sina;

   rotationMatrixZ.x = rotationAxis.z*rotationAxis.x*tmp - rotationAxis.y*sina;
   rotationMatrixZ.y = rotationAxis.z*rotationAxis.y*tmp + rotationAxis.x*sina;
   rotationMatrixZ.z = cosa + rotationAxis.z*rotationAxis.z*tmp;

   int threads = 1024;
   int blocks = numVertices_ / threads + 1;
   rotateGPU<<<threads, blocks>>>(verticesGPU_, numVertices_, rotationMatrixX, rotationMatrixY, rotationMatrixZ, axis_foot);

}


__global__ static void resetFractionFieldGPU(fracSize * RESTRICT const fractionFieldData, fracSize * RESTRICT const tmpFractionFieldData, int3 cellBBSize, int3 cellBBLocalMin, int3 stride_frac_field) {

   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z;
   if (x < cellBBSize.x && y < cellBBSize.y && z < cellBBSize.z )
   {
      const int idx = (x + cellBBLocalMin.x) + (y + cellBBLocalMin.y) * stride_frac_field.y + (z + cellBBLocalMin.z) * stride_frac_field.z;
      fractionFieldData[idx] -= tmpFractionFieldData[idx];
   }
}



__global__ void writeToActualFractionField(fracSize * RESTRICT const fractionFieldData, fracSize * RESTRICT const tmpFractionFieldData, int3 cellBBSize, int3 cellBBLocalMin, int3 stride_frac_field) {

   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z;
   if (x < cellBBSize.x && y < cellBBSize.y && z < cellBBSize.z )
   {
      const int idx = (x + cellBBLocalMin.x) + (y + cellBBLocalMin.y) * stride_frac_field.y + (z + cellBBLocalMin.z) * stride_frac_field.z;
      fractionFieldData[idx] += tmpFractionFieldData[idx];
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


__device__ fracSize recursiveSuperSampling(int* triangles, float* vertices, int numTriangles, float3 cellCenter, float dx, int depth, int maxSuperSamplingDepth, curandState state)
{
   // if only one cell left, split cell into 8 cell centers to get these cellCenter distances

   fracSize fraction        = 0.0;
   const fracSize fracValue = float(1.0 / pow(8, float(depth)));
   const float offsetMod    = float(1.0 / pow(2, float(depth + 2)));

   if (depth == maxSuperSamplingDepth)
   {
      int raySamples = 2;
      int cellInside = 0;
      for (int r = 0; r < raySamples; ++r) {
         float3 rayDirection = { float(curand_uniform(&state)), float(curand_uniform(&state)), float(curand_uniform(&state)) };
         int intersections   = 0;
         for (int i = 0; i < numTriangles; ++i)
         {
            float3 vertex0 = { vertices[3 * triangles[i * 3]], vertices[3 * triangles[i * 3] + 1], vertices[3 * triangles[i * 3] + 2] };
            float3 vertex1 = { vertices[3 * triangles[i * 3 + 1]], vertices[3 * triangles[i * 3 + 1] + 1], vertices[3 * triangles[i * 3 + 1] + 2] };
            float3 vertex2 = { vertices[3 * triangles[i * 3 + 2]], vertices[3 * triangles[i * 3 + 2] + 1], vertices[3 * triangles[i * 3 + 2] + 2] };
            if (RayIntersectsTriangleGPU(cellCenter, rayDirection, vertex0, vertex1, vertex2))
               intersections++;
         }
         if (intersections % 2 == 1)
            cellInside++;
         else
            break;
      }
      if (cellInside > (raySamples / 2))
         fraction = fracValue;
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
         fraction += recursiveSuperSampling(triangles, vertices, numTriangles, octreeCenter, dx, depth + 1, maxSuperSamplingDepth, state);
      }
   }
   return fraction;
}


__global__ void voxelizeRayTracingGPUSuperSampling(fracSize * RESTRICT const fractionFieldData, float3 minAABB, int3 cellBBSize, int3 cellBBLocalMin, int3 stride_frac_field,
                                         float dx, int* triangles, float* vertices, int numTriangles, int maxSuperSamplingDepth, curandState* state)
{
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < cellBBSize.x  && y < cellBBSize.y  && z < cellBBSize.z )
   {
      const int idx = (x + cellBBLocalMin.x) + (y + cellBBLocalMin.y) * stride_frac_field.y + (z + cellBBLocalMin.z) * stride_frac_field.z;
      //const int idx = x + y * stride_frac_field.y + z * stride_frac_field.z;

      const int xyz = x+y+z;
      curand_init(1234, xyz, 7, &state[xyz]);

      float dxHalf        = 0.0;//0.5f * dx;
      float3 cellCenter = { minAABB.x + float(x) * dx + dxHalf, minAABB.y + float(y) * dx + dxHalf,
                            minAABB.z + float(z) * dx + dxHalf };
      fracSize fraction;
      fraction = recursiveSuperSampling(triangles, vertices, numTriangles, cellCenter, dx, 0, maxSuperSamplingDepth, state[xyz]);
      fractionFieldData[idx] = fraction;

   }
}


void ObjectRotatorGPU::voxelizeRayTracingGPUCall() {
   for (auto& block : *blocks_)
   {
      auto tmpFractionFieldGPU = block.getData< gpu::GPUField<fracSize> >(tmpFracFieldGPUId);
      fracSize * RESTRICT const _data_tmpFractionFieldGPU = tmpFractionFieldGPU->dataAt(0, 0, 0, 0);

      auto fractionFieldGPU = block.getData< gpu::GPUField<fracSize> >(fractionFieldGPUId_);
      fracSize * RESTRICT const _data_fractionFieldGPU = fractionFieldGPU->dataAt(0, 0, 0, 0);

      auto level         = blocks_->getLevel(block);
      auto dx     = float(blocks_->dx(level));
      auto blockAABB = block.getAABB();

      //blockAABB.intersect(meshAABB); //TODO get meshAABB after every rotation???
      if(blockAABB.empty())
         continue;
      
      CellInterval cellBB = blocks_->getCellBBFromAABB(blockAABB);
      int3 cellBBSize = {int(cellBB.xSize() + 1), int(cellBB.ySize() + 1), int(cellBB.zSize() + 1)};
      Cell cellBBGlobalMin = cellBB.min();
      blocks_->transformGlobalToBlockLocalCell(cellBBGlobalMin, block);
      int3 cellBBLocalMin = {int(cellBBGlobalMin[0]), int(cellBBGlobalMin[1]), int(cellBBGlobalMin[2])};
      
      float3 minAABB = {float(blockAABB.minCorner()[0]), float(blockAABB.minCorner()[1]), float(blockAABB.minCorner()[2])};
      int3 stride_frac_field = {int(tmpFractionFieldGPU->xStride()), int(tmpFractionFieldGPU->yStride()), int(tmpFractionFieldGPU->zStride())};
      
      //WALBERLA_LOG_INFO("stride_frac_field is (" <<  stride_frac_field.x << "," << stride_frac_field.y << "," << stride_frac_field.z << ")" << " fstride " << tmpFractionFieldGPU->fStride())
      //WALBERLA_LOG_INFO("Cell BB is (" <<  cellBBSize.x << "," << cellBBSize.y << "," << cellBBSize.z << ")" )


      dim3 _block(uint64_c(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)), uint64_c(((1024 < ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))) ? 1024 : ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))), uint64_c(((64 < ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))))) ? 64 : ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))))));
      dim3 _grid(uint64_c(( (cellBBSize.x - 2) % (((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)) == 0 ? (int64_t)(cellBBSize.x - 2) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)) : ( (int64_t)(cellBBSize.x - 2) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)) ) +1 )), uint64_c(( (cellBBSize.y - 2) % (((1024 < ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))) ? 1024 : ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))) == 0 ? (int64_t)(cellBBSize.y - 2) / (int64_t)(((1024 < ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))) ? 1024 : ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))) : ( (int64_t)(cellBBSize.y - 2) / (int64_t)(((1024 < ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))) ? 1024 : ((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))) ) +1 )), uint64_c(( (cellBBSize.z - 2) % (((64 < ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))))) ? 64 : ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))))) == 0 ? (int64_t)(cellBBSize.z - 2) / (int64_t)(((64 < ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))))) ? 64 : ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))))) : ( (int64_t)(cellBBSize.z - 2) / (int64_t)(((64 < ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))))))) ? 64 : ((cellBBSize.z - 2 < ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))) ? cellBBSize.z - 2 : ((int64_t)(256) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)*((cellBBSize.y - 2 < 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2)))) ? cellBBSize.y - 2 : 16*((int64_t)(16) / (int64_t)(((16 < cellBBSize.x - 2) ? 16 : cellBBSize.x - 2))))))))) ) +1 )));
      
      //WALBERLA_LOG_INFO("Grid is (" <<  _grid.x << "," << _grid.y << "," << _grid.z << ") and Block is (" << _block.x << "," << _block.y << "," << _block.z << ")")
      //WALBERLA_LOG_INFO("Num trinagles is " << numTriangles_ << " num vertices is " << numVertices_)

      resetFractionFieldGPU<<<_grid, _block>>>(_data_fractionFieldGPU, _data_tmpFractionFieldGPU, cellBBSize, cellBBLocalMin, stride_frac_field);

      voxelizeRayTracingGPUSuperSampling<<<_grid, _block>>>(_data_tmpFractionFieldGPU, minAABB, cellBBSize, cellBBLocalMin, stride_frac_field, dx, trianglesGPU_, verticesGPU_, numTriangles_, maxSuperSamplingDepth_, dev_curand_states);

      writeToActualFractionField<<<_grid, _block>>>(_data_fractionFieldGPU, _data_tmpFractionFieldGPU, cellBBSize, cellBBLocalMin, stride_frac_field);



   }
}


} //namespace walberla