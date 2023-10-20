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


#define DOT(v1,v2) (v1.x*v2.x+v1.y*v2.y+v1.z*v2.z)

#define DOT2(v1,v2) (v1.x*v2.x+v1.y*v2.y)

#define SQNORM2(v1) (sqrtf(v1.x*v1.x + v1.y*v1.y))

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

/*
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
*/

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


__device__ float getSqSignedDistance(DistancePropertiesGPU * distancePropertiesGpu, int numFaces, float3 cellCenter) {
   // get sqDistance
   float final_sqDistance = INFINITY;
   float3 final_closestPoint;
   float3 final_normal;

   for (int i = 0; i < numFaces; ++i) {
      DistancePropertiesGPU dp = distancePropertiesGpu[i];
      float3 closestPoint;
      int region;
      float3 temp;
      float2 temp2;

      float3 pt3;
      ADD(temp, cellCenter, dp.translation)
      MATVECMUL(pt3, dp.rotation, temp)

      closestPoint.z = 0.0;

      float sqDistance = pt3.z * pt3.z;
      //printf("dp.translation[0] is %f, [1] is %f, [2] is %f, [3] is %f, [4] is %f, [5] is %f, [6] is %f, [7] is %f, [8] is %f \n", dp.rotation[0], dp.rotation[1], dp.rotation[2], dp.rotation[3], dp.rotation[4], dp.rotation[5], dp.rotation[6], dp.rotation[7], dp.rotation[8]);

      float2 pt = {pt3.x, pt3.y};

      float e0p = DOT2(dp.e0_normalized, pt);
      float e1p = DOT2(dp.e1_normalized, pt);
      SUB2(temp2, pt, dp.e0)
      float e2p = DOT2(dp.e2_normalized, temp2);

      temp2 = {-1,0};
      float e0d = DOT2(temp2, pt);
      float e1d = DOT2(dp.e1_normal, pt);
      SUB2(temp2, pt, dp.e0)
      float e2d = DOT2(dp.e2_normal, temp2);

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
         float transpose[9];
         TRANSPOSE(transpose, dp.rotation)
         MATVECMUL(temp, transpose, closestPoint)
         SUB(final_closestPoint, temp, dp.translation)
         final_normal = dp.region_normal[region];
      }
   }
   //printf("final_sqDistance is %f\n", final_sqDistance);
   float3 temp;
   SUB(temp, cellCenter, final_closestPoint)
   float dot = DOT(temp, final_normal);
   //return dot >= 0.0 ? final_sqDistance : -final_sqDistance;
   return final_sqDistance;

}


__global__ void voxelizeGPU(DistancePropertiesGPU * distancePropertiesGpu, fracSize * RESTRICT const fractionFieldData, float3 minAABB, int3 cellBBSize, int3 cellBBLocalMin, int3 stride_frac_field, float dx, int numFaces)
{
   const int64_t x = blockDim.x*blockIdx.x + threadIdx.x ;
   const int64_t y = blockDim.y*blockIdx.y + threadIdx.y ;
   const int64_t z = blockDim.z*blockIdx.z + threadIdx.z ;
   if (x < cellBBSize.x  && y < cellBBSize.y  && z < cellBBSize.z )
   {
      const int idx = (x + cellBBLocalMin.x) + (y + cellBBLocalMin.y) * stride_frac_field.y + (z + cellBBLocalMin.z) * stride_frac_field.z;

      float dxHalf        = 0.0;    //0.5f * dx;
      float3 cellCenter = { minAABB.x + float(x) * dx + dxHalf, minAABB.y + float(y) * dx + dxHalf,
                            minAABB.z + float(z) * dx + dxHalf };

      float sqSignedDistance = getSqSignedDistance(distancePropertiesGpu, numFaces, cellCenter);
      //printf("sqSignedDistance is %f \n", sqSignedDistance);

      fracSize fraction;
      fraction = max(0.0, min(1.0, (dx - (sqrt(sqSignedDistance) + (dx * 0.5)) ) / dx));
      fractionFieldData[idx] = sqSignedDistance;
   }
}



void ObjectRotatorGPU::voxelizeGPUCall() {
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
      int3 cellBBSize = {int(cellBB.xSize() + 2), int(cellBB.ySize() + 2), int(cellBB.zSize() + 2)}; //TODO +2 ??
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

      voxelizeGPU<<<_grid, _block>>>(distancePropertiesGPUPtr, _data_tmpFractionFieldGPU, minAABB, cellBBSize, cellBBLocalMin, stride_frac_field, dx, numFaces_);

      writeToActualFractionField<<<_grid, _block>>>(_data_fractionFieldGPU, _data_tmpFractionFieldGPU, cellBBSize, cellBBLocalMin, stride_frac_field);
   }
}

} //namespace walberla