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
//! \file ParticleAndVolumeFractionMappingKernel.cu
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#include "lbm_mesapd_coupling/DataTypesGPU.h"

#include <assert.h>

#include "ParticleAndVolumeFractionMappingKernel.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace cuda
{

/*__global__ void resetKernelAoS(walberla::cuda::FieldAccessor< ParticleAndVolumeFractionAoS_T > field)
{
   field.set(blockIdx, threadIdx);
   for (uint i = 0; i < MaxParticlesPerCell; i++)
   {
      field.get().overlapFractions[i] = 0.0;
      field.get().uids[i]             = id_t(0);
   }
   field.get().index = 0;
}

// TODO: look for better mapping method
__global__ void
   particleAndVolumeFractionMappingKernelAoS(walberla::cuda::FieldAccessor< ParticleAndVolumeFractionAoS_T > field,
                                             double3 spherePosition, real_t sphereRadius, double3 blockStart,
                                             double3 dx, int3 nSamples, id_t uid)
{
   field.set(blockIdx, threadIdx);
   double3 sampleDistance       = { 1.0 / (nSamples.x + 1) * dx.x, 1.0 / (nSamples.y + 1) * dx.y,
                                    1.0 / (nSamples.z + 1) * dx.z };
   double3 startSamplingPoint   = { (blockStart.x + threadIdx.x * dx.x + sampleDistance.x),
                                    (blockStart.y + blockIdx.x * dx.y + sampleDistance.y),
                                    (blockStart.z + blockIdx.y * dx.z + sampleDistance.z) };
   double3 currentSamplingPoint = startSamplingPoint;

   double3 minCornerSphere = { spherePosition.x - sphereRadius, spherePosition.y - sphereRadius,
                               spherePosition.z - sphereRadius };
   double3 maxCornerSphere = { spherePosition.x + sphereRadius, spherePosition.y + sphereRadius,
                               spherePosition.z + sphereRadius };

   if (startSamplingPoint.x + dx.x > minCornerSphere.x && startSamplingPoint.x < maxCornerSphere.x &&
       startSamplingPoint.y + dx.y > minCornerSphere.y && startSamplingPoint.y < maxCornerSphere.y &&
       startSamplingPoint.z + dx.z > minCornerSphere.z && startSamplingPoint.z < maxCornerSphere.z)
   {
      for (uint_t z = 0; z < nSamples.z; z++)
      {
         currentSamplingPoint.y = startSamplingPoint.y;
         for (uint_t y = 0; y < nSamples.y; y++)
         {
            currentSamplingPoint.x = startSamplingPoint.x;
            for (uint_t x = 0; x < nSamples.x; x++)
            {
               if ((currentSamplingPoint.x - spherePosition.x) * (currentSamplingPoint.x - spherePosition.x) +
                      (currentSamplingPoint.y - spherePosition.y) * (currentSamplingPoint.y - spherePosition.y) +
                      (currentSamplingPoint.z - spherePosition.z) * (currentSamplingPoint.z - spherePosition.z) <=
                   sphereRadius * sphereRadius)
               {
                  field.get().overlapFractions[field.get().index] += 1.0;
               }
               currentSamplingPoint.x += sampleDistance.x;
            }
            currentSamplingPoint.y += sampleDistance.y;
         }
         currentSamplingPoint.z += sampleDistance.z;
      }

      field.get().overlapFractions[field.get().index] *= 1.0 / (nSamples.x * nSamples.y * nSamples.z);
      if (field.get().overlapFractions[field.get().index] > 0)
      {
         field.get().uids[field.get().index] = uid;
         field.get().index += 1;
      }
      assert(field.get().index < MaxParticlesPerCell);
   }
}*/

// functions to calculate Bs
template< int Weighting_T >
__device__ void calculateWeighting(real_t* weighting, const real_t& /*epsilon*/, const real_t& /*tau*/)
{
   WALBERLA_STATIC_ASSERT(Weighting_T == 1 || Weighting_T == 2);
}
template<>
__device__ void calculateWeighting< 1 >(real_t* weighting, const real_t& epsilon, const real_t& /*tau*/)
{
   *weighting = epsilon;
}
template<>
__device__ void calculateWeighting< 2 >(real_t* weighting, const real_t& epsilon, const real_t& tau)
{
   *weighting = epsilon * (tau - real_t(0.5)) / ((real_t(1) - epsilon) + (tau - real_t(0.5)));
}

__global__ void resetKernelSoA(walberla::cuda::FieldAccessor< uint_t > indicesField,
                               walberla::cuda::FieldAccessor< real_t > overlapFractionsField,
                               walberla::cuda::FieldAccessor< id_t > uidsField,
                               walberla::cuda::FieldAccessor< real_t > bnField)
{
   indicesField.set(blockIdx, threadIdx);
   overlapFractionsField.set(blockIdx, threadIdx);
   uidsField.set(blockIdx, threadIdx);
   bnField.set(blockIdx, threadIdx);

   for (uint i = 0; i < MaxParticlesPerCell; i++)
   {
      overlapFractionsField.get(i) = 0.0;
      uidsField.get(i)             = id_t(0);
   }
   indicesField.get() = 0;
   bnField.get()      = 0.0;
}

// TODO: look for better mapping method
template< int Weighting_T >
__global__ void particleAndVolumeFractionMappingKernelSoA(walberla::cuda::FieldAccessor< uint_t > indicesField,
                                                          walberla::cuda::FieldAccessor< real_t > overlapFractionsField,
                                                          walberla::cuda::FieldAccessor< id_t > uidsField,
                                                          walberla::cuda::FieldAccessor< real_t > bnField, real_t omega,
                                                          double3 spherePosition, real_t sphereRadius,
                                                          double3 blockStart, double3 dx, int3 nSamples, id_t uid)
{
   indicesField.set(blockIdx, threadIdx);
   overlapFractionsField.set(blockIdx, threadIdx);
   uidsField.set(blockIdx, threadIdx);
   bnField.set(blockIdx, threadIdx);

   double3 sampleDistance       = { 1.0 / (nSamples.x + 1) * dx.x, 1.0 / (nSamples.y + 1) * dx.y,
                                    1.0 / (nSamples.z + 1) * dx.z };
   double3 startSamplingPoint   = { (blockStart.x + threadIdx.x * dx.x + sampleDistance.x),
                                    (blockStart.y + blockIdx.x * dx.y + sampleDistance.y),
                                    (blockStart.z + blockIdx.y * dx.z + sampleDistance.z) };
   double3 currentSamplingPoint = startSamplingPoint;

   double3 minCornerSphere = { spherePosition.x - sphereRadius, spherePosition.y - sphereRadius,
                               spherePosition.z - sphereRadius };
   double3 maxCornerSphere = { spherePosition.x + sphereRadius, spherePosition.y + sphereRadius,
                               spherePosition.z + sphereRadius };

   if (startSamplingPoint.x + dx.x > minCornerSphere.x && startSamplingPoint.x < maxCornerSphere.x &&
       startSamplingPoint.y + dx.y > minCornerSphere.y && startSamplingPoint.y < maxCornerSphere.y &&
       startSamplingPoint.z + dx.z > minCornerSphere.z && startSamplingPoint.z < maxCornerSphere.z)
   {
      for (uint_t z = 0; z < nSamples.z; z++)
      {
         currentSamplingPoint.y = startSamplingPoint.y;
         for (uint_t y = 0; y < nSamples.y; y++)
         {
            currentSamplingPoint.x = startSamplingPoint.x;
            for (uint_t x = 0; x < nSamples.x; x++)
            {
               if ((currentSamplingPoint.x - spherePosition.x) * (currentSamplingPoint.x - spherePosition.x) +
                      (currentSamplingPoint.y - spherePosition.y) * (currentSamplingPoint.y - spherePosition.y) +
                      (currentSamplingPoint.z - spherePosition.z) * (currentSamplingPoint.z - spherePosition.z) <=
                   sphereRadius * sphereRadius)
               {
                  overlapFractionsField.get(indicesField.get()) += 1.0;
               }
               currentSamplingPoint.x += sampleDistance.x;
            }
            currentSamplingPoint.y += sampleDistance.y;
         }
         currentSamplingPoint.z += sampleDistance.z;
      }

      overlapFractionsField.get(indicesField.get()) *= 1.0 / (nSamples.x * nSamples.y * nSamples.z);
      calculateWeighting< Weighting_T >(&overlapFractionsField.get(indicesField.get()),
                                        overlapFractionsField.get(indicesField.get()), real_t(1.0) / omega);
      if (overlapFractionsField.get(indicesField.get()) > 0)
      {
         uidsField.get(indicesField.get()) = uid;
         indicesField.get() += 1;
         bnField.get() += overlapFractionsField.get(indicesField.get());
      }
      assert(indicesField.get() < MaxParticlesPerCell);
   }
}

// TODO: find better solution for template kernels
auto instance_with_weighting_1 = particleAndVolumeFractionMappingKernelSoA< 1 >;
auto instance_with_weighting_2 = particleAndVolumeFractionMappingKernelSoA< 2 >;

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
