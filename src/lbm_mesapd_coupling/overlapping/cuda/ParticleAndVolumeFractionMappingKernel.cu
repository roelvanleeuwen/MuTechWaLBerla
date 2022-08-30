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

__global__ void normalizeFractionFieldKernelSoA(walberla::cuda::FieldAccessor< uint_t > nOverlappingParticlesField,
                                                walberla::cuda::FieldAccessor< real_t > BsField,
                                                walberla::cuda::FieldAccessor< real_t > BField)
{
   nOverlappingParticlesField.set(blockIdx, threadIdx);
   BsField.set(blockIdx, threadIdx);
   BField.set(blockIdx, threadIdx);

   if (BField.get() > 1)
   {
      for (uint i = 0; i < nOverlappingParticlesField.get(); i++)
      {
         BsField.get(i) /= BField.get();
      }
      BField.get() = 1.0;
   }
}

// functions to calculate Bs
template< int Weighting_T >
__device__ void calculateWeighting(real_t* __restrict__ const weighting, const real_t& /*epsilon*/,
                                   const real_t& /*tau*/)
{
   WALBERLA_STATIC_ASSERT(Weighting_T == 1 || Weighting_T == 2);
}
template<>
__device__ void calculateWeighting< 1 >(real_t* __restrict__ const weighting, const real_t& epsilon,
                                        const real_t& /*tau*/)
{
   *weighting = epsilon;
}
template<>
__device__ void calculateWeighting< 2 >(real_t* __restrict__ const weighting, const real_t& epsilon, const real_t& tau)
{
   *weighting = epsilon * (tau - real_t(0.5)) / ((real_t(1) - epsilon) + (tau - real_t(0.5)));
}

__global__ void resetKernelSoA(walberla::cuda::FieldAccessor< uint_t > nOverlappingParticlesField,
                               walberla::cuda::FieldAccessor< real_t > BsField,
                               walberla::cuda::FieldAccessor< id_t > idxField,
                               walberla::cuda::FieldAccessor< real_t > BField)
{
   nOverlappingParticlesField.set(blockIdx, threadIdx);
   BsField.set(blockIdx, threadIdx);
   idxField.set(blockIdx, threadIdx);
   BField.set(blockIdx, threadIdx);

   for (uint i = 0; i < MaxParticlesPerCell; i++)
   {
      BsField.get(i)  = 0.0;
      idxField.get(i) = id_t(0);
   }
   nOverlappingParticlesField.get() = 0;
   BField.get()                     = 0.0;
}

template< int Weighting_T >
__global__ void particleAndVolumeFractionMappingKernelSoA(
   walberla::cuda::FieldAccessor< uint_t > nOverlappingParticlesField, walberla::cuda::FieldAccessor< real_t > BsField,
   walberla::cuda::FieldAccessor< id_t > idxField, walberla::cuda::FieldAccessor< real_t > BField, real_t omega,
   double3 spherePosition, real_t sphereRadius, double3 blockStart, real_t dx, int3 nSamples, size_t idx)
{
   nOverlappingParticlesField.set(blockIdx, threadIdx);
   BsField.set(blockIdx, threadIdx);
   idxField.set(blockIdx, threadIdx);
   BField.set(blockIdx, threadIdx);

   double3 sampleDistance = { 1.0 / (nSamples.x + 1) * dx, 1.0 / (nSamples.y + 1) * dx, 1.0 / (nSamples.z + 1) * dx };
   double3 startSamplingPoint   = { (blockStart.x + threadIdx.x * dx + sampleDistance.x),
                                    (blockStart.y + blockIdx.x * dx + sampleDistance.y),
                                    (blockStart.z + blockIdx.y * dx + sampleDistance.z) };
   double3 currentSamplingPoint = startSamplingPoint;

   double3 minCornerSphere = { spherePosition.x - sphereRadius, spherePosition.y - sphereRadius,
                               spherePosition.z - sphereRadius };
   double3 maxCornerSphere = { spherePosition.x + sphereRadius, spherePosition.y + sphereRadius,
                               spherePosition.z + sphereRadius };

   double overlapFraction = 0.0;

   if (startSamplingPoint.x + dx > minCornerSphere.x && startSamplingPoint.x < maxCornerSphere.x &&
       startSamplingPoint.y + dx > minCornerSphere.y && startSamplingPoint.y < maxCornerSphere.y &&
       startSamplingPoint.z + dx > minCornerSphere.z && startSamplingPoint.z < maxCornerSphere.z)
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
                  overlapFraction += 1.0;
               }
               currentSamplingPoint.x += sampleDistance.x;
            }
            currentSamplingPoint.y += sampleDistance.y;
         }
         currentSamplingPoint.z += sampleDistance.z;
      }

      // store overlap fraction only if there is an intersection
      if (overlapFraction > 0.0)
      {
         assert(nOverlappingParticlesField.get() < MaxParticlesPerCell);
         BsField.get(nOverlappingParticlesField.get()) = overlapFraction;
         BsField.get(nOverlappingParticlesField.get()) *= 1.0 / (nSamples.x * nSamples.y * nSamples.z);
         calculateWeighting< Weighting_T >(&BsField.get(nOverlappingParticlesField.get()),
                                           BsField.get(nOverlappingParticlesField.get()), real_t(1.0) / omega);
         idxField.get(nOverlappingParticlesField.get()) = idx;
         BField.get() += BsField.get(nOverlappingParticlesField.get());
         nOverlappingParticlesField.get() += 1;
      }
   }
}

// Based on the following paper: https://doi.org/10.1108/EC-02-2016-0052
// TODO: why does the paper say: "requires only a single addition operation to estimate the intersection volume"
template< int Weighting_T >
__global__ void linearApproximation(walberla::cuda::FieldAccessor< uint_t > nOverlappingParticlesField,
                                    walberla::cuda::FieldAccessor< real_t > BsField,
                                    walberla::cuda::FieldAccessor< id_t > idxField,
                                    walberla::cuda::FieldAccessor< real_t > BField, real_t omega,
                                    real_t* __restrict__ const spherePositions, real_t* __restrict__ const sphereRadii,
                                    real_t* __restrict__ const f_rs, double3 blockStart, real_t dx, size_t numParticles)
{
   nOverlappingParticlesField.set(blockIdx, threadIdx);
   BsField.set(blockIdx, threadIdx);
   idxField.set(blockIdx, threadIdx);
   BField.set(blockIdx, threadIdx);

   const double3 cellCenter = { (blockStart.x + (threadIdx.x + 0.5) * dx), (blockStart.y + (blockIdx.x + 0.5) * dx),
                                (blockStart.z + (blockIdx.y + 0.5) * dx) };

   for (int idxMapped = 0; idxMapped < numParticles; idxMapped++)
   {
      double3 minCornerSphere = { spherePositions[idxMapped * 3] - sphereRadii[idxMapped],
                                  spherePositions[idxMapped * 3 + 1] - sphereRadii[idxMapped],
                                  spherePositions[idxMapped * 3 + 2] - sphereRadii[idxMapped] };
      double3 maxCornerSphere = { spherePositions[idxMapped * 3] + sphereRadii[idxMapped],
                                  spherePositions[idxMapped * 3 + 1] + sphereRadii[idxMapped],
                                  spherePositions[idxMapped * 3 + 2] + sphereRadii[idxMapped] };
      if (cellCenter.x + dx > minCornerSphere.x && cellCenter.x - dx < maxCornerSphere.x &&
          cellCenter.y + dx > minCornerSphere.y && cellCenter.y - dx < maxCornerSphere.y &&
          cellCenter.z + dx > minCornerSphere.z && cellCenter.z - dx < maxCornerSphere.z)
      {
         const double3 cellSphereVector = { spherePositions[idxMapped * 3] - cellCenter.x,
                                            spherePositions[idxMapped * 3 + 1] - cellCenter.y,
                                            spherePositions[idxMapped * 3 + 2] - cellCenter.z };

         const real_t D = sqrt(cellSphereVector.x * cellSphereVector.x + cellSphereVector.y * cellSphereVector.y +
                               cellSphereVector.z * cellSphereVector.z) -
                          sphereRadii[idxMapped];

         real_t epsilon = -D + f_rs[idxMapped];
         epsilon        = max(epsilon, 0.0);
         epsilon        = min(epsilon, 1.0);

         // store overlap fraction only if there is an intersection
         if (epsilon > 0.0)
         {
            assert(nOverlappingParticlesField.get() < MaxParticlesPerCell);
            BsField.get(nOverlappingParticlesField.get()) = epsilon;
            calculateWeighting< Weighting_T >(&BsField.get(nOverlappingParticlesField.get()),
                                              BsField.get(nOverlappingParticlesField.get()), real_t(1.0) / omega);
            idxField.get(nOverlappingParticlesField.get()) = idxMapped;
            BField.get() += BsField.get(nOverlappingParticlesField.get());
            nOverlappingParticlesField.get() += 1;
         }
      }
   }
}

// TODO: find better solution for template kernels
auto instance0_with_weighting_1 = particleAndVolumeFractionMappingKernelSoA< 1 >;
auto instance1_with_weighting_2 = particleAndVolumeFractionMappingKernelSoA< 2 >;
auto instance2_with_weighting_1 = linearApproximation< 1 >;
auto instance3_with_weighting_2 = linearApproximation< 2 >;

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
