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
//! \file PSMKernel.cu
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//! \brief Provide two kernels that need to be called before and after the PSM sweep
//
//======================================================================================================================

#include "PSMKernel.h"
#include "PSMUtilityGPU.cuh"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace cuda
{

template< int StencilSize >
__global__ void SetParticleVelocities(walberla::cuda::FieldAccessor< uint_t > nOverlappingParticles,
                                      walberla::cuda::FieldAccessor< real_t > particleVelocitiesField,
                                      double3* __restrict__ const linearVelocities,
                                      double3* __restrict__ const angularVelocities,
                                      double3* __restrict__ const positions, const double3 blockStart, const real_t dx)
{
   nOverlappingParticles.set(blockIdx, threadIdx);
   particleVelocitiesField.set(blockIdx, threadIdx);

   // Cell center is needed in order to compute the particle velocity at this WF point
   double3 cellCenter = { (blockStart.x + (threadIdx.x + 0.5) * dx), (blockStart.y + (blockIdx.x + 0.5) * dx),
                          (blockStart.z + (blockIdx.y + 0.5) * dx) };

   // Compute the particle velocity at this WF point for all overlapping particles
   for (uint_t p = 0; p < nOverlappingParticles.get(); p++)
   {
      double3 particleVelocityAtWFPoint{ 0.0, 0.0, 0.0 };
      getVelocityAtWFPoint(&particleVelocityAtWFPoint, linearVelocities[p], angularVelocities[p], positions[p],
                           cellCenter);
      // TODO: change hard coded 3 into dimension
      particleVelocitiesField.get(p * 3 + 0) = particleVelocityAtWFPoint.x;
      particleVelocitiesField.get(p * 3 + 1) = particleVelocityAtWFPoint.y;
      particleVelocitiesField.get(p * 3 + 2) = particleVelocityAtWFPoint.z;
   }
}

template< int StencilSize >
__global__ void ReduceParticleForces(walberla::cuda::FieldAccessor< uint_t > nOverlappingParticles,
                                     walberla::cuda::FieldAccessor< id_t > uidsField,
                                     walberla::cuda::FieldAccessor< real_t > particleForcesField,
                                     double3* __restrict__ const hydrodynamicForces,
                                     double3* __restrict__ const hydrodynamicTorques,
                                     double3* __restrict__ const positions, const double3 blockStart, const real_t dx,
                                     const real_t forceScalingFactor)
{
   nOverlappingParticles.set(blockIdx, threadIdx);
   uidsField.set(blockIdx, threadIdx);
   particleForcesField.set(blockIdx, threadIdx);

   // Cell center is needed in order to compute the particle velocity at this WF point
   double3 cellCenter = { (blockStart.x + (threadIdx.x + 0.5) * dx), (blockStart.y + (blockIdx.x + 0.5) * dx),
                          (blockStart.z + (blockIdx.y + 0.5) * dx) };

   // Reduce the forces for all overlapping particles
   for (uint_t p = 0; p < nOverlappingParticles.get(); p++)
   {
      // TODO: change hard coded 3 into dimension
      double3 forceOnParticle = { particleForcesField.get(p * 3 + 0), particleForcesField.get(p * 3 + 1),
                                  particleForcesField.get(p * 3 + 2) };
      forceOnParticle.x *= forceScalingFactor;
      forceOnParticle.y *= forceScalingFactor;
      forceOnParticle.z *= forceScalingFactor;
      // TODO: use index for hydrodynamicForces and hydrodynamicTorques?
      addHydrodynamicForceAtWFPosAtomic(p, hydrodynamicForces, hydrodynamicTorques, forceOnParticle, positions[p],
                                        cellCenter);
   }
}

// TODO: find better solution for template kernels
/*auto instance_with_stencil_19  = PSMKernel< 19 >;*/
auto instance1_with_stencil_19 = SetParticleVelocities< 19 >;
auto instance2_with_stencil_19 = ReduceParticleForces< 19 >;

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
