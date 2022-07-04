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
//! \file PSMUtilityGPU.cuh
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "cuda/FieldAccessor.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace cuda
{

__device__ void cross(double3* __restrict__ const crossResult, const double3& lhs, const double3& rhs)
{
   *crossResult = { lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x };
}

__device__ void getVelocityAtWFPoint(double3* __restrict__ const velocityAtWFPoint, const double3& linearVelocity,
                                     const double3& angularVelocity, const double3& position, const double3& wf_pt)
{
   double3 crossResult;
   cross(&crossResult, angularVelocity, double3{ wf_pt.x - position.x, wf_pt.y - position.y, wf_pt.z - position.z });
   *velocityAtWFPoint = { linearVelocity.x + crossResult.x, linearVelocity.y + crossResult.y,
                          linearVelocity.z + crossResult.z };
}

__device__ void addHydrodynamicForceAtWFPosAtomic(double3& particleForce, double3& particleTorque, const double3& f,
                                                  const double3& pos, const double3& wf_pt)
{
   // TODO: uncomment atomicAdds and find solution to set CMAKE_CUDA_ARCHITECTURES in .gitlab-ci.yml (maybe using
   // nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
   /*atomicAdd(&(particleForce.x), f.x);
   atomicAdd(&(particleForce.y), f.y);
   atomicAdd(&(particleForce.z), f.z);*/

   double3 torque;
   cross(&torque, { wf_pt.x - pos.x, wf_pt.y - pos.y, wf_pt.z - pos.z }, f);

   /*atomicAdd(&(particleTorque.x), torque.x);
   atomicAdd(&(particleTorque.y), torque.y);
   atomicAdd(&(particleTorque.z), torque.z);*/
}

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
