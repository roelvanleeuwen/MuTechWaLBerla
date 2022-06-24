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
//! \file PSMUtility.cuh
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

__device__ const int cx[27] = {
   // C   N   S   W   E   T   B  NW  NE  SW  SE  TN  TS  TW  TE  BN  BS  BW  BE TNE TNW TSE TSW BNE BNW BSE BSW
   0, 0, 0, -1, 1, 0, 0, -1, 1, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1
};

__device__ const int cy[27] = {
   // C   N   S   W   E   T   B  NW  NE  SW  SE  TN  TS  TW  TE  BN  BS  BW  BE TNE TNW TSE TSW BNE BNW BSE BSW
   0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1
};

__device__ const int cz[27] = {
   // C   N   S   W   E   T   B  NW  NE  SW  SE  TN  TS  TW  TE  BN  BS  BW  BE TNE TNW TSE TSW BNE BNW BSE BSW
   0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1
};

__device__ void cross(double3* __restrict__ const crossResult, const double3& lhs, const double3& rhs)
{
   *crossResult = { lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x };
}

template< int StencilSize >
__device__ void getVelocityAtWFPoint(double3* __restrict__ const velocityAtWFPoint, const double3& linearVelocity,
                                     const double3& angularVelocity, const double3& position, const double3& wf_pt)
{
   double3 crossResult;
   cross(&crossResult, angularVelocity, double3{ wf_pt.x - position.x, wf_pt.y - position.y, wf_pt.z - position.z });
   *velocityAtWFPoint = { linearVelocity.x + crossResult.x, linearVelocity.y + crossResult.y,
                          linearVelocity.z + crossResult.z };
}

__device__ void addHydrodynamicForceAtWFPosAtomic(const size_t p_idx, double3* __restrict__ const particleForces,
                                                  double3* __restrict__ const particleTorques, const double3& f,
                                                  const double3& pos, const double3& wf_pt)
{
   // TODO: uncomment atomicAdds and find solution to set CMAKE_CUDA_ARCHITECTURES in .gitlab-ci.yml (maybe using nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
   /*atomicAdd(&particleForces[p_idx].x, f.x);
   atomicAdd(&particleForces[p_idx].y, f.y);
   atomicAdd(&particleForces[p_idx].z, f.z);*/

   double3 t;
   cross(&t, { wf_pt.x - pos.x, wf_pt.y - pos.y, wf_pt.z - pos.z }, f);

   /*atomicAdd(&particleTorques[p_idx].x, t.x);
   atomicAdd(&particleTorques[p_idx].y, t.y);
   atomicAdd(&particleTorques[p_idx].z, t.z);*/
}

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
