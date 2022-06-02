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
//! \file BodyAndVolumeFractionMappingKernel.cu
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#include "ParticleAndVolumeFractionMappingKernel.h"

namespace walberla
{
__global__ void resetKernel(cuda::FieldAccessor< real_t > field)
{
   field.set(blockIdx, threadIdx);
   field.get() = 0.0;
}

__global__ void particleAndVolumeFractionMappingKernel(cuda::FieldAccessor< real_t > field, double3 spherePosition,
                                                       real_t sphereRadius, double3 blockStart)
{
   field.set(blockIdx, threadIdx);
   double3 point = { blockStart.x + threadIdx.x, blockStart.y + blockIdx.x, blockStart.z + blockIdx.y };

   if ((point.x - spherePosition.x) * (point.x - spherePosition.x) +
          (point.y - spherePosition.y) * (point.y - spherePosition.y) +
          (point.z - spherePosition.z) * (point.z - spherePosition.z) <=
       sphereRadius * sphereRadius)
   {
      field.get() = 1.0;
   }
   else { field.get() = 0.0; }
}

} // namespace walberla
