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
//! \file ParticleAndVolumeFractionMappingKernel.h
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/DataTypes.h"
#include "core/math/Vector3.h"

#include "cuda/FieldAccessor.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace cuda
{

/*__global__ void resetKernelAoS(walberla::cuda::FieldAccessor< ParticleAndVolumeFractionAoS_T > field);
__global__ void
   particleAndVolumeFractionMappingKernelAoS(walberla::cuda::FieldAccessor< ParticleAndVolumeFractionAoS_T > field,
                                             double3 spherePosition, real_t sphereRadius, double3 blockStart,
                                             double3 dx, int3 nSamples, id_t uid);*/

__global__ void resetKernelSoA(walberla::cuda::FieldAccessor< uint_t > indicesField,
                               walberla::cuda::FieldAccessor< real_t > overlapFractionsField,
                               walberla::cuda::FieldAccessor< id_t > uidsField,
                               walberla::cuda::FieldAccessor< real_t > bnField);

template< int Weighting_T >
__global__ void particleAndVolumeFractionMappingKernelSoA(walberla::cuda::FieldAccessor< uint_t > indicesField,
                                                          walberla::cuda::FieldAccessor< real_t > overlapFractionsField,
                                                          walberla::cuda::FieldAccessor< id_t > uidsField,
                                                          walberla::cuda::FieldAccessor< real_t > bnField, real_t omega,
                                                          double3 spherePosition, real_t sphereRadius,
                                                          double3 blockStart, double3 dx, int3 nSamples, id_t uid);

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
