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
//! \file ParticleAndVolumeFractionMappingKernels.h
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

template< int Weighting_T >
__global__ void particleAndVolumeFractionMappingKernelSoA(
   walberla::cuda::FieldAccessor< uint_t > nOverlappingParticlesField, walberla::cuda::FieldAccessor< real_t > BsField,
   walberla::cuda::FieldAccessor< id_t > idxField, walberla::cuda::FieldAccessor< real_t > BField, real_t omega,
   double3 spherePosition, real_t sphereRadius, double3 blockStart, real_t dx, int3 nSamples, id_t uid);

template< int Weighting_T >
__global__ void linearApproximation(walberla::cuda::FieldAccessor< uint_t > nOverlappingParticlesField,
                                    walberla::cuda::FieldAccessor< real_t > BsField,
                                    walberla::cuda::FieldAccessor< id_t > idxField,
                                    walberla::cuda::FieldAccessor< real_t > BField, real_t omega,
                                    real_t* __restrict__ const spherePositions, real_t* __restrict__ const sphereRadii,
                                    real_t* __restrict__ const f_rs, double3 blockStart, real_t dx,
                                    size_t* __restrict__ const numParticlesSubBlocks,
                                    size_t* __restrict__ const particleIDsSubBlocks, const size_t subBlocksPerDim);

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
