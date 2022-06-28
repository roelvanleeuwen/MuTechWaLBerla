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
//! \file PSMKernel.h
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

template< int StencilSize >
__global__ void PSMKernel(walberla::cuda::FieldAccessor< uint_t > nOverlappingParticles,
                          walberla::cuda::FieldAccessor< real_t > BsField,
                          walberla::cuda::FieldAccessor< id_t > uidsField,
                          walberla::cuda::FieldAccessor< real_t > BField, walberla::cuda::FieldAccessor< real_t > pdfs,
                          walberla::cuda::FieldAccessor< real_t > solidCollisionField,
                          double* __restrict__ const solidCollisionFieldData, ulong3* __restrict__ const size,
                          int4* __restrict__ const stride, double3* __restrict__ const hydrodynamicForces,
                          double3* __restrict__ const hydrodynamicTorques, double3* __restrict__ const linearVelocities,
                          double3* __restrict__ const angularVelocities, double3* __restrict__ const positions,
                          const double3 blockStart, const real_t dx, const real_t forceScalingFactor);

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
