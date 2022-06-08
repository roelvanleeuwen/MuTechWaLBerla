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

// TODO: fix namespaces
namespace walberla
{

__global__ void resetKernel(cuda::FieldAccessor< real_t > field, cuda::FieldAccessor< uint_t > indexField);
__global__ void particleAndVolumeFractionMappingKernel(cuda::FieldAccessor< real_t > field,
                                                       cuda::FieldAccessor< uint_t > indexField, double3 spherePosition,
                                                       real_t sphereRadius, double3 blockStart, double3 dx,
                                                       int3 nSamples);

} // namespace walberla
