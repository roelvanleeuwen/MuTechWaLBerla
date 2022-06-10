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
//! \file DataTypes.h
//! \ingroup lbm_mesapd_coupling
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/DataTypes.h"

#include "cuda/GPUField.h"

#include "field/GhostLayerField.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{

/*
 * Typedefs specific to the lbm - mesa_pd coupling
 */
using ParticleField_T = walberla::GhostLayerField< walberla::id_t, 1 >;

namespace psm
{
// store the particle uid together with the overlap fraction
using ParticleAndVolumeFraction_T      = std::pair< id_t, real_t >;
using ParticleAndVolumeFractionField_T = GhostLayerField< std::vector< ParticleAndVolumeFraction_T >, 1 >;

namespace cuda
{
const uint MaxParticlesPerCell = 7;

struct PSMCell_T
{
   uint_t index = 0;
   real_t overlapFractions[MaxParticlesPerCell];
   id_t uids[MaxParticlesPerCell];
   // TODO: find a better solution for the padding of the struct size to 128 B. This is used since the GPU
   // field asserts that the pitch size is a multiple of the struct size
   real_t padding[(128 - (sizeof(uint_t) + sizeof(real_t[MaxParticlesPerCell]) + sizeof(id_t[MaxParticlesPerCell]))) /
                  sizeof(real_t)];

   bool operator==(PSMCell_T const& cell) const
   {
      if (index != cell.index) { return false; }
      for (uint_t i = 0; i < MaxParticlesPerCell; ++i)
      {
         if (!realIsEqual(overlapFractions[i], cell.overlapFractions[i], real_c(1e-4))) { return false; }
         if (uids[i] != cell.uids[i]) { return false; }
      }
      return true;
   };
};

using ParticleAndVolumeFractionField_T    = GhostLayerField< PSMCell_T, 1 >;
using ParticleAndVolumeFractionFieldGPU_T = walberla::cuda::GPUField< PSMCell_T >;
} // namespace cuda

} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
