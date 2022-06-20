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
//! \file DataTypesGPU.h
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/DataTypes.h"

#include "cuda/AddGPUFieldToStorage.h"
#include "cuda/GPUField.h"

#include "field/GhostLayerField.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace cuda
{

const uint MaxParticlesPerCell = 7;

struct ParticleAndVolumeFractionAoS_T
{
   uint_t index = 0;
   real_t overlapFractions[MaxParticlesPerCell];
   id_t uids[MaxParticlesPerCell];

   bool operator==(ParticleAndVolumeFractionAoS_T const& cell) const
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

using ParticleAndVolumeFractionField_T    = GhostLayerField< ParticleAndVolumeFractionAoS_T, 1 >;
using ParticleAndVolumeFractionFieldGPU_T = walberla::cuda::GPUField< ParticleAndVolumeFractionAoS_T >;

using indicesField_T             = GhostLayerField< uint_t, 1 >;
using indicesFieldGPU_T          = walberla::cuda::GPUField< uint_t >;
using overlapFractionsField_T    = GhostLayerField< real_t, MaxParticlesPerCell >;
using overlapFractionsFieldGPU_T = walberla::cuda::GPUField< real_t >;
using uidsField_T                = GhostLayerField< id_t, MaxParticlesPerCell >;
using uidsFieldGPU_T             = walberla::cuda::GPUField< id_t >;

template< uint_t StencilSize >
struct ParticleAndVolumeFractionSoA_T
{
   BlockDataID indicesFieldID;
   BlockDataID overlapFractionsFieldID;
   BlockDataID uidsFieldID;

   ParticleAndVolumeFractionSoA_T(const shared_ptr< StructuredBlockStorage >& bs, const BlockDataID& indicesFieldCPUID,
                                  const BlockDataID& overlapFractionsFieldCPUID, const BlockDataID& uidsFieldCPUID)
   {
      indicesFieldID =
         walberla::cuda::addGPUFieldToStorage< indicesField_T >(bs, indicesFieldCPUID, "indices field GPU");
      overlapFractionsFieldID = walberla::cuda::addGPUFieldToStorage< overlapFractionsField_T >(
         bs, overlapFractionsFieldCPUID, "overlapFractions field GPU");
      uidsFieldID = walberla::cuda::addGPUFieldToStorage< uidsField_T >(bs, uidsFieldCPUID, "uids field GPU");
   }

   ParticleAndVolumeFractionSoA_T(const shared_ptr< StructuredBlockStorage >& bs)
   {
      indicesFieldID = walberla::cuda::addGPUFieldToStorage< indicesFieldGPU_T >(bs, "indices field GPU", uint_t(1),
                                                                                 field::fzyx, uint_t(0), true);
      overlapFractionsFieldID = walberla::cuda::addGPUFieldToStorage< overlapFractionsFieldGPU_T >(
         bs, "overlapFractions field GPU", MaxParticlesPerCell, field::fzyx, uint_t(0), true);
      uidsFieldID = walberla::cuda::addGPUFieldToStorage< uidsFieldGPU_T >(bs, "uids field GPU", MaxParticlesPerCell,
                                                                           field::fzyx, uint_t(0), true);
   }
};

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
