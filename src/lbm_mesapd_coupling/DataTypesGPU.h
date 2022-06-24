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
   real_t Bs[MaxParticlesPerCell];
   id_t uids[MaxParticlesPerCell];

   bool operator==(ParticleAndVolumeFractionAoS_T const& cell) const
   {
      if (index != cell.index) { return false; }
      for (uint_t i = 0; i < MaxParticlesPerCell; ++i)
      {
         if (!realIsEqual(Bs[i], cell.Bs[i], real_c(1e-4))) { return false; }
         if (uids[i] != cell.uids[i]) { return false; }
      }
      return true;
   };
};

using ParticleAndVolumeFractionField_T    = GhostLayerField< ParticleAndVolumeFractionAoS_T, 1 >;
using ParticleAndVolumeFractionFieldGPU_T = walberla::cuda::GPUField< ParticleAndVolumeFractionAoS_T >;

// B denotes the local weighting factor and is calculated by taking the sum of all local particle
// weighting factor Bs. The naming of the variables is based on the following paper:
// https://doi.org/10.1016/j.compfluid.2017.05.033

using nOverlappingParticlesField_T    = GhostLayerField< uint_t, 1 >;
using nOverlappingParticlesFieldGPU_T = walberla::cuda::GPUField< uint_t >;
using BsField_T                       = GhostLayerField< real_t, MaxParticlesPerCell >;
using BsFieldGPU_T                    = walberla::cuda::GPUField< real_t >;
using uidsField_T                     = GhostLayerField< id_t, MaxParticlesPerCell >;
using uidsFieldGPU_T                  = walberla::cuda::GPUField< id_t >;
using BFieldGPU_T                     = walberla::cuda::GPUField< real_t >;

template< int Weighting_T >
struct ParticleAndVolumeFractionSoA_T
{
   BlockDataID nOverlappingParticlesFieldID;
   BlockDataID BsFieldID;
   BlockDataID uidsFieldID;
   BlockDataID BFieldID;
   // relaxation rate omega is used for Weighting_T != 1
   real_t omega_;

   // TODO: set nrOfGhostLayers to 0 (requires changes of the generated kernels)
   ParticleAndVolumeFractionSoA_T(const shared_ptr< StructuredBlockStorage >& bs,
                                  const BlockDataID& nOverlappingParticlesFieldCPUID, const BlockDataID& BsFieldCPUID,
                                  const BlockDataID& uidsFieldCPUID, const real_t omega)
   {
      nOverlappingParticlesFieldID = walberla::cuda::addGPUFieldToStorage< nOverlappingParticlesField_T >(
         bs, nOverlappingParticlesFieldCPUID, "indices field GPU");
      BsFieldID   = walberla::cuda::addGPUFieldToStorage< BsField_T >(bs, BsFieldCPUID, "Bs field GPU");
      uidsFieldID = walberla::cuda::addGPUFieldToStorage< uidsField_T >(bs, uidsFieldCPUID, "uids field GPU");
      BFieldID =
         walberla::cuda::addGPUFieldToStorage< BFieldGPU_T >(bs, "bn field GPU", 1, field::fzyx, uint_t(1), true);
      omega_ = omega;
   }

   ParticleAndVolumeFractionSoA_T(const shared_ptr< StructuredBlockStorage >& bs, const real_t omega)
   {
      nOverlappingParticlesFieldID = walberla::cuda::addGPUFieldToStorage< nOverlappingParticlesFieldGPU_T >(
         bs, "indices field GPU", uint_t(1), field::fzyx, uint_t(1), true);
      BsFieldID   = walberla::cuda::addGPUFieldToStorage< BsFieldGPU_T >(bs, "Bs field GPU", MaxParticlesPerCell,
                                                                       field::fzyx, uint_t(1), true);
      uidsFieldID = walberla::cuda::addGPUFieldToStorage< uidsFieldGPU_T >(bs, "uids field GPU", MaxParticlesPerCell,
                                                                           field::fzyx, uint_t(1), true);
      BFieldID =
         walberla::cuda::addGPUFieldToStorage< BFieldGPU_T >(bs, "bn field GPU", 1, field::fzyx, uint_t(1), true);
      omega_ = omega;
   }
};

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
