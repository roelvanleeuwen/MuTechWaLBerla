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
//! \file ParticleAndVolumeFractionMapping.h
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "cuda/AddGPUFieldToStorage.h"
#include "cuda/FieldCopy.h"
#include "cuda/FieldIndexing.h"
#include "cuda/GPUField.h"
#include "cuda/HostFieldAllocator.h"
#include "cuda/Kernel.h"

#include "domain_decomposition/StructuredBlockStorage.h"

#include "field/GhostLayerField.h"

#include "lbm_mesapd_coupling/DataTypesGPU.h"
#include "lbm_mesapd_coupling/mapping/ParticleBoundingBox.h"
#include "lbm_mesapd_coupling/overlapping/cuda/OverlapFraction.h"
#include "lbm_mesapd_coupling/overlapping/cuda/ParticleAndVolumeFractionMappingKernel.h"
#include "lbm_mesapd_coupling/utility/ParticleSelector.h"

#include "mesa_pd/data/ParticleAccessorWithShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/kernel/SingleCast.h"

#include <functional>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/shape/Sphere.h>

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace cuda
{
// TODO: use superSamplingDepth
template< typename ParticleAccessor_T, typename ParticleSelector_T >
class ParticleAndVolumeFractionMappingGPU
{
 public:
   ParticleAndVolumeFractionMappingGPU(const shared_ptr< StructuredBlockStorage >& blockStorage,
                                       const shared_ptr< ParticleAccessor_T >& ac,
                                       const ParticleSelector_T& mappingParticleSelector,
                                       const BlockDataID& particleAndVolumeFractionFieldID,
                                       const uint_t superSamplingDepth = uint_t(4))
      : blockStorage_(blockStorage), ac_(ac), mappingParticleSelector_(mappingParticleSelector),
        particleAndVolumeFractionFieldID_(particleAndVolumeFractionFieldID), superSamplingDepth_(superSamplingDepth)
   {
      static_assert(std::is_base_of< mesa_pd::data::IAccessor, ParticleAccessor_T >::value,
                    "Provide a valid accessor as template");
   }

   void operator()()
   {
      // clear the fields
      for (auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt)
      {
         auto cudaField = blockIt->getData< walberla::cuda::GPUField< PSMCellAoS_T > >(particleAndVolumeFractionFieldID_);
         auto myKernel  = walberla::cuda::make_kernel(&resetKernelAoS);
         myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< PSMCellAoS_T >::xyz(*cudaField));
         myKernel();
      }

      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { update(idx); }
      }
   }

 private:
   void update(const size_t idx)
   {
      // update fraction mapping
      for (auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt)
      {
         singleCast_(idx, *ac_, overlapFractionFctr_, ac_, *blockIt, particleAndVolumeFractionFieldID_);
      }
   }

   shared_ptr< StructuredBlockStorage > blockStorage_;
   const shared_ptr< ParticleAccessor_T > ac_;
   ParticleSelector_T mappingParticleSelector_;
   const BlockDataID particleAndVolumeFractionFieldID_;
   const uint_t superSamplingDepth_;

   mesa_pd::kernel::SingleCast singleCast_;
   OverlapFractionFunctor overlapFractionFctr_;
};

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
