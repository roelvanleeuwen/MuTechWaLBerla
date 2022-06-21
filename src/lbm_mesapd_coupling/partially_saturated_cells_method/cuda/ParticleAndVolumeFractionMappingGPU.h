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

#include "mesa_pd/common/AABBConversion.h"
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

/*void clearField(const IBlock& blockIt, const BlockDataID& particleAndVolumeFractionField)
{
   auto cudaField =
      blockIt.getData< walberla::cuda::GPUField< ParticleAndVolumeFractionAoS_T > >(particleAndVolumeFractionField);
   auto myKernel = walberla::cuda::make_kernel(&resetKernelAoS);
   myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< ParticleAndVolumeFractionAoS_T >::xyz(*cudaField));
   myKernel();
}*/

void clearField(const IBlock& blockIt, const ParticleAndVolumeFractionSoA_T& particleAndVolumeFractionSoA)
{
   auto indicesField = blockIt.getData< indicesFieldGPU_T >(particleAndVolumeFractionSoA.indicesFieldID);
   auto overlapFractionsField =
      blockIt.getData< overlapFractionsFieldGPU_T >(particleAndVolumeFractionSoA.overlapFractionsFieldID);
   auto uidsField = blockIt.getData< uidsFieldGPU_T >(particleAndVolumeFractionSoA.uidsFieldID);

   auto myKernel = walberla::cuda::make_kernel(&resetKernelSoA);
   myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< uint_t >::xyz(*indicesField));
   myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*overlapFractionsField));
   myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< id_t >::xyz(*uidsField));
   myKernel();
}

// TODO: use superSamplingDepth
template< typename ParticleAccessor_T, typename ParticleSelector_T >
class ParticleAndVolumeFractionMappingGPU
{
 public:
   ParticleAndVolumeFractionMappingGPU(const shared_ptr< StructuredBlockStorage >& blockStorage,
                                       const shared_ptr< ParticleAccessor_T >& ac,
                                       const ParticleSelector_T& mappingParticleSelector,
                                       const ParticleAndVolumeFractionSoA_T& particleAndVolumeFractionField,
                                       const uint_t superSamplingDepth = uint_t(4))
      : blockStorage_(blockStorage), ac_(ac), mappingParticleSelector_(mappingParticleSelector),
        particleAndVolumeFractionField_(particleAndVolumeFractionField), superSamplingDepth_(superSamplingDepth)
   {
      static_assert(std::is_base_of< mesa_pd::data::IAccessor, ParticleAccessor_T >::value,
                    "Provide a valid accessor as template");
   }

   void operator()()
   {
      // clear the fields
      for (auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt)
      {
         clearField(*blockIt, particleAndVolumeFractionField_);
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
         // apply mapping only if block intersects with particle
         if (blockIt->getAABB().intersects(walberla::mesa_pd::getParticleAABB(idx, *ac_)))
         {
            singleCast_(idx, *ac_, overlapFractionFctr_, ac_, *blockIt, particleAndVolumeFractionField_);
         }
      }
   }

   shared_ptr< StructuredBlockStorage > blockStorage_;
   const shared_ptr< ParticleAccessor_T > ac_;
   ParticleSelector_T mappingParticleSelector_;
   const ParticleAndVolumeFractionSoA_T particleAndVolumeFractionField_;
   const uint_t superSamplingDepth_;

   mesa_pd::kernel::SingleCast singleCast_;
   OverlapFractionFunctor overlapFractionFctr_;
};

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
