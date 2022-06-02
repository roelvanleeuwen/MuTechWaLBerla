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

#include "lbm_mesapd_coupling/DataTypes.h"
#include "lbm_mesapd_coupling/mapping/ParticleBoundingBox.h"
#include "lbm_mesapd_coupling/overlapping/OverlapFraction.h"
#include "lbm_mesapd_coupling/utility/ParticleSelector.h"

#include "mesa_pd/data/ParticleAccessorWithShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/kernel/SingleCast.h"

#include <functional>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/shape/Sphere.h>

#include "ParticleAndVolumeFractionMappingKernel.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{

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
      // clear the field
      for (auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt)
      {
         auto cudaField = blockIt->getData< cuda::GPUField< real_t > >(particleAndVolumeFractionFieldID_);
         auto myKernel  = cuda::make_kernel(&resetKernel);
         myKernel.addFieldIndexingParam(cuda::FieldIndexing< real_t >::xyz(*cudaField));
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
      Vector3< real_t > particlePosition = ac_->getPosition(idx);

      // update fraction mapping
      for (auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt)
      {
         auto cudaField = blockIt->getData< cuda::GPUField< real_t > >(particleAndVolumeFractionFieldID_);

         auto myKernel = cuda::make_kernel(&particleAndVolumeFractionMappingKernel);
         myKernel.addFieldIndexingParam(cuda::FieldIndexing< real_t >::xyz(*cudaField));
         Vector3< real_t > blockStart = blockIt->getAABB().minCorner();
         myKernel.addParam(double3{ particlePosition[0], particlePosition[1], particlePosition[2] });
         myKernel.addParam(static_cast<mesa_pd::data::Sphere*>(ac_->getShape(idx))->getRadius());
         myKernel.addParam(double3{ blockStart[0], blockStart[1], blockStart[2] });
         myKernel();
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

} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
