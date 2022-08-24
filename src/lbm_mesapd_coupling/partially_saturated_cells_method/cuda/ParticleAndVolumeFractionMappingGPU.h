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

template< int Weighting_T >
void normalizeFractionField(const IBlock& blockIt,
                            const ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA)
{
   auto nOverlappingParticlesField =
      blockIt.getData< nOverlappingParticlesFieldGPU_T >(particleAndVolumeFractionSoA.nOverlappingParticlesFieldID);
   auto BsField = blockIt.getData< BsFieldGPU_T >(particleAndVolumeFractionSoA.BsFieldID);
   auto BField  = blockIt.getData< BFieldGPU_T >(particleAndVolumeFractionSoA.BFieldID);

   auto myKernel = walberla::cuda::make_kernel(&normalizeFractionFieldKernelSoA);
   myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< uint_t >::xyz(*nOverlappingParticlesField));
   myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*BsField));
   myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*BField));
   myKernel();
}

template< int Weighting_T >
void clearField(const IBlock& blockIt,
                const ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA)
{
   auto nOverlappingParticlesField =
      blockIt.getData< nOverlappingParticlesFieldGPU_T >(particleAndVolumeFractionSoA.nOverlappingParticlesFieldID);
   auto BsField  = blockIt.getData< BsFieldGPU_T >(particleAndVolumeFractionSoA.BsFieldID);
   auto idxField = blockIt.getData< idxFieldGPU_T >(particleAndVolumeFractionSoA.idxFieldID);
   auto BField   = blockIt.getData< BFieldGPU_T >(particleAndVolumeFractionSoA.BFieldID);

   auto myKernel = walberla::cuda::make_kernel(&resetKernelSoA);
   myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< uint_t >::xyz(*nOverlappingParticlesField));
   myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*BsField));
   myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< id_t >::xyz(*idxField));
   myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*BField));
   myKernel();
}

// TODO: use or remove superSamplingDepth
template< typename ParticleAccessor_T, typename ParticleSelector_T, int Weighting_T >
class ParticleAndVolumeFractionMappingGPU
{
 public:
   ParticleAndVolumeFractionMappingGPU(const shared_ptr< StructuredBlockStorage >& blockStorage,
                                       const shared_ptr< ParticleAccessor_T >& ac,
                                       const ParticleSelector_T& mappingParticleSelector,
                                       ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionField,
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

      size_t numMappedParticles = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { numMappedParticles++; }
      }
      const size_t scalarArraySize = numMappedParticles * sizeof(real_t);

      // Allocate unified memory for the particle information needed for computing the overlap fraction
      real_t* positions;
      cudaMallocManaged(&positions, 3 * scalarArraySize);
      cudaMemset(positions, 0, 3 * scalarArraySize);
      real_t* radii;
      cudaMallocManaged(&radii, scalarArraySize);
      cudaMemset(radii, 0, scalarArraySize);
      real_t* f_r;
      cudaMallocManaged(&f_r, scalarArraySize);
      cudaMemset(f_r, 0, scalarArraySize);

      // Store particle information inside unified memory to communicate information to the GPU
      size_t idxMapped = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_))
         {
            for (size_t d = 0; d < 3; ++d)
            {
               positions[idxMapped * 3 + d] = ac_->getPosition(idx)[d];
            }
            const real_t radius = static_cast< mesa_pd::data::Sphere* >(ac_->getShape(idx))->getRadius();
            radii[idxMapped]    = radius;
            real_t Va           = real_t(
               (1.0 / 12.0 - radius * radius) * atan((0.5 * sqrt(radius * radius - 0.5)) / (0.5 - radius * radius)) +
               1.0 / 3.0 * sqrt(radius * radius - 0.5) +
               (radius * radius - 1.0 / 12.0) * atan(0.5 / sqrt(radius * radius - 0.5)) -
               4.0 / 3.0 * radius * radius * radius * atan(0.25 / (radius * sqrt(radius * radius - 0.5))));
            f_r[idxMapped] = Va - radius + real_t(0.5);
            idxMapped++;
         }
      }

      // Store uids to check later on if the particles have changed
      particleAndVolumeFractionField_.mappingUIDs.clear();

      // TODO: simplify idx/idxMapped
      // update fraction mapping
      for (auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt)
      {
         auto nOverlappingParticlesField = blockIt->getData< nOverlappingParticlesFieldGPU_T >(
            particleAndVolumeFractionField_.nOverlappingParticlesFieldID);
         auto BsField  = blockIt->getData< BsFieldGPU_T >(particleAndVolumeFractionField_.BsFieldID);
         auto idxField = blockIt->getData< idxFieldGPU_T >(particleAndVolumeFractionField_.idxFieldID);
         auto BField   = blockIt->getData< BFieldGPU_T >(particleAndVolumeFractionField_.BFieldID);

         auto myKernel = walberla::cuda::make_kernel(&(linearApproximation< Weighting_T >) );
         myKernel.addFieldIndexingParam(
            walberla::cuda::FieldIndexing< uint_t >::xyz(*nOverlappingParticlesField));          // FieldAccessor
         myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*BsField)); // FieldAccessor
         myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< id_t >::xyz(*idxField));  // FieldAccessor
         myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*BField));  // FieldAccessor
         myKernel.addParam(particleAndVolumeFractionField_.omega_);                              // omega
         myKernel.addParam(positions);                                                           // spherePositions
         myKernel.addParam(radii);                                                               // sphereRadii
         myKernel.addParam(f_r);                                                                 // f_rs
         Vector3< real_t > blockStart = blockIt->getAABB().minCorner();
         myKernel.addParam(double3{ blockStart[0], blockStart[1], blockStart[2] });                   // blockStart
         myKernel.addParam(blockIt->getAABB().xSize() / real_t(nOverlappingParticlesField->xSize())); // dx
         // myKernel.addParam(int3{ 16, 16, 16 }); // nSamples
         myKernel.addParam(numMappedParticles); // numParticles
         myKernel();
      }

      cudaFree(positions);
      cudaFree(radii);
      cudaFree(f_r);

      idxMapped = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_))
         {
            /*update(idx, idxMapped);*/
            particleAndVolumeFractionField_.mappingUIDs.push_back(ac_->getUid(idx));
            idxMapped++;
         }
      }

      // normalize fraction field (Bs) if sum over all fractions (B) > 1
      for (auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt)
      {
         normalizeFractionField(*blockIt, particleAndVolumeFractionField_);
      }
      // This visualization is necessary so that the timeloop shows the correct time
      // TODO: maybe remove this synchronization when the particle mapping is no longer the bottleneck (then also remove
      // it from timeloop)
      cudaDeviceSynchronize();
   }

 private:
   void update(const size_t idx, const size_t idxMapped)
   {
      // update fraction mapping
      for (auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt)
      {
         // apply mapping only if block intersects with particle
         if (blockIt->getAABB().intersects(walberla::mesa_pd::getParticleAABB(idx, *ac_)))
         {
            singleCast_(idx, *ac_, overlapFractionFctr_, ac_, *blockIt, particleAndVolumeFractionField_,
                        particleAndVolumeFractionField_.omega_, idxMapped);
         }
      }
   }

   shared_ptr< StructuredBlockStorage > blockStorage_;
   const shared_ptr< ParticleAccessor_T > ac_;
   const ParticleSelector_T& mappingParticleSelector_;
   ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionField_;
   const uint_t superSamplingDepth_;

   mesa_pd::kernel::SingleCast singleCast_;
   OverlapFractionFunctor overlapFractionFctr_;
};

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
