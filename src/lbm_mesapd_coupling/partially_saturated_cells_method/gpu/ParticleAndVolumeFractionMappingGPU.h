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
//! \file ParticleAndVolumeFractionMappingGPU.h
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "domain_decomposition/StructuredBlockStorage.h"

#include "field/GhostLayerField.h"

#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/FieldCopy.h"
#include "gpu/FieldIndexing.h"
#include "gpu/GPUField.h"
#include "gpu/HostFieldAllocator.h"
#include "gpu/Kernel.h"

#include "lbm_mesapd_coupling/DataTypesGPU.h"
#include "lbm_mesapd_coupling/mapping/ParticleBoundingBox.h"
#include "lbm_mesapd_coupling/overlapping/gpu/ParticleAndVolumeFractionMappingKernels.h"
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
namespace gpu
{

template< int Weighting_T >
void mapParticles(const IBlock& blockIt,
                  const ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA,
                  const real_t* const spherePositions, const real_t* const sphereRadii, const real_t* const f_rs,
                  const size_t* const numParticlesSubBlocks, const size_t* const particleIDsSubBlocks,
                  const Vector3< uint_t > subBlocksPerDim)
{
   auto nOverlappingParticlesField =
      blockIt.getData< nOverlappingParticlesFieldGPU_T >(particleAndVolumeFractionSoA.nOverlappingParticlesFieldID);
   auto BsField  = blockIt.getData< BsFieldGPU_T >(particleAndVolumeFractionSoA.BsFieldID);
   auto idxField = blockIt.getData< idxFieldGPU_T >(particleAndVolumeFractionSoA.idxFieldID);
   auto BField   = blockIt.getData< BFieldGPU_T >(particleAndVolumeFractionSoA.BFieldID);

   auto myKernel = walberla::gpu::make_kernel(&(linearApproximation< Weighting_T >) );
   myKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< uint_t >::xyz(*nOverlappingParticlesField));
   myKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< real_t >::xyz(*BsField));
   myKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< id_t >::xyz(*idxField));
   myKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< real_t >::xyz(*BField));
   myKernel.addParam(particleAndVolumeFractionSoA.omega_);
   myKernel.addParam(spherePositions);
   myKernel.addParam(sphereRadii);
   myKernel.addParam(f_rs);
   Vector3< real_t > blockStart = blockIt.getAABB().minCorner();
   myKernel.addParam(double3{ blockStart[0], blockStart[1], blockStart[2] });
   myKernel.addParam(blockIt.getAABB().xSize() / real_t(nOverlappingParticlesField->xSize()));
   myKernel.addParam(numParticlesSubBlocks);
   myKernel.addParam(particleIDsSubBlocks);
   myKernel.addParam(uint3{ uint(subBlocksPerDim[0]), uint(subBlocksPerDim[1]), uint(subBlocksPerDim[2]) });
   myKernel();
}

template< typename ParticleAccessor_T, typename ParticleSelector_T, int Weighting_T >
class SphereFractionMappingGPU
{
 public:
   SphereFractionMappingGPU(const shared_ptr< StructuredBlockStorage >& blockStorage,
                            const shared_ptr< ParticleAccessor_T >& ac,
                            const ParticleSelector_T& mappingParticleSelector,
                            ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA,
                            const Vector3< uint_t > subBlocksPerDim)
      : blockStorage_(blockStorage), ac_(ac), mappingParticleSelector_(mappingParticleSelector),
        particleAndVolumeFractionSoA_(particleAndVolumeFractionSoA), subBlocksPerDim_(subBlocksPerDim)
   {
      static_assert(std::is_base_of< mesa_pd::data::IAccessor, ParticleAccessor_T >::value,
                    "Provide a valid accessor as template");
      for (auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt)
      {
         auto aabb = blockIt->getAABB();
         if (size_t(aabb.xSize()) % subBlocksPerDim_[0] != 0 || size_t(aabb.ySize()) % subBlocksPerDim_[1] != 0 ||
             size_t(aabb.zSize()) % subBlocksPerDim_[2] != 0)
         {
            WALBERLA_ABORT("Number of cells per block (" << aabb << ") is not divisible by subBlocksPerDim ("
                                                         << subBlocksPerDim_ << ").")
         }
      }
   }

   void operator()(IBlock* block)
   {
      size_t numMappedParticles = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { numMappedParticles++; }
      }

      // Allocate unified memory storing the particle information needed for the overlap fraction computations
      const size_t scalarArraySize = numMappedParticles * sizeof(real_t);

      if (particleAndVolumeFractionSoA_.positions != nullptr)
      {
         WALBERLA_GPU_CHECK(gpuFree(particleAndVolumeFractionSoA_.positions));
      }
      WALBERLA_GPU_CHECK(gpuMallocManaged(&(particleAndVolumeFractionSoA_.positions), 3 * scalarArraySize));
      real_t* radii;
      WALBERLA_GPU_CHECK(gpuMallocManaged(&radii, scalarArraySize));
      real_t* f_r; // f_r is described in https://doi.org/10.1108/EC-02-2016-0052
      WALBERLA_GPU_CHECK(gpuMallocManaged(&f_r, scalarArraySize));

      particleAndVolumeFractionSoA_.mappingUIDs.clear();

      // Store particle information inside the unified memory (can be accessed by both CPU and GPU)
      size_t idxMapped = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_))
         {
            // Store uids to make sure that the particles have not changed between the mapping and the PSM sweep
            particleAndVolumeFractionSoA_.mappingUIDs.push_back(ac_->getUid(idx));

            for (size_t d = 0; d < 3; ++d)
            {
               particleAndVolumeFractionSoA_.positions[idxMapped * 3 + d] = ac_->getPosition(idx)[d];
            }
            // If other shapes than spheres are mapped, ignore them here
            if (ac_->getShape(idx)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE)
            {
               const real_t radius = static_cast< mesa_pd::data::Sphere* >(ac_->getShape(idx))->getRadius();
               radii[idxMapped]    = radius;
               real_t Va           = real_t(
                  (1.0 / 12.0 - radius * radius) * atan((0.5 * sqrt(radius * radius - 0.5)) / (0.5 - radius * radius)) +
                  1.0 / 3.0 * sqrt(radius * radius - 0.5) +
                  (radius * radius - 1.0 / 12.0) * atan(0.5 / sqrt(radius * radius - 0.5)) -
                  4.0 / 3.0 * radius * radius * radius * atan(0.25 / (radius * sqrt(radius * radius - 0.5))));
               f_r[idxMapped] = Va - radius + real_t(0.5);
            }
            idxMapped++;
         }
      }

      // Update fraction mapping
      // Split the block into sub-blocks and sort the particle indices into each overlapping sub-block. This way, in
      // the particle mapping, each gpu thread only has to check the potentially overlapping particles.
      auto blockAABB            = block->getAABB();
      const size_t numSubBlocks = subBlocksPerDim_[0] * subBlocksPerDim_[1] * subBlocksPerDim_[2];
      std::vector< std::vector< size_t > > subBlocks(numSubBlocks);

      idxMapped = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_))
         {
            if (ac_->getShape(idx)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE)
            {
               auto sphereAABB = mesa_pd::getParticleAABB(idx, *ac_);
               if (blockAABB.intersects(sphereAABB))
               {
                  auto intersectionAABB = blockAABB.getIntersection(sphereAABB);
                  intersectionAABB.translate(-blockAABB.minCorner());
                  mesa_pd::Vec3 blockScaling = mesa_pd::Vec3(real_t(subBlocksPerDim_[0]) / blockAABB.sizes()[0],
                                                             real_t(subBlocksPerDim_[1]) / blockAABB.sizes()[1],
                                                             real_t(subBlocksPerDim_[2]) / blockAABB.sizes()[2]);

                  for (size_t z = size_t(intersectionAABB.zMin() * blockScaling[2]);
                       z < size_t(ceil(intersectionAABB.zMax() * blockScaling[2])); ++z)
                  {
                     for (size_t y = size_t(intersectionAABB.yMin() * blockScaling[1]);
                          y < size_t(ceil(intersectionAABB.yMax() * blockScaling[1])); ++y)
                     {
                        for (size_t x = size_t(intersectionAABB.xMin() * blockScaling[0]);
                             x < size_t(ceil(intersectionAABB.xMax() * blockScaling[0])); ++x)
                        {
                           size_t index = z * subBlocksPerDim_[0] * subBlocksPerDim_[1] + y * subBlocksPerDim_[0] + x;
                           subBlocks[index].push_back(idxMapped);
                        }
                     }
                  }
               }
            }
            idxMapped++;
         }
      }

      size_t maxParticlesPerSubBlock = 0;
      std::for_each(subBlocks.begin(), subBlocks.end(), [&maxParticlesPerSubBlock](std::vector< size_t >& subBlock) {
         maxParticlesPerSubBlock = std::max(maxParticlesPerSubBlock, subBlock.size());
      });

      size_t* numParticlesPerSubBlock;
      WALBERLA_GPU_CHECK(gpuMallocManaged(&numParticlesPerSubBlock, numSubBlocks * sizeof(size_t)));
      size_t* particleIDsSubBlocks;
      WALBERLA_GPU_CHECK(
         gpuMallocManaged(&particleIDsSubBlocks, numSubBlocks * maxParticlesPerSubBlock * sizeof(size_t)));

      // Copy data from std::vector to unified memory
      for (size_t z = 0; z < subBlocksPerDim_[2]; ++z)
      {
         for (size_t y = 0; y < subBlocksPerDim_[1]; ++y)
         {
            for (size_t x = 0; x < subBlocksPerDim_[0]; ++x)
            {
               size_t index = z * subBlocksPerDim_[0] * subBlocksPerDim_[1] + y * subBlocksPerDim_[0] + x;
               numParticlesPerSubBlock[index] = subBlocks[index].size();
               for (size_t k = 0; k < subBlocks[index].size(); k++)
               {
                  particleIDsSubBlocks[index + k * numSubBlocks] = subBlocks[index][k];
               }
            }
         }
      }

      mapParticles(*block, particleAndVolumeFractionSoA_, particleAndVolumeFractionSoA_.positions, radii, f_r,
                   numParticlesPerSubBlock, particleIDsSubBlocks, subBlocksPerDim_);

      WALBERLA_GPU_CHECK(gpuFree(numParticlesPerSubBlock));
      WALBERLA_GPU_CHECK(gpuFree(particleIDsSubBlocks));

      WALBERLA_GPU_CHECK(gpuFree(radii));
      WALBERLA_GPU_CHECK(gpuFree(f_r));
   }

   shared_ptr< StructuredBlockStorage > blockStorage_;
   const shared_ptr< ParticleAccessor_T > ac_;
   const ParticleSelector_T& mappingParticleSelector_;
   ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA_;
   const Vector3< uint_t > subBlocksPerDim_;
};

template< typename ParticleAccessor_T, typename ParticleSelector_T, int Weighting_T >
class BoxFractionMappingGPU
{
 public:
   BoxFractionMappingGPU(const shared_ptr< StructuredBlockStorage >& blockStorage,
                         const shared_ptr< ParticleAccessor_T >& ac, const uint_t boxUid,
                         const Vector3< real_t > boxEdgeLength,
                         ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA,
                         const ParticleSelector_T& mappingParticleSelector)
      : blockStorage_(blockStorage), ac_(ac), boxUid_(boxUid), boxEdgeLength_(boxEdgeLength),
        particleAndVolumeFractionSoA_(particleAndVolumeFractionSoA), mappingParticleSelector_(mappingParticleSelector)
   {
      static_assert(std::is_base_of< mesa_pd::data::IAccessor, ParticleAccessor_T >::value,
                    "Provide a valid accessor as template");
   }

   void operator()(IBlock* block)
   {
      auto nOverlappingParticlesField =
         block->getData< nOverlappingParticlesFieldGPU_T >(particleAndVolumeFractionSoA_.nOverlappingParticlesFieldID);
      auto BsField  = block->getData< BsFieldGPU_T >(particleAndVolumeFractionSoA_.BsFieldID);
      auto idxField = block->getData< idxFieldGPU_T >(particleAndVolumeFractionSoA_.idxFieldID);
      auto BField   = block->getData< BFieldGPU_T >(particleAndVolumeFractionSoA_.BFieldID);

      auto myKernel = walberla::gpu::make_kernel(&(boxMapping< Weighting_T >) );
      myKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< uint_t >::xyz(*nOverlappingParticlesField));
      myKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< real_t >::xyz(*BsField));
      myKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< id_t >::xyz(*idxField));
      myKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< real_t >::xyz(*BField));
      myKernel.addParam(particleAndVolumeFractionSoA_.omega_);
      const Vector3< real_t > boxPosition = ac_->getPosition(ac_->uidToIdx(boxUid_));
      myKernel.addParam(double3{ boxPosition[0] - boxEdgeLength_[0] / real_t(2),
                                 boxPosition[1] - boxEdgeLength_[1] / real_t(2),
                                 boxPosition[2] - boxEdgeLength_[2] / real_t(2) });
      myKernel.addParam(double3{ boxPosition[0] + boxEdgeLength_[0] / real_t(2),
                                 boxPosition[1] + boxEdgeLength_[1] / real_t(2),
                                 boxPosition[2] + boxEdgeLength_[2] / real_t(2) });
      Vector3< real_t > blockStart = block->getAABB().minCorner();
      myKernel.addParam(double3{ blockStart[0], blockStart[1], blockStart[2] });
      myKernel.addParam(block->getAABB().xSize() / real_t(nOverlappingParticlesField->xSize()));

      // Determine the index of the box among the mapped particles
      size_t idxMapped = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_))
         {
            if (ac_->getUid(idx) == boxUid_) { break; }
            idxMapped++;
         }
      }
      myKernel.addParam(idxMapped);
      myKernel();
   }

   shared_ptr< StructuredBlockStorage > blockStorage_;
   const shared_ptr< ParticleAccessor_T > ac_;
   const uint_t boxUid_;
   const Vector3< real_t > boxEdgeLength_;
   ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA_;
   const ParticleSelector_T& mappingParticleSelector_;
};

} // namespace gpu
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla