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
//! \file PSMWrapperSweepsGPU.h
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "cuda/FieldIndexing.h"
#include "cuda/GPUField.h"
#include "cuda/Kernel.h"
#include "cuda/sweeps/GPUSweepBase.h"

#include "domain_decomposition/StructuredBlockStorage.h"

#include "field/GhostLayerField.h"

#include "lbm/lattice_model/all.h"
#include "lbm/sweeps/StreamPull.h"
#include "lbm/sweeps/SweepBase.h"

#include "lbm_mesapd_coupling/DataTypesGPU.h"
#include "lbm_mesapd_coupling/utility/ParticleFunctions.h"

#include "mesa_pd/common/ParticleFunctions.h"

#include "timeloop/SweepTimeloop.h"

#include <cassert>

#include "PSMKernel.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace cuda
{

// The deviceSyncWrapper can be used so that the timeloop measures the correct device runtime
// TODO: check if the cudaDeviceSynchronize penalty is acceptable
auto deviceSyncWrapper = [](std::function< void(IBlock*) > sweep) {
   return [sweep](IBlock* b) {
      sweep(b);
      cudaDeviceSynchronize();
   };
};

template< typename LatticeModel_T, typename ParticleAccessor_T, typename ParticleSelector_T, int Weighting_T >
class SetParticleVelocitiesSweep
{
 public:
   SetParticleVelocitiesSweep(const shared_ptr< StructuredBlockStorage >& bs,
                              const shared_ptr< ParticleAccessor_T >& ac,
                              const ParticleSelector_T& mappingParticleSelector, BlockDataID& pdfFieldID,
                              const ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA)
      : bs_(bs), ac_(ac), mappingParticleSelector_(mappingParticleSelector), pdfFieldID_(pdfFieldID),
        particleAndVolumeFractionSoA_(particleAndVolumeFractionSoA)
   {}
   void operator()(IBlock* block)
   {
      assert(LatticeModel_T::Stencil::D == 3);
      // Check that uids have not changed since the last mapping to avoid incorrect indices
      std::vector< walberla::id_t > currentUIDs;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { currentUIDs.push_back(ac_->getUid(idx)); }
      }
      assert(particleAndVolumeFractionSoA_.mappingUIDs == currentUIDs);

      size_t numMappedParticles = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { numMappedParticles++; }
      }
      size_t arraySizes = numMappedParticles * sizeof(real_t) * 3;

      // Allocate unified memory for the particle information needed for computing the velocity at a WF point (needed by
      // the solid collision operator)
      real_t* linearVelocities;
      cudaMallocManaged(&linearVelocities, arraySizes);
      cudaMemset(linearVelocities, 0, arraySizes);
      real_t* angularVelocities;
      cudaMallocManaged(&angularVelocities, arraySizes);
      cudaMemset(angularVelocities, 0, arraySizes);
      real_t* positions;
      cudaMallocManaged(&positions, arraySizes);
      cudaMemset(positions, 0, arraySizes);

      // Store particle information inside unified memory to communicate information to the GPU
      size_t idxMapped = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_))
         {
            for (size_t d = 0; d < 3; ++d)
            {
               linearVelocities[idxMapped * 3 + d]  = ac_->getLinearVelocity(idx)[d];
               angularVelocities[idxMapped * 3 + d] = ac_->getAngularVelocity(idx)[d];
               positions[idxMapped * 3 + d]         = ac_->getPosition(idx)[d];
            }
            idxMapped++;
         }
      }

      auto nOverlappingParticlesField =
         block->getData< nOverlappingParticlesFieldGPU_T >(particleAndVolumeFractionSoA_.nOverlappingParticlesFieldID);
      auto idxField = block->getData< idxFieldGPU_T >(particleAndVolumeFractionSoA_.idxFieldID);
      auto particleVelocitiesField =
         block->getData< particleVelocitiesFieldGPU_T >(particleAndVolumeFractionSoA_.particleVelocitiesFieldID);

      // For every cell, set the particle velocities of the overlapping particles at the cell center
      auto velocitiesKernel = walberla::cuda::make_kernel(&(SetParticleVelocities));
      velocitiesKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< uint_t >::xyz(*nOverlappingParticlesField));
      velocitiesKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< id_t >::xyz(*idxField));
      velocitiesKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*particleVelocitiesField));
      velocitiesKernel.addParam(linearVelocities);
      velocitiesKernel.addParam(angularVelocities);
      velocitiesKernel.addParam(positions);
      __device__ double3 blockStart = { block->getAABB().minCorner()[0], block->getAABB().minCorner()[1],
                                        block->getAABB().minCorner()[2] };
      velocitiesKernel.addParam(blockStart);
      velocitiesKernel.addParam(block->getAABB().xSize() / real_t(nOverlappingParticlesField->xSize()));
      velocitiesKernel();

      cudaFree(linearVelocities);
      cudaFree(angularVelocities);
      cudaFree(positions);
   }

 private:
   shared_ptr< StructuredBlockStorage > bs_;
   const shared_ptr< ParticleAccessor_T > ac_;
   const ParticleSelector_T& mappingParticleSelector_;
   BlockDataID pdfFieldID_;
   const ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA_;
};

template< typename LatticeModel_T, typename ParticleAccessor_T, typename ParticleSelector_T, int Weighting_T >
class ReduceParticleForcesSweep
{
 public:
   ReduceParticleForcesSweep(const shared_ptr< StructuredBlockStorage >& bs, const shared_ptr< ParticleAccessor_T >& ac,
                             const ParticleSelector_T& mappingParticleSelector, BlockDataID& pdfFieldID,
                             const ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA)
      : bs_(bs), ac_(ac), mappingParticleSelector_(mappingParticleSelector), pdfFieldID_(pdfFieldID),
        particleAndVolumeFractionSoA_(particleAndVolumeFractionSoA)
   {}
   void operator()(IBlock* block)
   {
      assert(LatticeModel_T::Stencil::D == 3);
      // Check that uids have not changed since the last mapping to avoid incorrect indices
      std::vector< walberla::id_t > currentUIDs;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { currentUIDs.push_back(ac_->getUid(idx)); }
      }
      assert(particleAndVolumeFractionSoA_.mappingUIDs == currentUIDs);

      const real_t dxCurrentLevel      = bs_->dx(bs_->getLevel(*block));
      const real_t lengthScalingFactor = dxCurrentLevel;
      const real_t forceScalingFactor  = lengthScalingFactor * lengthScalingFactor;

      size_t numMappedParticles = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { numMappedParticles++; }
      }
      size_t arraySizes = numMappedParticles * sizeof(real_t) * 3;

      // Allocate unified memory for the reduction of the particle forces and torques on the GPU
      real_t* hydrodynamicForces;
      cudaMallocManaged(&hydrodynamicForces, arraySizes);
      cudaMemset(hydrodynamicForces, 0, arraySizes);
      real_t* hydrodynamicTorques;
      cudaMallocManaged(&hydrodynamicTorques, arraySizes);
      cudaMemset(hydrodynamicTorques, 0, arraySizes);

      // Allocate unified memory for the particle information needed for computing the velocity at a WF point (needed by
      // the solid collision operator)
      real_t* positions;
      cudaMallocManaged(&positions, arraySizes);
      cudaMemset(positions, 0, arraySizes);

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
            idxMapped++;
         }
      }

      auto nOverlappingParticlesField =
         block->getData< nOverlappingParticlesFieldGPU_T >(particleAndVolumeFractionSoA_.nOverlappingParticlesFieldID);
      auto idxField = block->getData< idxFieldGPU_T >(particleAndVolumeFractionSoA_.idxFieldID);
      auto particleForcesField =
         block->getData< particleForcesGPU_T >(particleAndVolumeFractionSoA_.particleForcesFieldID);

      __device__ double3 blockStart = { block->getAABB().minCorner()[0], block->getAABB().minCorner()[1],
                                        block->getAABB().minCorner()[2] };

      // For every cell, reduce the hydrodynamic forces and torques of the overlapping particles
      auto forcesKernel = walberla::cuda::make_kernel(&(ReduceParticleForces));
      forcesKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< uint_t >::xyz(*nOverlappingParticlesField));
      forcesKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< id_t >::xyz(*idxField));
      forcesKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*particleForcesField));
      forcesKernel.addParam(hydrodynamicForces);
      forcesKernel.addParam(hydrodynamicTorques);
      forcesKernel.addParam(positions);
      forcesKernel.addParam(blockStart);
      forcesKernel.addParam(block->getAABB().xSize() / real_t(nOverlappingParticlesField->xSize()));
      forcesKernel.addParam(forceScalingFactor);
      forcesKernel();

      cudaDeviceSynchronize();

      // Copy forces and torques of particles from GPU to CPU
      idxMapped = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_))
         {
            for (size_t d = 0; d < 3; ++d)
            {
               ac_->getHydrodynamicForceRef(idx)[d] += hydrodynamicForces[idxMapped * 3 + d];
               ac_->getHydrodynamicTorqueRef(idx)[d] += hydrodynamicTorques[idxMapped * 3 + d];
            }
            idxMapped++;
         }
      }

      cudaFree(hydrodynamicForces);
      cudaFree(hydrodynamicTorques);
      cudaFree(positions);
   }

 private:
   shared_ptr< StructuredBlockStorage > bs_;
   const shared_ptr< ParticleAccessor_T > ac_;
   const ParticleSelector_T& mappingParticleSelector_;
   BlockDataID pdfFieldID_;
   const ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA_;
};

template< typename Mapping, typename VelocitiesSweep, typename PSMSweep, typename ForcesSweep >
void addPSMSweepsToTimeloop(SweepTimeloop& timeloop, Mapping& particleMapping,
                            VelocitiesSweep& setParticleVelocitiesSweep, PSMSweep& psmSweep,
                            ForcesSweep& reduceParticleForcesSweep, bool synchronize = true)
{
   if (synchronize)
   {
      timeloop.add() << BeforeFunction(particleMapping, "Particle mapping")
                     << Sweep(deviceSyncWrapper(setParticleVelocitiesSweep), "Set particle velocities");
      timeloop.add() << Sweep(deviceSyncWrapper(psmSweep), "PSM sweep");
      timeloop.add() << Sweep(deviceSyncWrapper(reduceParticleForcesSweep), "Reduce particle forces");
   }
   else
   {
      timeloop.add() << BeforeFunction(particleMapping, "Particle mapping")
                     << Sweep(setParticleVelocitiesSweep, "Set particle velocities");
      timeloop.add() << Sweep(psmSweep, "PSM sweep");
      timeloop.add() << Sweep(reduceParticleForcesSweep, "Reduce particle forces");
   };
}

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla