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
//! \file HydrodynamicForcesSweepGPU.h
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

#include "PSMKernel.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace cuda
{

template< typename LatticeModel_T, typename ParticleAccessor_T, typename Sweep_T, typename ParticleSelector_T,
          int Weighting_T >
class PSMWrapperSweepCUDA
{
 public:
   PSMWrapperSweepCUDA(const shared_ptr< StructuredBlockStorage >& bs, const shared_ptr< ParticleAccessor_T >& ac,
                       const ParticleSelector_T& mappingParticleSelector, const Sweep_T& sweep, BlockDataID& pdfFieldID,
                       const ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA)
      : bs_(bs), sweep_(sweep), ac_(ac), mappingParticleSelector_(mappingParticleSelector), pdfFieldID_(pdfFieldID),
        particleAndVolumeFractionSoA_(particleAndVolumeFractionSoA)
   {}
   void operator()(IBlock* block)
   {
      const real_t dxCurrentLevel      = bs_->dx(bs_->getLevel(*block));
      const real_t lengthScalingFactor = dxCurrentLevel;
      const real_t forceScalingFactor  = lengthScalingFactor * lengthScalingFactor;

      size_t numMappedParticles = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { numMappedParticles++; }
      }
      size_t arraySizes = numMappedParticles * sizeof(double3);

      // Allocate unified memory for the reduction of the particle forces and torques on the GPU
      double3* hydrodynamicForces;
      cudaMallocManaged(&hydrodynamicForces, arraySizes);
      cudaMemset(hydrodynamicForces, 0, arraySizes);
      double3* hydrodynamicTorques;
      cudaMallocManaged(&hydrodynamicTorques, arraySizes);
      cudaMemset(hydrodynamicTorques, 0, arraySizes);

      // Allocate unified memory for the particle information needed for the velocity of the solid collision operator
      double3* linearVelocities;
      cudaMallocManaged(&linearVelocities, arraySizes);
      cudaMemset(linearVelocities, 0, arraySizes);
      double3* angularVelocities;
      cudaMallocManaged(&angularVelocities, arraySizes);
      cudaMemset(angularVelocities, 0, arraySizes);
      double3* positions;
      cudaMallocManaged(&positions, arraySizes);
      cudaMemset(positions, 0, arraySizes);

      // Store particle information inside unified memory
      for (size_t idx = 0; idx < ac_->size();)
      {
         if (mappingParticleSelector_(idx, *ac_))
         {
            linearVelocities[idx]  = { ac_->getLinearVelocity(idx)[0], ac_->getLinearVelocity(idx)[1],
                                       ac_->getLinearVelocity(idx)[2] };
            angularVelocities[idx] = { ac_->getAngularVelocity(idx)[0], ac_->getAngularVelocity(idx)[1],
                                       ac_->getAngularVelocity(idx)[2] };
            positions[idx]         = { ac_->getPosition(idx)[0], ac_->getPosition(idx)[1], ac_->getPosition(idx)[2] };
            ++idx;
         }
      }

      auto nOverlappingParticlesField =
         block->getData< nOverlappingParticlesFieldGPU_T >(particleAndVolumeFractionSoA_.nOverlappingParticlesFieldID);
      auto uidsField = block->getData< uidsFieldGPU_T >(particleAndVolumeFractionSoA_.uidsFieldID);
      auto particleVelocitiesField =
         block->getData< particleVelocitiesFieldGPU_T >(particleAndVolumeFractionSoA_.particleVelocitiesFieldID);
      auto particleForcesField =
         block->getData< particleForcesGPU_T >(particleAndVolumeFractionSoA_.particleForcesFieldID);

      auto velocitiesKernel = walberla::cuda::make_kernel(&(SetParticleVelocities< LatticeModel_T::Stencil::Size >) );
      velocitiesKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< uint_t >::xyz(*nOverlappingParticlesField));
      velocitiesKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*particleVelocitiesField));
      velocitiesKernel.addParam(linearVelocities);
      velocitiesKernel.addParam(angularVelocities);
      velocitiesKernel.addParam(positions);
      Vector3< real_t > blockStart = block->getAABB().minCorner();
      // TODO: why does this work? Why is the data on the GPU?
      velocitiesKernel.addParam(double3{ blockStart[0], blockStart[1], blockStart[2] });
      velocitiesKernel.addParam(block->getAABB().xSize() / real_t(nOverlappingParticlesField->xSize()));
      velocitiesKernel();

      sweep_(block);

      auto forcesKernel = walberla::cuda::make_kernel(&(ReduceParticleForces< LatticeModel_T::Stencil::Size >) );
      forcesKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< uint_t >::xyz(*nOverlappingParticlesField));
      forcesKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< id_t >::xyz(*uidsField));
      forcesKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*particleForcesField));
      forcesKernel.addParam(hydrodynamicForces);
      forcesKernel.addParam(hydrodynamicTorques);
      forcesKernel.addParam(positions);
      //  TODO: why does this work? Why is the data on the GPU?
      forcesKernel.addParam(double3{ blockStart[0], blockStart[1], blockStart[2] });
      forcesKernel.addParam(block->getAABB().xSize() / real_t(nOverlappingParticlesField->xSize()));
      forcesKernel.addParam(forceScalingFactor);
      forcesKernel();

      // Copy forces and torques of particles from GPU to CPU
      for (size_t idx = 0; idx < ac_->size();)
      {
         if (mappingParticleSelector_(idx, *ac_))
         {
            linearVelocities[idx]  = { ac_->getLinearVelocity(idx)[0], ac_->getLinearVelocity(idx)[1],
                                       ac_->getLinearVelocity(idx)[2] };
            angularVelocities[idx] = { ac_->getAngularVelocity(idx)[0], ac_->getAngularVelocity(idx)[1],
                                       ac_->getAngularVelocity(idx)[2] };
            positions[idx]         = { ac_->getPosition(idx)[0], ac_->getPosition(idx)[1], ac_->getPosition(idx)[2] };
            ac_->getHydrodynamicForceRef(idx)[0] += real_t(hydrodynamicForces[idx].x);
            ac_->getHydrodynamicForceRef(idx)[1] += real_t(hydrodynamicForces[idx].y);
            ac_->getHydrodynamicForceRef(idx)[2] += real_t(hydrodynamicForces[idx].z);

            ac_->getHydrodynamicTorqueRef(idx)[0] += real_t(hydrodynamicTorques[idx].x);
            ac_->getHydrodynamicTorqueRef(idx)[1] += real_t(hydrodynamicTorques[idx].y);
            ac_->getHydrodynamicTorqueRef(idx)[2] += real_t(hydrodynamicTorques[idx].z);
            ++idx;
         }
      }

      cudaFree(hydrodynamicForces);
      cudaFree(hydrodynamicTorques);
      cudaFree(linearVelocities);
      cudaFree(angularVelocities);
      cudaFree(positions);
   }

 private:
   shared_ptr< StructuredBlockStorage > bs_;
   Sweep_T sweep_;
   const shared_ptr< ParticleAccessor_T > ac_;
   ParticleSelector_T mappingParticleSelector_;
   BlockDataID pdfFieldID_;
   ParticleAndVolumeFractionSoA_T< Weighting_T > particleAndVolumeFractionSoA_;
};

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla