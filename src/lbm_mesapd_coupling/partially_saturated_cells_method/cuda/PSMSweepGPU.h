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

template< typename LatticeModel_T, typename ParticleAccessor_T, typename ParticleSelector_T, int Weighting_T >
class PSMSweepCUDA
{
 public:
   PSMSweepCUDA(const shared_ptr< StructuredBlockStorage >& bs, const shared_ptr< ParticleAccessor_T >& ac,
                const ParticleSelector_T& mappingParticleSelector, BlockDataID& pdfFieldID,
                const ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA)
      : bs_(bs), ac_(ac), mappingParticleSelector_(mappingParticleSelector), pdfFieldID_(pdfFieldID),
        particleAndVolumeFractionSoA_(particleAndVolumeFractionSoA)
   {}
   void operator()(IBlock* block)
   {
      size_t numMappedParticles = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { numMappedParticles++; }
      }

      // TODO: add torques
      double3* hydrodynamicForces;
      cudaMallocManaged(&hydrodynamicForces, numMappedParticles * sizeof(hydrodynamicForces));
      cudaMemset(hydrodynamicForces, 0, numMappedParticles * sizeof(hydrodynamicForces));

      double3* linearVelocities;
      cudaMallocManaged(&linearVelocities, numMappedParticles * sizeof(linearVelocities));
      cudaMemset(linearVelocities, 0, numMappedParticles * sizeof(linearVelocities));
      double3* angularVelocities;
      cudaMallocManaged(&angularVelocities, numMappedParticles * sizeof(angularVelocities));
      cudaMemset(angularVelocities, 0, numMappedParticles * sizeof(angularVelocities));
      double3* positions;
      cudaMallocManaged(&positions, numMappedParticles * sizeof(positions));
      cudaMemset(positions, 0, numMappedParticles * sizeof(positions));

      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_))
         {
            linearVelocities[idx]  = { ac_->getLinearVelocity(idx)[0], ac_->getLinearVelocity(idx)[1],
                                       ac_->getLinearVelocity(idx)[2] };
            angularVelocities[idx] = { ac_->getAngularVelocity(idx)[0], ac_->getAngularVelocity(idx)[1],
                                       ac_->getAngularVelocity(idx)[2] };
            positions[idx]         = { ac_->getPosition(idx)[0], ac_->getPosition(idx)[1], ac_->getPosition(idx)[2] };
         }
      }

      uint_t stencilSize = LatticeModel_T::Stencil::Size;
      real_t* w;
      cudaMallocManaged(&w, stencilSize * sizeof(real_t));
      for (uint_t i = 0; i < stencilSize; i++)
      {
         w[i] = LatticeModel_T::w[i];
      }

      auto indicesField = block->getData< indicesFieldGPU_T >(particleAndVolumeFractionSoA_.indicesFieldID);
      auto overlapFractionsField =
         block->getData< overlapFractionsFieldGPU_T >(particleAndVolumeFractionSoA_.overlapFractionsFieldID);
      auto uidsField = block->getData< uidsFieldGPU_T >(particleAndVolumeFractionSoA_.uidsFieldID);
      auto bnField   = block->getData< bnFieldGPU_T >(particleAndVolumeFractionSoA_.bnFieldID);
      auto pdfField  = block->getData< walberla::cuda::GPUField< real_t > >(pdfFieldID_);

      auto myKernel = walberla::cuda::make_kernel(&PSMKernel);
      myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< uint_t >::xyz(*indicesField));
      myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*overlapFractionsField));
      myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< id_t >::xyz(*uidsField));
      myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*bnField));
      myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*pdfField));
      myKernel.addParam(particleAndVolumeFractionSoA_.omega_);
      myKernel.addParam(hydrodynamicForces);
      myKernel.addParam(linearVelocities);
      myKernel.addParam(angularVelocities);
      myKernel.addParam(positions);
      myKernel.addParam(stencilSize);
      myKernel.addParam(w);
      myKernel();

      // TODO: add forces and torques on particles

      cudaFree(hydrodynamicForces);
      cudaFree(linearVelocities);
      cudaFree(angularVelocities);
      cudaFree(positions);
      cudaFree(w);
   }

 private:
   shared_ptr< StructuredBlockStorage > bs_;
   const shared_ptr< ParticleAccessor_T > ac_;
   ParticleSelector_T mappingParticleSelector_;
   BlockDataID pdfFieldID_;
   ParticleAndVolumeFractionSoA_T< Weighting_T > particleAndVolumeFractionSoA_;
};

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla