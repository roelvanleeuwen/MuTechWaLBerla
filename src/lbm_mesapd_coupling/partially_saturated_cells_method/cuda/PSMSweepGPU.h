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
//! \file PSMSweepGPU.h
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

template< typename GPUField_T, int Weighting_T >
class PSMSweepCUDA : public walberla::cuda::GPUSweepBase< GPUField_T >
{
 public:
   PSMSweepCUDA(BlockDataID& pdfFieldID,
                const ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA)
      : pdfFieldID_(pdfFieldID), particleAndVolumeFractionSoA_(particleAndVolumeFractionSoA)
   {}
   void operator()(IBlock* block)
   {
      auto pdfField     = block->getData< GPUField_T >(pdfFieldID_);
      auto indicesField = block->getData< indicesFieldGPU_T >(particleAndVolumeFractionSoA_.indicesFieldID);
      auto overlapFractionsField =
         block->getData< overlapFractionsFieldGPU_T >(particleAndVolumeFractionSoA_.overlapFractionsFieldID);
      auto uidsField = block->getData< uidsFieldGPU_T >(particleAndVolumeFractionSoA_.uidsFieldID);

      auto myKernel = walberla::cuda::make_kernel(&PSMKernel);
      myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*pdfField));
      myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< uint_t >::xyz(*indicesField));
      myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*overlapFractionsField));
      myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< id_t >::xyz(*uidsField));
      myKernel();
   }

 private:
   BlockDataID pdfFieldID_;
   ParticleAndVolumeFractionSoA_T< Weighting_T > particleAndVolumeFractionSoA_;
};

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
