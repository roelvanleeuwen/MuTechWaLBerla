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
//! \file PSMKernel.cu
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//! \brief Modification of pe_coupling/partially_saturated_cells_method/PSMSweep.h
//
//======================================================================================================================

#include "lbm_mesapd_coupling/DataTypesGPU.h"

#include "PSMKernel.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace cuda
{

__device__ void getEquilibriumDistribution(real_t* equilibrium, double3& velocity, const real_t rho) {}

template< int StencilSize >
__global__ void PSMKernel(walberla::cuda::FieldAccessor< uint_t > nOverlappingParticles,
                          walberla::cuda::FieldAccessor< real_t > BsField,
                          walberla::cuda::FieldAccessor< id_t > uidsField,
                          walberla::cuda::FieldAccessor< real_t > BField, walberla::cuda::FieldAccessor< real_t > pdfs,
                          real_t /*omega*/, double3* /*hydrodynamicForces*/, double3* /*linearVelocities*/,
                          double3* /*angularVelocities*/, double3* /*positions*/, real_t* /*w*/)
{
   nOverlappingParticles.set(blockIdx, threadIdx);
   BsField.set(blockIdx, threadIdx);
   uidsField.set(blockIdx, threadIdx);
   BField.set(blockIdx, threadIdx);
   pdfs.set(blockIdx, threadIdx);
}

// TODO: find better solution for template kernels
auto instance_with_stencil_19 = PSMKernel< 19 >;

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
