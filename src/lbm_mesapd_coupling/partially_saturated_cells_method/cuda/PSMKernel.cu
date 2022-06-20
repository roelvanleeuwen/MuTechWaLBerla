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

// functions to calculate the cell coverage weighting B
template< int Weighting_T >
__device__ void calculateWeighting(real_t* weighting, const real_t& /*epsilon*/, const real_t& /*tau*/)
{
   WALBERLA_STATIC_ASSERT(Weighting_T == 1 || Weighting_T == 2);
}
template<>
__device__ void calculateWeighting< 1 >(real_t* weighting, const real_t& epsilon, const real_t& /*tau*/)
{
   *weighting = epsilon;
}
template<>
__device__ void calculateWeighting< 2 >(real_t* weighting, const real_t& epsilon, const real_t& tau)
{
   *weighting = epsilon * (tau - real_t(0.5)) / ((real_t(1) - epsilon) + (tau - real_t(0.5)));
}

__global__ void addHydrodynamicForcesKernel(
   walberla::cuda::FieldAccessor< uint_t > indicesField, walberla::cuda::FieldAccessor< real_t > overlapFractionsField,
   walberla::cuda::FieldAccessor< id_t > uidsField, walberla::cuda::FieldAccessor< real_t > bnField,
   walberla::cuda::FieldAccessor< real_t > omegaNField, walberla::cuda::FieldAccessor< real_t > pdfs, real_t omega,
   double3* /*hydrodynamicForces*/, double3* /*linearVelocities*/, double3* /*angularVelocities*/,
   double3* /*positions*/, uint_t stencilSize, real_t* /*w*/)
{
   indicesField.set(blockIdx, threadIdx);
   overlapFractionsField.set(blockIdx, threadIdx);
   uidsField.set(blockIdx, threadIdx);
   bnField.set(blockIdx, threadIdx);
   omegaNField.set(blockIdx, threadIdx);
   pdfs.set(blockIdx, threadIdx);

   bnField.get() = 0.0;
   for (uint_t i = 0; i < stencilSize; i++)
   {
      omegaNField.get() = 0.0;
   }
   for (uint_t i = 0; i < indicesField.get(); i++)
   {
      real_t Bs;
      calculateWeighting< 1 >(&Bs, overlapFractionsField.get(i), 1.0 / omega);
      bnField.get() += Bs;
   }
}

__global__ void PSMKernel(walberla::cuda::FieldAccessor< real_t > pdfField,
                          walberla::cuda::FieldAccessor< uint_t > indicesField,
                          walberla::cuda::FieldAccessor< real_t > overlapFractionsField,
                          walberla::cuda::FieldAccessor< id_t > uidsField)
{
   pdfField.set(blockIdx, threadIdx);
   indicesField.set(blockIdx, threadIdx);
   overlapFractionsField.set(blockIdx, threadIdx);
   uidsField.set(blockIdx, threadIdx);
}

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
