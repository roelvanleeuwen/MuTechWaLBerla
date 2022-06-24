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
//! \file OverlapFraction.h
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//! \brief Functor that provides overlap fraction computations for different MESA-PD shapes (used by SingleCast)
//
//======================================================================================================================

#pragma once

#include "mesa_pd/data/shape/Sphere.h"

#include "ParticleAndVolumeFractionMappingKernel.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace cuda
{

struct OverlapFractionFunctor
{
   /*   template< typename ParticleAccessor_T, typename Shape_T >
      void operator()(const size_t */
   /*particleIdx*/ /*, const Shape_T& */ /*shape*/ /*,
const shared_ptr< ParticleAccessor_T >& */
   /*ac*/ /*, const IBlock& */ /*blockIt*/         /*,
const BlockDataID& */
   /*particleAndVolumeFractionFieldID*/            /*)
{
WALBERLA_ABORT("OverlapFraction not implemented!");
}*/

   template< typename ParticleAccessor_T, typename Shape_T, int Weighting_T >
   void operator()(const size_t /*particleIdx*/, const Shape_T& /*shape*/,
                   const shared_ptr< ParticleAccessor_T >& /*ac*/, const IBlock& /*blockIt*/,
                   const ParticleAndVolumeFractionSoA_T< Weighting_T >& /*particleAndVolumeFractionSoA*/,
                   const real_t /*omega*/)
   {
      WALBERLA_ABORT("OverlapFraction not implemented!");
   }

   /*   template< typename ParticleAccessor_T >
      void operator()(const size_t particleIdx, const mesa_pd::data::Sphere& */
   /*sphere*/ /*,
const shared_ptr< ParticleAccessor_T >& ac, const IBlock& blockIt,
const BlockDataID& particleAndVolumeFractionFieldID)
{
WALBERLA_STATIC_ASSERT((std::is_base_of< mesa_pd::data::IAccessor, ParticleAccessor_T >::value));

Vector3< real_t > particlePosition = ac->getPosition(particleIdx);

auto cudaField = blockIt.getData< walberla::cuda::GPUField< ParticleAndVolumeFractionAoS_T > >(
particleAndVolumeFractionFieldID);

auto myKernel = walberla::cuda::make_kernel(&particleAndVolumeFractionMappingKernelAoS);
myKernel.addFieldIndexingParam(
walberla::cuda::FieldIndexing< ParticleAndVolumeFractionAoS_T >::xyz(*cudaField)); // FieldAccessor
Vector3< real_t > blockStart = blockIt.getAABB().minCorner();
myKernel.addParam(double3{ particlePosition[0], particlePosition[1], particlePosition[2] }); // spherePosition
myKernel.addParam(static_cast< mesa_pd::data::Sphere* >(ac->getShape(particleIdx))->getRadius()); // sphereRadius
myKernel.addParam(double3{ blockStart[0], blockStart[1], blockStart[2] });                        // blockStart
myKernel.addParam(double3{ 1, 1, 1 });                                                            // dx
myKernel.addParam(int3{ 16, 16, 16 });                                                            // nSamples
myKernel.addParam(ac->getUid(particleIdx));
myKernel();
}*/

   template< typename ParticleAccessor_T, int Weighting_T >
   void operator()(const size_t particleIdx, const mesa_pd::data::Sphere& /*sphere*/,
                   const shared_ptr< ParticleAccessor_T >& ac, const IBlock& blockIt,
                   const ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA,
                   const real_t omega)
   {
      WALBERLA_STATIC_ASSERT((std::is_base_of< mesa_pd::data::IAccessor, ParticleAccessor_T >::value));

      Vector3< real_t > particlePosition = ac->getPosition(particleIdx);

      auto nOverlappingParticlesField =
         blockIt.getData< nOverlappingParticlesFieldGPU_T >(particleAndVolumeFractionSoA.nOverlappingParticlesFieldID);
      auto BsField   = blockIt.getData< BsFieldGPU_T >(particleAndVolumeFractionSoA.BsFieldID);
      auto uidsField = blockIt.getData< uidsFieldGPU_T >(particleAndVolumeFractionSoA.uidsFieldID);
      auto BField    = blockIt.getData< BFieldGPU_T >(particleAndVolumeFractionSoA.BFieldID);

      auto myKernel = walberla::cuda::make_kernel(&(particleAndVolumeFractionMappingKernelSoA< Weighting_T >) );
      myKernel.addFieldIndexingParam(
         walberla::cuda::FieldIndexing< uint_t >::xyz(*nOverlappingParticlesField));          // FieldAccessor
      myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*BsField)); // FieldAccessor
      myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< id_t >::xyz(*uidsField)); // FieldAccessor
      myKernel.addFieldIndexingParam(walberla::cuda::FieldIndexing< real_t >::xyz(*BField));  // FieldAccessor
      myKernel.addParam(omega);
      Vector3< real_t > blockStart = blockIt.getAABB().minCorner();
      myKernel.addParam(double3{ particlePosition[0], particlePosition[1], particlePosition[2] }); // spherePosition
      myKernel.addParam(static_cast< mesa_pd::data::Sphere* >(ac->getShape(particleIdx))->getRadius()); // sphereRadius
      myKernel.addParam(double3{ blockStart[0], blockStart[1], blockStart[2] });                        // blockStart
      myKernel.addParam(blockIt.getAABB().xSize() / real_t(nOverlappingParticlesField->xSize()));       // dx
      myKernel.addParam(int3{ 16, 16, 16 }); // nSamples
      myKernel.addParam(ac->getUid(particleIdx));
      myKernel();
   }
};

} // namespace cuda
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
