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
//! \file   ParticleSelectors.h
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "mesa_pd/data/shape/Sphere.h"

namespace walberla
{
namespace piping
{

struct SphereSelector
{
   template< typename ParticleAccessor_T >
   bool inline operator()(const size_t particleIdx, const ParticleAccessor_T& ac) const
   {
      return ac.getBaseShape(particleIdx)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE;
   }
};

struct SphereSphereSelector
{
   template< typename ParticleAccessor_T >
   bool inline operator()(const size_t particleIdx0, const size_t particleIdx1, const ParticleAccessor_T& ac) const
   {
      return ac.getBaseShape(particleIdx0)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE &&
             ac.getBaseShape(particleIdx1)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE;
   }
};

struct SphereSelectorExcludeGhost
{
   template< typename ParticleAccessor_T >
   bool inline operator()(const size_t particleIdx, const ParticleAccessor_T& ac) const
   {
      using namespace walberla::mesa_pd::data::particle_flags;
      return (ac.getBaseShape(particleIdx)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE) &&
             !isSet(ac.getFlags(particleIdx), GHOST);
   }
};

class SelectBoxEdgeLength
{
 public:
   using return_type = walberla::mesa_pd::Vec3;
   walberla::mesa_pd::Vec3 operator()(mesa_pd::data::Particle& p) const
   {
      return static_cast< mesa_pd::data::Box* >(p->getBaseShape().get())->getEdgeLength();
   }
   walberla::mesa_pd::Vec3 operator()(mesa_pd::data::Particle&& p) const
   {
      return static_cast< mesa_pd::data::Box* >(p->getBaseShape().get())->getEdgeLength();
   }
   walberla::mesa_pd::Vec3 const operator()(const mesa_pd::data::Particle& p) const
   {
      auto p_tmp = p;
      return static_cast< mesa_pd::data::Box* >(p_tmp->getBaseShape().get())->getEdgeLength();
   }
};

} // namespace piping
} // namespace walberla
