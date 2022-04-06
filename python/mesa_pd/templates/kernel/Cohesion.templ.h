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
//! \file
//! \author Lukas Werner <lks.werner@fau.de>
//
//======================================================================================================================

//======================================================================================================================
//
//  THIS FILE IS GENERATED - PLEASE CHANGE THE TEMPLATE !!!
//
//======================================================================================================================

#pragma once

#include <core/logging/Logging.h>
#include <functional>
#include <math.h>
#include <mesa_pd/common/ParticleFunctions.h>
#include <mesa_pd/data/DataTypes.h>
#include <mesa_pd/data/IAccessor.h>
#include <mesa_pd/collision_detection/GeneralContactDetection.h>
#include <mesa_pd/data/shape/Sphere.h>

namespace walberla
{
namespace mesa_pd
{
namespace kernel
{
/**
 * Cohesion Initialization Kernel
 *
 * \code
   {%- for prop in interface %}
   {%- if 'g' in prop.access %}
 * const {{prop.type}}& get{{prop.name | capFirst}}(const size_t p_idx) const;
   {%- endif %}
   {%- if 's' in prop.access %}
 * void set{{prop.name | capFirst}}(const size_t p_idx, const {{prop.type}}& v);
   {%- endif %}
   {%- if 'r' in prop.access %}
 * {{prop.type}}& get{{prop.name | capFirst}}Ref(const size_t p_idx);
   {%- endif %}
 *
   {%- endfor %}
 * \endcode
 * \ingroup mesa_pd_kernel
 */

class Cohesion
{
 public:
   Cohesion(const uint_t numParticleTypes);
   Cohesion(const Cohesion& other) = default;
   Cohesion(Cohesion&& other)      = default;
   Cohesion& operator=(const Cohesion& other) = default;
   Cohesion& operator=(Cohesion&& other) = default;

   template <typename Accessor>
   bool operator()(const size_t p_idx1,
                   const size_t p_idx2,
                   Accessor& ac,
                   const bool contactExists,
                   Vec3 contactNormal,
                   real_t penetrationDepth,
                   real_t dt);

   {% for param in parameters %}
   /// assumes this parameter is symmetric
   void set{{param | capFirst}}(const size_t type1, const size_t type2, const real_t& val);
   {%- endfor %}

   {% for param in parameters %}
   real_t get{{param | capFirst}}(const size_t type1, const size_t type2) const;
   {%- endfor %}
 private:
   uint_t numParticleTypes_;
   {% for param in parameters %}
   std::vector<real_t> {{param}}_ {};
   {%- endfor %}
};

Cohesion::Cohesion(const uint_t numParticleTypes)
{
   numParticleTypes_ = numParticleTypes;
   {% for param in parameters %}
   {{param}}_.resize(numParticleTypes * numParticleTypes, real_t(0));
   {%- endfor %}
}

{% for param in parameters %}
inline void Cohesion::set{{param | capFirst}}(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
{{param}}_[numParticleTypes_*type1 + type2] = val;
{{param}}_[numParticleTypes_*type2 + type1] = val;
}
{%- endfor %}

{% for param in parameters %}
inline real_t Cohesion::get{{param | capFirst}}(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( {{param}}_[numParticleTypes_*type1 + type2],
                             {{param}}_[numParticleTypes_*type2 + type1],
                             "parameter matrix for {{param}} not symmetric!");
return {{param}}_[numParticleTypes_*type1 + type2];
}
{%- endfor %}


template <typename Accessor>
inline bool Cohesion::operator()(const size_t p_idx1,
                                 const size_t p_idx2,
                                 Accessor& ac,
                                 const bool contactExists,
                                 Vec3 contactNormal,
                                 real_t penetrationDepth,
                                 real_t dt) {
   //WALBERLA_LOG_INFO("Checking cohesion between " << p_idx1 << " and " << p_idx2 << ".");

   auto uid_p1 = ac.getUid(p_idx1);
   auto uid_p2 = ac.getUid(p_idx2);


   // existing contact history of particle 1 -> particle 2
   const auto& och1 = ac.getOldContactHistoryRef(p_idx1)[uid_p2];
   // existing contact history of particle 2 -> particle 1
   const auto& och2 = ac.getOldContactHistoryRef(p_idx2)[uid_p1];


}

} // namespace kernel
} // namespace mesa_pd
} // namespace walberla
