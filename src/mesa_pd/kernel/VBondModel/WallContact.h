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
//! \author Igor Ostanin <i.ostanin@skoltech.ru>
//! \author Grigorii Drozdov <drozd013@umn.edu>
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

//======================================================================================================================
//
//  THIS FILE IS GENERATED - PLEASE CHANGE THE TEMPLATE !!!
//
//======================================================================================================================

#pragma once

#include <mesa_pd/common/ParticleFunctions.h>
#include <mesa_pd/data/DataTypes.h>
#include <mesa_pd/data/IAccessor.h>

#include <core/math/Constants.h>
#include <core/logging/Logging.h>

#include <vector>

namespace walberla {
namespace mesa_pd {
namespace kernel {
namespace VBondModel {

/**
 * Advanced DEM kernel
 */
class WallContact
{
public:
   template<typename Accessor>
   void operator()(const size_t p_idx1,
                   const size_t p_idx2,
                   Accessor &ac) const;

   static constexpr real_t r = 6.78; ///< A
   static constexpr real_t eps = 0.254e-3; ///< eV/amu
   static constexpr real_t m = 2648.8; ///< amu
   static constexpr real_t s = 3.6; ///< A
   static constexpr real_t s12 = ((s * s) * (s * s) * (s * s)) * ((s * s) * (s * s) * (s * s));
};

template<typename Accessor>
inline void WallContact::operator()(const size_t p_idx1,
                                    const size_t p_idx2,
                                    Accessor &ac) const
{
   Vec3 pos = ac->getPosition(p_idx1) - ac->getPosition(p_idx2);
   double x = pos[2];

   Vec3 force(0, 0, 0);
   if (x >= 0)
   {
      const auto tmp = x - r;
      const auto pow = ((tmp * tmp) * (tmp * tmp) * (tmp * tmp)) * ((tmp * tmp) * (tmp * tmp) * (tmp * tmp)) * tmp;
      if (x < r + 10)
         force[2] = m * eps * s12 * 12 / pow;
      else if (x < r + 12)
         force[2] = m * eps * s12 * (3 * (12 + r) * (14 + r) - 6 * (13 + r) * x + 3 * x * x) * (20e-12);
   } else
   {
      const auto tmp = -x - r;
      const auto pow = ((tmp * tmp) * (tmp * tmp) * (tmp * tmp)) * ((tmp * tmp) * (tmp * tmp) * (tmp * tmp)) * tmp;
      if (x > -(r + 10))
         force[2] = -m * eps * s12 * 12 / pow;
      else if (x > -(r + 12))
         force[2] = -m * eps * s12 * (3 * (12 + r) * (14 + r) - 6 * (13 + r) * (-x) + 3 * x * x) * (20e-12);
   }

   addForceAtomic( p_idx1, ac,  force );
   addForceAtomic( p_idx2, ac, -force );
}

} //namespace VBondModel
} //namespace kernel
} //namespace mesa_pd
} //namespace walberla