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
class IsotropicVDWContact
{
public:
   template<typename Accessor>
   real_t operator()(const size_t p_idx1,
                     const size_t p_idx2,
                     Accessor &ac) const;

   static constexpr real_t eps = 0.07124;
   static constexpr real_t A = 0.0223;
   static constexpr real_t B = 1.31;
   static constexpr real_t alpha = 9.5;
   static constexpr real_t beta = 4.0;
   static constexpr real_t Rinv = 1.0 / 6.78;
   static constexpr real_t Dc = 0.4;
};

/**
 *
 * @tparam Accessor
 * @param p_idx1
 * @param p_idx2
 * @param ac
 * @return vdW adhesion energy
 */
template<typename Accessor>
inline
real_t IsotropicVDWContact::operator()(const size_t p_idx1,
                                       const size_t p_idx2,
                                       Accessor &ac) const
{
   Vec3 n = ac.getPosition(p_idx2) - ac.getPosition(p_idx1); ///< contact normal
   real_t L = n.length();
   n *= (1_r/L);

   real_t D = L * Rinv - 2_r;
   real_t F = 0.01; //default value
   real_t U = 0; //default value

   if (D > Dc)
   {
      F = 4 * eps * (-(alpha * A) / pow(D, (alpha + 1_r)) + (beta * B) / pow(D, (beta + 1_r)));
      U = 4 * eps * (A / pow(D, alpha) - B / pow(D, beta));
   }

   Vec3 force = n * F;
   addForceAtomic(p_idx1, ac,  force);
   addForceAtomic(p_idx2, ac, -force);

   return U; // vdW adhesion energy
}

} //namespace VBondModel
} //namespace kernel
} //namespace mesa_pd
} //namespace walberla