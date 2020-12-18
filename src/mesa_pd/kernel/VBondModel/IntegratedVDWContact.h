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
class IntegratedVDWContact
{
public:
   template<typename Accessor>
   real_t operator()(const size_t p_idx1,
                     const size_t p_idx2,
                     Accessor &ac) const;

   static constexpr double eps_ = 0.07124;
   static constexpr double A_ = 0.0223;
   static constexpr double B_ = 1.31;
   static constexpr double alf_ = 9.5;
   static constexpr double bet_ = 4.0;
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
real_t IntegratedVDWContact::operator()(const size_t p_idx1,
                                        const size_t p_idx2,
                                        Accessor &ac) const
{
   constexpr real_t K_n = 200.0; // Good for fabrics modeling

   // particle centers
   Vec3 O1 = ac.getPosition(p_idx1);
   Vec3 O2 = ac.getPosition(p_idx2);

   // axial vectors
   Vec3 b1 = ac.getRotation(p_idx1) * Vec3(1.0, 0.0, 0.0);
   Vec3 b2 = ac.getRotation(p_idx2) * Vec3(1.0, 0.0, 0.0);

   // vdW adhesion + linear repulsion potential.
   const real_t r0 = R_CNT * (2 + std::pow((alf_ * A_ / (bet_ * B_)), 1 / (alf_ - bet_)));
   const real_t u0 = 4 * eps_ * (A_ / pow(r0 / R_CNT - 2, (alf_)) - B_ / pow(r0 / R_CNT - 2, (bet_)));

   real_t U = 0, dU = 0; ///< total potential
   Vec3 force12(0), dforce12(0);  // Force 12
   Vec3 force21(0), dforce21(0);  // Force 21
   Vec3 moment12(0), dmoment12(0); // Total torque 12
   Vec3 moment21(0), dmoment21(0); // Total torque 21

   constexpr int Np = 5; // Number of integration points over each axis
   constexpr double Npinv = 1.0 / Np;
   for (int i = 0; i < Np; ++i) // integral dl1
   {
      for (int j = 0; j < Np; ++j) // integral dl2
      {
         // Levers
         Vec3 l1 = (-0.5 * T + (0.5 + i) * T * Npinv) * b1;
         Vec3 l2 = (-0.5 * T + (0.5 + j) * T * Npinv) * b2;

         /// radius vector between dl1 and dl2
         Vec3 R12 = (O2 + l2) - (O1 + l1);

         Vec3 n12 = R12.getNormalized();
         real_t r12 = R12.length();

         if (r12 < r0) // elastic interaction
         {
            dU = u0 + K_n * (r12 - r0) * (r12 - r0) * Npinv * Npinv;
            dforce12 = n12 * K_n * ((r12 - r0)) * Npinv * Npinv;
            dforce21 = -dforce12;
         }
         if (r12 >= r0) // vdW interaction
         {
            const double powAlpha1 = pow(r12 / R_CNT - 2.0, (alf_ + 1.0));
            const double powBeta1 = pow(r12 / R_CNT - 2.0, (bet_ + 1));
            dU = 4.0 * eps_ * (A_ / pow(r12 / R_CNT - 2.0, (alf_)) - B_ / pow(r12 / R_CNT - 2.0, (bet_))) * Npinv *
                 Npinv;
            dforce12 =
                  n12 * 4.0 * eps_ * (-(alf_ * A_) / powAlpha1 + (bet_ * B_) / powBeta1) / R_CNT * Npinv * Npinv;
            dforce21 =
                  -n12 * 4.0 * eps_ * (-(alf_ * A_) / powAlpha1 + (bet_ * B_) / powBeta1) / R_CNT * Npinv * Npinv;
         }

         dmoment12 = l2 % dforce12;
         dmoment21 = l1 % dforce21;

         U += dU;
         force12 += dforce12;
         force21 += dforce21;
         moment12 += dmoment12;
         moment21 += dmoment21;
      }
   }

   addForceAtomic(p_idx1, ac, force12);
   addForceAtomic(p_idx2, ac, force21);
   addTorqueAtomic(p_idx1, ac,  -moment21);
   addTorqueAtomic(p_idx2, ac,  -moment12);

   // Viscous addition
   constexpr real_t dampFactor = 1052.0; // Calibrated in accordance with 2014 JAM
   real_t damp = beta_ * dampFactor;
   Vec3 relVelocity = ac.getLinearVel(p_idx1) - ac.getLinearVel(p_idx2);
   Vec3 visc_force = -damp * relVelocity;
   //WALBERLA_LOG_DEVEL( "Visc force: = " << visc_force );
   addForceAtomic(p_idx1, ac,  visc_force);
   addForceAtomic(p_idx2, ac, -visc_force);

   return U; // vdW adhesion energy
}

} //namespace VBondModel
} //namespace kernel
} //namespace mesa_pd
} //namespace walberla