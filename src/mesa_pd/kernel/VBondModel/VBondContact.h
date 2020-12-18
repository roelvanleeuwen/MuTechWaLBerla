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
class VBondContact
{
public:
   template<typename Accessor>
   real_t operator()(const size_t p_idx1,
                     const size_t p_idx2,
                     Accessor &ac) const;

   static constexpr real_t S = 142.7; // Area of the bond,
   static constexpr real_t E = 1029 * 0.006242; // Conversion from GPa to eV/A^3
   static constexpr real_t G = 459 * 0.006242;
   static constexpr real_t a = 2 * 6.78; // (10,10) CNTS
   static constexpr real_t J = 3480;
   static constexpr real_t Jp = 2 * J;

   // Stiffnesses, equilibrium length etc
   static constexpr real_t B1 = E * S / a; // Need to calibrate
   static constexpr real_t B2 = 12 * E * J / a;
   static constexpr real_t B3 = -2 * E * J / a - G * Jp / (2 * a);
   static constexpr real_t B4 = G * Jp / a;


   // Calibration through atomistic simulation
   //constexpr real_t B1 = 60.1;
   //constexpr real_t B2 = 17100;
   //constexpr real_t B3 = 3610;
   //constexpr real_t B4 = 1107;
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
real_t VBondContact::operator()(const size_t p_idx1,
                                const size_t p_idx2,
                                Accessor &ac) const
{
   srand(1);

   // Particle centres vectors
   Vec3 ri = ac.getPosition(p_idx1);
   Vec3 rj = ac.getPosition(p_idx2);

   // Vectors nik, njk

   // Fix for the issue of segment's undefined sides
   real_t sign = 1;
   if (SID_1 > SID_2) sign = -1;

   if (((period_X) && (abs(SID_2 - SID_1) == max_segsX - 1)) or
       ((period_Y) && (abs(SID_2 - SID_1) == max_segsY - 1)))
      sign = -sign; // special case for periodic fibers

   Vec3 ni1 = ac.getRotation(p_idx1) * Vec3(sign * 1.0, 0.0, 0.0);
   Vec3 ni2 = ac.getRotation(p_idx1) * Vec3(0.0, 1.0, 0.0);
   Vec3 ni3 = ac.getRotation(p_idx1) * Vec3(0.0, 0.0, 1.0);

   Vec3 nj1 = ac.getRotation(p_idx2) * Vec3(-sign * 1.0, 0.0, 0.0);
   Vec3 nj2 = ac.getRotation(p_idx2) * Vec3(0.0, 1.0, 0.0);
   Vec3 nj3 = ac.getRotation(p_idx2) * Vec3(0.0, 0.0, 1.0);

   // Vectors Dij and dij
   Vec3 Dij = rj - ri;
   real_t s = Dij.length();
   Vec3 dij = Dij * (1_r / s);

   real_t C = 1;

   /*
           //srand(SID_1);
           double AA = 0.2 + 0.2 * ((rand() % 1000)/1000); //strain of initiation of stress corrosion
           //double BB = 0.01; // width of stress corrosion band
           //double DD = 1. + AA/BB;
           //double KK = 1 / BB;
           //if (((std::fabs((s-a)/s))>AA)&&((std::fabs((s-a)/s))<=AA+BB)) C = DD - std::fabs((s-a)/s) * KK;
           //if ((std::fabs((s-a)/s))>AA+BB) C = 0;
           if ((std::fabs((s-a)/s))>AA) C = 0;

           // Potential energy
           real_t U_1 = C * 0.5 * B1 * (s - a) * (s - a);
           real_t U_2 = C * 0.5 * B2 * (nj1 - ni1) * dij + B2;
           real_t U_3 = C * B3 * ni1 * nj1 + B3;
           real_t U_4 = C * -0.5 * B4 * (ni2 * nj2 + ni3 * nj3) + B4;
           real_t U = U_1 + U_2 + U_3 + U_4;
   */
   real_t U_1 = 0.5 * B1 * (s - a) * (s - a);
   real_t U_2 = B2 * (0.5 * (nj1 - ni1) * dij - 0.25 * ni1 * nj1 + 0.75);
   real_t U_3 = (0.25 * B2 + B3 + 0.5 * B4) * (ni1 * nj1 + 1);
   real_t U_4 = -0.5 * B4 * (ni1 * nj1 + ni2 * nj2 + ni3 * nj3 - 1);
   real_t U = U_1 + U_2 + U_3 + U_4;

   c->setValue1(U); // Total Strain energy

   c->setValue2(U_1); // Separate strain energy components
   c->setValue3(U_2);
   c->setValue4(U_3);
   c->setValue5(U_4);


   //WALBERLA_LOG_DEVEL("\Delta U = " << U + 3);
   Vec3 rij = dij;

   //Forces and torques
   Vec3 Fij = C * B1 * (s - a) * rij + B2 / (2 * s) * ((nj1 - ni1) - ((nj1 - ni1) * dij) * dij);
   Vec3 Fji = -Fij;
   Vec3 M_TB = C * B3 * (nj1 % ni1) - 0.5 * B4 * (nj2 % ni2 + nj3 % ni3);
   Vec3 Mij = -C * 0.5 * B2 * (dij % ni1) + M_TB;
   Vec3 Mji = C * 0.5 * B2 * (dij % nj1) - M_TB;

   addForceAtomic(p_idx1, ac, Fij);
   addForceAtomic(p_idx2, ac, Fji);
   addTorqueAtomic(p_idx1, ac, Mij);
   addTorqueAtomic(p_idx2, ac, Mji);




   // Viscous damping of rotations and translations

   // Viscous addition
   constexpr real_t dampFactor = 1052.0; // Calibrated in accordance with 2014 JAM
   real_t damp = beta_ * dampFactor;

   Vec3 relVelocity = ac.getLinearVel(p_idx1) - ac.getLinearVel(p_idx2);
   Vec3 visc_force = -damp * relVelocity;

   addForceAtomic(p_idx1, ac, visc_force);
   addForceAtomic(p_idx2, ac, -visc_force);
}

} //namespace VBondModel
} //namespace kernel
} //namespace mesa_pd
} //namespace walberla