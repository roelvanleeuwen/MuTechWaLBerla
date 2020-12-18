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
class AnisotropicVDWContact
{
public:
   template<typename Accessor>
   real_t operator()(const size_t p_idx1,
                     const size_t p_idx2,
                     Accessor &ac) const;

   static constexpr real_t eps_ = 0.03733;
   static constexpr real_t A_ = 0.0223;
   static constexpr real_t B_ = 1.31;
   static constexpr real_t alf_ = 9.5;
   static constexpr real_t bet_ = 4.0;
   static constexpr real_t Cg_ = 90;
   static constexpr real_t del_ = -7.5;

   static constexpr real_t C1_ = 0.35819;
   static constexpr real_t C2_ = 0.03263;
   static constexpr real_t C3_ = -0.00138;
   static constexpr real_t C4_ = -0.00017;
   static constexpr real_t C5_ = 0.00024;
   static constexpr real_t R_ = R_CNT;
   static constexpr real_t r_cut_ = CutoffFactor * R_;
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
real_t AnisotropicVDWContact::operator()(const size_t p_idx1,
                                         const size_t p_idx2,
                                         Accessor &ac) const
{
   //===Adaptation of PFC5 vdW contact model implementation====

   // Getting the orientations of segments
   Vec3 b1 = ac.getRotation(p_idx1) * Vec3(1.0, 0.0, 0.0);
   Vec3 b2 = ac.getRotation(p_idx2) * Vec3(1.0, 0.0, 0.0);


   // Distance between segments

   Vec3 n = ac.getPosition(p_idx2) - ac.getPosition(p_idx1); ///< contact normal
   double L = n.length();
   n *= (1_r/L);


   //WALBERLA_LOG_DEVEL( "Normal: n = " << n );
   //WALBERLA_LOG_DEVEL( "Orientation of seg 2: b2 = " << b2 );
   //WALBERLA_LOG_DEVEL( "Orientation of seg 1: b1 = " << b1 );
   //WALBERLA_LOG_DEVEL( "Length of rad vect: L = " << L );


   constexpr real_t TOL = 10e-8;
   //---------------------
   // NORMALS CALCULATION
   //---------------------
   // names convention:
   // c1 - contact 1-2 normal
   // b1 - ball 1 axial direction
   // b2 - ball 2 axial direction
   // b3 - neytral direction
   // g - alighning torque direction
   // d - neytral plane normal direction
   // s - shear force direction

   // angle gamma - angle between two axial directions
   double cos_gamma = b1 * b2;



   // if the angle between two axal directions is blunt, then inverce b2
   if (cos_gamma < 0)
   {
      b2 = -b2;
      cos_gamma = -cos_gamma;
   }
   // check that cosine belongs [-1,1]
   cos_gamma = std::min(1.0, cos_gamma);
   cos_gamma = std::max(-1.0, cos_gamma);
   if (L < 20 && L > 16)
   {
      const double gamma = acos(cos_gamma);
      if (gamma < DEGtoRAD(10) || gamma > DEGtoRAD(170)) c->setValue6(1);
   }
   //WALBERLA_LOG_DEVEL( "cos_gamma: = " << cos_gamma );


   // calculate functions of double argument
   double sin_gamma = std::sqrt(1.0 - cos_gamma * cos_gamma);
   double cos_2gamma = cos_gamma * cos_gamma - sin_gamma * sin_gamma;
   double sin_2gamma = 2.0 * sin_gamma * cos_gamma;

   //WALBERLA_LOG_DEVEL( "sin_gamma: = " << sin_gamma );
   //WALBERLA_LOG_DEVEL( "cos_2gamma: = " << cos_2gamma );
   //WALBERLA_LOG_DEVEL( "sin_2gamma: = " << sin_2gamma );


   // g - direction of the aligning torques - b1 X b2
   Vec3 g(0.0, 0.0, 0.0);
   if (sin_gamma > TOL)
   {
      g = b1 % b2;
      g = g * (1.0 / g.length());
   }
   //WALBERLA_LOG_DEVEL( "Aligning moment direction: g = " << g );

   // b3 - vector defining the neutral plane ( plane of shear forces )
   Vec3 b3 = b1 + b2;
   b3 = b3 * (1.0 / b3.length());
   //WALBERLA_LOG_DEVEL( "Neutral plane defined by b3 = " << b3 );

   // angle theta - angle between b3 and c1
   double cos_theta = b3 * n;
   // check that cosine belongs [-1,1]
   cos_theta = std::min(1.0, cos_theta);
   cos_theta = std::max(-1.0, cos_theta);
   //WALBERLA_LOG_DEVEL( "cos_theta: = " << cos_theta );

   // calculation of shear force direction
   Vec3 s(0.0, 0.0, 0.0);
   Vec3 d(0.0, 0.0, 0.0);

   if ((cos_theta > -1.0 + TOL) || (cos_theta < 1.0 - TOL))
      d = n % b3;
   s = n % d;
   s = s * (1.0 / s.length());

   //WALBERLA_LOG_DEVEL( "Shear force direction: = " << s );
   //--------------------------------
   // NORMALS CALCULATION - END
   //--------------------------------

   // Fittable constants
   constexpr size_t M = 5;
   double C[M];

   // Fast calculation of trigonometric functions ( Chebyshev formulas )
   double coss[M], sinn[M];
   double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

   coss[0] = cos_theta * cos_theta - sin_theta * sin_theta;
   sinn[0] = 2.0 * sin_theta * cos_theta;

   for (int i = 0; i < M - 1; i++)
   {
      coss[i + 1] = coss[i] * coss[0] - sinn[i] * sinn[0];
      sinn[i + 1] = sinn[i] * coss[0] + sinn[0] * coss[i];
   }

   //WALBERLA_LOG_DEVEL( "coss: = " << coss[0] <<" "<< coss[1] <<" "<< coss[2] <<" "<< coss[3] <<" "<< coss[4]);
   //WALBERLA_LOG_DEVEL( "sinn: = " << sinn[0] <<" "<< sinn[1] <<" "<< sinn[2] <<" "<< sinn[3] <<" "<< sinn[4]);


   // Set of fitted constants, that determine the level of "smoothness" of vdW potential -
   // magnitube of shear forces between sliding CNTs
   C[0] = C1_;
   C[1] = C2_;
   C[2] = C3_;
   C[3] = C4_;
   C[4] = C5_;

   //WALBERLA_LOG_DEVEL( "C: = " << C[0] <<" "<< C[1] <<" "<< C[2] <<" "<< C[3] <<" "<< C[4]);

   // Cutoff for theta adjustment
   double W_th = 1;
   double W_th_L = 0;
   double W_th_LL = 0;

   // Adjustment w/r to O
   double TH = 1;
   double TH_L = 0;
   double TH_LL = 0;
   double TH_O = 0;
   double TH_OO = 0;

   int sign = 1;
   for (int i = 0; i < M; i++)
   {
      TH = TH + W_th * C[i] * (sign + coss[i]);
      TH_L = TH_L + W_th_L * C[i] * (sign + coss[i]);
      TH_LL = TH_LL + W_th_LL * C[i] * (sign + coss[i]);
      TH_O = TH_O - W_th * 2 * (i + 1) * C[i] * (sinn[i]);
      TH_OO = TH_OO - W_th * 4 * (i + 1) * (i + 1) * C[i] * (coss[i]);
      sign *= -1;
   }

   //WALBERLA_LOG_DEVEL( "TH: = " << TH );
   //WALBERLA_LOG_DEVEL( "TH_L: = " << TH_L );
   //WALBERLA_LOG_DEVEL( "TH_LL: = " << TH_LL );
   //WALBERLA_LOG_DEVEL( "TH_O: = " << TH_O );
   //WALBERLA_LOG_DEVEL( "TH_OO: = " << TH_OO );


   //------------------------------------------------------------------
   // THIS BLOCK IMPLEMENTS IF THE DISTANCE L IS WITHIN WORKING RANGE
   //------------------------------------------------------------------
   if ((L < 2 * r_cut_) && (L > 2 * R_ * 1.2 * TH))
   {
      //-----Constants that appear in the potential--------------------------
      // This set of constants is described in our paper.
      //---------------------------------------------------------------------

      //-----Function D and its derivatives-----------------------
      double D, D_L, D_LL, D_O, D_OO;
      D = L / (R_ * TH) - 2;
      D_L = 1 / (R_ * TH) - (L * TH_L) / (R_ * TH * TH);
      D_O = -(L * TH_O) / (R_ * TH * TH);
      D_LL = -(TH_L) / (R_ * TH * TH);
      D_LL = D_LL - ((R_ * TH * TH) * (TH_L + L * TH_LL)) / (R_ * R_ * TH * TH * TH * TH);
      D_LL = D_LL + (2 * R_ * L * TH * TH_L * TH_L) / (R_ * R_ * TH * TH * TH * TH);
      D_OO = -(R_ * L * TH_OO * TH * TH) / (R_ * R_ * TH * TH * TH * TH);
      D_OO = D_OO + (2 * R_ * L * TH * TH_O * TH_O) / (R_ * R_ * TH * TH * TH * TH);
      //-----------------------------------------------------------

      //WALBERLA_LOG_DEVEL( "D: = " << D );
      //WALBERLA_LOG_DEVEL( "D_L: = " << D_L );
      //WALBERLA_LOG_DEVEL( "D_LL: = " << D_LL );
      //WALBERLA_LOG_DEVEL( "D_O: = " << D_O );
      //WALBERLA_LOG_DEVEL( "D_OO: = " << D_OO );


      //----Function Vc and its derivatives---------------------------------------
      double Vc, Vc_L, Vc_LL, Vc_O, Vc_OO;
      const double DpowAlpha1 = std::pow(D, -(alf_ + 1));
      const double DpowAlpha2 = std::pow(D, -(alf_ + 2));
      const double DpowBeta1 = std::pow(D, -(bet_ + 1));
      const double DpowBeta2 = std::pow(D, -(bet_ + 2));
      Vc = 4 * eps_ * (A_ * std::pow(D, -alf_) - B_ * std::pow(D, -bet_));
      Vc_L = 4 * eps_ * (-alf_ * A_ * DpowAlpha1 + bet_ * B_ * DpowBeta1) * D_L;
      Vc_O = 4 * eps_ * (-alf_ * A_ * DpowAlpha1 + bet_ * B_ * DpowBeta1) * D_O;
      Vc_LL = 4 * eps_ * (alf_ * (alf_ + 1) * A_ * DpowAlpha2 - bet_ * (bet_ + 1) * B_ * DpowBeta2) * D_L;
      Vc_LL = Vc_LL + 4 * eps_ * (-alf_ * A_ * DpowAlpha1 + bet_ * B_ * DpowBeta1) * D_LL;
      Vc_OO = 4 * eps_ * (alf_ * (alf_ + 1) * A_ * DpowAlpha2 - bet_ * (bet_ + 1) * B_ * DpowBeta2) * D_O;
      Vc_OO = Vc_OO + 4 * eps_ * (-alf_ * A_ * DpowAlpha1 + bet_ * B_ * DpowBeta1) * D_OO;
      //--------------------------------------------------------------------------

      //WALBERLA_LOG_DEVEL( "VC = " << Vc );
      //WALBERLA_LOG_DEVEL( "VC_L = " << Vc_L );
      //WALBERLA_LOG_DEVEL( "VC_LL = " << Vc_LL );

      //WALBERLA_LOG_DEVEL( "VC_O = " << Vc_O );
      //WALBERLA_LOG_DEVEL( "VC_OO = " << Vc_OO );


      // Cutoff for u adjustment
      double W_u, W_u_L, W_u_LL;

      // Cubic cutoff function 3T->4T (hardcoded since we do not need to mess w these parameters)
      constexpr double Q1_ = -80.0;
      constexpr double Q2_ = 288.0;
      constexpr double Q3_ = -336.0;
      constexpr double Q4_ = 128.0;
      const double rcut2inv = 1 / (2.0 * r_cut_);
      double nd = L * rcut2inv;
      if ((nd > 0.75) && (nd < 1.0))
      {
         W_u = Q1_ + Q2_ * nd + Q3_ * nd * nd + Q4_ * nd * nd * nd;
         W_u_L = (Q2_ + 2.0 * Q3_ * nd + 3.0 * Q4_ * nd * nd) * rcut2inv;
         W_u_LL = (2.0 * Q3_ + 6.0 * Q4_ * nd) * rcut2inv * rcut2inv;
      } else
      {
         W_u = 1.0;
         W_u_L = 0;
         W_u_LL = 0;
      }
      //--------------------------------------------------------------------------

      //WALBERLA_LOG_DEVEL( "W_u = " << W_u );
      //WALBERLA_LOG_DEVEL( "W_u_L = " << W_u_L );
      //WALBERLA_LOG_DEVEL( "W_u_LL = " << W_u_LL );


      // Cutoff for gamma adjustment

      double W_ga, W_ga_L, W_ga_LL;
      if (L / R_ > 2.75)
      {
         W_ga = Cg_ * std::pow((L / R_), del_);
         W_ga_L = ((del_ * Cg_) / R_) * std::pow((L / R_), del_ - 1);
         W_ga_LL = ((del_ * (del_ - 1) * Cg_) / (R_ * R_)) * std::pow((L / R_), del_ - 2);
      } else
      {
         W_ga = Cg_ * std::pow((2.75), del_);
         W_ga_L = 0;
         W_ga_LL = 0;
      }

      //WALBERLA_LOG_DEVEL( "W_ga = " << W_ga );
      //WALBERLA_LOG_DEVEL( "W_ga_L = " << W_ga_L );
      //WALBERLA_LOG_DEVEL( "W_ga_LL = " << W_ga_LL );


      double GA, GA_L, GA_LL, GA_G, GA_GG;
      if (fabs(sin_gamma) > TOL)
      {
         GA = 1. + W_ga * (1 - cos_2gamma);
         GA_L = W_ga_L * (1 - cos_2gamma);
         GA_LL = W_ga_LL * (1 - cos_2gamma);
         GA_G = 2 * W_ga * sin_2gamma;
         GA_GG = 4 * W_ga * cos_2gamma;
      } else
      {
         GA = 1.;
         GA_L = 0;
         GA_LL = 0;
         GA_G = 0;
         GA_GG = 0;
      }

      //----Forces and torque-----------------------
      double FL, FO, MG;

      FL = -GA_L * W_u * Vc - GA * W_u_L * Vc - GA * W_u * Vc_L;
      FO = -(1 / L) * GA * W_u * Vc_O;
      MG = -GA_G * W_u * Vc;

      //WALBERLA_LOG_DEVEL( "FL = " << FL );
      //WALBERLA_LOG_DEVEL( "FO = " << FO );
      //WALBERLA_LOG_DEVEL( "MG = " << MG );


      Vec3 force = FL * n + FO * s;
      Vec3 moment = MG * g;


      //WALBERLA_LOG_DEVEL( "Contact force: = " << force );
      //WALBERLA_LOG_DEVEL( "Contact moment: = " << moment );

      addForceAtomic(p_idx1, ac, -force);
      addForceAtomic(p_idx2, ac,  force);

      addTorqueAtomic(p_idx1, ac, -moment);
      addTorqueAtomic(p_idx2, ac,  moment);

      // Potential energy
      real_t U = GA * W_u * Vc;
      c->setValue0(U); // vdW adhesion energy
      // WALBERLA_LOG_DEVEL( "U_vdw = " << U );

      // Viscous damping


      // Viscous addition
      constexpr real_t dampFactor = 1052.0; // Calibrated in accordance with 2014 JAM
      const real_t damp = beta_ * dampFactor;

      Vec3 relVelocity = ac.getLinearVel(p_idx1) - ac.getLinearVel(p_idx2);
      Vec3 visc_force = -damp * relVelocity;
      //WALBERLA_LOG_DEVEL( "Visc force: = " << visc_force );
      addForceAtomic(p_idx1, ac,  visc_force);
      addForceAtomic(p_idx2, ac, -visc_force);
   } else if (L <= 2 * R_ * 1.2 * TH)
   { // Small distance
      //WALBERLA_LOG_DEVEL( "Small distance");
      real_t F = -1;
      addForceAtomic(p_idx1, ac,  F * n);
      addForceAtomic(p_idx2, ac, -F * n);
   }
}

} //namespace VBondModel
} //namespace kernel
} //namespace mesa_pd
} //namespace walberla