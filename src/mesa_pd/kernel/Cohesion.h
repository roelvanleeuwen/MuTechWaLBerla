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
 * const walberla::id_t& getUid(const size_t p_idx) const;
 *
 * const walberla::mesa_pd::Vec3& getPosition(const size_t p_idx) const;
 *
 * const walberla::mesa_pd::Vec3& getLinearVelocity(const size_t p_idx) const;
 *
 * walberla::mesa_pd::Vec3& getForceRef(const size_t p_idx);
 *
 * const walberla::mesa_pd::Vec3& getAngularVelocity(const size_t p_idx) const;
 *
 * walberla::mesa_pd::Vec3& getTorqueRef(const size_t p_idx);
 *
 * const uint_t& getType(const size_t p_idx) const;
 *
 * const std::map<walberla::id_t, walberla::mesa_pd::Vec3>& getContactHistory(const size_t p_idx) const;
 * void setContactHistory(const size_t p_idx, const std::map<walberla::id_t, walberla::mesa_pd::Vec3>& v);
 *
 * \endcode
 * \ingroup mesa_pd_kernel
 */

class Cohesion
{
 public:
   Cohesion(const real_t dt,
            const real_t E = real_t(1e5),
            const real_t damp = real_t(0),
            const real_t b_c = real_t(10));
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
                   real_t penetrationDepth);
   template <typename Accessor>
   bool wereParticlesBound(const size_t idx1, const size_t idx2, Accessor& ac);

 private:
   template <typename Accessor>
   inline void persistContactCohesion(const size_t idx1,
                                      const size_t idx2,
                                      Accessor& ac,
                                      const Vec3 Ut,
                                      const Vec3 Ur,
                                      const Vec3 Uo);

   real_t dt_;
   real_t E_;
   real_t damp_;
   real_t b_c_;
};

Cohesion::Cohesion(const real_t dt,
                   const real_t E,
                   const real_t damp,
                   const real_t b_c) : dt_(dt), E_(E), damp_(damp), b_c_(b_c) {}

template <typename Accessor>
inline bool Cohesion::operator()(const size_t p_idx1,
                                 const size_t p_idx2,
                                 Accessor& ac,
                                 const bool contactExists,
                                 Vec3 contactNormal,
                                 real_t penetrationDepth) {

   auto uid_p1 = ac.getUid(p_idx1);
   auto uid_p2 = ac.getUid(p_idx2);



   // existing contact history of particle 1 -> particle 2
   const auto& och1 = ac.getOldContactHistoryRef(p_idx1)[uid_p2];
   // existing contact history of particle 2 -> particle 1
   const auto& och2 = ac.getOldContactHistoryRef(p_idx2)[uid_p1];


   if (wereParticlesBound(p_idx1, p_idx2, ac))
   {
      const auto shape1 = ac.getShape(p_idx1);
      const auto shape2 = ac.getShape(p_idx2);

      WALBERLA_CHECK_EQUAL(shape1->getShapeType(), data::Sphere::SHAPE_TYPE, "Cohesion needs sphere shapes");
      WALBERLA_CHECK_EQUAL(shape2->getShapeType(), data::Sphere::SHAPE_TYPE, "Cohesion needs sphere shapes");

      const auto sphere1 = *static_cast<data::Sphere*>(shape1);
      const auto sphere2 = *static_cast<data::Sphere*>(shape2);

      real_t radius1 = sphere1.getRadius();
      real_t radius2 = sphere2.getRadius();

      contactNormal = (ac.getPosition(p_idx1) - ac.getPosition(p_idx2)).getNormalized();
      penetrationDepth = (ac.getPosition(p_idx2) - ac.getPosition(p_idx1)).length() - radius1 - radius2;
      real_t initialDn = och1.getInitialPenetrationDepth();
      WALBERLA_CHECK(realIsEqual(initialDn, och2.getInitialPenetrationDepth()), "Initial dn inconsistent in contact histories.");
      real_t initialDnDiff = penetrationDepth - initialDn;

      // like in CohesionModel.py:
      Vec3 relVel = ac.getLinearVelocity(p_idx1) - ac.getLinearVelocity(p_idx2);
      Vec3 angVel1 = ac.getAngularVelocity(p_idx1);
      Vec3 angVel2 = ac.getAngularVelocity(p_idx2);


      // params

      const auto kn = 2. * E_ * radius1*radius2 / (radius1+radius2);
      const auto ks = kn * real_t(0.2);
      const auto kr = kn * real_t(0.);
      const auto ko = kn * real_t(0.);
      auto mass1 = real_t(1)/ac.getInvMass(p_idx1);
      auto mass2 = real_t(1)/ac.getInvMass(p_idx2);
      const auto m_eff  =  mass1*mass2/(mass1 + mass2);
      const auto nug = real_t(2) * sqrt(real_t(2)*kn*m_eff) * damp_;
      const auto nu = nug * real_t(0.5);
      const auto nus = nug * real_t(0.1);
      const auto nur = nug * real_t(0.5) * real_t(0.01);
      const auto nuo = nug * real_t(0.5) * real_t(0.01);

      // calculations
      real_t fNabs = real_t(0);
      if(initialDnDiff >= real_t(0))
      {
         fNabs = real_t(-1) * kn * initialDnDiff;
      }
      else
      {

         fNabs = real_t(-1) * kn * initialDnDiff;
         if(fNabs < real_t(0)) fNabs = real_t(0);
      }

      auto fn = fNabs * contactNormal + nu * (real_t(-1)*relVel * contactNormal) * contactNormal;
      auto a_ij = radius1*radius2 / (radius1+radius2);

      // Sliding
      auto Vij = relVel
                 + (radius1 - real_t(0.5)*fabs(initialDnDiff))
                      * (contactNormal % angVel1)
                 + (radius2 - real_t(0.5)*fabs(initialDnDiff))
                      * (contactNormal % angVel2);
//      auto Vij = relVel;
      auto Vt = Vij - contactNormal*(dot(contactNormal, Vij));
      auto Ut_pre = och1.getSlidingDisplacement();
      Ut_pre = Ut_pre - contactNormal*(contactNormal*Ut_pre);
      auto Ut = Ut_pre + dt_*Vt;
      auto fs = real_t(-1)*ks*Ut - nus*Vt;

      // Rolling
      auto a_prime_ij = ((radius1 - real_t(0.5)*fabs(initialDnDiff)) * (radius2 - real_t(0.5)*fabs(initialDnDiff)))
                        / ((radius1 - real_t(0.5)*fabs(initialDnDiff)) + (radius2 - real_t(0.5)*fabs(initialDnDiff)));
      auto Vr = real_t(-1) * a_prime_ij * ((contactNormal % angVel1)
                                           - (contactNormal % angVel2));

      auto Ur_pre = och1.getRollingDisplacement();
      Ur_pre = Ur_pre - contactNormal*(contactNormal*Ur_pre);
      auto Ur = Ur_pre + dt_*Vr;
      auto fr = real_t(-1)*kr*Ur - nur*Vr;
      auto torque = a_ij * (contactNormal % fr);

      // Torsion
      auto Vo = a_ij * (contactNormal*angVel1 - contactNormal*angVel2) * contactNormal;
      auto Uo_pre = och1.getTorsionDisplacement();

      Uo_pre = contactNormal*(contactNormal*Uo_pre);
      auto Uo = Uo_pre + dt_*Vo;
      auto fo = real_t(-1)*ko*Uo - nuo*Vo;

      auto torsion = a_ij * fo;

      // rupture
      auto y_n = b_c_ * real_t(-1);
      auto y_s = b_c_ * real_t(0.5);
      auto y_r = b_c_ * real_t(0.4);
      auto y_o = b_c_ * real_t(0.4);

      auto rupture = pow(fs.length() / y_s, real_t(2))
                     + (fNabs / y_n)
                     + pow(torque.length() / y_r, real_t(2))
                     + pow(torsion.length() / y_o, real_t(2))
                     - real_t(1);

//      WALBERLA_LOG_INFO(ac.getUid(p_idx1) << "-" << ac.getUid(p_idx2) << ": Rupture: " << rupture);

      auto& nch1 = ac.getNewContactHistoryRef(p_idx1)[uid_p2];
      auto& nch2 = ac.getNewContactHistoryRef(p_idx2)[uid_p1];
      nch1.setRupture(rupture);
      nch2.setRupture(rupture);

      if (rupture >= real_t(0)) {
         WALBERLA_LOG_INFO("Cohesion bond breaks (rupture: " << rupture << ") between particle "
                           << ac.getUid(p_idx1) << " and " << ac.getUid(p_idx2) << ".");
         fn[0] = 0_r;
         fn[1] = 0_r;
         fn[2] = 0_r;
         fs[0] = 0_r;
         fs[1] = 0_r;
         fs[2] = 0_r;

         torque[0] = 0_r;
         torque[1] = 0_r;
         torque[2] = 0_r;

         torque[0] = 0_r;
         torque[1] = 0_r;
         torque[2] = 0_r;

         torsion[0] = 0_r;
         torsion[1] = 0_r;
         torsion[2] = 0_r;

      }

         auto force1 = fn + fs;
         auto force2 = - fn - fs;
         addForceAtomic(p_idx1, ac, force1);
         addForceAtomic(p_idx2, ac, force2);

         auto torque1 = real_t(-1) * (radius1 - real_t(0.5)*fabs(initialDnDiff)) * (contactNormal % force1)
                        + torque
                        + torsion;
         auto torque2 = (radius2 - real_t(0.5)*fabs(initialDnDiff)) / (radius1 - real_t(0.5)*fabs(initialDnDiff))
                        * (real_t(-1) * (radius1 - real_t(0.5)*fabs(initialDnDiff)) * (contactNormal % force1))
                        - torque
                        - torsion;
         addTorqueAtomic(p_idx1, ac, torque1);
         addTorqueAtomic(p_idx2, ac, torque2);

         if(rupture < real_t(0))
         {
            persistContactCohesion(p_idx1, p_idx2, ac, Ut, Ur, Uo);
         }


      return true;
   }
   else if (!wereParticlesBound(p_idx1, p_idx2, ac) && contactExists)
   {
      const auto shape1 = ac.getShape(p_idx1);
      const auto shape2 = ac.getShape(p_idx2);

      WALBERLA_CHECK_EQUAL(shape1->getShapeType(), data::Sphere::SHAPE_TYPE, "Cohesion needs sphere shapes");
      WALBERLA_CHECK_EQUAL(shape2->getShapeType(), data::Sphere::SHAPE_TYPE, "Cohesion needs sphere shapes");

      const auto sphere1 = *static_cast<data::Sphere*>(shape1);
      const auto sphere2 = *static_cast<data::Sphere*>(shape2);

      real_t radius1 = sphere1.getRadius();
      real_t radius2 = sphere2.getRadius();


      contactNormal = (ac.getPosition(p_idx1) - ac.getPosition(p_idx2)).getNormalized();
      penetrationDepth = (ac.getPosition(p_idx2) - ac.getPosition(p_idx1)).length() - radius1 - radius2;
      real_t dn = penetrationDepth;

      Vec3 relVel = ac.getLinearVelocity(p_idx1) - ac.getLinearVelocity(p_idx2);
      Vec3 angVel1 = ac.getAngularVelocity(p_idx1);
      Vec3 angVel2 = ac.getAngularVelocity(p_idx2);

      const auto kn = 2. * E_ * radius1*radius2 / (radius1+radius2);
      const auto ks = kn * real_t(1);
      const auto kr = kn * real_t(0.);
      const auto ko = kn * real_t(0.);
      auto mass1 = real_t(1)/ac.getInvMass(p_idx1);
      auto mass2 = real_t(1)/ac.getInvMass(p_idx2);
      const auto m_eff  =  mass1*mass2/(mass1 + mass2);

      const auto nug = real_t(2) * sqrt(real_t(2)*kn*m_eff) * damp_;
      const auto nu = nug * real_t(0.5);
      const auto nus = nug * real_t(0.1);
      const auto nur = nug * real_t(0.5) * real_t(0.01);
      const auto nuo = nug * real_t(0.5) * real_t(0.01);


      auto fNabs = real_t(-1) * kn * dn;
      if(fNabs < real_t(0)) fNabs = real_t(0);
      auto fn = fNabs * contactNormal + nu * (real_t(-1)*relVel * contactNormal) * contactNormal;
      auto a_ij = radius1*radius2 / (radius1+radius2);
      auto Vij = relVel
                 + (radius1 - real_t(0.5)*fabs(dn))
                   * (contactNormal % angVel1)
                 + (radius2 - real_t(0.5)*fabs(dn))
                   * (contactNormal % angVel2);

//      auto Vij = relVel;
      auto Vt = Vij - contactNormal*(dot(contactNormal, Vij));
      auto Ut_pre = och1.getSlidingDisplacement();
      Ut_pre = Ut_pre - contactNormal*(contactNormal*Ut_pre);
      auto Ut = Ut_pre + dt_*Vt;
      auto fs = real_t(-1)*ks*Ut - nus*Vt;

      // Rolling
      auto a_prime_ij = ((radius1 - real_t(0.5)*fabs(dn)) * (radius2 - real_t(0.5)*fabs(dn)))
                        / ((radius1 - real_t(0.5)*fabs(dn)) + (radius2 - real_t(0.5)*fabs(dn)));
      auto Vr = real_t(-1) * a_prime_ij * ((contactNormal % angVel1)
                                           - (contactNormal % angVel2));
      auto Ur_pre = och1.getRollingDisplacement();
      Ur_pre = Ur_pre - contactNormal*(contactNormal*Ur_pre);
      auto Ur = Ur_pre + dt_*Vr;
      auto fr = real_t(-1)*kr*Ur - nur*Vr;
      auto torque = a_ij * (contactNormal % fr);

      // Torsion
      auto Vo = a_ij * (contactNormal*angVel1 - contactNormal*angVel2) * contactNormal;
      auto Uo_pre = och1.getTorsionDisplacement();

      Uo_pre = contactNormal*(contactNormal*Uo_pre);
      auto Uo = Uo_pre + dt_*Vo;
      auto fo = real_t(-1)*ko*Uo - nuo*Vo;
      auto torsion = a_ij * fo;


      auto force1 = fn + fs;
      auto force2 = - fn - fs;
      addForceAtomic(p_idx1, ac, force1);
      addForceAtomic(p_idx2, ac, force2);

      auto torque1 = real_t(-1) * (radius1 - real_t(0.5)*fabs(dn)) * (contactNormal % force1)
                     + torque
                     + torsion;
      auto torque2 = (radius2 - real_t(0.5)*fabs(dn)) / (radius1 - real_t(0.5)*fabs(dn))
                     * (real_t(-1) * (radius1 - real_t(0.5)*fabs(dn)) * (contactNormal % force1))
                     - torque
                     - torsion;
      addTorqueAtomic(p_idx1, ac, torque1);
      addTorqueAtomic(p_idx2, ac, torque2);


      // here the non-cohesive DEM can be implemented, if a contact exists
      return false; // might have to run normal DEM, if a contact exists
   }
}

template <typename Accessor>
inline void Cohesion::persistContactCohesion(const size_t p_idx1,
                                             const size_t p_idx2,
                                             Accessor& ac,
                                             const Vec3 Ut,
                                             const Vec3 Ur,
                                             const Vec3 Uo) {
   auto uid_p1 = ac.getUid(p_idx1);
   auto uid_p2 = ac.getUid(p_idx2);

   auto& och1 = ac.getOldContactHistoryRef(p_idx1)[uid_p2];
   auto& och2 = ac.getOldContactHistoryRef(p_idx2)[uid_p1];

   auto& nch1 = ac.getNewContactHistoryRef(p_idx1)[uid_p2];
   auto& nch2 = ac.getNewContactHistoryRef(p_idx2)[uid_p1];

   nch1.setCohesionBound(true);
   nch2.setCohesionBound(true);

   real_t initialDn = och1.getInitialPenetrationDepth();
   nch1.setInitialPenetrationDepth(initialDn);
   nch2.setInitialPenetrationDepth(initialDn);

   nch1.setId1(och1.getId1());
   nch1.setId2(och1.getId2());

   nch2.setId1(och2.getId1());
   nch2.setId2(och2.getId2());

//   WALBERLA_LOG_INFO(" och1.getId1(): "<< och1.getId1()<<"  och1.getId2(): "<<och1.getId2());

   nch1.setSlidingDisplacement(Ut);
   nch2.setSlidingDisplacement(Ut);

   nch1.setRollingDisplacement(Ur);
   nch2.setRollingDisplacement(Ur);

   nch1.setTorsionDisplacement(Uo);
   nch2.setTorsionDisplacement(Uo);
}

template <typename Accessor>
inline bool Cohesion::wereParticlesBound(const size_t idx1, const size_t idx2,
                                        Accessor& ac) {
   auto uid_p1 = ac.getUid(idx1);
   auto uid_p2 = ac.getUid(idx2);

   bool cohesionBound_1to2 = false;
   bool cohesionBound_2to1 = false;

   /// Retrieve old contact histories of both particles to check if they are bound by cohesion
   auto oldContactHistory_p1 = ac.getOldContactHistoryRef(idx1);
   auto contact_1to2 = oldContactHistory_p1.find(uid_p2);
   if (contact_1to2 != oldContactHistory_p1.end()) {
      cohesionBound_1to2 = contact_1to2->second.getCohesionBound();
   }
   auto oldContactHistory_p2 = ac.getOldContactHistoryRef(idx2);
   auto contact_2to1 = oldContactHistory_p2.find(uid_p1);
   if (contact_2to1 != oldContactHistory_p2.end()) {
      cohesionBound_2to1 = contact_2to1->second.getCohesionBound();
   }

   // Ensure consistency of contact histories.
   WALBERLA_CHECK(cohesionBound_1to2 == cohesionBound_2to1,
                  "Inconsistency in old cohesion bounds detected.\n"
                     << "Old Contact History of particle 1: \n" << contact_1to2->second
                     << "Old Contact History of particle 2: \n" << contact_2to1->second);

   // Now the current state of cohesion is known.
   return cohesionBound_1to2;
}

} // namespace kernel
} // namespace mesa_pd
} // namespace walberla