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
   Cohesion(const uint_t numParticleTypes);
   Cohesion(const Cohesion& other) = default;
   Cohesion(Cohesion&& other)      = default;
   Cohesion& operator=(const Cohesion& other) = default;
   Cohesion& operator=(Cohesion&& other) = default;

   template <typename Accessor>
   bool operator()(const size_t p_idx1,
                   const size_t p_idx2,
                   Accessor& ac,
                   real_t dt);

   template <typename Accessor>
   void nonCohesiveInteraction(const size_t p_idx1, const size_t p_idx2,
                               Accessor& ac, Vec3 contactPoint,
                               Vec3 contactNormal, real_t gapSize, real_t dt);

   template <typename Accessor>
   bool isCohesiveBondActive(size_t p_idx1, size_t p_idx2,  Accessor& ac) const;

   
   /// assumes this parameter is symmetric
   void setKn(const size_t type1, const size_t type2, const real_t& val);
   /// assumes this parameter is symmetric
   void setNun(const size_t type1, const size_t type2, const real_t& val);
   /// assumes this parameter is symmetric
   void setKsFactor(const size_t type1, const size_t type2, const real_t& val);
   /// assumes this parameter is symmetric
   void setKrFactor(const size_t type1, const size_t type2, const real_t& val);
   /// assumes this parameter is symmetric
   void setKoFactor(const size_t type1, const size_t type2, const real_t& val);
   /// assumes this parameter is symmetric
   void setNusFactor(const size_t type1, const size_t type2, const real_t& val);
   /// assumes this parameter is symmetric
   void setNurFactor(const size_t type1, const size_t type2, const real_t& val);
   /// assumes this parameter is symmetric
   void setNuoFactor(const size_t type1, const size_t type2, const real_t& val);
   /// assumes this parameter is symmetric
   void setFrictionCoefficient(const size_t type1, const size_t type2, const real_t& val);
   /// assumes this parameter is symmetric
   void setYn(const size_t type1, const size_t type2, const real_t& val);
   /// assumes this parameter is symmetric
   void setYs(const size_t type1, const size_t type2, const real_t& val);
   /// assumes this parameter is symmetric
   void setYr(const size_t type1, const size_t type2, const real_t& val);
   /// assumes this parameter is symmetric
   void setYo(const size_t type1, const size_t type2, const real_t& val);

   
   real_t getKn(const size_t type1, const size_t type2) const;
   real_t getNun(const size_t type1, const size_t type2) const;
   real_t getKsFactor(const size_t type1, const size_t type2) const;
   real_t getKrFactor(const size_t type1, const size_t type2) const;
   real_t getKoFactor(const size_t type1, const size_t type2) const;
   real_t getNusFactor(const size_t type1, const size_t type2) const;
   real_t getNurFactor(const size_t type1, const size_t type2) const;
   real_t getNuoFactor(const size_t type1, const size_t type2) const;
   real_t getFrictionCoefficient(const size_t type1, const size_t type2) const;
   real_t getYn(const size_t type1, const size_t type2) const;
   real_t getYs(const size_t type1, const size_t type2) const;
   real_t getYr(const size_t type1, const size_t type2) const;
   real_t getYo(const size_t type1, const size_t type2) const;
 private:
   uint_t numParticleTypes_;
   
   std::vector<real_t> kn_ {};
   std::vector<real_t> nun_ {};
   std::vector<real_t> ksFactor_ {};
   std::vector<real_t> krFactor_ {};
   std::vector<real_t> koFactor_ {};
   std::vector<real_t> nusFactor_ {};
   std::vector<real_t> nurFactor_ {};
   std::vector<real_t> nuoFactor_ {};
   std::vector<real_t> frictionCoefficient_ {};
   std::vector<real_t> yn_ {};
   std::vector<real_t> ys_ {};
   std::vector<real_t> yr_ {};
   std::vector<real_t> yo_ {};
};

Cohesion::Cohesion(const uint_t numParticleTypes)
{
   numParticleTypes_ = numParticleTypes;
   
   kn_.resize(numParticleTypes * numParticleTypes, real_t(0));
   nun_.resize(numParticleTypes * numParticleTypes, real_t(0));
   ksFactor_.resize(numParticleTypes * numParticleTypes, real_t(0));
   krFactor_.resize(numParticleTypes * numParticleTypes, real_t(0));
   koFactor_.resize(numParticleTypes * numParticleTypes, real_t(0));
   nusFactor_.resize(numParticleTypes * numParticleTypes, real_t(0));
   nurFactor_.resize(numParticleTypes * numParticleTypes, real_t(0));
   nuoFactor_.resize(numParticleTypes * numParticleTypes, real_t(0));
   frictionCoefficient_.resize(numParticleTypes * numParticleTypes, real_t(0));
   yn_.resize(numParticleTypes * numParticleTypes, real_t(0));
   ys_.resize(numParticleTypes * numParticleTypes, real_t(0));
   yr_.resize(numParticleTypes * numParticleTypes, real_t(0));
   yo_.resize(numParticleTypes * numParticleTypes, real_t(0));
}


inline void Cohesion::setKn(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
kn_[numParticleTypes_*type1 + type2] = val;
kn_[numParticleTypes_*type2 + type1] = val;
}
inline void Cohesion::setNun(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
nun_[numParticleTypes_*type1 + type2] = val;
nun_[numParticleTypes_*type2 + type1] = val;
}
inline void Cohesion::setKsFactor(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
ksFactor_[numParticleTypes_*type1 + type2] = val;
ksFactor_[numParticleTypes_*type2 + type1] = val;
}
inline void Cohesion::setKrFactor(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
krFactor_[numParticleTypes_*type1 + type2] = val;
krFactor_[numParticleTypes_*type2 + type1] = val;
}
inline void Cohesion::setKoFactor(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
koFactor_[numParticleTypes_*type1 + type2] = val;
koFactor_[numParticleTypes_*type2 + type1] = val;
}
inline void Cohesion::setNusFactor(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
nusFactor_[numParticleTypes_*type1 + type2] = val;
nusFactor_[numParticleTypes_*type2 + type1] = val;
}
inline void Cohesion::setNurFactor(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
nurFactor_[numParticleTypes_*type1 + type2] = val;
nurFactor_[numParticleTypes_*type2 + type1] = val;
}
inline void Cohesion::setNuoFactor(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
nuoFactor_[numParticleTypes_*type1 + type2] = val;
nuoFactor_[numParticleTypes_*type2 + type1] = val;
}
inline void Cohesion::setFrictionCoefficient(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
frictionCoefficient_[numParticleTypes_*type1 + type2] = val;
frictionCoefficient_[numParticleTypes_*type2 + type1] = val;
}
inline void Cohesion::setYn(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
yn_[numParticleTypes_*type1 + type2] = val;
yn_[numParticleTypes_*type2 + type1] = val;
}
inline void Cohesion::setYs(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
ys_[numParticleTypes_*type1 + type2] = val;
ys_[numParticleTypes_*type2 + type1] = val;
}
inline void Cohesion::setYr(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
yr_[numParticleTypes_*type1 + type2] = val;
yr_[numParticleTypes_*type2 + type1] = val;
}
inline void Cohesion::setYo(const size_t type1, const size_t type2, const real_t& val)
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
yo_[numParticleTypes_*type1 + type2] = val;
yo_[numParticleTypes_*type2 + type1] = val;
}


inline real_t Cohesion::getKn(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( kn_[numParticleTypes_*type1 + type2],
                             kn_[numParticleTypes_*type2 + type1],
                             "parameter matrix for kn not symmetric!");
return kn_[numParticleTypes_*type1 + type2];
}
inline real_t Cohesion::getNun(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( nun_[numParticleTypes_*type1 + type2],
                             nun_[numParticleTypes_*type2 + type1],
                             "parameter matrix for nun not symmetric!");
return nun_[numParticleTypes_*type1 + type2];
}
inline real_t Cohesion::getKsFactor(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( ksFactor_[numParticleTypes_*type1 + type2],
                             ksFactor_[numParticleTypes_*type2 + type1],
                             "parameter matrix for ksFactor not symmetric!");
return ksFactor_[numParticleTypes_*type1 + type2];
}
inline real_t Cohesion::getKrFactor(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( krFactor_[numParticleTypes_*type1 + type2],
                             krFactor_[numParticleTypes_*type2 + type1],
                             "parameter matrix for krFactor not symmetric!");
return krFactor_[numParticleTypes_*type1 + type2];
}
inline real_t Cohesion::getKoFactor(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( koFactor_[numParticleTypes_*type1 + type2],
                             koFactor_[numParticleTypes_*type2 + type1],
                             "parameter matrix for koFactor not symmetric!");
return koFactor_[numParticleTypes_*type1 + type2];
}
inline real_t Cohesion::getNusFactor(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( nusFactor_[numParticleTypes_*type1 + type2],
                             nusFactor_[numParticleTypes_*type2 + type1],
                             "parameter matrix for nusFactor not symmetric!");
return nusFactor_[numParticleTypes_*type1 + type2];
}
inline real_t Cohesion::getNurFactor(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( nurFactor_[numParticleTypes_*type1 + type2],
                             nurFactor_[numParticleTypes_*type2 + type1],
                             "parameter matrix for nurFactor not symmetric!");
return nurFactor_[numParticleTypes_*type1 + type2];
}
inline real_t Cohesion::getNuoFactor(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( nuoFactor_[numParticleTypes_*type1 + type2],
                             nuoFactor_[numParticleTypes_*type2 + type1],
                             "parameter matrix for nuoFactor not symmetric!");
return nuoFactor_[numParticleTypes_*type1 + type2];
}
inline real_t Cohesion::getFrictionCoefficient(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( frictionCoefficient_[numParticleTypes_*type1 + type2],
                             frictionCoefficient_[numParticleTypes_*type2 + type1],
                             "parameter matrix for frictionCoefficient not symmetric!");
return frictionCoefficient_[numParticleTypes_*type1 + type2];
}
inline real_t Cohesion::getYn(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( yn_[numParticleTypes_*type1 + type2],
                             yn_[numParticleTypes_*type2 + type1],
                             "parameter matrix for yn not symmetric!");
return yn_[numParticleTypes_*type1 + type2];
}
inline real_t Cohesion::getYs(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( ys_[numParticleTypes_*type1 + type2],
                             ys_[numParticleTypes_*type2 + type1],
                             "parameter matrix for ys not symmetric!");
return ys_[numParticleTypes_*type1 + type2];
}
inline real_t Cohesion::getYr(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( yr_[numParticleTypes_*type1 + type2],
                             yr_[numParticleTypes_*type2 + type1],
                             "parameter matrix for yr not symmetric!");
return yr_[numParticleTypes_*type1 + type2];
}
inline real_t Cohesion::getYo(const size_t type1, const size_t type2) const
{
WALBERLA_ASSERT_LESS( type1, numParticleTypes_ );
WALBERLA_ASSERT_LESS( type2, numParticleTypes_ );
WALBERLA_ASSERT_FLOAT_EQUAL( yo_[numParticleTypes_*type1 + type2],
                             yo_[numParticleTypes_*type2 + type1],
                             "parameter matrix for yo not symmetric!");
return yo_[numParticleTypes_*type1 + type2];
}


template <typename Accessor>
inline bool Cohesion::operator()(const size_t p_idx1,
                                 const size_t p_idx2,
                                 Accessor& ac,
                                 real_t dt) {

   auto uid_p1 = ac.getUid(p_idx1);
   auto uid_p2 = ac.getUid(p_idx2);

   const auto shape1 = ac.getShape(p_idx1);
   const auto shape2 = ac.getShape(p_idx2);
   WALBERLA_CHECK_EQUAL(shape1->getShapeType(), data::Sphere::SHAPE_TYPE, "Cohesion needs sphere shapes");
   WALBERLA_CHECK_EQUAL(shape2->getShapeType(), data::Sphere::SHAPE_TYPE, "Cohesion needs sphere shapes");
   const auto sphere1 = *static_cast<data::Sphere*>(shape1);
   const auto sphere2 = *static_cast<data::Sphere*>(shape2);
   real_t radius1 = sphere1.getRadius();
   real_t radius2 = sphere2.getRadius();

   auto type1 = ac.getType(p_idx1);
   auto type2 = ac.getType(p_idx2);

   const auto& och1 = ac.getOldContactHistoryRef(p_idx1)[uid_p2];
   const auto& och2 = ac.getOldContactHistoryRef(p_idx2)[uid_p1];

   auto& nch1 = ac.getNewContactHistoryRef(p_idx1)[uid_p2];
   auto& nch2 = ac.getNewContactHistoryRef(p_idx2)[uid_p1];

   Vec3 relVelLinear = ac.getLinearVelocity(p_idx1) - ac.getLinearVelocity(p_idx2);
   Vec3 angVel1 = ac.getAngularVelocity(p_idx1);
   Vec3 angVel2 = ac.getAngularVelocity(p_idx2);

   auto contactNormal = (ac.getPosition(p_idx1) - ac.getPosition(p_idx2)).getNormalized();
   auto gapSize = (ac.getPosition(p_idx2) - ac.getPosition(p_idx1)).length() - radius1 - radius2;

   real_t dn0 = och1.getInitialGapSize();
   WALBERLA_CHECK(realIsEqual(dn0, och2.getInitialGapSize()), "Initial dn inconsistent in contact histories.");
   real_t dn = gapSize - dn0;

   // normal part
   real_t fN_norm = -getKn(type1,type2) * dn - getNun(type1,type2) * (relVelLinear * contactNormal);
   Vec3 fn = fN_norm * contactNormal;

   // tangential part
   auto vij = relVelLinear + (radius1 - 0.5_r * abs(dn)) * (contactNormal % angVel1)
              + (radius2 - 0.5_r * abs(dn)) * (contactNormal % angVel2);
   auto vt = vij - contactNormal*(dot(contactNormal, vij));

   auto us = och1.getTangentialSpringDisplacement();
   auto us_projected = us - contactNormal * (contactNormal * us);
   auto us_new = us_projected + dt * vt;

   real_t ks = getKsFactor(type1,type2) * getKn(type1,type2);
   real_t nus = getNusFactor(type1,type2) * getNun(type1,type2);

   Vec3 fs = -ks * us_new - nus * vt;

   nch1.setTangentialSpringDisplacement(us_new);
   nch2.setTangentialSpringDisplacement(us_new);

   // rolling part
   auto a_prime_ij = ((radius1 - 0.5_r*abs(dn)) * (radius2 - 0.5_r*abs(dn))) / ((radius1 - 0.5_r*abs(dn)) + (radius2 - 0.5_r*abs(dn)));
   auto vr = -a_prime_ij * ((contactNormal % angVel1) - (contactNormal % angVel2));

   auto ur = och1.getRollingDisplacement();
   auto ur_projected = ur - contactNormal * (contactNormal * ur);
   auto ur_new = ur_projected + dt * vr;

   real_t kr = getKrFactor(type1,type2) * getKn(type1,type2);
   real_t nur = getNurFactor(type1,type2) * getNun(type1,type2);

   Vec3 fr = -kr * ur_new - nur * vr;
   auto a_ij = radius1*radius2 / (radius1+radius2);
   auto torqueRolling = a_ij * (contactNormal % fr);

   nch1.setRollingDisplacement(ur_new);
   nch2.setRollingDisplacement(ur_new);

   // torsion
   auto vo = a_ij * (contactNormal*angVel1 - contactNormal*angVel2) * contactNormal;
   auto uo = och1.getTorsionDisplacement();
   auto uo_projected = uo - contactNormal * (contactNormal * uo);
   auto uo_new = uo_projected + dt * vo;

   real_t ko = getKoFactor(type1,type2) * getKn(type1,type2);
   real_t nuo = getNuoFactor(type1,type2) * getNun(type1,type2);

   Vec3 fo = -ko * uo_new - nuo * vo;
   auto torsion = a_ij * fo;

   real_t ruptureParameter = - fN_norm / getYn(type1,type2)
                             + pow(fs.length() / getYs(type1,type2), 2)
                             + pow(torqueRolling.length() / getYr(type1,type2), 2)
                             + pow(torsion.length() / getYo(type1,type2), 2)
                             - 1_r;

   WALBERLA_CHECK(!std::isnan(ruptureParameter));

   /*
   WALBERLA_LOG_INFO("In cohesion kernel:"  << dn);
   WALBERLA_LOG_INFO("Normal: " << fn << " " << relVelLinear << " " << fN_norm );
   WALBERLA_LOG_INFO("Tangential: " << fs << " " << us_new << " " <<  vt);
   WALBERLA_LOG_INFO("Rolling: " << torqueRolling);
   WALBERLA_LOG_INFO("Torsion: " << torsion);
   WALBERLA_LOG_INFO("Rupture: " << ruptureParameter);
   */

   if(ruptureParameter >= 0_r)
   {
      // bond breaks -> use regular DEM
      nch1.setCohesionBound(false);
      nch2.setCohesionBound(false);
      return false;

   } else
   {
      // bond remains
      nch1.setCohesionBound(true);
      nch2.setCohesionBound(true);

      nch1.setInitialGapSize(dn0);
      nch2.setInitialGapSize(dn0);

      Vec3 force1 = fn + fs;
      Vec3 force2 = -force1;

      auto torqueTemp = - (radius1 - 0.5_r*abs(dn)) * (contactNormal % force1);
      auto torque1 = torqueTemp + torqueRolling + torsion;
      auto torque2 = (radius2 - 0.5_r*abs(dn)) / (radius1 - 0.5_r*abs(dn)) * torqueTemp - torqueRolling - torsion;

      addForceAtomic(p_idx1, ac, force1);
      addForceAtomic(p_idx2, ac, force2);

      addTorqueAtomic(p_idx1, ac, torque1);
      addTorqueAtomic(p_idx2, ac, torque2);

      return true;
   }

}

template <typename Accessor>
inline void Cohesion::nonCohesiveInteraction(const size_t p_idx1, const size_t p_idx2,
                                             Accessor& ac, Vec3 contactPoint, Vec3 contactNormal, real_t gapSize, real_t dt) {
   if(gapSize > 0_r) return; // no overlap -> no interaction

   auto uid_p1 = ac.getUid(p_idx1);
   auto uid_p2 = ac.getUid(p_idx2);

   auto type1 = ac.getType(p_idx1);
   auto type2 = ac.getType(p_idx2);

   const auto& och1 = ac.getOldContactHistoryRef(p_idx1)[uid_p2];
   const auto& och2 = ac.getOldContactHistoryRef(p_idx2)[uid_p1];

   auto& nch1 = ac.getNewContactHistoryRef(p_idx1)[uid_p2];
   auto& nch2 = ac.getNewContactHistoryRef(p_idx2)[uid_p1];

   Vec3 relVelLinear = ac.getLinearVelocity(p_idx1) - ac.getLinearVelocity(p_idx2);

   real_t dn = gapSize;

   // normal
   real_t fN_norm = -getKn(type1,type2) * dn - getNun(type1,type2) * (relVelLinear * contactNormal);
   Vec3 fn = fN_norm * contactNormal;

   // tangential
   auto vij = -(getVelocityAtWFPoint(p_idx1, ac, contactPoint) - getVelocityAtWFPoint(p_idx2, ac, contactPoint));
   auto vt = vij - contactNormal*(dot(contactNormal, vij));

   auto us = och1.getTangentialSpringDisplacement();
   auto us_projected = us - contactNormal * (contactNormal * us);
   auto us_new = us_projected + dt * vt;

   real_t ks = getKsFactor(type1,type2) * getKn(type1,type2);
   real_t nus = getNusFactor(type1,type2) * getNun(type1,type2);

   Vec3 fs = ks * us_new + nus * vt;

   /*
   WALBERLA_LOG_INFO("In non-cohesion kernel: " << dn);
   WALBERLA_LOG_INFO("Normal: " << fn );
   WALBERLA_LOG_INFO("Tangential (SD): " << fs << " " << us_new << vt);
    */

   const Vec3 t = fs.getNormalizedIfNotZero(); // tangential unit vector
   const real_t fFriction_norm = getFrictionCoefficient(type1,type2) * fn.length();

   real_t f_tangential_norm;
   if( fs.length() < fFriction_norm )
   {
      f_tangential_norm = fs.length();
   }
   else
   {
      f_tangential_norm = fFriction_norm;
      // reset displacement vector
      us_new = ( fFriction_norm * t - nus * vt ) / ks;
   }

   Vec3 fTangential = f_tangential_norm * t;

   /*
   WALBERLA_LOG_INFO("Tangential (friction): " << fFriction_norm);
   WALBERLA_LOG_INFO("Tangential (actual): " << fTangential);
   */

   nch1.setTangentialSpringDisplacement(us_new);
   nch2.setTangentialSpringDisplacement(us_new);

   //note: rolling and torsion is not considered!

   Vec3 force1 = fn + fTangential;
   Vec3 force2 = -force1;

   addForceAtWFPosAtomic( p_idx1, ac, force1, contactPoint ); // sets force and torque
   addForceAtWFPosAtomic( p_idx2, ac, force2, contactPoint );

}


template <typename Accessor>
inline bool Cohesion::isCohesiveBondActive(size_t p_idx1, size_t p_idx2,  Accessor& ac) const {

   auto uid_p1 = ac.getUid(p_idx1);
   auto uid_p2 = ac.getUid(p_idx2);

   bool cohesionBound_1to2 = false;
   bool cohesionBound_2to1 = false;

   /// Retrieve new contact histories of both particles to check if they are bound by cohesion
   auto och1 = ac.getOldContactHistory(p_idx1);
   auto contact_1to2 = och1.find(uid_p2);
   if (contact_1to2 != och1.end()) {
      cohesionBound_1to2 = contact_1to2->second.getCohesionBound();
   }
   auto och2 = ac.getOldContactHistory(p_idx2);
   auto contact_2to1 = och2.find(uid_p1);
   if (contact_2to1 != och2.end()) {
      cohesionBound_2to1 = contact_2to1->second.getCohesionBound();
   }

   // Ensure consistency of contact histories.
   WALBERLA_CHECK(cohesionBound_1to2 == cohesionBound_2to1,
                  "Inconsistency in cohesion bounds detected.\n"
                        << "Contact History of particle 1: \n" << contact_1to2->second
                        << "Contact History of particle 2: \n" << contact_2to1->second);

   // Now the current state of cohesion is known.
   return cohesionBound_1to2;
}

} // namespace kernel
} // namespace mesa_pd
} // namespace walberla