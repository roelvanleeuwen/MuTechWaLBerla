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
                   const bool contactExists,
                   Vec3 contactNormal,
                   real_t penetrationDepth,
                   real_t dt);

   
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
                                 const bool contactExists,
                                 Vec3 contactNormal,
                                 real_t penetrationDepth,
                                 real_t dt) {
   WALBERLA_LOG_INFO(p_idx1 << " " << p_idx2 << " " << contactNormal);
   //WALBERLA_LOG_INFO("Checking cohesion between " << p_idx1 << " and " << p_idx2 << ".");
/*
   auto uid_p1 = ac.getUid(p_idx1);
   auto uid_p2 = ac.getUid(p_idx2);


   // existing contact history of particle 1 -> particle 2
   const auto& och1 = ac.getOldContactHistoryRef(p_idx1)[uid_p2];
   // existing contact history of particle 2 -> particle 1
   const auto& och2 = ac.getOldContactHistoryRef(p_idx2)[uid_p1];
*/
   return true;

}

} // namespace kernel
} // namespace mesa_pd
} // namespace walberla