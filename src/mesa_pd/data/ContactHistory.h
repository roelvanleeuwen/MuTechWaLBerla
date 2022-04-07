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
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

//======================================================================================================================
//
//  THIS FILE IS GENERATED - PLEASE CHANGE THE TEMPLATE !!!
//
//======================================================================================================================

#pragma once

#include <mesa_pd/data/DataTypes.h>
#include <mesa_pd/data/STLOverloads.h>

#include <core/Abort.h>
#include <core/debug/Debug.h>
#include <core/math/AABB.h>
#include <core/mpi/Datatype.h>
#include <core/mpi/RecvBuffer.h>
#include <core/mpi/SendBuffer.h>
#include <core/STLIO.h>

#include <vector>

namespace walberla {
namespace mesa_pd {
namespace data {

class ContactHistory
{
public:
   const walberla::mesa_pd::Vec3& getTangentialSpringDisplacement() const {return tangentialSpringDisplacement_;}
   walberla::mesa_pd::Vec3& getTangentialSpringDisplacementRef() {return tangentialSpringDisplacement_;}
   void setTangentialSpringDisplacement(const walberla::mesa_pd::Vec3& v) { tangentialSpringDisplacement_ = v;}
   
   const bool& getIsSticking() const {return isSticking_;}
   bool& getIsStickingRef() {return isSticking_;}
   void setIsSticking(const bool& v) { isSticking_ = v;}
   
   const real_t& getImpactVelocityMagnitude() const {return impactVelocityMagnitude_;}
   real_t& getImpactVelocityMagnitudeRef() {return impactVelocityMagnitude_;}
   void setImpactVelocityMagnitude(const real_t& v) { impactVelocityMagnitude_ = v;}
   
   const bool& getCohesionBound() const {return cohesionBound_;}
   bool& getCohesionBoundRef() {return cohesionBound_;}
   void setCohesionBound(const bool& v) { cohesionBound_ = v;}
   
   const real_t& getInitialGapSize() const {return initialGapSize_;}
   real_t& getInitialGapSizeRef() {return initialGapSize_;}
   void setInitialGapSize(const real_t& v) { initialGapSize_ = v;}
   
   const walberla::mesa_pd::Vec3& getSlidingDisplacement() const {return slidingDisplacement_;}
   walberla::mesa_pd::Vec3& getSlidingDisplacementRef() {return slidingDisplacement_;}
   void setSlidingDisplacement(const walberla::mesa_pd::Vec3& v) { slidingDisplacement_ = v;}
   
   const walberla::mesa_pd::Vec3& getRollingDisplacement() const {return rollingDisplacement_;}
   walberla::mesa_pd::Vec3& getRollingDisplacementRef() {return rollingDisplacement_;}
   void setRollingDisplacement(const walberla::mesa_pd::Vec3& v) { rollingDisplacement_ = v;}
   
   const walberla::mesa_pd::Vec3& getTorsionDisplacement() const {return torsionDisplacement_;}
   walberla::mesa_pd::Vec3& getTorsionDisplacementRef() {return torsionDisplacement_;}
   void setTorsionDisplacement(const walberla::mesa_pd::Vec3& v) { torsionDisplacement_ = v;}
   
private:
   walberla::mesa_pd::Vec3 tangentialSpringDisplacement_ {};
   bool isSticking_ {};
   real_t impactVelocityMagnitude_ {};
   bool cohesionBound_ {};
   real_t initialGapSize_ {};
   walberla::mesa_pd::Vec3 slidingDisplacement_ {};
   walberla::mesa_pd::Vec3 rollingDisplacement_ {};
   walberla::mesa_pd::Vec3 torsionDisplacement_ {};
};

inline
std::ostream& operator<<( std::ostream& os, const ContactHistory& ch )
{
   os << "==========  Contact History  ==========" << "\n" <<
         "tangentialSpringDisplacement: " << ch.getTangentialSpringDisplacement() << "\n" <<
         "isSticking          : " << ch.getIsSticking() << "\n" <<
         "impactVelocityMagnitude: " << ch.getImpactVelocityMagnitude() << "\n" <<
         "cohesionBound       : " << ch.getCohesionBound() << "\n" <<
         "initialGapSize      : " << ch.getInitialGapSize() << "\n" <<
         "slidingDisplacement : " << ch.getSlidingDisplacement() << "\n" <<
         "rollingDisplacement : " << ch.getRollingDisplacement() << "\n" <<
         "torsionDisplacement : " << ch.getTorsionDisplacement() << "\n" <<
         "================================" << std::endl;
   return os;
}

} //namespace data
} //namespace mesa_pd
} //namespace walberla

//======================================================================================================================
//
//  Send/Recv Buffer Serialization Specialization
//
//======================================================================================================================

namespace walberla {
namespace mpi {

template< typename T,    // Element type of SendBuffer
          typename G>    // Growth policy of SendBuffer
mpi::GenericSendBuffer<T,G>& operator<<( mpi::GenericSendBuffer<T,G> & buf, const mesa_pd::data::ContactHistory& obj )
{
   buf.addDebugMarker( "ch" );
   buf << obj.getTangentialSpringDisplacement();
   buf << obj.getIsSticking();
   buf << obj.getImpactVelocityMagnitude();
   buf << obj.getCohesionBound();
   buf << obj.getInitialGapSize();
   buf << obj.getSlidingDisplacement();
   buf << obj.getRollingDisplacement();
   buf << obj.getTorsionDisplacement();
   return buf;
}

template< typename T>    // Element type  of RecvBuffer
mpi::GenericRecvBuffer<T>& operator>>( mpi::GenericRecvBuffer<T> & buf, mesa_pd::data::ContactHistory& objparam )
{
   buf.readDebugMarker( "ch" );
   buf >> objparam.getTangentialSpringDisplacementRef();
   buf >> objparam.getIsStickingRef();
   buf >> objparam.getImpactVelocityMagnitudeRef();
   buf >> objparam.getCohesionBoundRef();
   buf >> objparam.getInitialGapSizeRef();
   buf >> objparam.getSlidingDisplacementRef();
   buf >> objparam.getRollingDisplacementRef();
   buf >> objparam.getTorsionDisplacementRef();
   return buf;
}

} // mpi
} // walberla