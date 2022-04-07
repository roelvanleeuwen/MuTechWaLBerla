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
 * Cohesio Initialization Kernel
 *
 * \code
 * const walberla::id_t& getUid(const size_t p_idx) const;
 *
 * const std::map<walberla::id_t, walberla::mesa_pd::Vec3>& getContactHistory(const size_t p_idx) const;
 * void setContactHistory(const size_t p_idx, const std::map<walberla::id_t, walberla::mesa_pd::Vec3>& v);
 *
 * \endcode
 * \ingroup mesa_pd_kernel
 */

class CohesionInitialization
{
 public:
   CohesionInitialization();
   CohesionInitialization(const CohesionInitialization& other) = default;
   CohesionInitialization(CohesionInitialization&& other)      = default;
   CohesionInitialization& operator=(const CohesionInitialization& other) = default;
   CohesionInitialization& operator=(CohesionInitialization&& other) = default;

   template <typename Accessor>
   void operator()(const size_t p_idx1, const size_t p_idx2,
                   Accessor& ac,
                   const real_t gapSize);
};

CohesionInitialization::CohesionInitialization() {}

template <typename Accessor>
inline void CohesionInitialization::operator()(const size_t p_idx1, const size_t p_idx2,
                                               Accessor& ac,
                                               const real_t gapSize) {
   auto shape1 = ac.getShape(p_idx1);
   auto shape2 = ac.getShape(p_idx2);

   WALBERLA_CHECK_EQUAL(shape1->getShapeType(), data::Sphere::SHAPE_TYPE, "Cohesion needs sphere shapes");
   WALBERLA_CHECK_EQUAL(shape2->getShapeType(), data::Sphere::SHAPE_TYPE, "Cohesion needs sphere shapes");

   // penetration depth has a negative sign if particles overlap
   WALBERLA_CHECK_LESS(gapSize, real_t(0), "Particles don't overlap.");
   WALBERLA_LOG_INFO("Initializing contact between pidxs " << p_idx1 << " and " << p_idx2 << ".");

   // contact history of particle 1 -> particle 2
   auto& nch1 = ac.getNewContactHistoryRef(p_idx1)[ac.getUid(p_idx2)];
   // contact history of particle 2 -> particle 1
   auto& nch2 = ac.getNewContactHistoryRef(p_idx2)[ac.getUid(p_idx1)];
   // save for each of the particles that they are bound to the other by cohesion
   nch1.setCohesionBound(true);
   nch2.setCohesionBound(true);

   nch1.setInitialGapSize(gapSize);
   nch2.setInitialGapSize(gapSize);

   nch1.setId1(ac.getUid(p_idx1));
   nch1.setId2(ac.getUid(p_idx2));

   nch2.setId1(ac.getUid(p_idx1));
   nch2.setId2(ac.getUid(p_idx2));
   WALBERLA_LOG_INFO("init id1: "<<ac.getUid(p_idx1)<<" init id2: "<<ac.getUid(p_idx2));
   WALBERLA_LOG_INFO("init id21: "<< nch2.getId1()<<" init id22: "<<nch2.getId2());
}

} // namespace kernel
} // namespace mesa_pd
} // namespace walberla