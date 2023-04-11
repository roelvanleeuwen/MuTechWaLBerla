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
//! \file   Utility.h
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//
//======================================================================================================================

#pragma once

#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/ShapeStorage.h"


namespace walberla {
namespace mesa_pd {


std::vector<walberla::id_t> createBucket(
   mesa_pd::data::ParticleStorage& ps, mesa_pd::data::ShapeStorage& ss,
   real_t radius,
   real_t wallThickness,
   real_t height,
   const Vector3<real_t> bottomCenterPosition,
   size_t numWallElements,
   const Vector3<real_t> linearVelocity = Vector3<real_t>{0}) {
   WALBERLA_CHECK(numWallElements%2 == 0, "numWallElements must be divisible by two.");

   real_t sliceAngle = 2*math::pi / real_t(numWallElements);
   real_t gapFillDelta = tan(sliceAngle/2_r) * wallThickness;
   real_t wallBoxWidth = sliceAngle*radius + 2_r*gapFillDelta;

   Vector3<real_t> wallBoxDims{
      wallThickness,
      wallBoxWidth,
      height
   };

   std::vector<walberla::id_t> boxes{numWallElements+numWallElements/2};

   auto boxWallShapeId = ss.create<mesa_pd::data::Box>(wallBoxDims);

   for(size_t i = 0; i < numWallElements; ++i) {
      real_t localSliceAngle = sliceAngle*(real_t(i) + .5_r);
      Vector3<real_t> position{
         bottomCenterPosition[0] + radius*cos(localSliceAngle),
         bottomCenterPosition[1] + radius*sin(localSliceAngle),
         bottomCenterPosition[2] + height/2_r
      };
      math::Rot3<real_t> rotation{Vector3<real_t>(0_r, 0_r, 1_r), localSliceAngle};

      auto boxWallParticle = ps.create(true);
      boxWallParticle->setShapeID(boxWallShapeId);
      boxWallParticle->setPosition(position);
      boxWallParticle->setRotation(rotation);
      boxWallParticle->setOwner(walberla::MPIManager::instance()->rank());
      boxWallParticle->setInteractionRadius(sqrt((wallBoxWidth/2_r * wallBoxWidth/2_r)
                                                 +(height/2_r * height/2_r)
                                                 +(wallThickness/2_r * wallThickness/2_r)));
      boxWallParticle->setLinearVelocity(linearVelocity);
      mesa_pd::data::particle_flags::set(boxWallParticle->getFlagsRef(), mesa_pd::data::particle_flags::FIXED);
      mesa_pd::data::particle_flags::set(boxWallParticle->getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);

      boxes.push_back(boxWallParticle->getUid());
   }

   Vector3<real_t> topBoxDims{
      wallBoxWidth,
      2_r* radius + wallThickness,
      wallThickness
   };
   auto boxTopShapeId = ss.create<mesa_pd::data::Box>(topBoxDims);

   for(size_t i = 0; i < numWallElements/2; ++i) {
      real_t localSliceAngle = sliceAngle*(real_t(i) + real_t((numWallElements/2+1)%2)*.5_r);
      Vector3<real_t> position{
         bottomCenterPosition[0],
         bottomCenterPosition[1],
         bottomCenterPosition[2] + height - wallThickness/2_r
      };
      math::Rot3<real_t> rotation{Vector3<real_t>(0_r, 0_r, 1_r), localSliceAngle};

      auto boxTopParticle = ps.create(true);
      boxTopParticle->setShapeID(boxTopShapeId);
      boxTopParticle->setPosition(position);
      boxTopParticle->setRotation(rotation);
      boxTopParticle->setOwner(walberla::MPIManager::instance()->rank());
      boxTopParticle->setInteractionRadius(sqrt((wallBoxWidth/2_r * wallBoxWidth/2_r)
                                                +((2_r*radius + wallThickness)/2_r * (2_r*radius + wallThickness)/2_r)
                                                +(wallThickness/2_r * wallThickness/2_r)));
      boxTopParticle->setLinearVelocity(linearVelocity);
      mesa_pd::data::particle_flags::set(boxTopParticle->getFlagsRef(), mesa_pd::data::particle_flags::FIXED);
      mesa_pd::data::particle_flags::set(boxTopParticle->getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);

      boxes.push_back(boxTopParticle->getUid());
   }

   return boxes;
}

void updateBucketPosition(mesa_pd::data::ParticleAccessorWithShape & accessor,
                          const std::vector<walberla::id_t>& bucketBoxUids, real_t dt) {
   for (const auto& bucketBoxUid: bucketBoxUids) {
      auto boxIdx = accessor.uidToIdx(bucketBoxUid);
      if (boxIdx == accessor.getInvalidIdx()) continue;
      auto newBoxPosition = accessor.getPosition(boxIdx) + dt * accessor.getLinearVelocity(boxIdx);
      accessor.setPosition(boxIdx, newBoxPosition);
   }
}

} // namespace mesa_pd
} // namespace walberla
