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
//! \file PredefinedMovingGeometry.h
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================
#pragma once

#include "MovingGeometry.h"

namespace walberla
{




template< typename FractionField_T, typename VectorField_T, typename GeometryField_T = field::GhostLayerField< real_t, 1 > >
class PredefinedMovingGeometry : public MovingGeometry< FractionField_T, VectorField_T, GeometryField_T >
{
   using MGBase = MovingGeometry< FractionField_T, VectorField_T, GeometryField_T >;
   using GeometryFieldData_T = typename GeometryField_T::value_type;
   #if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      typedef gpu::GPUField< GeometryFieldData_T > GeometryFieldGPU_T;
   #endif

 public:
   PredefinedMovingGeometry(shared_ptr< StructuredBlockForest >& blocks, shared_ptr< mesh::TriangleMesh >& mesh,
                            const BlockDataID fractionFieldId, const BlockDataID objectVelocityId,
                            const BlockDataID forceFieldId, shared_ptr< mesh::DistanceOctree< mesh::TriangleMesh > >& distOctree,
                            const std::string meshName, const uint_t superSamplingDepth, bool useTauInFractionField,
                            real_t omega, real_t dt, AABB movementBoundingBox, Vector3<real_t> initialVelocity, Vector3<real_t> initialRotation, bool moving = true)
      : MGBase(blocks, mesh, fractionFieldId, objectVelocityId, forceFieldId, distOctree,
                       meshName, superSamplingDepth, useTauInFractionField, 1.0 / omega, dt, movementBoundingBox, initialVelocity, initialRotation, moving)
   { }


   void updateObjectPosition()
   {
      auto newPosition      = MGBase::particleAccessor_->getPosition(0) + MGBase::particleAccessor_->getLinearVelocity(0) * MGBase::dt_;
      MGBase::particleAccessor_->setPosition(0, newPosition);
      auto newParticleRotation = MGBase::particleAccessor_->getRotation(0);
      newParticleRotation.rotate(-MGBase::particleAccessor_->getAngularVelocity(0) * MGBase::dt_);
      MGBase::particleAccessor_->setRotation(0, newParticleRotation);
   }


   // reset velocity (in evey timestep) for time-dependent movement
   void setObjectLinearVelocity(Vector3<real_t> vel) {
      MGBase::particleAccessor_->setLinearVelocity(0, vel);
   }

   void setObjectAngularVelocity(Vector3<real_t> vel) {
      MGBase::particleAccessor_->setAngularVelocity(0, vel);
   }

 private:

   };
}//namespace waLBerla
