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
//! \file FreeMovingGeometry.h
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================
#pragma once

#include "MovingGeometry.h"
#include "mesa_pd/kernel/ExplicitEuler.h"

namespace walberla
{




template< typename FractionField_T, typename VectorField_T > >
class FreeMovingGeometry : public MovingGeometry< FractionField_T, VectorField_T >
{
   using MGBase = MovingGeometry< FractionField_T, VectorField_T >;

 public:
   FreeMovingGeometry(shared_ptr< StructuredBlockForest >& blocks, shared_ptr< mesh::TriangleMesh >& mesh,
                            const BlockDataID fractionFieldId, const BlockDataID objectVelocityId,
                            const BlockDataID forceFieldId, shared_ptr< mesh::DistanceOctree< mesh::TriangleMesh > >& distOctree,
                            const std::string meshName, const uint_t superSamplingDepth, bool useTauInFractionField,
                            real_t omega, real_t dt, AABB movementBoundingBox, Vector3<real_t> initialVelocity, Vector3<real_t> initialRotation, real_t fluidDensity, real_t objectDensity, bool moving = true)
      : MGBase(blocks, mesh, fractionFieldId, objectVelocityId, forceFieldId, distOctree,
                       meshName, superSamplingDepth, useTauInFractionField, 1.0 / omega, dt, movementBoundingBox, initialVelocity, initialRotation, fluidDensity, moving)
   {
      explEulerIntegrator_ = make_shared<mesa_pd::kernel::ExplicitEuler>(dt);
      const Vector3< real_t > dxyz = Vector3< real_t >(MGBase::blocks_->dx(0), MGBase::blocks_->dy(0),
                                                       MGBase::blocks_->dz(0));
      real_t objectMass = mesh::computeVolume(*MGBase::mesh_) * objectDensity;
      WALBERLA_LOG_INFO_ON_ROOT("Mass of object " << MGBase::meshName_ << " is " << objectMass)
      MGBase::particleAccessor_->setInvMass(0, 1.0 / objectMass);
      //TODO inertia for arbitrary object
      const math::Matrix3<real_t> inertia = math::Matrix3<real_t>::makeDiagonalMatrix( real_c(0.4) * objectMass * 0.5 * 0.5 );
      MGBase::particleAccessor_->setInvInertiaBF(0, inertia.getInverse());
   }


   void updateObjectPosition()
   {
      MGBase::calculateForcesOnBody();
      (*explEulerIntegrator_)(0, *MGBase::particleAccessor_);
   }


 private:

   shared_ptr<mesa_pd::kernel::ExplicitEuler> explEulerIntegrator_;

   };
}//namespace waLBerla
