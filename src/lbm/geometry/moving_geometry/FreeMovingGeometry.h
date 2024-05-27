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


template< typename FractionField_T, typename VectorField_T >
class FreeMovingGeometry : public MovingGeometry< FractionField_T, VectorField_T >
{
   using MGBase = MovingGeometry< FractionField_T, VectorField_T >;

 public:
   FreeMovingGeometry(shared_ptr< StructuredBlockForest >& blocks, shared_ptr< mesh::TriangleMesh >& mesh,
                            const BlockDataID fractionFieldId, const BlockDataID objectVelocityId,
                            const BlockDataID forceFieldId, shared_ptr< mesh::DistanceOctree< mesh::TriangleMesh > >& distOctree,
                            const std::string meshName, const uint_t superSamplingDepth, bool useTauInFractionField,
                            real_t omega, real_t dt, AABB movementBoundingBox, Vector3<real_t> initialVelocity, Vector3<real_t> initialRotation, real_t fluidDensity, real_t objectDensity, Vector3<real_t> externalForce, bool moving = true)
      : MGBase(blocks, mesh, fractionFieldId, objectVelocityId, forceFieldId, distOctree, meshName, superSamplingDepth, useTauInFractionField, 1.0 / omega, dt, movementBoundingBox, initialVelocity, initialRotation, fluidDensity, moving),
        objectDensity_(objectDensity), externalForce_(externalForce)
   {
      explEulerIntegrator_ = make_shared<mesa_pd::kernel::ExplicitEuler>(dt);
      real_t objectMass = mesh::computeVolume(*MGBase::mesh_) * objectDensity_;
      MGBase::particleAccessor_->setInvMass(0, 1.0 / objectMass);

      auto inertia = mesh::computeInertiaTensor(*MGBase::mesh_) * objectDensity_;
      MGBase::particleAccessor_->setInvInertiaBF(0, inertia.getInverse());
   }


   void updateObjectPosition()
   {
      MGBase::calculateForcesOnBody();
      addHydrodynamicForce();
      (*explEulerIntegrator_)(0, *MGBase::particleAccessor_);
   }

   void addHydrodynamicForce() {
      Vector3<real_t> combinedForces = externalForce_ + MGBase::particleAccessor_->getHydrodynamicForce(0);
      MGBase::particleAccessor_->setForce(0, combinedForces);
   }


 private:

   real_t objectDensity_;
   Vector3<real_t> externalForce_;

   shared_ptr<mesa_pd::kernel::ExplicitEuler> explEulerIntegrator_;

   };
}//namespace waLBerla
