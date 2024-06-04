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
//! \file MovingGeometry.h
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================
#pragma once

#include "core/math/Constants.h"
#include "blockforest/all.h"
#include "geometry/containment_octree/ContainmentOctree.h"
#include "field/AddToStorage.h"
#include "mesh/boundary/BoundarySetup.h"
#include "mesh_common/DistanceComputations.h"
#include "mesh_common/DistanceFunction.h"
#include "mesh_common/MatrixVectorOperations.h"
#include "mesh_common/MeshIO.h"
#include "mesh_common/MeshOperations.h"
#include "mesh_common/TriangleMeshes.h"
#include "mesh_common/distance_octree/DistanceOctree.h"
#include "mesh_common/vtk/CommonDataSources.h"
#include "mesh_common/vtk/VTKMeshWriter.h"
#include "mesh/blockforest/BlockExclusion.h"
#include "mesh/blockforest/BlockForestInitialization.h"
#include "mesh/boundary/BoundaryInfo.h"
#include "mesh/boundary/BoundarySetup.h"
#include "mesh/boundary/BoundaryUIDFaceDataSource.h"
#include "mesh/boundary/ColorToBoundaryMapper.h"

#include "mesa_pd/data/ParticleAccessor.h"
#include "mesa_pd/data/ParticleStorage.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#   include "gpu/AddGPUFieldToStorage.h"
#   include "gpu/DeviceSelectMPI.h"
#   include "gpu/FieldCopy.h"
#   include "gpu/GPUWrapper.h"
#   include "gpu/HostFieldAllocator.h"
#   include "gpu/ParallelStreams.h"
#   include "gpu/communication/UniformGPUScheme.h"
//#   include <gpu/Atomic.h>
#endif


namespace walberla
{


template < typename FractionField_T, typename VectorField_T>
class MovingGeometry
{

   typedef field::GhostLayerField< bool, 1 > GeometryField_T;
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   typedef gpu::GPUField< bool > GeometryFieldGPU_T;
#endif

 public:

   MovingGeometry(shared_ptr< StructuredBlockForest >& blocks, shared_ptr< mesh::TriangleMesh >& mesh,
                  const BlockDataID fractionFieldId, const BlockDataID objectVelocityId,  const BlockDataID forceFieldId,
                  shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>>& distOctree, const std::string meshName,
                  const uint_t superSamplingDepth, bool useTauInFractionField, real_t omega, real_t dt, AABB movementBoundingBox, 
                  Vector3<real_t> initialVelocity, Vector3<real_t> initialRotation, real_t fluidDensity, bool moving = true)
      : blocks_(blocks), mesh_(mesh), fractionFieldId_(fractionFieldId), objectVelocityId_(objectVelocityId), forceFieldId_(forceFieldId),
        distOctree_(distOctree), meshName_(meshName), superSamplingDepth_(superSamplingDepth),
        useTauInFractionField_(useTauInFractionField), tau_(1.0 / omega), dt_(dt), movementBoundingBox_(movementBoundingBox), fluidDensity_(fluidDensity), moving_(moving) {

      if(!moving_ && initialVelocity.length() > 0.0) {
         WALBERLA_LOG_WARNING_ON_ROOT("You created a non moving geometry, but your specified the initial velocity to " << initialVelocity)
      }
      if(!moving_ && initialRotation.length() > 0.0) {
         WALBERLA_LOG_WARNING_ON_ROOT("You created a non moving geometry, but your specified the initial velocity to " << initialRotation)
      }

      if (useTauInFractionField_)
         if (omega > 2.0 || omega < 0.0)
            WALBERLA_ABORT(
               "If you want to use the relaxation rate for building the fraction field, use a valid value for omega")
      
      auto meshCenterPoint         = computeCentroid(*mesh_);
      meshCenter_  = Vector3< real_t >(meshCenterPoint[0], meshCenterPoint[1], meshCenterPoint[2]);
      meshAABB_    = computeAABB(*mesh_);
      const Vector3< real_t > dxyz = Vector3< real_t >(blocks_->dx(0), blocks_->dy(0),
                                                       blocks_->dz(0));
      meshAABB_.extend(2. * dxyz);

      // Create one particle to use it for holding velocity and force information and for Euler integration
      particleStorage_            = walberla::make_shared< mesa_pd::data::ParticleStorage >(1);
      particleAccessor_           = walberla::make_shared< mesa_pd::data::ParticleAccessor >(particleStorage_);
      mesa_pd::data::Particle&& p = *particleStorage_->create();
      p.setPosition(meshCenter_);
      p.setLinearVelocity(initialVelocity);
      p.setAngularVelocity(initialRotation);

      WcTimer simTimer;
      simTimer.start();
      buildGeometryField();
      simTimer.end();
      double time = simTimer.max();
      WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
      WALBERLA_LOG_INFO_ON_ROOT("Finished building Geometry Mesh in " << time << "s")

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      if (geometryField_)
      {
         geometryFieldGPU_ = new GeometryFieldGPU_T(
            geometryField_->xSize(), geometryField_->ySize(), geometryField_->zSize(), geometryField_->fSize(),
            geometryField_->nrOfGhostLayers(), geometryField_->layout(), true);
         gpu::fieldCpy(*geometryFieldGPU_, *geometryField_);
      }
#endif
      getFractionFieldFromGeometryMesh();

      real_t volumeFromFracField = getVolumeFromFractionField() * dxyz[0] * dxyz[1] * dxyz[2];
      real_t meshVolume = mesh::computeVolume(*mesh_);
      real_t discretisationError = pow(volumeFromFracField - meshVolume, 2) / pow(meshVolume, 2);
      WALBERLA_LOG_INFO_ON_ROOT("Mesh volume is " << meshVolume << ", fraction Field volume is " << volumeFromFracField << ", discretisation Error is " << discretisationError << " (only accurate for fraction field without omega weighting)")

      if(moving_) {
         updateObjectVelocityField();
      }
   }


   void operator()() {

      getFractionFieldFromGeometryMesh();

      if(moving_) {
         updateObjectVelocityField();
         updateObjectPosition();
      }
   }

   virtual void updateObjectPosition() = 0;

   //TODO speed this up, maybe also with some supersampling etc
   void buildGeometryField() {
      WALBERLA_LOG_PROGRESS("Getting max level for geometry field size")
      int maxLevel = -1;
      WALBERLA_LOG_PROGRESS("Size of blocks_ is " << blocks_->size())

      for (auto& block : *blocks_) {
         if(movementBoundingBox_.intersects(block.getAABB())) {
            WALBERLA_LOG_PROGRESS("Getting level for block " << block.getId())
            const int level = int(blocks_->getLevel(block));
            if (level > maxLevel)
            {
               maxLevel           = level;
               maxRefinementDxyz_ = Vector3< real_t >(blocks_->dx(level), blocks_->dy(level), blocks_->dz(level));
            }
         }
      }

      WALBERLA_LOG_PROGRESS("Testing maxlevel")
      if(maxLevel <= -1) {
         WALBERLA_LOG_PROGRESS("Maxlevel <= -1")
         return;
      }

      uint_t interpolationStencilSize = uint_t(pow(2, real_t(superSamplingDepth_))) + 1;
      Vector3<real_t> dxyzSS = maxRefinementDxyz_ / real_t(pow(2, real_t(superSamplingDepth_)));
      auto fieldSize = Vector3<uint_t> (uint_t(meshAABB_.xSize() / dxyzSS[0] ), uint_t(meshAABB_.ySize() / dxyzSS[1] ), uint_t(meshAABB_.zSize() / dxyzSS[2] ));
      WALBERLA_LOG_PROGRESS("Building geometry field with size " << fieldSize)
      geometryField_= make_shared< GeometryField_T >(fieldSize[0], fieldSize[1], fieldSize[2], uint_t(real_t(interpolationStencilSize) * 0.5 ), false, field::fzyx);

      const auto distFunct = make_shared<MeshDistanceFunction<mesh::DistanceOctree<mesh::TriangleMesh>>>( distOctree_ );

      CellInterval geoFieldBB = geometryField_->xyzSizeWithGhostLayer();

      WALBERLA_LOG_PROGRESS("Filling geometry field")

      std::queue< CellInterval > ciQueue;
      ciQueue.push(geoFieldBB);

      while (!ciQueue.empty())
      {
         const CellInterval& curCi = ciQueue.front();

         WALBERLA_ASSERT(!curCi.empty(), "Cell Interval: " << curCi);


         const AABB curAABB = AABB(meshAABB_.min()[0] + curCi.min()[0] * dxyzSS[0],
                                   meshAABB_.min()[1] + curCi.min()[1] * dxyzSS[1],
                                   meshAABB_.min()[2] + curCi.min()[2] * dxyzSS[2],
                                   meshAABB_.min()[0] + curCi.max()[0] * dxyzSS[0] + dxyzSS[0],
                                   meshAABB_.min()[1] + curCi.max()[1] * dxyzSS[1] + dxyzSS[1],
                                   meshAABB_.min()[2] + curCi.max()[2] * dxyzSS[2] + dxyzSS[2]);

         WALBERLA_ASSERT(!curAABB.empty(), "AABB: " << curAABB);

         Vector3< real_t > cellCenter = curAABB.center();

         const real_t sqSignedDistance = (*distFunct)(cellCenter);

         //only one cell left in the cell interval
         if (curCi.numCells() == uint_t(1))
         {
            bool fraction = sqSignedDistance < 0.0 ? true : false;
            geometryField_->get(curCi.min()) = fraction;
            ciQueue.pop();
            continue;
         }

         const real_t circumRadius   = curAABB.sizes().length() * real_t(0.5);
         const real_t sqCircumRadius = circumRadius * circumRadius;

         // the cell interval is fully covered by the mesh
         if (sqSignedDistance < -sqCircumRadius)
         {
            std::fill(geometryField_->beginSliceXYZ(curCi), geometryField_->end(), true);

            ciQueue.pop();
            continue;
         }
         // the cell interval is fully outside of mesh
         if (sqSignedDistance > sqCircumRadius)
         {
            std::fill(geometryField_->beginSliceXYZ(curCi), geometryField_->end(), false);
            ciQueue.pop();
            continue;
         }

         WALBERLA_ASSERT_GREATER(curCi.numCells(), uint_t(1));
         mesh::BoundarySetup::divideAndPushCellInterval(curCi, ciQueue);
         ciQueue.pop();
      }
   }

   void moveTriangleMesh(uint_t timestep, uint_t vtk_frequency)
   {
      if (timestep == 0) {
         oldRotation_ = particleAccessor_->getRotation(0).getMatrix();
         return;
      }

      if (vtk_frequency > 0 && timestep % vtk_frequency == 0 && moving_)
      {
         auto meshCenter = computeCentroid(*mesh_);
         Vector3< real_t > translationVector = particleAccessor_->getPosition(0) - Vector3<real_t>(meshCenter[0], meshCenter[1], meshCenter[2]);
         mesh::translate(*mesh_, translationVector);

         auto rotationMatrix = particleAccessor_->getRotation(0).getMatrix().getInverse() * oldRotation_;
         const Vector3< mesh::TriangleMesh::Scalar > axis_foot(particleAccessor_->getPosition(0));
         mesh::rotate(*mesh_, rotationMatrix, axis_foot);
         oldRotation_ = particleAccessor_->getRotation(0).getMatrix();
      }
   }

   Vector3<real_t> getHydrodynamicForce() {
      return particleAccessor_->getHydrodynamicForce(0);
   }

   Vector3<real_t> getTorque() {
      return particleAccessor_->getTorque(0);
   }

   Vector3<real_t> getLinearVelocity() {
      return particleAccessor_->getLinearVelocity(0);
   }

   void getFractionFieldFromGeometryMesh();
   void resetFractionField();
   virtual void updateObjectVelocityField();
   void calculateForcesOnBody();
   real_t getVolumeFromFractionField();
   //Vector3<real_t> getInertiaFromFractionField(real_t objectDensity);

 protected:
   shared_ptr< StructuredBlockForest > blocks_;
   shared_ptr< mesh::TriangleMesh > mesh_;

   BlockDataID fractionFieldId_;
   shared_ptr <GeometryField_T> geometryField_; //One Field on every MPI process, not on every block
   BlockDataID objectVelocityId_;
   BlockDataID forceFieldId_;

#if defined(WALBERLA_BUILD_WITH_CUDA)
   GeometryFieldGPU_T *geometryFieldGPU_;
#endif

   shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>> distOctree_;
   std::string meshName_;
   uint_t superSamplingDepth_;
   bool useTauInFractionField_;
   real_t tau_;
   real_t dt_;
   AABB movementBoundingBox_;
   real_t fluidDensity_;
   bool moving_;

   Vector3<real_t> meshCenter_;
   AABB meshAABB_;
   Vector3<real_t> maxRefinementDxyz_;

   //particle objects
   shared_ptr< mesa_pd::data::ParticleStorage > particleStorage_;
   shared_ptr< mesa_pd::data::ParticleAccessor > particleAccessor_;
   Matrix3<real_t> oldRotation_;
};


}//namespace waLBerla