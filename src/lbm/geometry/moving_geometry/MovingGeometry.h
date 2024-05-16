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


template < typename FractionField_T, typename VectorField_T, typename GeometryField_T = field::GhostLayerField< real_t, 1 > >
class MovingGeometry
{
   using GeometryFieldData_T = typename GeometryField_T::value_type;
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   typedef gpu::GPUField< GeometryFieldData_T > GeometryFieldGPU_T;
#endif

 public:

   MovingGeometry(shared_ptr< StructuredBlockForest >& blocks, shared_ptr< mesh::TriangleMesh >& mesh,
                  const BlockDataID fractionFieldId, const BlockDataID objectVelocityId,  const BlockDataID forceFieldId,
                  shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>>& distOctree, const std::string meshName,
                  const uint_t superSamplingDepth, bool useTauInFractionField, real_t omega, real_t dt, AABB movementBoundingBox, 
                  Vector3<real_t> initialVelocity, Vector3<real_t> initialRotation, bool moving = true)
      : blocks_(blocks), mesh_(mesh), fractionFieldId_(fractionFieldId), objectVelocityId_(objectVelocityId), forceFieldId_(forceFieldId),
        distOctree_(distOctree), meshName_(meshName), superSamplingDepth_(superSamplingDepth),
        useTauInFractionField_(useTauInFractionField), tau_(1.0 / omega), dt_(dt), movementBoundingBox_(movementBoundingBox), moving_(moving) {

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
      WALBERLA_LOG_INFO_ON_ROOT("Finished coppying geometry field")
#endif
      getFractionFieldFromGeometryMesh();
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
      geometryField_= make_shared< GeometryField_T >(fieldSize[0], fieldSize[1], fieldSize[2], uint_t(real_t(interpolationStencilSize) * 0.5 ), GeometryFieldData_T(0), field::fzyx);

      const auto distFunct = make_shared<MeshDistanceFunction<mesh::DistanceOctree<mesh::TriangleMesh>>>( distOctree_ );
      real_t sqDx = dxyzSS[0] * dxyzSS[0];
      real_t sqDxHalf = (0.5 * dxyzSS[0]) * (0.5 * dxyzSS[0]);

      CellInterval blockCi = geometryField_->xyzSizeWithGhostLayer();

      WALBERLA_LOG_PROGRESS("Filling geometry field")

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (cell_idx_t cellZ = blockCi.zMin(); cellZ < blockCi.zMax(); ++cellZ) {
         for (cell_idx_t cellY = blockCi.yMin(); cellY < blockCi.yMax(); ++cellY) {
            for (cell_idx_t cellX = blockCi.xMin(); cellX < blockCi.xMax(); ++cellX) {

               Cell cell(cellX, cellY, cellZ);

               Vector3< real_t > cellCenter = meshAABB_.min() + Vector3< real_t >(cell.x() * dxyzSS[0], cell.y() * dxyzSS[1], cell.z() * dxyzSS[2]) + dxyzSS * 0.5;
               const real_t sqSignedDistance = (*distFunct)(cellCenter);

               GeometryFieldData_T fraction = GeometryFieldData_T(std::max(0.0, std::min(1.0, (sqDx - (sqSignedDistance + sqDxHalf)) / sqDx)));
               geometryField_->get(cell) = fraction;
            }
         }
      }
   }

   void moveTriangleMesh(uint_t timestep, uint_t vtk_frequency)
   {
      if (timestep == 0) return;
      if (vtk_frequency > 0 && timestep % vtk_frequency == 0 && moving_)
      {
         Vector3< real_t > translationVector = particleAccessor_->getLinearVelocity(0)* dt_ * real_t(vtk_frequency);
         Vector3< real_t > rotationVector = particleAccessor_->getAngularVelocity(0) * dt_ * real_t(vtk_frequency);
         auto rotationMatrix = math::Rot3<real_t> (rotationVector);
         mesh::translate(*mesh_, translationVector);
         const Vector3< mesh::TriangleMesh::Scalar > axis_foot(particleAccessor_->getPosition(0));
         mesh::rotate(*mesh_, rotationMatrix.getMatrix(), axis_foot);
      }
   }

   void writeForceToFile(uint_t timestep, std::string fileName) {
      WALBERLA_ROOT_SECTION(){
         auto force = particleAccessor_->getForce(0);
         std::ofstream myFile(fileName, std::ios::app);
         myFile << real_c(timestep)*dt_ << " " << force[0] << " " << force[1] << " " << force[2] << "\n";
         myFile.close();
      }
   }
   void getFractionFieldFromGeometryMesh();
   void resetFractionField();
   virtual void updateObjectVelocityField();
   void calculateForcesOnBody(real_t physForceFactor);
   real_t getVolumeFromFractionField();

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
   bool moving_;

   Vector3<real_t> meshCenter_;
   AABB meshAABB_;
   Vector3<real_t> maxRefinementDxyz_;

   //particle objects
   shared_ptr< mesa_pd::data::ParticleStorage > particleStorage_;
   shared_ptr< mesa_pd::data::ParticleAccessor > particleAccessor_;
};


}//namespace waLBerla