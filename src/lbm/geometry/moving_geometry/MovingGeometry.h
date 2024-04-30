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
#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/FieldCopy.h"
#include "gpu/GPUWrapper.h"
#include "gpu/HostFieldAllocator.h"
#include "gpu/ParallelStreams.h"
#include "gpu/communication/UniformGPUScheme.h"
#include <cuda_runtime.h>
#endif


namespace walberla
{
struct GeometryMovementStruct{
   //Translation Vector for timestep t for converting cell point from LBM space to geometry space in cells/timestep
   Vector3<real_t> translationVector;
   //Rotation Angle for timestep t for converting cell point from LBM space to geometry space in rad/timestep per dimension
   Vector3<real_t> rotationVector;
   //Maximum bounding box of the geometry over all timesteps. Used to only update blocks with movement in it. Set to domainAABB if unknown.
   AABB movementBoundingBox;
   //If translationVector or rotationAngle is dependent on the timestep, set to true. Update objectVelocityField only once if set to false.
   bool timeDependentMovement;
};


template < typename FractionField_T, typename VectorField_T, typename GeometryField_T = field::GhostLayerField< real_t, 1 > >
class MovingGeometry
{
   using GeometryFieldData_T = typename GeometryField_T::value_type;
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   typedef gpu::GPUField< GeometryFieldData_T > GeometryFieldGPU_T;
#endif
   using GeometryMovementFunction = std::function<GeometryMovementStruct (uint_t)>;

 public:

   MovingGeometry(shared_ptr< StructuredBlockForest >& blocks, shared_ptr< mesh::TriangleMesh >& mesh,
                  const BlockDataID fractionFieldId, const BlockDataID objectVelocityId,  const BlockDataID forceFieldId,
                  const GeometryMovementFunction & movementFunction,
                  shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>>& distOctree, const std::string meshName,
                  const uint_t superSamplingDepth, const uint_t ghostLayers, bool moving, bool useTauInFractionField, real_t omega, real_t dt)
      : blocks_(blocks), mesh_(mesh), fractionFieldId_(fractionFieldId), objectVelocityId_(objectVelocityId), forceFieldId_(forceFieldId),
        movementFunction_(movementFunction), distOctree_(distOctree), meshName_(meshName),
        superSamplingDepth_(superSamplingDepth), ghostLayers_(ghostLayers), moving_(moving), useTauInFractionField_(useTauInFractionField), tau_(1.0 / omega), dt_(dt)
   {

      auto geometryMovement = movementFunction_(0);
      if(useTauInFractionField_)
         if (omega > 2.0 || omega < 0.0)
            WALBERLA_ABORT("If you want to use the relaxation rate for building the fraction field, use a valid value for omega")

      auto meshCenterPoint = computeCentroid(*mesh_);
      meshCenter = Vector3<real_t> (meshCenterPoint[0], meshCenterPoint[1], meshCenterPoint[2]);
      meshAABB_ = computeAABB(*mesh_);
      const Vector3<real_t> dxyz = Vector3<real_t>(blocks_->dx(0), blocks_->dy(0), blocks_->dz(0));
      meshAABB_.extend(10.0 * dxyz);



      //Create one particle to use it for holding velocity and force information and for Euler integration
      particleStorage_             = walberla::make_shared< mesa_pd::data::ParticleStorage >(1);
      particleAccessor_            = walberla::make_shared< mesa_pd::data::ParticleAccessor >(particleStorage_);
      mesa_pd::data::Particle&& p = *particleStorage_->create();
      p.setPosition(meshCenter);



      if(moving_) {
         WcTimer simTimer;
         simTimer.start();
         buildGeometryField(geometryMovement.movementBoundingBox);
         simTimer.end();
         double time = simTimer.max();
         WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
         WALBERLA_LOG_INFO_ON_ROOT("Finished building Geometry Mesh in " << time << "s")

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         if(geometryField_) {
            geometryFieldGPU_ = new GeometryFieldGPU_T(geometryField_->xSize(), geometryField_->ySize(), geometryField_->zSize(), geometryField_->fSize(), geometryField_->nrOfGhostLayers(), geometryField_->layout(), true);
            gpu::fieldCpy(*geometryFieldGPU_, *geometryField_);
         }
#endif
         getFractionFieldFromGeometryMesh(0);
         updateObjectVelocityField(0);
      }
      else {
         staticFractionFieldId_ = field::addToStorage< FractionField_T >(blocks, "staticFractionField_" + meshName_, real_t(0.0), field::fzyx, ghostLayers_);
         buildStaticFractionField();
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         staticFractionFieldGPUId_ = gpu::addGPUFieldToStorage< FractionField_T >(blocks_, staticFractionFieldId_, "staticFractionFieldGPU_" + meshName_, true );
         gpu::fieldCpy< gpu::GPUField< real_t >, FractionField_T >(blocks, staticFractionFieldGPUId_, staticFractionFieldId_);
#endif
         addStaticGeometryToFractionField();
      }
   }


   void operator()(uint_t timestep) {
      if(moving_) {
         updateObjectPosition(timestep);
         getFractionFieldFromGeometryMesh(timestep);
         updateObjectVelocityField(timestep);
      }
      else {
         addStaticGeometryToFractionField();
      }
   }


   void updateObjectPosition(uint_t timestep) {
      auto geometryMovement = movementFunction_(timestep);
      auto newPosition = particleAccessor_->getPosition(0) + geometryMovement.translationVector * dt_;
      particleAccessor_->setPosition(0, newPosition);
      auto newParticleRotation = particleAccessor_->getRotation(0);
      newParticleRotation.rotate(-geometryMovement.rotationVector * dt_);
      particleAccessor_->setRotation(0, newParticleRotation);
   }



   void buildStaticFractionField() {
      const auto distFunct = make_shared<MeshDistanceFunction<mesh::DistanceOctree<mesh::TriangleMesh>>>( distOctree_ );

      for (auto& block : *blocks_)
      {
         FractionField_T* staticFractionField = block.getData< FractionField_T >(staticFractionFieldId_);
         auto level = blocks_->getLevel(block);
         const Vector3<real_t> dxyz = Vector3<real_t>(blocks_->dx(level), blocks_->dy(level), blocks_->dz(level));
         auto cellBBMesh = blocks_->getCellBBFromAABB( meshAABB_, level );

         CellInterval blockCi = staticFractionField->xyzSizeWithGhostLayer();
         blocks_->transformBlockLocalToGlobalCellInterval(blockCi, block);
         cellBBMesh.intersect(blockCi);

         std::queue< CellInterval > ciQueue;
         ciQueue.push(cellBBMesh);

         while (!ciQueue.empty())
         {
            const CellInterval& curCi = ciQueue.front();

            WALBERLA_ASSERT(!curCi.empty(), "Cell Interval: " << curCi);

            const AABB curAABB = blocks_->getAABBFromCellBB(curCi, level);

            WALBERLA_ASSERT(!curAABB.empty(), "AABB: " << curAABB);

            Vector3< real_t > cellCenter = curAABB.center();

            blocks_->mapToPeriodicDomain(cellCenter);
            const real_t sqSignedDistance = (*distFunct)(cellCenter);

            if (curCi.numCells() == uint_t(1))
            {
               //real_t fraction = recursiveSuperSampling(distFunct, cellCenter, dx, 0);
               real_t fraction;
               real_t sqDx = dxyz[0] * dxyz[0];
               real_t sqDxHalf = (0.5 * dxyz[0]) * (0.5 * dxyz[0]);
               fraction = std::max(0.0, std::min(1.0, (sqDx - (sqSignedDistance + sqDxHalf) ) / sqDx));

               //B2 from "A comparative study of fluid-particle coupling methods for fully resolved lattice Boltzmann simulations" from Rettinger et al
               if (useTauInFractionField_)
                  fraction = fraction * (tau_ - 0.5) / ((1.0 - fraction) + (tau_ - 0.5));

               Cell localCell;
               blocks_->transformGlobalToBlockLocalCell(localCell, block, curCi.min());
               staticFractionField->get(localCell) = fraction;

               ciQueue.pop();
               continue;
            }

            const real_t circumRadius   = curAABB.sizes().length() * real_t(0.5);
            const real_t sqCircumRadius = circumRadius * circumRadius;

            // the cell interval is fully covered by the mesh
            if (sqSignedDistance < -sqCircumRadius)
            {
               CellInterval localCi;
               blocks_->transformGlobalToBlockLocalCellInterval(localCi, block, curCi);
               std::fill(staticFractionField->beginSliceXYZ(localCi), staticFractionField->end(), 1.0);

               ciQueue.pop();
               continue;
            }
            // the cell interval is fully outside of mesh
            if (sqSignedDistance > sqCircumRadius)
            {
               ciQueue.pop();
               continue;
            }

            WALBERLA_ASSERT_GREATER(curCi.numCells(), uint_t(1));
            mesh::BoundarySetup::divideAndPushCellInterval(curCi, ciQueue);
            ciQueue.pop();
         }
      }
   }

   void buildGeometryField(AABB movementBoundingBox) {
      WALBERLA_LOG_PROGRESS("Getting max level for geometry field size")
      int maxLevel = -1;
      WALBERLA_LOG_PROGRESS("Size of blocks_ is " << blocks_->size())

      for (auto& block : *blocks_) {
         if(movementBoundingBox.intersects(block.getAABB())) {
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

      uint_t stencilSize = uint_t(pow(2, real_t(superSamplingDepth_)));
      Vector3<real_t> dxyzSS = maxRefinementDxyz_ / real_t(stencilSize);
      auto fieldSize = Vector3<uint_t> (uint_t(meshAABB_.xSize() / dxyzSS[0] ), uint_t(meshAABB_.ySize() / dxyzSS[1] ), uint_t(meshAABB_.zSize() / dxyzSS[2] ));
      WALBERLA_LOG_PROGRESS("Building geometry field with size " << fieldSize)
      geometryField_= make_shared< GeometryField_T >(fieldSize[0], fieldSize[1], fieldSize[2], uint_t(std::ceil(real_t(stencilSize) * 0.5 )), GeometryFieldData_T(0), field::fzyx);

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
         auto geometryMovement = movementFunction_(timestep);
         Vector3< real_t > translationVector = geometryMovement.translationVector * dt_ * real_t(vtk_frequency);
         Vector3< real_t > rotationVector = geometryMovement.rotationVector * dt_ * real_t(vtk_frequency);
         auto rotationMatrix = math::Rot3<real_t> (rotationVector);
         mesh::translate(*mesh_, translationVector);
         const Vector3< mesh::TriangleMesh::Scalar > axis_foot(particleAccessor_->getPosition(0));
         mesh::rotate(*mesh_, rotationMatrix.getMatrix(), axis_foot);
      }
   }


   Vector3<real_t> calculateForceOnBody() {
      Vector3<real_t> summedForceOnObject;
      for (auto &block : *blocks_) {
         VectorField_T* forceField = block.getData< VectorField_T >(forceFieldId_);
         FractionField_T* fractionField = block.getData< FractionField_T >(fractionFieldId_);
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(forceField,
            if(fractionField->get(x,y,z,0) > 0.0) {
               summedForceOnObject += Vector3(forceField->get(x,y,z,0), forceField->get(x,y,z,1), forceField->get(x,y,z,2));
            }
         )
      }
      WALBERLA_MPI_SECTION() {
         walberla::mpi::reduceInplace(summedForceOnObject, walberla::mpi::SUM);
      }
      return summedForceOnObject;
   }

   //TODO add distance from object center
   Vector3<real_t> calculateTorqueOnBody() {
      Vector3<real_t> summedForceOnObject;
      for (auto &block : *blocks_) {
         VectorField_T* forceField = block.getData< VectorField_T >(forceFieldId_);
         FractionField_T* fractionField = block.getData< FractionField_T >(fractionFieldId_);
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(forceField,
            if(fractionField->get(x,y,z,0) > 0.0) {
               summedForceOnObject += Vector3(forceField->get(x,y,z,0), forceField->get(x,y,z,1), forceField->get(x,y,z,2));
            }
         )
      }
      WALBERLA_MPI_SECTION() {
         walberla::mpi::reduceInplace(summedForceOnObject, walberla::mpi::SUM);
      }
      return summedForceOnObject;
   }


   void getFractionFieldFromGeometryMesh(uint_t timestep);
   void addStaticGeometryToFractionField();
   void resetFractionField();
   void updateObjectVelocityField(uint_t timestep);

 private:
   shared_ptr< StructuredBlockForest > blocks_;
   shared_ptr< mesh::TriangleMesh > mesh_;

   BlockDataID fractionFieldId_;
   shared_ptr <GeometryField_T> geometryField_; //One Field on every MPI process, not on every block
   BlockDataID staticFractionFieldId_;
   BlockDataID objectVelocityId_;
   BlockDataID forceFieldId_;


#if defined(WALBERLA_BUILD_WITH_CUDA)
   BlockDataID staticFractionFieldGPUId_;
   BlockDataID forceFieldGPUId_;
   GeometryFieldGPU_T *geometryFieldGPU_;
#endif

   GeometryMovementFunction movementFunction_;
   shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>> distOctree_;
   std::string meshName_;
   uint_t superSamplingDepth_;
   uint_t ghostLayers_;
   bool moving_;
   bool useTauInFractionField_;
   real_t tau_;
   real_t dt_;

   Vector3<real_t> meshCenter;
   AABB meshAABB_;
   Vector3<real_t> maxRefinementDxyz_;

   //particle objects
   shared_ptr< mesa_pd::data::ParticleStorage > particleStorage_;
   shared_ptr< mesa_pd::data::ParticleAccessor > particleAccessor_;
};

}//namespace waLBerla

#if defined(WALBERLA_BUILD_WITH_CUDA)
   #include "MovingGeometry.impl.gpu.h"
#else
   #include "MovingGeometry.impl.h"
#endif