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

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif


namespace walberla
{
using geoSize = bool;
typedef field::GhostLayerField< real_t, 3 > VectorField_T;

typedef field::GhostLayerField< real_t, 1 > FracField_T;
typedef field::GhostLayerField< geoSize, 1 > GeometryField_T;
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
typedef gpu::GPUField< geoSize > GeometryFieldGPU_T;
#endif

struct GeometryMovementStruct{

   //Translation Vector for timestep t for converting cell point from LBM space to geometry space
   Vector3<real_t> translationVector;
   //Rotation Angle for timestep t for converting cell point from LBM space to geometry space
   real_t rotationAngle;
   Vector3< mesh::TriangleMesh::Scalar > rotationAxis;
   //Maximum bounding box of the geometry over all timesteps. Set to domainAABB if unknown
   AABB movementBoundingBox;
   //If translationVector or rotationAngle is dependent on the timestep, set to true
   bool timeDependentMovement;
};



class MovingGeometry
{
 public:
   enum MovementType { STILL = 0, ROTATING = 1, TRANSLATING = 2 };
   using GeometryMovementFunction = std::function<GeometryMovementStruct (uint_t)>;

   MovingGeometry(shared_ptr< StructuredBlockForest >& blocks, shared_ptr< mesh::TriangleMesh >& mesh,
                  const BlockDataID fractionFieldId, const BlockDataID objectVelocityId,
                  const GeometryMovementFunction & movementFunction,
                  shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>>& distOctree, const std::string meshName,
                  const uint_t superSamplingDepth, const uint_t ghostLayers, MovementType movementType)
      : blocks_(blocks), mesh_(mesh), fractionFieldId_(fractionFieldId), objectVelocityId_(objectVelocityId),
        movementFunction_(movementFunction), distOctree_(distOctree), meshName_(meshName),
        superSamplingDepth_(superSamplingDepth), ghostLayers_(ghostLayers), movementType_(movementType)
   {

      auto geometryMovement = movementFunction_(0);
      if (movementType_ == 0  && geometryMovement.rotationAngle > 0)
         WALBERLA_ABORT("Geometry Mevement Type is " << movementType_ << " but rotation angle is " << geometryMovement.rotationAngle)
      if (movementType_ == 0  && geometryMovement.translationVector.length() > 0)
         WALBERLA_ABORT("Geometry Mevement Type is " << movementType_ << " but translation vector is " << geometryMovement.translationVector)

      auto meshCenterPoint = computeCentroid(*mesh_);
      meshCenter = Vector3<real_t> (meshCenterPoint[0], meshCenterPoint[1], meshCenterPoint[2]);
      meshAABB_ = computeAABB(*mesh_);
      const Vector3<real_t> dxyz = Vector3<real_t>(blocks_->dx(0), blocks_->dy(0), blocks_->dz(0));
      meshAABB_.extend(dxyz);

      if(movementType_ > 0) {
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
         staticFractionFieldId_ = field::addToStorage< FracField_T >(blocks, "staticFractionField_" + meshName_, real_t(0.0), field::fzyx, ghostLayers_);
         buildStaticFractionField();
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
         staticFractionFieldGPUId_ = gpu::addGPUFieldToStorage< FracField_T >(blocks_, staticFractionFieldId_, "staticFractionFieldGPU_" + meshName_, true );
         gpu::fieldCpy< gpu::GPUField< real_t >, FracField_T >(blocks, staticFractionFieldGPUId_, staticFractionFieldId_);
#endif
         addStaticGeometryToFractionField();
      }
   }

   void operator()(uint_t timestep) {
      if(movementType_  > 0) {
         getFractionFieldFromGeometryMesh(timestep);
         updateObjectVelocityField(timestep);
      }
      else {
         addStaticGeometryToFractionField();
      }
   }

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   void getFractionFieldFromGeometryMesh(uint_t timestep);
   void addStaticGeometryToFractionField();
   void resetFractionField();
   void updateObjectVelocityField(uint_t timestep);
#else

   void getFractionFieldFromGeometryMesh(uint_t timestep) {
      auto geometryMovement = movementFunction_(timestep);
      Matrix3<real_t>rotationMatrix(geometryMovement.rotationAxis, real_t(timestep) * -geometryMovement.rotationAngle);

      uint_t interpolationStencilSize = uint_t( pow(2, real_t(superSamplingDepth_)) + 1);
      auto oneOverInterpolArea = 1.0 / real_t( interpolationStencilSize * interpolationStencilSize * interpolationStencilSize);

      for (auto& block : *blocks_)
      {
         if(!geometryMovement.movementBoundingBox.intersects(block.getAABB()) )
            continue;

         FracField_T* fractionField = block.getData< FracField_T >(fractionFieldId_);
         auto level = blocks_->getLevel(block);
         Vector3<real_t> dxyzSS = maxRefinementDxyz_ / pow(2, real_t(superSamplingDepth_));
         CellInterval blockCi = fractionField->xyzSizeWithGhostLayer();

#ifdef _OPENMP
         #pragma omp parallel for schedule(static)
#endif
         for (cell_idx_t cellZ = blockCi.zMin(); cellZ < blockCi.zMax(); ++cellZ) {
            for (cell_idx_t cellY = blockCi.yMin(); cellY < blockCi.yMax(); ++cellY) {
               for (cell_idx_t cellX = blockCi.xMin(); cellX < blockCi.xMax(); ++cellX) {
                  Cell cell(cellX, cellY, cellZ);
                  Cell globalCell;
                  blocks_->transformBlockLocalToGlobalCell(globalCell, block, cell);
                  Vector3< real_t > cellCenter = blocks_->getCellCenter(globalCell, level);

                  // translation
                  cellCenter -= geometryMovement.translationVector;

                  // rotation
                  cellCenter -= meshCenter;
                  cellCenter = rotationMatrix * cellCenter;
                  cellCenter += meshCenter;

                  // get corresponding geometryField_ cell
                  auto pointInGeometrySpace = cellCenter - meshAABB_.min() - 0.5 * dxyzSS;
                  auto cellInGeometrySpace  = Vector3< int32_t >(int32_t(pointInGeometrySpace[0] / dxyzSS[0]),
                                                                int32_t(pointInGeometrySpace[1] / dxyzSS[1]),
                                                                int32_t(pointInGeometrySpace[2] / dxyzSS[2]));
                  real_t fraction           = 0.0;

                  // rotated cell outside of geometry field
                  if (cellInGeometrySpace[0] < 0 ||
                      cellInGeometrySpace[0] >= int32_t(geometryField_->xSize()) ||
                      cellInGeometrySpace[1] < 0 ||
                      cellInGeometrySpace[1] >= int32_t(geometryField_->ySize()) ||
                      cellInGeometrySpace[2] < 0 || cellInGeometrySpace[2] >= int32_t(geometryField_->zSize()))
                  {
                     fraction = 0.0;
                  }
                  // 2x2x2 interpolation for superSamplingDepth_=0
                  else if (superSamplingDepth_ == 0)
                  {
                     auto cellCenterInGeometrySpace = Vector3< real_t > (cellInGeometrySpace[0] * dxyzSS[0], cellInGeometrySpace[1] * dxyzSS[1], cellInGeometrySpace[2] * dxyzSS[2]);
                     auto distanceToCellCenter      = pointInGeometrySpace - cellCenterInGeometrySpace;
                     auto offset = Vector3< int >(int(distanceToCellCenter[0] / abs(distanceToCellCenter[0])),
                                                  int(distanceToCellCenter[1] / abs(distanceToCellCenter[1])),
                                                  int(distanceToCellCenter[2] / abs(distanceToCellCenter[2])));

                     Vector3< int > iterationStart =
                        Vector3< int >((offset[0] < 0) ? -1 : 0, (offset[1] < 0) ? -1 : 0, (offset[2] < 0) ? -1 : 0);
                     Vector3< int > iterationEnd = iterationStart + Vector3< int >(interpolationStencilSize);

                     for (int z = iterationStart[2]; z < iterationEnd[2]; ++z) {
                        for (int y = iterationStart[1]; y < iterationEnd[1]; ++y) {
                           for (int x = iterationStart[0]; x < iterationEnd[0]; ++x) {
                              fraction += geometryField_->get(cellInGeometrySpace[0] + x, cellInGeometrySpace[1] + y, cellInGeometrySpace[2] + z);
                           }
                        }
                     }
                     fraction *= oneOverInterpolArea;
                  }
                  // interpolate with stencil sizes 3x3x3 , 5x5x5, ...
                  else
                  {
                     int halfInterpolationStencilSize = int(real_t(interpolationStencilSize) * 0.5);
                     for (int z = -halfInterpolationStencilSize; z <= halfInterpolationStencilSize; ++z)
                     {
                        for (int y = -halfInterpolationStencilSize; y <= halfInterpolationStencilSize; ++y)
                        {
                           for (int x = -halfInterpolationStencilSize; x <= halfInterpolationStencilSize; ++x)
                           {
                              fraction += geometryField_->get(
                                 cellInGeometrySpace[0] + x, cellInGeometrySpace[1] + y, cellInGeometrySpace[2] + z);
                           }
                        }
                     }
                     fraction *= oneOverInterpolArea;
                  }
                  fractionField->get(cell) = std::min(1.0, fractionField->get(cell) + fraction);
               }
            }
         }
      }
   }

   void addStaticGeometryToFractionField() {
      for (auto& block : *blocks_)
      {
         FracField_T* fractionField       = block.getData< FracField_T >(fractionFieldId_);
         FracField_T* staticFractionField = block.getData< FracField_T >(staticFractionFieldId_);
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(
            fractionField, fractionField->get(x, y, z) =
                              std::min(1.0, fractionField->get(x, y, z) + staticFractionField->get(x, y, z));)
      }
   }

   void resetFractionField() {
      for (auto& block : *blocks_)
      {
         FracField_T* fractionField = block.getData< FracField_T >(fractionFieldId_);
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(fractionField, fractionField->get(x, y, z) = 0.0;)
      }
   }


   void updateObjectVelocityField(uint_t timestep) {
      auto geometryMovement = movementFunction_(timestep+1);
      auto geometryMovementLastTimestep = movementFunction_(timestep);
      const Vector3<real_t> dxyz_root = Vector3<real_t>(blocks_->dx(0), blocks_->dy(0), blocks_->dz(0));
      geometryMovement.movementBoundingBox.extend(dxyz_root);

      //update object velocity field only on 0th timestep for time independent movement
      if(!geometryMovement.timeDependentMovement && timestep > 0)
         return;

      auto rotationSpeed = geometryMovement.rotationAngle - geometryMovementLastTimestep.rotationAngle;
      auto translationSpeed = geometryMovement.translationVector - geometryMovementLastTimestep.translationVector;
      for (auto& block : *blocks_)
      {
         if(!geometryMovement.movementBoundingBox.intersects(block.getAABB()) )
            continue;

         auto level = blocks_->getLevel(block);
         const Vector3<real_t> dxyz = Vector3<real_t>(blocks_->dx(level), blocks_->dy(level), blocks_->dz(level));

         auto objVelField = block.getData< VectorField_T >(objectVelocityId_);
         auto fractionField = block.getData< FracField_T >(fractionFieldId_);

         Vector3< real_t > angularVel;

         angularVel = Vector3< real_t > (geometryMovement.rotationAxis[0] * rotationSpeed, geometryMovement.rotationAxis[1] * rotationSpeed, geometryMovement.rotationAxis[2] * rotationSpeed);

         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(objVelField,
            Cell cell(x,y,z);
            if(geometryMovement.timeDependentMovement) {
               if (fractionField->get(cell, 0) <= 0.0)
               {
                  objVelField->get(cell, 0) = 0;
                  objVelField->get(cell, 1) = 0;
                  objVelField->get(cell, 2) = 0;
                  continue;
               }
            }

            blocks_->transformBlockLocalToGlobalCell(cell, block);
            if(!geometryMovement.timeDependentMovement) {
               auto cellAABB = blocks_->getCellAABB(cell, level);
               if(!cellAABB.intersects(geometryMovement.movementBoundingBox))
                  continue;
            }


            Vector3< real_t > cellCenter = blocks_->getCellCenter(cell, level);
            Vector3< real_t > distance((cellCenter[0] - meshCenter[0]) / dxyz[0], (cellCenter[1] - meshCenter[1]) / dxyz[1], (cellCenter[2] - meshCenter[2]) / dxyz[2]);

            real_t velX = angularVel[1] * distance[2] - angularVel[2] * distance[1];
            real_t velY = angularVel[2] * distance[0] - angularVel[0] * distance[2];
            real_t velZ = angularVel[0] * distance[1] - angularVel[1] * distance[0];

            blocks_->transformGlobalToBlockLocalCell(cell, block);
            objVelField->get(cell, 0) = velX + translationSpeed[0] / dxyz[0];
            objVelField->get(cell, 1) = velY + translationSpeed[1] / dxyz[1];
            objVelField->get(cell, 2) = velZ + translationSpeed[2] / dxyz[2];
         )
      }
   }

#endif
   //TODO
   void moveTriangleMesh(uint_t timestep, uint_t vtk_frequency) {
      if(vtk_frequency > 0 && timestep % vtk_frequency == 0 && movementType_ > 0) {
         auto geometryMovement = movementFunction_(timestep);
         uint_t old_timestep;
         if (timestep == 0) old_timestep = 0;
         else old_timestep = timestep - vtk_frequency;
         auto geometryMovementLastTimestep = movementFunction_( old_timestep );
         auto rotationSpeed = geometryMovement.rotationAngle - geometryMovementLastTimestep.rotationAngle;
         auto translationSpeed = geometryMovement.translationVector - geometryMovementLastTimestep.translationVector;

         mesh::translate(*mesh_, translationSpeed);
         const Vector3< mesh::TriangleMesh::Scalar > axis_foot(meshCenter[0] + geometryMovement.translationVector[0],
                                                               meshCenter[1] + geometryMovement.translationVector[1],
                                                               meshCenter[2] + geometryMovement.translationVector[2]);
         mesh::rotate(*mesh_, geometryMovement.rotationAxis, rotationSpeed, axis_foot);
      }
   }



   void buildStaticFractionField() {
      const auto distFunct = make_shared<MeshDistanceFunction<mesh::DistanceOctree<mesh::TriangleMesh>>>( distOctree_ );

      for (auto& block : *blocks_)
      {
         FracField_T* staticFractionField = block.getData< FracField_T >(staticFractionFieldId_);
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
      geometryField_= make_shared< GeometryField_T >(fieldSize[0], fieldSize[1], fieldSize[2], uint_t(std::ceil(real_t(stencilSize) * 0.5 )), geoSize(0), field::fzyx);

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

               geoSize fraction = geoSize(std::max(0.0, std::min(1.0, (sqDx - (sqSignedDistance + sqDxHalf)) / sqDx)));
               geometryField_->get(cell) = fraction;
            }
         }
      }
   }


 private:
   shared_ptr< StructuredBlockForest > blocks_;
   shared_ptr< mesh::TriangleMesh > mesh_;

   BlockDataID fractionFieldId_;
   shared_ptr <GeometryField_T> geometryField_;
   BlockDataID staticFractionFieldId_;
   BlockDataID objectVelocityId_;

#if defined(WALBERLA_BUILD_WITH_CUDA)
   BlockDataID staticFractionFieldGPUId_;
   GeometryFieldGPU_T *geometryFieldGPU_;
#endif

   GeometryMovementFunction movementFunction_;
   shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>> distOctree_;
   std::string meshName_;
   uint_t superSamplingDepth_;
   uint_t ghostLayers_;
   MovementType movementType_;
   Vector3<real_t> meshCenter;
   AABB meshAABB_;
   Vector3<real_t> maxRefinementDxyz_;
};
}//namespace waLBerla