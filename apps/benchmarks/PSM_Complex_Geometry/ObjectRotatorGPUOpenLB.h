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
//! \file ObjectRotatorGPUOpenLB.h
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
typedef field::GhostLayerField< real_t, 1 > ScalarField_T;
typedef field::GhostLayerField< real_t, 3 > VectorField_T;

typedef field::GhostLayerField< real_t, 1 > FracField_T;
typedef field::GhostLayerField< real_t, 1 > GeometryField_T;

template< typename GeometryField_T >
class GeometryFieldHandling : public field::BlockDataHandling< GeometryField_T >
{
 public:
   GeometryFieldHandling(const weak_ptr< StructuredBlockStorage >& blocks, AABB meshAABB)
      : blocks_(blocks), meshAABB_(meshAABB)
   {}

 protected:
   GeometryField_T* allocate(IBlock* const block) override { return allocateDispatch(block); }

   GeometryField_T* reallocate(IBlock* const block) override { return allocateDispatch(block); }

 private:
   weak_ptr< StructuredBlockStorage > blocks_;
   AABB meshAABB_;

   GeometryField_T* allocateDispatch(IBlock* const block)
   {
      WALBERLA_ASSERT_NOT_NULLPTR(block)
      auto blocks = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(blocks)
      auto level = blocks->getLevel(*block);
      const real_t dx = blocks->dx(level);

      auto fieldSize = Vector3<uint_t> (uint_t(meshAABB_.xSize() / dx ), uint_t(meshAABB_.ySize() / dx ), uint_t(meshAABB_.zSize() / dx ));
      return new GeometryField_T(fieldSize[0], fieldSize[1], fieldSize[2], uint_t(1), real_t(0), field::fzyx);
   }
}; // class GeometryFieldHandling


class ObjectRotatorGPUOpenLB
{
 public:
   ObjectRotatorGPUOpenLB(shared_ptr< StructuredBlockForest >& blocks, shared_ptr< mesh::TriangleMesh >& mesh,
                          BlockDataID fractionFieldGPUId, const BlockDataID objectVelocityId,
                          Vector3<real_t> translation, const real_t rotationAngle, const uint_t frequency,
                          Vector3<uint_t> rotationAxis, shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>>& distOctree,
                          std::string meshName, const bool isRotating = true)
      : blocks_(blocks), mesh_(mesh), fractionFieldGPUId_(fractionFieldGPUId), objectVelocityId_(objectVelocityId),translation_(translation),
        rotationAngle_(rotationAngle), frequency_(frequency), rotationAxis_(rotationAxis), distOctree_(distOctree),
        meshName_(meshName), isRotating_(isRotating)
   {
      auto meshCenterPoint = computeCentroid(*mesh_);
      meshCenter = Vector3<real_t> (meshCenterPoint[0], meshCenterPoint[1], meshCenterPoint[2]);
      meshAABB_ = computeAABB(*mesh_);
      const real_t dx = blocks_->dx(0);
      meshAABB_.extend(dx);

      if(isRotating_) {
         initObjectVelocityField();
         std::shared_ptr< GeometryFieldHandling< GeometryField_T > > geometryFieldDataHandling = std::make_shared< GeometryFieldHandling< GeometryField_T > >(blocks_, meshAABB_);
         geometryFieldId_ = blocks_->addBlockData( geometryFieldDataHandling, "geometryField_" + meshName_ );
         geometryFieldGPUId_ = gpu::addGPUFieldToStorage< FracField_T >(blocks_, geometryFieldId_, "geometryFieldGPU_" + meshName_, true );
         buildGeometryMesh();
         gpu::fieldCpy< gpu::GPUField< real_t >, GeometryField_T >(blocks, geometryFieldGPUId_, geometryFieldId_);
         getFractionFieldFromGeometryMesh(0);
      }
      else {
         staticFractionFieldId_ = field::addToStorage< FracField_T >(blocks, "staticFractionField_" + meshName_, real_t(0.0), field::fzyx, uint_c(1));
         staticFractionFieldGPUId_ = gpu::addGPUFieldToStorage< FracField_T >(blocks_, staticFractionFieldId_, "staticFractionFieldGPU_" + meshName_, true );
         buildStaticFractionField();
         gpu::fieldCpy< gpu::GPUField< real_t >, FracField_T >(blocks, staticFractionFieldGPUId_, staticFractionFieldId_);
         addStaticGeometryToFractionField();
      }
   }

   void operator()(uint_t timestep) {
      if(isRotating_) {
         if (timestep % frequency_ == 0)
            getFractionFieldFromGeometryMesh(timestep);
      }
      else {
         if (timestep % frequency_ == 0)
            addStaticGeometryToFractionField();
      }
   }

   void getFractionFieldFromGeometryMesh(uint_t timestep);

   void addStaticGeometryToFractionField();

   void moveTriangleMesh(uint_t timestep, uint_t vtk_frequency) {
      if(vtk_frequency > 0 && timestep % vtk_frequency == 0 && isRotating_) {
         mesh::translate(*mesh_, translation_ * vtk_frequency);
         const Vector3< mesh::TriangleMesh::Scalar > axis_foot(meshCenter[0] + real_t(timestep+vtk_frequency) * translation_[0],
                                                               meshCenter[1] + real_t(timestep+vtk_frequency) * translation_[1],
                                                               meshCenter[2] + real_t(timestep+vtk_frequency) * translation_[2]);
         mesh::rotate(*mesh_, rotationAxis_, rotationAngle_ * vtk_frequency, axis_foot);
      }
   }

   void initObjectVelocityField() {
      for (auto& block : *blocks_)
      {
         auto level = blocks_->getLevel(block);
         auto cellBB = blocks_->getCellBBFromAABB( meshAABB_, level );

         auto objVelField = block.getData< VectorField_T >(objectVelocityId_);
         Vector3< real_t > angularVel;

         if (isRotating_ == false || frequency_ == 0) {
            angularVel = Vector3< real_t >(0,0,0);
         }
         else
            angularVel = Vector3< real_t > (rotationAxis_[0] * rotationAngle_ / real_c(frequency_), rotationAxis_[1] * rotationAngle_ / real_c(frequency_), rotationAxis_[2] * rotationAngle_ / real_c(frequency_));

         const real_t dx = blocks_->dx(level);
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(objVelField,
                                                          Cell cell(x,y,z);
                                                          blocks_->transformBlockLocalToGlobalCell(cell, block);

                                                          // Set velocity only in aabb of mesh for x dir
                                                          if(cell[0] < cellBB.xMin() || cell[0] > cellBB.xMax())
                                                             continue;

                                                          Vector3< real_t > cellCenter = blocks_->getCellCenter(cell, level);
                                                          Vector3< real_t > distance((cellCenter[0] - meshCenter[0]) / dx, (cellCenter[1] - meshCenter[1]) / dx, (cellCenter[2] - meshCenter[2]) / dx);
                                                          real_t velX = angularVel[1] * distance[2] - angularVel[2] * distance[1];
                                                          real_t velY = angularVel[2] * distance[0] - angularVel[0] * distance[2];
                                                          real_t velZ = angularVel[0] * distance[1] - angularVel[1] * distance[0];

                                                          blocks_->transformGlobalToBlockLocalCell(cell, block);
                                                          objVelField->get(cell, 0) = velX;
                                                          objVelField->get(cell, 1) = velY;
                                                          objVelField->get(cell, 2) = velZ;
         )
      }
   }

   void resetFractionField();

   void buildStaticFractionField() {
      const auto distFunct = make_shared<MeshDistanceFunction<mesh::DistanceOctree<mesh::TriangleMesh>>>( distOctree_ );

      for (auto& block : *blocks_)
      {
         FracField_T* staticFractionField = block.getData< FracField_T >(staticFractionFieldId_);
         auto level = blocks_->getLevel(block);
         const real_t dx = blocks_->dx(level);
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
               real_t sqDx = dx * dx;
               real_t sqDxHalf = (0.5 * dx) * (0.5 * dx);
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

   void buildGeometryMesh() {
      const auto distFunct = make_shared<MeshDistanceFunction<mesh::DistanceOctree<mesh::TriangleMesh>>>( distOctree_ );
      for (auto& block : *blocks_)
      {
         GeometryField_T * geometryField = block.getData< GeometryField_T >(geometryFieldId_);

         CellInterval blockCi = geometryField->xyzSizeWithGhostLayer();

         auto level = blocks_->getLevel(block);
         const real_t dx = blocks_->dx(level);
         real_t sqDx = dx * dx;
         real_t sqDxHalf = (0.5 * dx) * (0.5 * dx);

         for (auto cellIt = blockCi.begin(); cellIt != blockCi.end(); cellIt++) {
            Vector3<real_t> cellCenter = meshAABB_.min() + Vector3<real_t> (cellIt->x() * dx, cellIt->y() * dx, cellIt->z() * dx) + Vector3<real_t> (0.5 * dx);
            const real_t sqSignedDistance = (*distFunct)(cellCenter);
            real_t fraction = std::max(0.0, std::min(1.0, (sqDx - (sqSignedDistance + sqDxHalf) ) / sqDx));
            geometryField->get(*cellIt) = fraction;
         }
      }
   }

 private:
   shared_ptr< StructuredBlockForest > blocks_;
   shared_ptr< mesh::TriangleMesh > mesh_;

   BlockDataID fractionFieldGPUId_;
   BlockDataID geometryFieldId_;
   BlockDataID staticFractionFieldId_;

   BlockDataID geometryFieldGPUId_;
   BlockDataID staticFractionFieldGPUId_;

   const BlockDataID objectVelocityId_;
   Vector3<real_t> translation_;
   const real_t rotationAngle_;
   const uint_t frequency_;
   Vector3< mesh::TriangleMesh::Scalar > rotationAxis_;
   shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>> distOctree_;
   std::string meshName_;
   const bool isRotating_;
   Vector3<real_t> meshCenter;
   AABB meshAABB_;
};
}//namespace waLBerla