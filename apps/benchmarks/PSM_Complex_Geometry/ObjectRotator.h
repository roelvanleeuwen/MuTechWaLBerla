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
//! \file ObjectRotator.h
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================
#pragma once

#include "blockforest/all.h"

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/FieldCopy.h"
#include "gpu/GPUWrapper.h"
#include "gpu/HostFieldAllocator.h"
#include "gpu/ParallelStreams.h"
#include "gpu/communication/UniformGPUScheme.h"
#endif

#include "geometry/containment_octree/ContainmentOctree.h"

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
#include "mesh/boundary/BoundaryLocation.h"
#include "mesh/boundary/BoundaryLocationFunction.h"
#include "mesh/boundary/BoundarySetup.h"
#include "mesh/boundary/BoundaryUIDFaceDataSource.h"
#include "mesh/boundary/ColorToBoundaryMapper.h"



namespace walberla
{
typedef field::GhostLayerField< real_t, 1 > ScalarField_T;
using fracSize = float;
typedef field::GhostLayerField< fracSize, 1 > FracField_T;

# define M_PI           3.14159265358979323846


class ObjectRotator
{
   typedef std::function< real_t(const Vector3< real_t >&) > DistanceFunction;
   typedef field::GhostLayerField< real_t, 3 > VectorField_T;

 public:
   ObjectRotator(shared_ptr< StructuredBlockForest >& blocks, shared_ptr< mesh::TriangleMesh >& mesh, const BlockDataID fractionFieldId, const BlockDataID fractionFieldGPUId,
                 const BlockDataID objectVelocityId, const real_t rotationAngle, const uint_t frequency, Vector3<int> rotationAxis, shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>>& distOctree, const bool preProcessedFractionFields, const bool rotate = true)
      : blocks_(blocks), mesh_(mesh), fractionFieldId_(fractionFieldId), fractionFieldGPUId_(fractionFieldGPUId),
        objectVelocityId_(objectVelocityId), rotationAngle_(rotationAngle), frequency_(frequency), distOctree_(distOctree),
        preProcessedFractionFields_(preProcessedFractionFields), rotate_(rotate), rotationAxis_(rotationAxis)
   {
      meshCenter = computeCentroid(*mesh_);
      initObjectVelocityField();
      if(!preProcessedFractionFields_) {
         getFractionFieldFromMesh(fractionFieldId_);
         //getFractionFieldFromMeshWithContainmentOctree(fractionFieldId_);

      }
   }

   void operator()(uint_t timestep) {
      if (timestep % frequency_ == 0)
      {
         if(rotate_) {
            WcTimer simTimer;
            simTimer.start();
            rotate();
            simTimer.end();
            double time = simTimer.max();
            WALBERLA_LOG_PROGRESS_ON_ROOT("Rotation needed " << time << " s")
            simTimer.reset();
            simTimer.start();
            resetFractionField(fractionFieldId_);
            simTimer.end();
            time = simTimer.max();
            WALBERLA_LOG_PROGRESS_ON_ROOT("Reset Fraction Field needed " << time << " s")
            simTimer.reset();
            simTimer.start();
            getFractionFieldFromMesh(fractionFieldId_);
            simTimer.end();
            time = simTimer.max();
            WALBERLA_LOG_PROGRESS_ON_ROOT("Voxelize Fraction Field needed " << time << " s")
         }
      }
   }

   void rotate()
   {
      const Vector3< mesh::TriangleMesh::Scalar > axis_foot(meshCenter[0], meshCenter[1], meshCenter[2]);
      mesh::rotate(*mesh_, rotationAxis_, rotationAngle_, axis_foot);

      distOctree_ = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(
         make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh_));
   }


   void resetFractionField(BlockDataID fractionFieldId)
   {
      auto aabbMesh = computeAABB(*mesh_);
      for (auto& block : *blocks_)
      {
         auto fractionField = block.getData< FracField_T >(fractionFieldId);
         auto level = blocks_->getLevel(block);
         auto cellBBMesh = blocks_->getCellBBFromAABB( aabbMesh, level );
         blocks_->transformGlobalToBlockLocalCellInterval(cellBBMesh, block);
         CellInterval blockCi = fractionField->xyzSizeWithGhostLayer();
         cellBBMesh.intersect(blockCi);
         std::fill(fractionField->beginSliceXYZ(cellBBMesh), fractionField->end(), 0.0);
      }
   }

   void initObjectVelocityField() {
      for (auto& block : *blocks_)
      {
         auto aabbMesh = computeAABB(*mesh_);
         auto level = blocks_->getLevel(block);
         auto cellBB = blocks_->getCellBBFromAABB( aabbMesh, level );

         auto objVelField = block.getData< VectorField_T >(objectVelocityId_);
         Vector3< real_t > angularVel;
         if (!rotate_)
            angularVel = Vector3< real_t >(0,0,0);
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

   void getFractionFieldFromMesh(BlockDataID fractionFieldId)
   {
      auto aabbMesh = computeAABB(*mesh_);
      DistanceFunction distFunct = makeMeshDistanceFunction(distOctree_);
      const real_t ONEOVEREIGHT = real_t(1) / real_t(8);
      const real_t ONEOVERFOUR = real_t(1) / real_t(4);

      for (auto& block : *blocks_)
      {
         FracField_T* fractionField = block.getData< FracField_T >(fractionFieldId);
         auto level = blocks_->getLevel(block);
         const real_t dx = blocks_->dx(level);
         auto cellBBMesh = blocks_->getCellBBFromAABB( aabbMesh, level );

         CellInterval blockCi = fractionField->xyzSizeWithGhostLayer();
         blocks_->transformBlockLocalToGlobalCellInterval(blockCi, block);
         cellBBMesh.intersect(blockCi);

         std::queue< CellInterval > ciQueue;
         ciQueue.push(cellBBMesh);

         while (!ciQueue.empty())
         {
            const CellInterval& curCi = ciQueue.front();

            WALBERLA_ASSERT(!curCi.empty(), "Cell Interval: " << curCi);

            AABB curAABB = blocks_->getAABBFromCellBB(curCi, level);

            WALBERLA_ASSERT(!curAABB.empty(), "AABB: " << curAABB);

            Vector3< real_t > cellCenter = curAABB.center();
            blocks_->mapToPeriodicDomain(cellCenter);
            real_t sqSignedDistance = distFunct(cellCenter);

            if (curCi.numCells() == uint_t(1))
            {
               //if only one cell left, split cell into 8 cell centers to get these cellCenter distances
               std::vector<int> xOffset{-1,-1,-1,-1, 1, 1, 1, 1};
               std::vector<int> yOffset{-1,-1, 1, 1,-1,-1, 1, 1};
               std::vector<int> zOffset{-1, 1,-1, 1,-1, 1,-1, 1};
               real_t fraction = 0;
               Vector3<real_t> octreeCenter;
               for (uint_t i = 0; i < 8; ++i) {
                  octreeCenter = Vector3<real_t>(cellCenter[0] + xOffset[i] * dx * ONEOVERFOUR, cellCenter[1] + yOffset[i] * dx * ONEOVERFOUR, cellCenter[2] + zOffset[i] * dx * ONEOVERFOUR);
                  sqSignedDistance = distFunct( octreeCenter );
                  if(sqSignedDistance  < real_t(0))
                     fraction += ONEOVEREIGHT;
               }
               Cell localCell;
               blocks_->transformGlobalToBlockLocalCell(localCell, block, curCi.min());
               fractionField->get(localCell) = fracSize(fraction);

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
               std::fill(fractionField->beginSliceXYZ(localCi), fractionField->end(), 1.0);

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
            divideAndPushCellInterval(curCi, ciQueue);

            ciQueue.pop();
         }
      }
   }


   void divideAndPushCellInterval(const CellInterval& ci, std::queue< CellInterval >& outputQueue)
   {
      WALBERLA_ASSERT(!ci.empty());

      Cell newMax(ci.xMin() + std::max(cell_idx_c(ci.xSize()) / cell_idx_t(2) - cell_idx_t(1), cell_idx_t(0)),
                  ci.yMin() + std::max(cell_idx_c(ci.ySize()) / cell_idx_t(2) - cell_idx_t(1), cell_idx_t(0)),
                  ci.zMin() + std::max(cell_idx_c(ci.zSize()) / cell_idx_t(2) - cell_idx_t(1), cell_idx_t(0)));

      WALBERLA_ASSERT(ci.contains(newMax));

      Cell newMin(newMax[0] + cell_idx_c(1), newMax[1] + cell_idx_c(1), newMax[2] + cell_idx_c(1));

      outputQueue.push(CellInterval(ci.xMin(), ci.yMin(), ci.zMin(), newMax[0], newMax[1], newMax[2]));
      if (newMin[2] <= ci.zMax())
         outputQueue.push(CellInterval(ci.xMin(), ci.yMin(), newMin[2], newMax[0], newMax[1], ci.zMax()));
      if (newMin[1] <= ci.yMax())
      {
         outputQueue.push(CellInterval(ci.xMin(), newMin[1], ci.zMin(), newMax[0], ci.yMax(), newMax[2]));
         if (newMin[2] <= ci.zMax())
            outputQueue.push(CellInterval(ci.xMin(), newMin[1], newMin[2], newMax[0], ci.yMax(), ci.zMax()));
      }
      if (newMin[0] <= ci.xMax())
      {
         outputQueue.push(CellInterval(newMin[0], ci.yMin(), ci.zMin(), ci.xMax(), newMax[1], newMax[2]));
         if (newMin[2] <= ci.zMax())
            outputQueue.push(CellInterval(newMin[0], ci.yMin(), newMin[2], ci.xMax(), newMax[1], ci.zMax()));
         if (newMin[1] <= ci.yMax())
         {
            outputQueue.push(CellInterval(newMin[0], newMin[1], ci.zMin(), ci.xMax(), ci.yMax(), newMax[2]));
            if (newMin[2] <= ci.zMax())
               outputQueue.push(CellInterval(newMin[0], newMin[1], newMin[2], ci.xMax(), ci.yMax(), ci.zMax()));
         }
      }
   }

 private:
   shared_ptr< StructuredBlockForest > blocks_;
   shared_ptr< mesh::TriangleMesh > mesh_;
   const BlockDataID fractionFieldId_;
   const BlockDataID fractionFieldGPUId_;
   const BlockDataID objectVelocityId_;
   const real_t rotationAngle_;
   const uint_t frequency_;
   shared_ptr<mesh::DistanceOctree<mesh::TriangleMesh>> distOctree_;
   const bool preProcessedFractionFields_;
   const bool rotate_;
   mesh::TriangleMesh::Point meshCenter;
   Vector3< mesh::TriangleMesh::Scalar > rotationAxis_;
};

}//namespace waLBerla