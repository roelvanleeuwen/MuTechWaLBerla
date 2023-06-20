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
//! \file PSM_Test.cpp
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
   ObjectRotator(shared_ptr< StructuredBlockForest >& blocks, shared_ptr< mesh::TriangleMesh >& mesh, const BlockDataID fractionFieldId,
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
                 const BlockDataID fractionFieldGPUId,
#endif
                 const BlockDataID objectVelocityId, const real_t rotationAngle, const uint_t frequency, DistanceFunction distOctree, const bool preProcessedFractionFields)
      : blocks_(blocks), mesh_(mesh), fractionFieldId_(fractionFieldId),
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
        fractionFieldGPUId_(fractionFieldGPUId),
#endif
        objectVelocityId_(objectVelocityId), rotationAngle_(rotationAngle), frequency_(frequency), distOctree_(distOctree),
        preProcessedFractionFields_(preProcessedFractionFields), counter(0), rotationAxis(0,-1,0)
   {
      meshCenter = computeCentroid(*mesh_);
      initObjectVelocityField();
      if(preProcessedFractionFields_) {
         preprocessMesh();
      }
   }

   void operator()() { rotate(); }

   void rotate()
   {
      if (counter % frequency_ == 0)
      {
         if(preProcessedFractionFields_) {
            syncFractionFieldFromVector();
         }
         else {
            const Vector3< mesh::TriangleMesh::Scalar > axis_foot(meshCenter[0], meshCenter[1], meshCenter[2]);
            mesh::rotate(*mesh_, rotationAxis, rotationAngle_, axis_foot);
            distOctree_ = makeMeshDistanceFunction(make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(
               make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh_)));
            getFractionFieldFromMesh(fractionFieldId_);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
            gpu::fieldCpy< gpu::GPUField< fracSize >, FracField_T >(blocks_, fractionFieldGPUId_, fractionFieldId_);
#endif
         }


      }
      counter += 1;
   }

   void resetFractionField()
   {
      for (auto& block : *blocks_)
      {
         auto fractionField = block.getData< FracField_T >(fractionFieldId_);
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(fractionField, fractionField->get(x, y, z) = 0.0;)
      }
   }

   void initObjectVelocityField() {
      for (auto& block : *blocks_)
      {
         auto level = blocks_->getLevel(block);
         auto objVelField = block.getData< VectorField_T >(objectVelocityId_);
         const Vector3< real_t > angularVel(rotationAxis[0] * rotationAngle_, rotationAxis[1] * rotationAngle_, rotationAxis[2] * rotationAngle_);
         const real_t dx = blocks_->dx(level);
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(objVelField,
            Cell cell(x,y,z);
            Vector3< real_t > cellCenter = blocks_->getCellCenter(cell, level);
            Vector3< real_t > distance((cellCenter[0] - meshCenter[0]) / dx, (cellCenter[1] - meshCenter[1]) / dx, (cellCenter[2] - meshCenter[2]) / dx);
            real_t velX = angularVel[1] * distance[2] - angularVel[2] * distance[1];
            real_t velY = angularVel[2] * distance[0] - angularVel[0] * distance[2];
            real_t velZ = angularVel[0] * distance[1] - angularVel[1] * distance[0];

            objVelField->get(cell, 0) = velX;
            objVelField->get(cell, 1) = velY;
            objVelField->get(cell, 2) = velZ;

         )
      }
   }


   void getFractionFieldFromMesh(BlockDataID fractionFieldId)
   {
      for (auto& block : *blocks_)
      {
         FracField_T* fractionField = block.getData< FracField_T >(fractionFieldId);

         CellInterval blockCi = fractionField->xyzSizeWithGhostLayer();
         blocks_->transformBlockLocalToGlobalCellInterval(blockCi, block);
         auto level = blocks_->getLevel(block);

         std::queue< CellInterval > ciQueue;
         ciQueue.push(blockCi);

         while (!ciQueue.empty())
         {
            const CellInterval& curCi = ciQueue.front();

            WALBERLA_ASSERT(!curCi.empty(), "Cell Interval: " << curCi);

            AABB curAABB = blocks_->getAABBFromCellBB(curCi, level);

            WALBERLA_ASSERT(!curAABB.empty(), "AABB: " << curAABB);

            Vector3< real_t > cellCenter = curAABB.center();
            blocks_->mapToPeriodicDomain(cellCenter);
            const real_t sqSignedDistance = distOctree_(cellCenter);

            if (curCi.numCells() == uint_t(1))
            {
               real_t distance;
               if (sqSignedDistance < 0)
               {
                  distance = sqrt(-sqSignedDistance);
                  distance *= -1;
               }
               else { distance = sqrt(sqSignedDistance); }
               Cell localCell;
               blocks_->transformGlobalToBlockLocalCell(localCell, block, curCi.min());
               fractionField->get(localCell) = fracSize(std::min(1.0, std::max(0.0, 1.0 - distance / blocks_->dx(level) + 0.5)));
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
               CellInterval localCi;
               blocks_->transformGlobalToBlockLocalCellInterval(localCi, block, curCi);
               std::fill(fractionField->beginSliceXYZ(localCi), fractionField->end(), 0.0);

               ciQueue.pop();
               continue;
            }

            WALBERLA_ASSERT_GREATER(curCi.numCells(), uint_t(1));
            divideAndPushCellInterval(curCi, ciQueue);

            ciQueue.pop();
         }
      }
   }


   void getBinaryFractionFieldFromMesh()
   {
      resetFractionField();

      for (auto& block : *blocks_)
      {
         FracField_T* fractionField = block.getData< FracField_T >(fractionFieldId_);

         CellInterval blockCi = fractionField->xyzSizeWithGhostLayer();
         blocks_->transformBlockLocalToGlobalCellInterval(blockCi, block);

         std::queue< CellInterval > ciQueue;
         ciQueue.push(blockCi);

         while (!ciQueue.empty())
         {
            const CellInterval& curCi = ciQueue.front();

            WALBERLA_ASSERT(!curCi.empty(), "Cell Interval: " << curCi);

            AABB curAABB = blocks_->getAABBFromCellBB(curCi, blocks_->getLevel(block));

            WALBERLA_ASSERT(!curAABB.empty(), "AABB: " << curAABB);

            Vector3< real_t > cellCenter = curAABB.center();
            blocks_->mapToPeriodicDomain(cellCenter);
            const real_t sqSignedDistance = distOctree_(cellCenter);

            if (curCi.numCells() == uint_t(1))
            {
               WALBERLA_LOG_INFO_ON_ROOT("Signed squared distance is " << sqSignedDistance)
               if ((sqSignedDistance < real_t(0)))
               {
                  Cell localCell;
                  blocks_->transformGlobalToBlockLocalCell(localCell, block, curCi.min());
                  fractionField->get(localCell) = uint8_t(1);
               }

               ciQueue.pop();
               continue;
            }

            const real_t circumRadius   = curAABB.sizes().length() * real_t(0.5);
            const real_t sqCircumRadius = circumRadius * circumRadius;

            if (sqSignedDistance < -sqCircumRadius)
            {
               // clearly the cell interval is fully covered by the mesh
               CellInterval localCi;
               blocks_->transformGlobalToBlockLocalCellInterval(localCi, block, curCi);
               // std::fill( fractionField->beginSliceXYZ( localCi ), fractionField->end(), uint8_t(1) );
               std::fill(fractionField->beginSliceXYZ(localCi), fractionField->end(), sqSignedDistance);

               ciQueue.pop();
               continue;
            }

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

   void preprocessMesh() {
      if(rotationAngle_ == 0.0 || frequency_ == 0)
         return;

      const Vector3< mesh::TriangleMesh::Scalar > axis_foot(meshCenter[0], meshCenter[1], meshCenter[2]);
      uint_t numFields = uint_c(std::round(2.0 * M_PI / rotationAngle_));
      for (uint_t i = 0; i < numFields; ++i) {

         mesh::rotate(*mesh_, rotationAxis, rotationAngle_, axis_foot);

         distOctree_ = makeMeshDistanceFunction(make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(
            make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh_)));

         BlockDataID fractionFieldId = field::addToStorage< FracField_T >(blocks_, "fractionFieldId_" + std::to_string(i), fracSize(0.0), field::fzyx);

         getFractionFieldFromMesh(fractionFieldId);
         fractionFieldIds_.push_back(fractionFieldId);
      }
   }

   void syncFractionFieldFromVector() {
      uint_t rotationState = (counter / frequency_) % fractionFieldIds_.size();

#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
      gpu::fieldCpy< gpu::GPUField< fracSize >, FracField_T >(blocks_, fractionFieldGPUId_, fractionFieldIds_[rotationState]);
#else
      for (auto & block : *blocks_) {
         FracField_T* realFractionField = block.getData< FracField_T >(fractionFieldId_);
         FracField_T* fractionFieldFromVector = block.getData< FracField_T >(fractionFieldIds_[rotationState]);
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(realFractionField,
            realFractionField->get(x,y,z) = fractionFieldFromVector->get(x,y,z);
         )
      }

#endif
   }

 private:
   shared_ptr< StructuredBlockForest > blocks_;
   shared_ptr< mesh::TriangleMesh > mesh_;
   const BlockDataID fractionFieldId_;
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   const BlockDataID fractionFieldGPUId_;
#endif
   const BlockDataID objectVelocityId_;
   const real_t rotationAngle_;
   const uint_t frequency_;
   DistanceFunction distOctree_;
   const bool preProcessedFractionFields_;
   uint_t counter;
   mesh::TriangleMesh::Point meshCenter;
   Vector3< mesh::TriangleMesh::Scalar > rotationAxis;
   std::vector<BlockDataID> fractionFieldIds_;
};

}//namespace waLBerla