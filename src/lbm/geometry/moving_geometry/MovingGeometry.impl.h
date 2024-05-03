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
//! \file MovingGeometry.impl.h
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================
#pragma once

#include "MovingGeometry.h"

namespace walberla
{

template < typename FractionField_T, typename VectorField_T, typename GeometryField_T >
void MovingGeometry<FractionField_T, VectorField_T, GeometryField_T>::getFractionFieldFromGeometryMesh(uint_t timestep) {
   auto geometryMovement = movementFunction_(timestep);
   Matrix3<real_t> rotationMatrix = particleAccessor_->getRotation(0).getMatrix();
   auto translationVector = particleAccessor_->getPosition(0);

   uint_t interpolationStencilSize = uint_t( pow(2, real_t(superSamplingDepth_)) + 1);
   auto oneOverInterpolArea = 1.0 / real_t( interpolationStencilSize * interpolationStencilSize * interpolationStencilSize);

   for (auto& block : *blocks_)
   {
      if(!geometryMovement.movementBoundingBox.intersects(block.getAABB()) )
         continue;

      FractionField_T* fractionField = block.getData< FractionField_T >(fractionFieldId_);
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
               auto pointInGeometrySpace = cellCenter - translationVector;
               // rotation
               pointInGeometrySpace -= meshCenter;
               pointInGeometrySpace = rotationMatrix * pointInGeometrySpace;
               pointInGeometrySpace += meshCenter;

               // get corresponding geometryField_ cell
               pointInGeometrySpace = pointInGeometrySpace - meshAABB_.min() - 0.5 * dxyzSS;
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
               //B2 from "A comparative study of fluid-particle coupling methods for fully resolved lattice Boltzmann simulations" from Rettinger et al
               if (useTauInFractionField_)
                  fraction = fraction * (tau_ - 0.5) / ((1.0 - fraction) + (tau_ - 0.5));

               fractionField->get(cell) = std::min(1.0, fractionField->get(cell) + fraction);
            }
         }
      }
   }
}



template < typename FractionField_T, typename VectorField_T, typename GeometryField_T >
void MovingGeometry<FractionField_T, VectorField_T, GeometryField_T>::addStaticGeometryToFractionField() {
   for (auto& block : *blocks_)
   {
      FractionField_T* fractionField       = block.getData< FractionField_T >(fractionFieldId_);
      FractionField_T* staticFractionField = block.getData< FractionField_T >(staticFractionFieldId_);
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(
         fractionField, fractionField->get(x, y, z) =
                           std::min(1.0, fractionField->get(x, y, z) + staticFractionField->get(x, y, z));)
   }
}



template < typename FractionField_T, typename VectorField_T, typename GeometryField_T >
void MovingGeometry<FractionField_T, VectorField_T, GeometryField_T>::resetFractionField() {
   for (auto& block : *blocks_)
   {
      FractionField_T* fractionField = block.getData< FractionField_T >(fractionFieldId_);
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(fractionField, fractionField->get(x, y, z) = 0.0;)
   }
}



template < typename FractionField_T, typename VectorField_T, typename GeometryField_T >
void MovingGeometry<FractionField_T, VectorField_T, GeometryField_T>::updateObjectVelocityField(uint_t timestep) {
   auto geometryMovement = movementFunction_(timestep);
   auto objectPosition = particleAccessor_->getPosition(0);
   const Vector3<real_t> dxyz_root = Vector3<real_t>(blocks_->dx(0), blocks_->dy(0), blocks_->dz(0));
   geometryMovement.movementBoundingBox.extend(dxyz_root);

   //update object velocity field only on 0th timestep for time independent movement
   if(!geometryMovement.timeDependentMovement && timestep > 0)
      return;

   for (auto& block : *blocks_)
   {
      if(!geometryMovement.movementBoundingBox.intersects(block.getAABB()) )
         continue;

      auto level = blocks_->getLevel(block);

      auto objVelField = block.getData< VectorField_T >(objectVelocityId_);
      auto fractionField = block.getData< FractionField_T >(fractionFieldId_);

      Vector3< real_t > angularVel = geometryMovement.rotationVector;

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
         Vector3< real_t > distance = cellCenter - objectPosition;

         real_t linearVelX = angularVel[1] * distance[2] - angularVel[2] * distance[1];
         real_t linearVelY = angularVel[2] * distance[0] - angularVel[0] * distance[2];
         real_t linearVelZ = angularVel[0] * distance[1] - angularVel[1] * distance[0];

         blocks_->transformGlobalToBlockLocalCell(cell, block);
         objVelField->get(cell, 0) = (linearVelX + geometryMovement.translationVector[0]) * dt_ / dxyz_root[0];
         objVelField->get(cell, 1) = (linearVelY + geometryMovement.translationVector[1]) * dt_ / dxyz_root[1];
         objVelField->get(cell, 2) = (linearVelZ + geometryMovement.translationVector[2]) * dt_ / dxyz_root[2];
      )
   }
}

template < typename FractionField_T, typename VectorField_T, typename GeometryField_T >
void MovingGeometry<FractionField_T, VectorField_T, GeometryField_T>::calculateForcesOnBody() {
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
   particleAccessor_->setHydrodynamicForce(0, summedForceOnObject);
   //TODO calculate Torque
}

} // namespace waLBerla