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

#include "MovingGeometry.h"

namespace walberla
{

template < typename FractionField_T, typename VectorField_T >
void MovingGeometry<FractionField_T, VectorField_T>::getFractionFieldFromGeometryMesh() {
   Matrix3<real_t> rotationMatrix = particleAccessor_->getRotation(0).getMatrix();
   auto translationVector = particleAccessor_->getPosition(0) - meshCenter_;

   uint_t interpolationStencilSize = uint_t( pow(2, real_t(superSamplingDepth_)) + 1);
   auto oneOverInterpolArea = 1.0 / real_t( interpolationStencilSize * interpolationStencilSize * interpolationStencilSize);

   for (auto& block : *blocks_)
   {
      if(!movementBoundingBox_.intersects(block.getAABB()) )
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
               pointInGeometrySpace -= meshCenter_;
               pointInGeometrySpace = rotationMatrix * pointInGeometrySpace;
               pointInGeometrySpace += meshCenter_;

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
                   cellInGeometrySpace[2] < 0 ||
                   cellInGeometrySpace[2] >= int32_t(geometryField_->zSize()))
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


template < typename FractionField_T, typename VectorField_T >
void MovingGeometry<FractionField_T, VectorField_T>::resetFractionField() {
   for (auto& block : *blocks_)
   {
      FractionField_T* fractionField = block.getData< FractionField_T >(fractionFieldId_);
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(fractionField, fractionField->get(x, y, z) = 0.0;)
   }
}

template < typename FractionField_T, typename VectorField_T >
void MovingGeometry<FractionField_T, VectorField_T>::updateObjectVelocityField() {
   auto objectPosition = particleAccessor_->getPosition(0);
   auto objectLinearVelocity = particleAccessor_->getLinearVelocity(0);
   auto objectAngularVelocity = particleAccessor_->getAngularVelocity(0);
   const Vector3<real_t> dxyz_root = Vector3<real_t>(blocks_->dx(0), blocks_->dy(0), blocks_->dz(0));

   for (auto& block : *blocks_)
   {
      if(!movementBoundingBox_.intersects(block.getAABB()) )
         continue;

      auto level = blocks_->getLevel(block);

      auto objVelField = block.getData< VectorField_T >(objectVelocityId_);
      auto fractionField = block.getData< FractionField_T >(fractionFieldId_);


      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(objVelField,
         Cell cell(x,y,z);
         if (fractionField->get(cell, 0) <= 0.0)
         {
            objVelField->get(cell, 0) = 0;
            objVelField->get(cell, 1) = 0;
            objVelField->get(cell, 2) = 0;
            continue;
         }

         blocks_->transformBlockLocalToGlobalCell(cell, block);
         auto cellAABB = blocks_->getCellAABB(cell, level);
         if(!cellAABB.intersects(movementBoundingBox_))
            continue;

         Vector3< real_t > cellCenter = blocks_->getCellCenter(cell, level);
         Vector3< real_t > distance = cellCenter - objectPosition;

         real_t linearVelX = objectAngularVelocity[1] * distance[2] - objectAngularVelocity[2] * distance[1];
         real_t linearVelY = objectAngularVelocity[2] * distance[0] - objectAngularVelocity[0] * distance[2];
         real_t linearVelZ = objectAngularVelocity[0] * distance[1] - objectAngularVelocity[1] * distance[0];

         blocks_->transformGlobalToBlockLocalCell(cell, block);
         objVelField->get(cell, 0) = (linearVelX + objectLinearVelocity[0]) * dt_ / dxyz_root[0];
         objVelField->get(cell, 1) = (linearVelY + objectLinearVelocity[1]) * dt_ / dxyz_root[1];
         objVelField->get(cell, 2) = (linearVelZ + objectLinearVelocity[2]) * dt_ / dxyz_root[2];
      )
   }
}


template < typename FractionField_T, typename VectorField_T >
void MovingGeometry<FractionField_T, VectorField_T>::calculateForcesOnBody() {
   Vector3<real_t> summedForceOnObject;
   Vector3<real_t> summedTorqueOnObject;

   auto objectPosition = particleAccessor_->getPosition(0);
   for (auto &block : *blocks_) {
      VectorField_T* forceField = block.getData< VectorField_T >(forceFieldId_);
      FractionField_T* fractionField = block.getData< FractionField_T >(fractionFieldId_);
      auto level = blocks_->getLevel(block);
      WALBERLA_FOR_ALL_CELLS_XYZ(forceField,
         if(fractionField->get(x,y,z,0) > 0.0) {
            auto force = Vector3(forceField->get(x,y,z,0), forceField->get(x,y,z,1), forceField->get(x,y,z,2));
            summedForceOnObject += force;

            Vector3< real_t > cellCenter = blocks_->getCellCenter(Cell(x,y,z), level);
            const auto torque = cross(( cellCenter - objectPosition ), force);
            summedTorqueOnObject += torque;
         }
      )
   }
   WALBERLA_MPI_SECTION() {
      walberla::mpi::allReduceInplace(summedForceOnObject, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(summedTorqueOnObject, walberla::mpi::SUM);

   }
   real_t forceFactor = fluidDensity_ * pow(blocks_->dx(0),4) / (dt_ * dt_); //(kg / m³ -> kg m / s²)
   Vector3<real_t> forceSI = summedForceOnObject * forceFactor;
   particleAccessor_->setHydrodynamicForce(0, forceSI);
   Vector3<real_t> torqueSI = summedTorqueOnObject * forceFactor;
   particleAccessor_->setTorque(0, torqueSI);
}

template < typename FractionField_T, typename VectorField_T >
real_t MovingGeometry<FractionField_T, VectorField_T>::getVolumeFromFractionField() {
   real_t summedVolume = 0.0;
   for (auto &block : *blocks_) {
      FractionField_T* fractionField = block.getData< FractionField_T >(fractionFieldId_);
      WALBERLA_FOR_ALL_CELLS_XYZ(fractionField,
                              summedVolume += fractionField->get(x,y,z,0);
      )
   }
   WALBERLA_MPI_SECTION() {
      walberla::mpi::allReduceInplace(summedVolume, walberla::mpi::SUM);
   }
   return summedVolume;
}

/*
template < typename FractionField_T, typename VectorField_T >
Vector3<real_t> MovingGeometry<FractionField_T, VectorField_T>::getInertiaFromFractionField(real_t objectDensity) {
   Vector3<real_t> inertia(0.0);
   auto objectPos = particleAccessor_->getPosition(0);
   auto dxyz = Vector3<real_t> (blocks_->dx(0), blocks_->dy(0), blocks_->dz(0));
   for (auto &block : *blocks_) {

      auto level = blocks_->getLevel(block);
      FractionField_T* fractionField = block.template getData< FractionField_T >(fractionFieldId_);

      WALBERLA_FOR_ALL_CELLS_XYZ(fractionField,
                                 Vector3< real_t > cellCenter = blocks_->getCellCenter(Cell(x,y,z), level);
                                 real_t sqXDist = pow(cellCenter[0] - objectPos[0], 2);
                                 real_t sqYDist = pow(cellCenter[1] - objectPos[1], 2);
                                 real_t sqZDist = pow(cellCenter[2] - objectPos[2], 2);
                                 inertia[0] += fractionField->get(x,y,z,0) * objectDensity * dxyz[0] * dxyz[1] * dxyz[2] * (sqYDist + sqZDist);
                                 inertia[1] += fractionField->get(x,y,z,0) * objectDensity * dxyz[0] * dxyz[1] * dxyz[2] * (sqXDist + sqZDist);
                                 inertia[2] += fractionField->get(x,y,z,0) * objectDensity * dxyz[0] * dxyz[1] * dxyz[2] * (sqXDist + sqYDist);
      )
   }
   WALBERLA_MPI_SECTION() {
      walberla::mpi::allReduceInplace(inertia, walberla::mpi::SUM);
   }
   return inertia;
}
*/


template class MovingGeometry<field::GhostLayerField< real_t, 1 >, field::GhostLayerField< real_t, 3 >>;
} // namespace waLBerla