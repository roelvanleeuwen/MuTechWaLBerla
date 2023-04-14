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
//! \file InitializerFunctions.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================
#include "InitSpherePacking.h"

namespace walberla
{
using FlagField_T  = FlagField< uint8_t >;

void InitSpherePacking(const shared_ptr< StructuredBlockStorage >& blocks, BlockDataID flagFieldID,
                       const field::FlagUID boundaryFlagUID, const real_t Radius, const real_t Shift, const Vector3<real_t> fillIn)
{
   // Check that too small spheres are not used
   if(Radius < 1e-10)
      return;

   const real_t Diameter = 2 * Radius + Shift;
   const real_t RadiusSquared = Radius * Radius;
   const real_t SphereDistanceX = (Diameter / 3) * std::sqrt(6);

   const cell_idx_t MinR = cell_idx_c(floor(Radius));
   const cell_idx_t MaxR = cell_idx_c(ceil(Radius));

   const uint_t Nx = uint_c(ceil((blocks->getDomainCellBB().xMax() / SphereDistanceX)));
   const uint_t Ny = uint_c(ceil(blocks->getDomainCellBB().yMax() / Diameter)) + 1;
   const uint_t Nz = uint_c(ceil(blocks->getDomainCellBB().zMax() / Diameter)) + 1;

   for (auto& block : *blocks)
   {
      auto flagField    = block.template getData< FlagField_T >(flagFieldID);
      auto boundaryFlag = flagField->getFlag(boundaryFlagUID);
      const CellInterval& BlockBB = blocks->getBlockCellBB( block );
      for (uint_t i = 0; i < Nx; ++i){
         for (uint_t j = 0; j < Ny; ++j){
            for (uint_t k = 0; k < Nz; ++k){

               const real_t Offset = (i % 2 == 0) ? 0.0 : Radius;
               Cell point(cell_idx_c(real_c(i) * SphereDistanceX), cell_idx_c(real_c(j) * Diameter + Offset), cell_idx_c(real_c(k) * Diameter + Offset));
               CellInterval SphereBB(point.x() - MinR, point.y() - MinR, point.z() - MinR, point.x() + MaxR, point.y() + MaxR, point.z() + MaxR);

               if(BlockBB.overlaps(SphereBB))
               {
                  SphereBB.intersect(BlockBB);
                  Cell localCell;
                  Cell localPoint;

                  for(auto it = SphereBB.begin(); it != SphereBB.end(); ++it)
                  {
                     if(it->x() > (real_c(blocks->getDomainCellBB().xSize()) * fillIn[0]) || it->y() > (real_c(blocks->getDomainCellBB().ySize()) * fillIn[1]) || it->z() > (real_c(blocks->getDomainCellBB().zSize()) * fillIn[2])) continue;
                     blocks->transformGlobalToBlockLocalCell(localCell, block, Cell(it->x(), it->y(), it->z()));
                     blocks->transformGlobalToBlockLocalCell(localPoint, block, point);
                     real_t Ri = (localCell[0] - localPoint.x()) * (localCell[0] - localPoint.x()) +
                                 (localCell[1] - localPoint.y()) * (localCell[1] - localPoint.y()) +
                                 (localCell[2] - localPoint.z()) * (localCell[2] - localPoint.z());

                     if(Ri < RadiusSquared)
                     {
                        addFlag(flagField->get(localCell), boundaryFlag);
                     }
                  }
               }
            }}}
   }
}
} // namespace walberla
