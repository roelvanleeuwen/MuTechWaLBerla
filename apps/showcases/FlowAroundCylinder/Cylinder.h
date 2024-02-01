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
//! \file Cylinder.h
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================
#pragma once

#include "blockforest/SetupBlock.h"
#include "blockforest/SetupBlockForest.h"
#include "blockforest/StructuredBlockForest.h"

#include "domain_decomposition/IBlock.h"

#include "core/DataTypes.h"
#include "core/math/AABB.h"
#include "core/math/Vector3.h"
#include "core/cell/Cell.h"

#include "Setup.h"

namespace walberla
{

class Cylinder
{
 public:
   Cylinder(const Setup& setup) : setup_(setup) {}

   bool operator()(const Vector3< real_t >& point) const { return contains(point); }

   bool contains(const Vector3< real_t >& point) const;
   bool contains(const AABB& aabb) const;

   bool intersects(const AABB& aabb, const real_t bufferDistance = real_t(0)) const;

   real_t delta(const Vector3< real_t >& fluid, const Vector3< real_t >& boundary) const;

 private:
   Setup setup_;

}; // class Cylinder

class CylinderRefinementSelection
{
 public:
   CylinderRefinementSelection(const Cylinder& cylinder, const uint_t level, const real_t bufferDistance)
      : cylinder_(cylinder), level_(level), bufferDistance_(bufferDistance)
   {}

   void operator()(SetupBlockForest& forest)
   {
      for (auto block = forest.begin(); block != forest.end(); ++block)
      {
         const AABB& aabb = block->getAABB();

         if (block->getLevel() < level_ && cylinder_.intersects(aabb, bufferDistance_) && !cylinder_.contains(aabb))
            block->setMarker(true);
      }
   }

 private:
   Cylinder cylinder_;
   uint_t level_;
   real_t bufferDistance_;

}; // class CylinderRefinementSelection

class CylinderBlockExclusion
{
 public:
   CylinderBlockExclusion(const Cylinder& cylinder) : cylinder_(cylinder) {}

   bool operator()(const blockforest::SetupBlock& block)
   {
      const AABB aabb = block.getAABB();
      return static_cast< bool >(cylinder_.contains(aabb));
   }

 private:
   Cylinder cylinder_;

}; // class CylinderBlockExclusion


class wallDistance
{
 public:
   wallDistance(const Cylinder& cylinder) : cylinder_(cylinder) {}

   real_t operator()(const Cell& fluidCell, const Cell& boundaryCell, const shared_ptr< StructuredBlockForest >& SbF,
                     IBlock& block) const;

 private:
   Cylinder cylinder_;
}; // class wallDistance

} // namespace walberla