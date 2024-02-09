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
//! \file Sphere.h
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

class Sphere
{
 public:
   Sphere(const Setup& setup) : setup_(setup) {}

   bool operator()(const Vector3< real_t >& point) const { return contains(point); }

   bool contains(const Vector3< real_t >& point) const;
   bool contains(const AABB& aabb) const;

   bool intersects(const AABB& aabb, const real_t bufferDistance = real_t(0)) const;

   real_t delta(const Vector3< real_t >& fluid, const Vector3< real_t >& boundary) const;

 private:
   Setup setup_;

}; // class Sphere

class SphereRefinementSelection
{
 public:
   SphereRefinementSelection(const Sphere& sphere, const uint_t level, const real_t bufferDistance)
      : sphere_(sphere), level_(level), bufferDistance_(bufferDistance)
   {}

   void operator()(SetupBlockForest& forest)
   {
      for (auto block = forest.begin(); block != forest.end(); ++block)
      {
         const AABB& aabb = block->getAABB();

         if (block->getLevel() < level_ && sphere_.intersects(aabb, bufferDistance_) && !sphere_.contains(aabb))
            block->setMarker(true);
      }
   }

 private:
   Sphere sphere_;
   uint_t level_;
   real_t bufferDistance_;

}; // class SphereRefinementSelection

class SphereBlockExclusion
{
 public:
   SphereBlockExclusion(const Sphere& sphere) : sphere_(sphere) {}

   bool operator()(const blockforest::SetupBlock& block)
   {
      const AABB aabb = block.getAABB();
      return static_cast< bool >(sphere_.contains(aabb));
   }

 private:
   Sphere sphere_;

}; // class SphereBlockExclusion


class wallDistance
{
 public:
   wallDistance(const Sphere& sphere) : sphere_(sphere) {}

   real_t operator()(const Cell& fluidCell, const Cell& boundaryCell, const shared_ptr< StructuredBlockForest >& SbF,
                     IBlock& block) const;

 private:
   Sphere sphere_;
}; // class wallDistance

} // namespace walberla