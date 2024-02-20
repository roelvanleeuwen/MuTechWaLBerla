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
//! \file Cylinder.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#include "Cylinder.h"

namespace walberla
{

bool Cylinder::contains(const Vector3< real_t >& point) const
{
   const real_t px = setup_.cylinderXPosition;
   const real_t py = setup_.cylinderYPosition;
   const real_t H  = setup_.H;
   const real_t r  = setup_.cylinderRadius;

   if (setup_.circularCrossSection)
   {
      const real_t xd = point[0] - px;
      const real_t yd = point[1] - py;
      const real_t d  = xd * xd + yd * yd;
      return point[2] >= real_t(0.0) && point[2] <= H && d <= (r * r);
   }
   else
   {
      const AABB cylinder(px - r, py - r, real_t(0), px + r, py + r, H);
      return cylinder.contains(point);
   }
}

bool Cylinder::contains(const AABB& aabb) const
{
   if (setup_.circularCrossSection)
   {
      Vector3< real_t > p[8];
      p[0].set(aabb.xMin(), aabb.yMin(), aabb.zMin());
      p[1].set(aabb.xMax(), aabb.yMin(), aabb.zMin());
      p[2].set(aabb.xMin(), aabb.yMax(), aabb.zMin());
      p[3].set(aabb.xMax(), aabb.yMax(), aabb.zMin());
      p[4].set(aabb.xMin(), aabb.yMin(), aabb.zMax());
      p[5].set(aabb.xMax(), aabb.yMin(), aabb.zMax());
      p[6].set(aabb.xMin(), aabb.yMax(), aabb.zMax());
      p[7].set(aabb.xMax(), aabb.yMax(), aabb.zMax());
      return contains(p[0]) && contains(p[1]) && contains(p[2]) && contains(p[3]) && contains(p[4]) && contains(p[5]) &&
             contains(p[6]) && contains(p[7]);
   }
   else { return contains(aabb.min()) && contains(aabb.max()); }
}

bool Cylinder::intersects(const AABB& aabb, const real_t bufferDistance) const
{
   const real_t px = setup_.cylinderXPosition;
   const real_t py = setup_.cylinderYPosition;
   const real_t r  = setup_.cylinderRadius;

   if (setup_.circularCrossSection)
   {
      Vector3< real_t > p(px, py, real_t(0));

      if (p[0] < aabb.xMin())
         p[0] = aabb.xMin();
      else if (p[0] > aabb.xMax())
         p[0] = aabb.xMax();
      if (p[1] < aabb.yMin())
         p[1] = aabb.yMin();
      else if (p[1] > aabb.yMax())
         p[1] = aabb.yMax();

      const real_t xd = p[0] - px;
      const real_t yd = p[1] - py;
      const real_t d  = xd * xd + yd * yd;
      const real_t rr = (r + bufferDistance) * (r + bufferDistance);
      return d <= rr;
   }
   else
   {
      const AABB cylinder(px - r, py - r, real_t(0), px + r, py + r, setup_.H);
      return cylinder.intersects(aabb, bufferDistance);
   }
}

real_t Cylinder::delta(const Vector3< real_t >& fluid, const Vector3< real_t >& boundary) const
{
   WALBERLA_CHECK(!contains(fluid))
   WALBERLA_CHECK(contains(boundary))

   const real_t px = setup_.cylinderXPosition;
   const real_t py = setup_.cylinderYPosition;
   const real_t r  = setup_.cylinderRadius;

   if (setup_.circularCrossSection)
   {
      // http://devmag.org.za/2009/04/17/basic-collision-detection-in-2d-part-2/

      const Vector3< real_t > circle(px, py, real_t(0));

      const Vector3< real_t > f = fluid - circle;
      const Vector3< real_t > d = (boundary - circle) - f;

      const real_t a = d[0] * d[0] + d[1] * d[1];
      const real_t b = real_t(2) * (d[0] * f[0] + d[1] * f[1]);
      const real_t c = f[0] * f[0] + f[1] * f[1] - r * r;

      const real_t bb4ac = b * b - (real_t(4) * a * c);
      WALBERLA_CHECK_GREATER_EQUAL(bb4ac, real_t(0))

      const real_t sqrtbb4ac = std::sqrt(bb4ac);

      const real_t alpha = std::min((-b + sqrtbb4ac) / (real_t(2) * a), (-b - sqrtbb4ac) / (real_t(2) * a));

      WALBERLA_CHECK_GREATER_EQUAL(alpha, real_t(0))
      WALBERLA_CHECK_LESS_EQUAL(alpha, real_t(1))

      return alpha;
   }

   const AABB cylinder(px - r, py - r, real_t(0), px + r, py + r, setup_.H);

   if (fluid[0] <= cylinder.xMin())
   {
      const real_t xdiff = cylinder.xMin() - fluid[0];

      if (fluid[1] <= cylinder.yMin())
      {
         const real_t ydiff = cylinder.yMin() - fluid[1];
         if (xdiff >= ydiff)
         {
            WALBERLA_CHECK_LESS_EQUAL(fluid[0], boundary[0])
            return xdiff / (boundary[0] - fluid[0]);
         }
         WALBERLA_CHECK_LESS_EQUAL(fluid[1], boundary[1])
         return ydiff / (boundary[1] - fluid[1]);
      }
      else if (fluid[1] >= cylinder.yMax())
      {
         const real_t ydiff = fluid[1] - cylinder.yMax();
         if (xdiff >= ydiff)
         {
            WALBERLA_CHECK_LESS_EQUAL(fluid[0], boundary[0])
            return xdiff / (boundary[0] - fluid[0]);
         }
         WALBERLA_CHECK_GREATER_EQUAL(fluid[1], boundary[1])
         return ydiff / (fluid[1] - boundary[1]);
      }

      WALBERLA_CHECK_LESS_EQUAL(fluid[0], boundary[0])
      return xdiff / (boundary[0] - fluid[0]);
   }
   else if (fluid[0] >= cylinder.xMax())
   {
      const real_t xdiff = fluid[0] - cylinder.xMax();

      if (fluid[1] <= cylinder.yMin())
      {
         const real_t ydiff = cylinder.yMin() - fluid[1];
         if (xdiff >= ydiff)
         {
            WALBERLA_CHECK_GREATER_EQUAL(fluid[0], boundary[0])
            return xdiff / (fluid[0] - boundary[0]);
         }
         WALBERLA_CHECK_LESS_EQUAL(fluid[1], boundary[1])
         return ydiff / (boundary[1] - fluid[1]);
      }
      else if (fluid[1] >= cylinder.yMax())
      {
         const real_t ydiff = fluid[1] - cylinder.yMax();
         if (xdiff >= ydiff)
         {
            WALBERLA_CHECK_GREATER_EQUAL(fluid[0], boundary[0])
            return xdiff / (fluid[0] - boundary[0]);
         }
         WALBERLA_CHECK_GREATER_EQUAL(fluid[1], boundary[1])
         return ydiff / (fluid[1] - boundary[1]);
      }

      WALBERLA_CHECK_GREATER_EQUAL(fluid[0], boundary[0])
      return xdiff / (fluid[0] - boundary[0]);
   }

   if (fluid[1] <= cylinder.yMin())
   {
      WALBERLA_CHECK_LESS_EQUAL(fluid[1], boundary[1])
      return (cylinder.yMin() - fluid[1]) / (boundary[1] - fluid[1]);
   }

   WALBERLA_CHECK_GREATER_EQUAL(fluid[1], cylinder.yMax())
   WALBERLA_CHECK_GREATER_EQUAL(fluid[1], boundary[1])
   return (fluid[1] - cylinder.yMax()) / (fluid[1] - boundary[1]);
}

real_t wallDistance::operator()(const Cell& fluidCell, const Cell& boundaryCell,
                                const shared_ptr< StructuredBlockForest >& SbF, IBlock& block) const
{
   const Vector3< real_t > boundary = SbF->getBlockLocalCellCenter( block, boundaryCell );
   const Vector3< real_t > fluid = SbF->getBlockLocalCellCenter( block, fluidCell );
   WALBERLA_ASSERT(cylinder_.contains(boundary), "Boundary cell must be inside the cylinder!")
   WALBERLA_ASSERT(!cylinder_.contains(fluid), "Fluid cell must not be inside the cylinder!")

   return cylinder_.delta( fluid, boundary );
}
} // namespace walberla