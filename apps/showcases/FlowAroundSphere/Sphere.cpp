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

#include "Sphere.h"

namespace walberla
{

bool Sphere::contains(const Vector3< real_t >& point) const
{
   const real_t px = setup_.sphereXPosition;
   const real_t py = setup_.sphereYPosition;
   const real_t pz = setup_.sphereZPosition;
   const real_t r  = setup_.sphereRadius;

   const real_t xd = point[0] - px;
   const real_t yd = point[1] - py;
   const real_t zd = point[2] - pz;

   const real_t d  = xd * xd + yd * yd + zd * zd;
   return d <= (r * r);
}

bool Sphere::contains(const AABB& aabb) const
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

bool Sphere::intersects(const AABB& aabb, const real_t bufferDistance) const
{
   const real_t px = setup_.sphereXPosition;
   const real_t py = setup_.sphereYPosition;
   const real_t pz = setup_.sphereZPosition;
   const real_t r  = setup_.sphereRadius;

      Vector3< real_t > p(px, py, real_t(0));

      if (p[0] < aabb.xMin())
         p[0] = aabb.xMin();
      else if (p[0] > aabb.xMax())
         p[0] = aabb.xMax();
      if (p[1] < aabb.yMin())
         p[1] = aabb.yMin();
      else if (p[1] > aabb.yMax())
         p[1] = aabb.yMax();
      if (p[2] < aabb.zMin())
         p[2] = aabb.zMin();
      else if (p[2] > aabb.zMax())
         p[2] = aabb.zMax();

      const real_t xd = p[0] - px;
      const real_t yd = p[1] - py;
      const real_t zd = p[2] - pz;
      const real_t d  = xd * xd + yd * yd + zd * zd;
      const real_t rr = (r + bufferDistance) * (r + bufferDistance);
      return d <= rr;
}

real_t Sphere::delta(const Vector3< real_t >& fluid, const Vector3< real_t >& boundary) const
{
   WALBERLA_CHECK(!contains(fluid));
   WALBERLA_CHECK(contains(boundary));

   const real_t px = setup_.sphereXPosition;
   const real_t py = setup_.sphereYPosition;
   const real_t pz = setup_.sphereZPosition;
   const real_t r  = setup_.sphereRadius;

   // http://devmag.org.za/2009/04/17/basic-collision-detection-in-2d-part-2/

   const Vector3< real_t > circle(px, py, pz);

   const Vector3< real_t > f = fluid - circle;
   const Vector3< real_t > d = (boundary - circle) - f;

   const real_t dDotd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
   const real_t fDotf = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
   const real_t dDotf = d[0] * f[0] + d[1] * f[1] + d[2] * f[2];

   const real_t b = real_c(2.0) * dDotf;
   const real_t c = fDotf - r * r;

   const real_t bb4ac = b * b - (real_c(4.0) * dDotd * c);
   WALBERLA_CHECK_GREATER_EQUAL(bb4ac, real_c(0.0));

   const real_t sqrtbb4ac = std::sqrt(bb4ac);
   const real_t alpha = std::min((-b + sqrtbb4ac) / (real_c(2.0) * dDotd), (-b - sqrtbb4ac) / (real_c(2.0) * dDotd));

   WALBERLA_CHECK_GREATER_EQUAL(alpha, real_c(0.0));
   WALBERLA_CHECK_LESS_EQUAL(alpha, real_c(1.0));

   return alpha;
}

real_t wallDistance::operator()(const Cell& fluidCell, const Cell& boundaryCell,
                                const shared_ptr< StructuredBlockForest >& SbF, IBlock& block) const
{
   const Vector3< real_t > boundary = SbF->getBlockLocalCellCenter( block, boundaryCell );
   const Vector3< real_t > fluid = SbF->getBlockLocalCellCenter( block, fluidCell );
   WALBERLA_ASSERT(sphere_.contains(boundary), "Boundary cell must be inside the sphere!")
   WALBERLA_ASSERT(!sphere_.contains(fluid), "Fluid cell must not be inside the sphere!")

   return sphere_.delta( fluid, boundary );
}
} // namespace walberla