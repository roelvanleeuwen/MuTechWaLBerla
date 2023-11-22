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
//! \file wallDistance.h
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================
#pragma once

#include "blockforest/StructuredBlockForest.h"

#include "core/DataTypes.h"
#include "core/cell/Cell.h"
#include "core/logging/Logging.h"
#include "core/math/Vector3.h"

#include <cstddef>

#include "mesh_common/TriangleMeshes.h"

using namespace walberla;

class wallDistance
{
 public:
   wallDistance(const std::shared_ptr< walberla::mesh::TriangleMesh >& mesh) : mesh_(mesh)
   {
      auto n_of_faces = mesh->n_faces();
      if (n_of_faces == 0) WALBERLA_LOG_INFO_ON_ROOT("The mesh contains no triangles!")

      std::size_t index_tr = 0;
      for (auto it_f = mesh->faces_begin(); it_f != mesh->faces_end(); ++it_f)
      {
         auto vertexIt = mesh->fv_iter(*it_f);
         Vector3< real_t > vert(0.0);
         triangles_.emplace_back();
         while (vertexIt.is_valid())
         {
            auto v = mesh->point(*vertexIt);
            for (size_t i = 0; i != 3; ++i)
            {
               vert[i] = v[i];
            }
            triangles_[index_tr].push_back(vert);
            ++vertexIt;
         }
         ++index_tr;
      }
      if (triangles_.size() != n_of_faces)
      {
         WALBERLA_LOG_INFO_ON_ROOT("Wrong number of found triangles!")
         WALBERLA_LOG_INFO_ON_ROOT("Return an empty triangles vector!")
         triangles_.clear();
      }
   }

   real_t operator()(const Cell& fluidCell, const Cell& boundaryCell, const shared_ptr< StructuredBlockForest >& SbF,
                     IBlock& block) const;
   bool computePointToMeshDistance(const Vector3< real_t > pf, const Vector3< real_t > ps,
                                   const std::vector< Vector3< real_t > >& triangle, real_t& q) const;
   Vector3< real_t > cell2GlobalCCPosition(const shared_ptr< StructuredBlockStorage >& blocks, const Cell loc,
                                           IBlock& block) const;

 private:
   const std::shared_ptr< mesh::TriangleMesh > mesh_;
   std::vector< std::vector< Vector3< real_t > > > triangles_;
}; // class wallDistance

real_t wallDistance::operator()(const Cell& fluidCell, const Cell& boundaryCell,
                                const shared_ptr< StructuredBlockForest >& SbF, IBlock& block) const
{
   real_t q = 0.0;

   const Vector3< real_t > pf = cell2GlobalCCPosition(SbF, fluidCell, block);
   const Vector3< real_t > ps = cell2GlobalCCPosition(SbF, boundaryCell, block);

   WALBERLA_CHECK_GREATER(triangles_.size(), std::size_t(0))

   for (std::size_t x = 0; x != triangles_.size(); ++x)
   {
      const bool intersects = computePointToMeshDistance(pf, ps, triangles_[x], q);
      if (intersects && q > -1.0) { break; }
   }

   WALBERLA_CHECK_GREATER_EQUAL(q, real_t(0))
   WALBERLA_CHECK_LESS_EQUAL(q, real_t(1))
   return q;
}

bool wallDistance::computePointToMeshDistance(const Vector3< real_t > pf, const Vector3< real_t > ps,
                                              const std::vector< Vector3< real_t > >& triangle, real_t& q) const
{
   Vector3< real_t > v0;
   Vector3< real_t > e0;
   Vector3< real_t > e1;

   Vector3< real_t > normal;
   Vector3< real_t > dir;
   Vector3< real_t > intersection;
   Vector3< real_t > tmp;

   real_t a[2][2];
   real_t b[2];
   real_t num;
   real_t den;
   real_t t;
   real_t u;
   real_t v;
   real_t det;
   real_t upv;
   real_t norm;
   const real_t eps = 100.0 * std::numeric_limits< real_t >::epsilon();

   v0     = triangle[0];
   e0     = triangle[1] - v0; // triangle edge from triangle[1] to triangle[0]
   e1     = triangle[2] - v0; // triangle edge from triangle[2] to triangle[0]
   normal = e0 % e1;
   norm   = std::sqrt(normal * normal);
   if (std::fabs(norm) < eps)
   {
      q = -1;
      return false;
   }
   normal /= norm;
   dir = ps - pf;
   num = v0 * normal - pf * normal;
   den = dir * normal;
   t   = num / den;
   //
   if (std::fabs(t) < eps || std::fabs(t - 1) < eps)
   {
      v0  = v0 + 2.0 * eps * normal;
      num = v0 * normal - pf * normal;
      t   = num / den;
   }
   //
   if (std::fabs(den) < eps) { return std::fabs(num) < eps; }

   if (t < 0.0 || t > 1.0)
   {
      q = -1.0;
      return false;
   }
   intersection = pf + dir * t;

   a[0][0]           = e0 * e0;
   a[0][1]           = e0 * e1;
   a[1][0]           = a[0][1];
   a[1][1]           = e1 * e1;
   tmp[0]            = intersection[0] - v0[0];
   tmp[1]            = intersection[1] - v0[1];
   tmp[2]            = intersection[2] - v0[2];
   b[0]              = tmp * e0;
   b[1]              = tmp * e1;
   det               = a[0][0] * a[1][1] - a[0][1] * a[1][0];
   u                 = (a[1][1] * b[0] - a[0][1] * b[1]) / det;
   v                 = (a[0][0] * b[1] - a[1][0] * b[0]) / det;
   upv               = u + v;
   const bool ueq0   = std::fabs(u) < eps;
   const bool ueq1   = std::fabs(u - 1.0) < eps;
   const bool veq0   = std::fabs(v) < eps;
   const bool veq1   = std::fabs(v - 1.0) < eps;
   const bool upveq1 = std::fabs(upv - 1) < eps;
   if ((u < 0.0 && !ueq0) || (u > 1.0 && !ueq1) || (v < 0.0 && !veq0) || (v > 1.0 && !veq1) || (upv > 1 && !upveq1))
   {
      q = -1;
      return false;
   }
   else
   {
      q = std::fabs(t);
      return true;
   }
   return true;
}

Vector3< real_t > wallDistance::cell2GlobalCCPosition(const shared_ptr< StructuredBlockStorage >& blocks,
                                                      const Cell loc, IBlock& block) const
{
   CellInterval globalCell(loc.x(), loc.y(), loc.z(), loc.x(), loc.y(),loc.z());
   blocks->transformBlockLocalToGlobalCellInterval(globalCell,block);
   math::GenericAABB< real_t > const cellAABB = blocks->getAABBFromCellBB(globalCell,blocks->getLevel(block));
   Vector3< real_t > p = cellAABB.center();
   blocks->mapToPeriodicDomain(p);
   return p;
} // end cell2GlobalCC_Position
