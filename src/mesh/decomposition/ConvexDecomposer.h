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
//! \file ConvexDecomposer.h
//! \author Tobias Leemann <tobias.leemann@fau.de>
//
//======================================================================================================================

#pragma once

#include "mesh/TriangleMeshes.h"
#include "CGALWrapper.h"
#include <vector>

namespace walberla {
namespace mesh {

   typedef walberla::cgalwraps::Exact_kernel Exact_kernel;
   typedef walberla::cgalwraps::Polyhedron Polyhedron;
   typedef walberla::cgalwraps::Surface_mesh Surface_mesh;
   typedef walberla::cgalwraps::Nef_polyhedron  Nef_polyhedron;
   typedef walberla::cgalwraps::Volume_const_iterator Volume_const_iterator;

   class ConvexDecomposer{
   public:
      /** Decompose a non-convex Triangle mesh into several convex parts.
       * \param mesh The mesh which is decomposed.
       * \return Vector containing the convex parts.
      */
      static std::vector<TriangleMesh> convexDecompose(const TriangleMesh& mesh);

   private:
      // Check decomposition result.
      static bool performDecompositionTests(const Nef_polyhedron &input, const std::vector<Nef_polyhedron> &convex_parts );

      // Convert OpenMesh to polyhedron
      static void openMeshToPoly(const TriangleMesh &mesh, Polyhedron &poly);

      // Convert Nef Polyhedron back to OpenMesh
      static void nefToOpenMesh(const Nef_polyhedron &nef, TriangleMesh &mesh);
   };


} // mesh
} // walberla

