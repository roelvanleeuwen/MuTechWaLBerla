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
#include "VHACDWrapper.h"
#include <vector>

namespace walberla {
namespace mesh {

   class ConvexDecomposer{
   public:
      /** Decompose a non-convex Triangle mesh into several convex parts.
       * \param mesh The mesh which is decomposed.
       * \return Vector containing the convex parts.
      */
      static std::vector<TriangleMesh> convexDecompose(const TriangleMesh& mesh);


      /** Decompose a non-convex Triangle mesh approximately into several convex parts.
       * For complex meshes this method results in considerably fewer parts and should therfore
       * be preferded over ConvexDecomposer::convexDecompose.
       * \param mesh The mesh which is decomposed.
       * \param max_concavity Maximum concavity allowed for the returned mesh.
       * \return Vector containing the convex parts.
      */
      static std::vector<TriangleMesh> approximateConvexDecompose(const TriangleMesh& mesh, real_t max_concavity = real_t(1e-3));

   private:

      // Check decomposition result.
      static bool performDecompositionTests(const cgalwraps::Nef_polyhedron &input, const std::vector<cgalwraps::Nef_polyhedron> &convex_parts );

      // Convert OpenMesh to CGAL-polyhedron
      static void openMeshToPoly(const TriangleMesh &mesh, cgalwraps::Polyhedron &poly);

      // Convert Nef-Polyhedron (CGAL) back to OpenMesh
      static void nefToOpenMesh(const cgalwraps:: Nef_polyhedron &nef, TriangleMesh &mesh);

      // Fill Mesh vectors for HACD processing
      static void openMeshToVectors(const TriangleMesh& mesh, std::vector<double> &points, std::vector<uint32_t> &triangles);

      // Convert VHACD output vectors back to openmesh
      static void vectorsToOpenMesh(const std::vector<double> &points, const std::vector<uint32_t> &triangles, TriangleMesh &mesh);
   };

} // mesh
} // walberla

