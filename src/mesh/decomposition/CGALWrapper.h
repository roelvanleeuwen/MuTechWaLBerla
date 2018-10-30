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
//! \file CGALWrapper.h
//! \author Tobias Leemann <tobias.leemann@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/Abort.h"

#define WALBERLA_CGAL_FUNCTION_ERROR WALBERLA_ABORT( "Invalid CGAL function call! In case of compiling without CGAL, CGAL functions are not available and shouldn't be called!" );

/// \cond internal

// CMake generated header, needed to check defintion of WALBERLA_BUILD_WITH_CGAL
#include "waLBerlaDefinitions.h"
#include <iostream>

// CGAL Headers
#ifdef WALBERLA_BUILD_WITH_CGAL

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
#include <CGAL/boost/graph/graph_traits_PolyMesh_ArrayKernelT.h>
#include <CGAL/IO/Nef_polyhedron_iostream_3.h>
#include <CGAL/Nef_3/SNC_indexed_items.h>
#include <CGAL/convex_decomposition_3.h>
#include <CGAL/convexity_check_3.h>
#include <CGAL/OFF_to_nef_3.h>

#pragma GCC diagnostic pop

#endif

namespace walberla {
namespace cgalwraps {

#ifdef WALBERLA_BUILD_WITH_CGAL
   typedef CGAL::Exact_predicates_exact_constructions_kernel Exact_kernel;
   typedef CGAL::Polyhedron_3<Exact_kernel> Polyhedron;
   typedef CGAL::Surface_mesh<Exact_kernel::Point_3> Surface_mesh;
   typedef CGAL::Nef_polyhedron_3<Exact_kernel> Nef_polyhedron;
   typedef Nef_polyhedron::Volume_const_iterator Volume_const_iterator;

   template<class Poly>
   inline void convex_decomposition_3(Poly& nef){
      CGAL::convex_decomposition_3(nef);
   }

   template<class Nefp, class SMesh>
   inline void convert_nef_polyhedron_to_polygon_mesh(const Nefp &nef, SMesh &smesh){
      CGAL::convert_nef_polyhedron_to_polygon_mesh(nef, smesh);
   }

   template<class PolygonMesh>
   inline bool is_strongly_convex_3(PolygonMesh &mesh){
      return CGAL::is_strongly_convex_3(mesh);
   }

#else
   struct Polyhedron{};

   struct Surface_mesh{};

   struct ShellsBeginResult{};

   struct Exact_kernel{};

   struct Volume{
      inline bool mark() const {WALBERLA_CGAL_FUNCTION_ERROR}
      inline ShellsBeginResult shells_begin() const {WALBERLA_CGAL_FUNCTION_ERROR}
   };

   struct Volume_const_iterator{
      inline Volume* operator->(){WALBERLA_CGAL_FUNCTION_ERROR}
      inline Volume_const_iterator operator++() {WALBERLA_CGAL_FUNCTION_ERROR}
      inline bool operator==(const Volume_const_iterator&) const {WALBERLA_CGAL_FUNCTION_ERROR}
      inline bool operator!=(const Volume_const_iterator&) const {WALBERLA_CGAL_FUNCTION_ERROR}
   };

   struct Nef_polyhedron {
      static const Polyhedron EMPTY;

      inline Nef_polyhedron(Polyhedron){WALBERLA_CGAL_FUNCTION_ERROR}

      inline Volume_const_iterator volumes_begin(){WALBERLA_CGAL_FUNCTION_ERROR}

      inline Volume_const_iterator volumes_end(){WALBERLA_CGAL_FUNCTION_ERROR}

      inline void convert_inner_shell_to_polyhedron(ShellsBeginResult, Polyhedron){WALBERLA_CGAL_FUNCTION_ERROR}

      inline Nef_polyhedron interior() const {WALBERLA_CGAL_FUNCTION_ERROR}

      inline Nef_polyhedron intersection(const Nef_polyhedron&) const {WALBERLA_CGAL_FUNCTION_ERROR}

      inline bool operator==(const Nef_polyhedron&) const {WALBERLA_CGAL_FUNCTION_ERROR}

      inline bool operator!=(const Nef_polyhedron&) const {WALBERLA_CGAL_FUNCTION_ERROR}

      inline Nef_polyhedron operator+=(const Nef_polyhedron&){WALBERLA_CGAL_FUNCTION_ERROR}
   };


   template<typename Poly>
   inline void convex_decomposition_3(Poly&){WALBERLA_CGAL_FUNCTION_ERROR}

   template<typename Nefp, typename SMesh>
   inline void convert_nef_polyhedron_to_polygon_mesh(const Nefp&, SMesh&){WALBERLA_CGAL_FUNCTION_ERROR}

   template<class PolygonMesh>
   inline bool is_strongly_convex_3(PolygonMesh&){WALBERLA_CGAL_FUNCTION_ERROR}

   inline std::ostream& operator<< (std::ostream&, const Surface_mesh&){WALBERLA_CGAL_FUNCTION_ERROR}

   inline std::istream& operator>> (std::istream&, const Polyhedron&){WALBERLA_CGAL_FUNCTION_ERROR}

#endif
} // namespace cgalwraps
} // namespace walberla


