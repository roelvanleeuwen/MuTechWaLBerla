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
//! This file implements a dummy minimal CGAL interface which compiles the code,
//! but throws runtime errors, if CGAL is not available and the functions 
//! are called.
//
//======================================================================================================================

#pragma once

#include "core/Abort.h"

#define WALBERLA_VHACD_FUNCTION_ERROR WALBERLA_ABORT( "Invalid VHACD function call! In case of compiling without V-HACD, V-HACD functions for approximate decomposition are not available and shouldn't be called!" );

/// \cond internal

// CMake generated header, needed to check defintion of WALBERLA_BUILD_WITH_VHACD
#include "waLBerlaDefinitions.h"
#include <iostream>

// CGAL Headers
#ifdef WALBERLA_BUILD_WITH_VHACD

#include "VHACD.h"

#endif

namespace walberla {
namespace vhacdwraps {

#ifdef WALBERLA_BUILD_WITH_VHACD
   typedef VHACD::IVHACD IVHACD;

   inline IVHACD* CreateVHACD(){
      return VHACD::CreateVHACD();
   }

#else

   struct IVHACD{
      struct ConvexHull {
         double* m_points;
         uint32_t* m_triangles;
         uint32_t m_nPoints;
         uint32_t m_nTriangles;
      };

      struct Parameters{
         double m_concavity;
         double m_alpha;
         double m_beta;
         double m_minVolumePerCH;
         uint32_t m_resolution;
         uint32_t m_maxNumVerticesPerCH;
         uint32_t m_planeDownsampling;
         uint32_t m_convexhullDownsampling;
         uint32_t m_pca;
         uint32_t m_mode;
         uint32_t m_convexhullApproximation;
         uint32_t m_oclAcceleration;
         uint32_t	m_maxConvexHulls;
		   bool	m_projectHullVertices;
      };

      inline bool Compute(const double* const,
         const uint32_t,
         const uint32_t*,
         const uint32_t,
         const Parameters& ){
         WALBERLA_VHACD_FUNCTION_ERROR
      }


      inline uint32_t GetNConvexHulls() const { WALBERLA_VHACD_FUNCTION_ERROR }

      inline void GetConvexHull(const uint32_t, ConvexHull& ) const { WALBERLA_VHACD_FUNCTION_ERROR }
      inline void Clean() { WALBERLA_VHACD_FUNCTION_ERROR }
      inline void Release() { WALBERLA_VHACD_FUNCTION_ERROR }
   };

   inline IVHACD* CreateVHACD(){ WALBERLA_VHACD_FUNCTION_ERROR }

#endif
} // namespace vhacdwraps
} // namespace walberla


