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
//! \file Decomposition.cpp
//! \author Tobias Leemann <tobias.leemann@fau.de>
//
//======================================================================================================================

// Test of the convex decomposition

#include "pe/basic.h"

#include <core/timing/Timer.h>
#include "core/debug/TestSubsystem.h"

#include <geometry/mesh/TriangleMesh.h>

#include <mesh/pe/rigid_body/ConvexPolyhedron.h>
#include "mesh/MeshOperations.h"
#include <mesh/PolyMeshes.h>
#include <mesh/QHull.h>
#include <mesh/TriangleMeshes.h>
#include <mesh/decomposition/ConvexDecomposer.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_on_sphere.hpp>

using namespace walberla;
using namespace walberla::pe;
using namespace walberla::mesh::pe;


std::vector<Vector3<real_t>> generatPointCloudOnSphere( boost::random::mt19937& rng, const real_t radius, const uint_t numPoints )
{
   boost::uniform_on_sphere<real_t> distribution(3);

   std::vector<Vector3<real_t>> pointCloud( numPoints );
   for( auto & p : pointCloud )
   {
      auto v = distribution(rng);
      p[0] = v[0] * radius;
      p[1] = v[1] * radius;
      p[2] = v[2] * radius;
      WALBERLA_LOG_DEVEL("Point: " << p );

   }

   return pointCloud;
}

std::vector<Vector3<real_t>> generateHexahedron( const real_t radius)
{

   std::vector<Vector3<real_t>> okta( 6 );
   for(int i = 0; i < 6; i++){
      auto &p = okta[i];
      p[i%3]=(i<3) ? radius: -radius;
      WALBERLA_LOG_DEVEL("Point: " << p );
   }

   return okta;
}

int main( int argc, char** argv )
{
   walberla::debug::enterTestMode();
   walberla::MPIManager::instance()->initializeMPI( &argc, &argv );

   boost::random::mt19937 rng(42);

   shared_ptr< mesh::TriangleMesh > mesh = make_shared<mesh::TriangleMesh>();
   mesh::QHull< mesh::TriangleMesh > qhull( generateHexahedron(real_t(1.0)), mesh );
   qhull.run();
   Vec3 centroid = mesh::toWalberla( mesh::computeCentroid( *mesh ) );
   mesh::translate( *mesh, -centroid );
   std::vector<mesh::TriangleMesh> convexParts = mesh::ConvexDecomposer::convexDecompose(*mesh);
   WALBERLA_CHECK_EQUAL( convexParts.size(), 1 );
   /*ConvexPolyhedron cp(0,
                       0,
                       centroid,
                       Vec3(0,0,0),
                       Quat(),
                       *mesh,
                       Material::find("iron"),
                       false,
                       true,
                       true);*/


   return EXIT_SUCCESS;
}
