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
//! \brief Tests for convex decomposition
//
//======================================================================================================================

// Test of the convex decomposition

#include <core/timing/Timer.h>
#include "core/debug/TestSubsystem.h"
#include "core/logging/Logging.h"

#include <geometry/mesh/TriangleMesh.h>

#include <mesh/PolyMeshes.h>
#include <mesh/QHull.h>
#include <mesh/TriangleMeshes.h>
#include <mesh/decomposition/ConvexDecomposer.h>
#include <OpenMesh/Core/Geometry/VectorT.hh>

#include <iostream>

using namespace walberla;
using namespace walberla::mesh;
using namespace walberla::math;


typedef typename TriangleMesh::VertexHandle VertexHandle;
typedef typename OpenMesh::VectorT<real_t, 3> OMVec3;

// Add a quadrangle surface in two triangular parts..
void generateQuadrangleSurface(TriangleMesh& mesh, const VertexHandle &v1, const VertexHandle &v2, const VertexHandle &v3, const VertexHandle &v4){
   mesh.add_face( v1, v2, v4 );
   mesh.add_face( v3, v4, v2 );
}

// Generate non-convex "cube" with one subcube missing
// C = [-cube_size, cube_size]^3 without [0, cube_size]^3
// with volume 7*cube_size^3;
// This body has to be partitioned into at least 3 convex parts.
void generateCubeTestMesh( TriangleMesh& mesh, real_t cube_size)
{
   mesh.clear();
   mesh.request_face_normals();
   mesh.request_face_status();
   mesh.request_edge_status();
   mesh.request_vertex_status();
   mesh.request_halfedge_status();

   const VertexHandle basennn = mesh.add_vertex( OMVec3(-cube_size, -cube_size, -cube_size));
   const VertexHandle basennp = mesh.add_vertex( OMVec3(-cube_size, -cube_size, cube_size));
   const VertexHandle basenpn = mesh.add_vertex( OMVec3(-cube_size, cube_size, -cube_size));
   const VertexHandle basenpp = mesh.add_vertex( OMVec3(-cube_size, cube_size, cube_size));

   const VertexHandle basepnn = mesh.add_vertex( OMVec3(cube_size, -cube_size, -cube_size));
   const VertexHandle basepnp = mesh.add_vertex( OMVec3(cube_size, -cube_size, cube_size));
   const VertexHandle baseppn = mesh.add_vertex( OMVec3(cube_size, cube_size, -cube_size));

   const VertexHandle basecpp = mesh.add_vertex( OMVec3(0.0, cube_size, cube_size));
   const VertexHandle basepcp = mesh.add_vertex( OMVec3(cube_size, 0.0, cube_size));
   const VertexHandle baseppc = mesh.add_vertex( OMVec3(cube_size, cube_size, 0.0));
   const VertexHandle baseccp = mesh.add_vertex( OMVec3(0.0, 0.0, cube_size));
   const VertexHandle basepcc = mesh.add_vertex( OMVec3(cube_size, 0.0, 0.0));
   const VertexHandle basecpc = mesh.add_vertex( OMVec3(0.0, cube_size, 0.0));

   const VertexHandle baseccc = mesh.add_vertex( OMVec3(0.0, 0.0, 0.0));

   //Y=-1
   generateQuadrangleSurface(mesh, basennn, basepnn, basepnp, basennp);

   //Z=-1
   generateQuadrangleSurface(mesh, basennn, basenpn, baseppn, basepnn);

   //X=-1
   generateQuadrangleSurface(mesh, basennn, basennp, basenpp, basenpn);

   //X=1
   generateQuadrangleSurface(mesh, basepnn, baseppn, baseppc, basepcc);
   generateQuadrangleSurface(mesh, basepnp, basepnn, basepcc, basepcp);

   //Y=1
   generateQuadrangleSurface(mesh, baseppn, basenpn,basecpc, baseppc);
   generateQuadrangleSurface(mesh, basecpc, basenpn, basenpp, basecpp);

   //Z=1
   generateQuadrangleSurface(mesh, basennp, basepnp, basepcp, baseccp);
   generateQuadrangleSurface(mesh, basennp, baseccp, basecpp, basenpp);

   // Z=0
   generateQuadrangleSurface(mesh, baseccc,basepcc, baseppc, basecpc);

   // X=0
   generateQuadrangleSurface(mesh, basecpp, baseccp, baseccc, basecpc);

   // Y=0
   generateQuadrangleSurface(mesh, baseccp, basepcp,  basepcc, baseccc);

   mesh.update_face_normals();
}

std::vector<Vector3<real_t>> generateOctahedron( const real_t radius)
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
   WALBERLA_LOG_INFO( "--- TESTING OCTAHEDRON ---");
   // Test an convex octahedron, which is not decompositioned.
   shared_ptr< mesh::TriangleMesh > mesh = make_shared<mesh::TriangleMesh>();
   mesh::QHull< mesh::TriangleMesh > qhull( generateOctahedron(real_t(1.0)), mesh );
   qhull.run();
   std::vector<mesh::TriangleMesh> convexParts = mesh::ConvexDecomposer::convexDecompose(*mesh);
   WALBERLA_CHECK_EQUAL( convexParts.size(), 1 );

   WALBERLA_LOG_INFO( "--- TESTING CUBE ---");
   // Test a cube, with one of its 8 subcubes missing.
   TriangleMesh cubeMesh;
   generateCubeTestMesh(cubeMesh, real_t(1.0));
   WALBERLA_LOG_INFO( "Building complete. Testing...");
   std::vector<mesh::TriangleMesh> convexPartsCube = mesh::ConvexDecomposer::convexDecompose(cubeMesh);
   WALBERLA_CHECK_GREATER_EQUAL( convexPartsCube.size(), 3 ); // Decompose in at least 3 parts

   return EXIT_SUCCESS;
}
