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
#include <boost/tuple/tuple.hpp>

#include "core/debug/TestSubsystem.h"
#include "core/logging/Logging.h"

#include "mesh/PolyMeshes.h"
#include "mesh/QHull.h"
#include "mesh/TriangleMeshes.h"
#include "mesh/decomposition/ConvexDecomposer.h"
#include <OpenMesh/Core/Geometry/VectorT.hh>

#include "core/all.h"
#include "blockforest/StructuredBlockForest.h"
#include "blockforest/Initialization.h"
#include "domain_decomposition/all.h"


#include "pe/basic.h"
#include "pe/rigidbody/Union.h"
#include "pe/rigidbody/UnionFactory.h"
#include "mesh/pe/rigid_body/ConvexPolyhedron.h"
#include "mesh/pe/rigid_body/ConvexPolyhedronFactory.h"
#include <iostream>
#include <memory>

using namespace walberla;
using namespace walberla::pe;
using namespace walberla::mesh::pe;

// Typdefs for OpenMesh
typedef typename mesh::TriangleMesh::VertexHandle OMVertexHandle;
typedef typename OpenMesh::VectorT<real_t, 3> OMVec3;

// Typdefs for Union
using UnionType = Union<boost::tuple<mesh::pe::ConvexPolyhedron>> ;
typedef boost::tuple<UnionType, mesh::pe::ConvexPolyhedron> BodyTuple ;

// Add a quadrangle surface in two triangular parts..
void generateQuadrangleSurface(mesh::TriangleMesh& mesh, const OMVertexHandle &v1, const OMVertexHandle &v2, const OMVertexHandle &v3, const OMVertexHandle &v4){
   mesh.add_face( v1, v2, v4 );
   mesh.add_face( v3, v4, v2 );
}

// Generate non-convex "cube" with one subcube missing
// C = [-cube_size, cube_size]^3 without [0, cube_size]^3
// with volume 7*cube_size^3;
// This body has to be partitioned into at least 3 convex parts.
void generateCubeTestMesh( mesh::TriangleMesh& mesh, real_t cube_size)
{
   mesh.clear();
   mesh.request_face_normals();
   mesh.request_face_status();
   mesh.request_edge_status();
   mesh.request_vertex_status();
   mesh.request_halfedge_status();

   const OMVertexHandle basennn = mesh.add_vertex( OMVec3(-cube_size, -cube_size, -cube_size));
   const OMVertexHandle basennp = mesh.add_vertex( OMVec3(-cube_size, -cube_size, cube_size));
   const OMVertexHandle basenpn = mesh.add_vertex( OMVec3(-cube_size, cube_size, -cube_size));
   const OMVertexHandle basenpp = mesh.add_vertex( OMVec3(-cube_size, cube_size, cube_size));

   const OMVertexHandle basepnn = mesh.add_vertex( OMVec3(cube_size, -cube_size, -cube_size));
   const OMVertexHandle basepnp = mesh.add_vertex( OMVec3(cube_size, -cube_size, cube_size));
   const OMVertexHandle baseppn = mesh.add_vertex( OMVec3(cube_size, cube_size, -cube_size));

   const OMVertexHandle basecpp = mesh.add_vertex( OMVec3(0.0, cube_size, cube_size));
   const OMVertexHandle basepcp = mesh.add_vertex( OMVec3(cube_size, 0.0, cube_size));
   const OMVertexHandle baseppc = mesh.add_vertex( OMVec3(cube_size, cube_size, 0.0));
   const OMVertexHandle baseccp = mesh.add_vertex( OMVec3(0.0, 0.0, cube_size));
   const OMVertexHandle basepcc = mesh.add_vertex( OMVec3(cube_size, 0.0, 0.0));
   const OMVertexHandle basecpc = mesh.add_vertex( OMVec3(0.0, cube_size, 0.0));

   const OMVertexHandle baseccc = mesh.add_vertex( OMVec3(0.0, 0.0, 0.0));

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
   const real_t cubehalflength = real_t(1.0);
   const real_t subcubeVol = cubehalflength *cubehalflength *cubehalflength;
   // Test a cube, with one of its 8 subcubes missing.
   mesh::TriangleMesh cubeMesh;
   generateCubeTestMesh(cubeMesh, cubehalflength);
   WALBERLA_LOG_INFO( "Building complete. Testing...");
   std::vector<mesh::TriangleMesh> convexPartsCube = mesh::ConvexDecomposer::convexDecompose(cubeMesh);
   WALBERLA_CHECK_GREATER_EQUAL( convexPartsCube.size(), 3 ); // Decompose in at least 3 parts

   WALBERLA_LOG_INFO( "--- TESTING CUBE AS PE-UNION ---");
   SetBodyTypeIDs<BodyTuple>::execute();
   //shared_ptr<BodyStorage> globalBodyStorage = make_shared<BodyStorage>();
   shared_ptr<BodyStorage> globalBodyStorage = make_shared<BodyStorage>();

   // create blocks
   shared_ptr< StructuredBlockForest > forest = blockforest::createUniformBlockGrid(
            uint_c( 1), uint_c( 1), uint_c( 1), // number of blocks in x,y,z direction
            uint_c( 1), uint_c( 1), uint_c( 1), // how many cells per block (x,y,z)
            real_c(10),                         // dx: length of one cell in physical coordinates
            0,                                  // max blocks per process
            false, false,                       // include metis / force metis
            false, false, false );                 // full periodicity
   auto storageID = forest->addBlockData(createStorageDataHandling<BodyTuple>(), "Storage");

   mesh::pe::TriangleMeshUnion* un = mesh::pe::createNonConvexUnion( *globalBodyStorage, forest->getBlockStorage(), storageID, 0, Vec3(), cubeMesh );

   WALBERLA_CHECK_GREATER_EQUAL( un->size(), 3 ); // Decompose in at least 3 parts
   //Check volume
   WALBERLA_CHECK_FLOAT_EQUAL(un->getVolume(), real_t(7.0)*subcubeVol, "Volume is incorrect.");

   //Check center of gravity
   WALBERLA_CHECK_FLOAT_EQUAL((un->getPosition()-Vec3(real_t(-1./14.),real_t(-1./14.),real_t(-1./14.))).sqrLength(), real_t(0.0), "Center of gravity is not valid.");

   //Check inertia
   const real_t fac  = un->getMass()*subcubeVol;
   const Mat3 analyticInertia = Mat3(real_t(fac*193./294.), real_t(fac*2./49.), real_t(fac*2./49.),
                                             real_t(fac*2./49.), real_t(fac*193./294.), real_t(fac*2./49.),
                                             real_t(fac*2./49.), real_t(fac*2./49.), real_t(fac*193./294.));

   WALBERLA_CHECK(un->getInertia() == analyticInertia, "Inertia is incorrect.");

   return EXIT_SUCCESS;
}
