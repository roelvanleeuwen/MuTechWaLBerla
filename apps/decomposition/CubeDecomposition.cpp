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
//! \file   01_ConfinedGas.cpp
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

//! [Includes]
#include <pe/basic.h>

#include <core/Environment.h>
#include <core/grid_generator/HCPIterator.h>
#include <core/grid_generator/SCIterator.h>
#include <core/logging/Logging.h>
#include <core/math/Random.h>
#include <OpenMesh/Core/Geometry/VectorT.hh>
#include "mesh/TriangleMeshes.h"
#include "mesh/pe/communication/ConvexPolyhedron.h"
#include "blockforest/StructuredBlockForest.h"
#include "blockforest/Initialization.h"
#include "pe/fcd/GJKEPACollideFunctor.h"
#include "mesh/pe/rigid_body/ConvexPolyhedron.h"
#include "mesh/pe/rigid_body/ConvexPolyhedronFactory.h"
#include "mesh/vtk/VTKMeshWriter.h"
#include "mesh/pe/tesselation/ConvexPolyhedron.h"
#include "mesh/pe/vtk/PeVTKMeshWriter.h"
#include "mesh/pe/DefaultTesselation.h"
#include "mesh/pe/vtk/CommonDataSources.h"
#include "mesh/PolyMeshes.h"

#include <vtk/VTKOutput.h>
#include <pe/vtk/SphereVtkOutput.h>
//! [Includes]

namespace walberla {
using namespace walberla::pe;


//! [BodyTypeTuple]
typedef boost::tuple<Sphere, Plane, mesh::pe::TriangleMeshUnion, mesh::pe::ConvexPolyhedron> BodyTypeTuple ;
//! [BodyTypeTuple]

// Typdefs for OpenMesh
typedef typename mesh::TriangleMesh::VertexHandle OMVertexHandle;
typedef typename OpenMesh::VectorT<real_t, 3> OMVec3;
using OutputMesh = mesh::PolyMesh;
using TesselationType=mesh::pe::DefaultTesselation<OutputMesh>;



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

int main( int argc, char ** argv )
{
   //! [Parameters]
   Environment env(argc, argv);
   WALBERLA_UNUSED(env);

   math::seedRandomGenerator( static_cast<unsigned int>(1337 * mpi::MPIManager::instance()->worldRank()) );

   real_t spacing          = real_c(1.0);
   real_t radius           = real_c(0.4);
   real_t vMax             = real_c(1.0);
   int    simulationSteps  = 1000;
   real_t dt               = real_c(0.01);
   //! [Parameters]

   WALBERLA_LOG_INFO_ON_ROOT("*** GLOBALBODYSTORAGE ***");
   //! [GlobalBodyStorage]
   shared_ptr<BodyStorage> globalBodyStorage = make_shared<BodyStorage>();
   //! [GlobalBodyStorage]

   WALBERLA_LOG_INFO_ON_ROOT("*** BLOCKFOREST ***");
   // create forest
   //! [BlockForest]
   shared_ptr< BlockForest > forest = createBlockForest( AABB(-5,-5,-5,20,20,20), // simulation domain
                                                         Vector3<uint_t>(2,2,2), // blocks in each direction
                                                         Vector3<bool>(false, false, false) // periodicity
                                                         );
   //! [BlockForest]
   if (!forest)
   {
      WALBERLA_LOG_INFO_ON_ROOT( "No BlockForest created ... exiting!");
      return EXIT_SUCCESS;
   }







   WALBERLA_LOG_INFO_ON_ROOT("*** STORAGEDATAHANDLING ***");
   // add block data
   //! [StorageDataHandling]
   auto storageID           = forest->addBlockData(createStorageDataHandling<BodyTypeTuple>(), "Storage");
   //! [StorageDataHandling]
   //! [AdditionalBlockData]
   auto ccdID               = forest->addBlockData(ccd::createHashGridsDataHandling( globalBodyStorage, storageID ), "CCD");
   auto fcdID               = forest->addBlockData(fcd::createGenericFCDDataHandling<BodyTypeTuple, fcd::GJKEPACollideFunctor>(), "FCD");
   //! [AdditionalBlockData]

   WALBERLA_LOG_INFO_ON_ROOT("*** INTEGRATOR ***");
   //! [Integrator]
   cr::HCSITS cr(globalBodyStorage, forest, storageID, ccdID, fcdID);
   cr.setMaxIterations( 10 );
   cr.setRelaxationModel( cr::HardContactSemiImplicitTimesteppingSolvers::ApproximateInelasticCoulombContactByDecoupling );
   cr.setRelaxationParameter( real_t(0.7) );
   cr.setGlobalLinearAcceleration( Vec3(6,6,6) );
   //! [Integrator]

   WALBERLA_LOG_INFO_ON_ROOT("*** BodyTypeTuple ***");
   // initialize body type ids
   //! [SetBodyTypeIDs]
   SetBodyTypeIDs<BodyTypeTuple>::execute();
   //! [SetBodyTypeIDs]

   WALBERLA_LOG_INFO_ON_ROOT("*** VTK OUTPUT ***");
   //! [VTK Domain Output]
   auto vtkDomainOutput = vtk::createVTKOutput_DomainDecomposition( forest, "domain_decomposition", 1, std::string("VTK"), "simulation_step" );
   //! [VTK Domain Output]
   //! [VTK Sphere Output]
   auto vtkSphereHelper = make_shared<SphereVtkOutput>(storageID, *forest) ;
   auto vtkSphereOutput = vtk::createVTKOutput_PointData(vtkSphereHelper, "Bodies", 1, std::string("VTK"), "simulation_step", false, false);

   TesselationType tesselation;
   auto vtkMeshWriter = shared_ptr<mesh::pe::PeVTKMeshWriter<OutputMesh, TesselationType> >( new mesh::pe::PeVTKMeshWriter<OutputMesh, TesselationType>(forest, storageID, tesselation, std::string("MeshOutput"), uint_t(1), std::string("VTK") ));
   vtkMeshWriter->setBodyFilter([](const RigidBody& rb){ return rb.getTypeID() == mesh::pe::TriangleMeshUnion::getStaticTypeID(); });
   vtkMeshWriter->addFacePropertyRank();
   shared_ptr<mesh::pe::PeVTKMeshWriter<OutputMesh, TesselationType>::FaceDataSource<uint64_t>> sidFace = make_shared<mesh::pe::SIDFaceDataSource<OutputMesh, TesselationType, uint64_t>>();
   shared_ptr<mesh::pe::PeVTKMeshWriter<OutputMesh, TesselationType>::FaceDataSource<uint64_t>> uidFace = make_shared<mesh::pe::UIDFaceDataSource<OutputMesh, TesselationType, uint64_t>>();
   vtkMeshWriter->addDataSource( sidFace );
   vtkMeshWriter->addDataSource( uidFace );

   const int VTKSpacing = 10;

   WALBERLA_LOG_INFO_ON_ROOT("*** SETUP - START ***");
   //! [Material]
   const real_t   static_cof  ( real_c(0.1) / 2 );   // Coefficient of static friction. Note: pe doubles the input coefficient of friction for material-material contacts.
   const real_t   dynamic_cof ( static_cof ); // Coefficient of dynamic friction. Similar to static friction for low speed friction.
   MaterialID     material = createMaterial( "granular", real_t( 1.0 ), 0, static_cof, dynamic_cof, real_t( 0.5 ), 1, 1, 0, 0 );
   //! [Material]

   auto simulationDomain = forest->getDomain();
   const auto& generationDomain = AABB(5,5,5,10,10,10); // simulationDomain.getExtended(-real_c(0.5) * spacing);
   //! [Planes]
   createPlane(*globalBodyStorage, 0, Vec3(1,0,0), simulationDomain.minCorner(), material );
   createPlane(*globalBodyStorage, 0, Vec3(-1,0,0), simulationDomain.maxCorner(), material );
   createPlane(*globalBodyStorage, 0, Vec3(0,1,0), simulationDomain.minCorner(), material );
   createPlane(*globalBodyStorage, 0, Vec3(0,-1,0), simulationDomain.maxCorner(), material );
   createPlane(*globalBodyStorage, 0, Vec3(0,0,1), simulationDomain.minCorner(), material );
   createPlane(*globalBodyStorage, 0, Vec3(0,0,-1), simulationDomain.maxCorner(), material );
   //! [Planes]

   // Non_convex-Cube
   const real_t cubehalflength = real_t(1.0);
   // Test a cube, with one of its 8 subcubes missing.
   mesh::TriangleMesh cubeMesh;
   generateCubeTestMesh(cubeMesh, cubehalflength);

   mesh::pe::createNonConvexUnion( *globalBodyStorage, *forest, storageID, 0, Vec3(), cubeMesh );

   //! [Gas]
   uint_t numParticles = uint_c(0);
   for (auto blkIt = forest->begin(); blkIt != forest->end(); ++blkIt)
   {
      IBlock & currentBlock = *blkIt;
      for (auto it = grid_generator::SCIterator(currentBlock.getAABB().getIntersection(generationDomain), Vector3<real_t>(spacing, spacing, spacing) * real_c(0.5), spacing); it != grid_generator::SCIterator(); ++it)
      {
         SphereID sp = createSphere( *globalBodyStorage, *forest, storageID, 0, *it, radius, material);
         Vec3 rndVel(math::realRandom<real_t>(-vMax, vMax), math::realRandom<real_t>(-vMax, vMax), math::realRandom<real_t>(-vMax, vMax));
         if (sp != nullptr) sp->setLinearVel(rndVel);
         if (sp != nullptr) ++numParticles;
      }
   }
   WALBERLA_LOG_INFO_ON_ROOT("#particles created: " << numParticles);
   syncNextNeighbors<BodyTypeTuple>(*forest, storageID);
   //! [Gas]

   WALBERLA_LOG_INFO_ON_ROOT("*** SETUP - END ***");

   WALBERLA_LOG_INFO_ON_ROOT("*** SIMULATION - START ***");
   //! [GameLoop]
   for (int i=0; i < simulationSteps; ++i)
   {
      if( i % 10 == 0 )
      {
         WALBERLA_LOG_INFO_ON_ROOT( "Timestep " << i << " / " << simulationSteps );
      }
      if( i % VTKSpacing == 0 )
      {
         vtkSphereOutput->write( true );
         vtkMeshWriter->operator()();
      }
      cr.timestep( real_c(dt) );
      syncNextNeighbors<BodyTypeTuple>(*forest, storageID);
   }
   //! [GameLoop]
   WALBERLA_LOG_INFO_ON_ROOT("*** SIMULATION - END ***");


   WALBERLA_LOG_INFO_ON_ROOT("*** GETTING STATISTICAL INFORMATION ***");
   //! [PostProcessing]
   Vec3 meanVelocity(0,0,0);
   for (auto blockIt = forest->begin(); blockIt != forest->end(); ++blockIt)
   {
      for (auto bodyIt = LocalBodyIterator::begin(*blockIt, storageID); bodyIt != LocalBodyIterator::end(); ++bodyIt)
      {
        meanVelocity += bodyIt->getLinearVel();
      }
   }
   meanVelocity /= numParticles;
   WALBERLA_LOG_INFO( "mean velocity: " << meanVelocity );
   //! [PostProcessing]

   return EXIT_SUCCESS;
}
} // namespace walberla

int main( int argc, char* argv[] )
{
  return walberla::main( argc, argv );
}
