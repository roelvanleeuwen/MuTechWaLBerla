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
#include "mesh/TriangleMeshes.h"

#include <core/Environment.h>
#include <core/grid_generator/HCPIterator.h>
#include <core/grid_generator/SCIterator.h>
#include <core/logging/Logging.h>
#include <core/math/Random.h>
#include <OpenMesh/Core/Geometry/VectorT.hh>
#include <OpenMesh/Core/IO/reader/OFFReader.hh>
#include <OpenMesh/Core/IO/importer/ImporterT.hh>


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

/*#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
#include <CGAL/IO/Nef_polyhedron_iostream_3.h>
#pragma GCC diagnostic pop*/

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>


//! [Includes]

namespace walberla {
using namespace walberla::pe;


//! [BodyTypeTuple]
typedef boost::tuple<Sphere, Plane, mesh::pe::TriangleMeshUnion, mesh::pe::ConvexPolyhedron> BodyTypeTuple ;
//! [BodyTypeTuple]

/*typedef CGAL::Exact_predicates_exact_constructions_kernel Exact_kernel;
typedef CGAL::Polyhedron_3<Exact_kernel> Polyhedron;
typedef CGAL::Surface_mesh<Exact_kernel::Point_3> Surface_mesh;
typedef CGAL::Nef_polyhedron_3<Exact_kernel> Nef_polyhedron;
typedef Nef_polyhedron::Vector_3  NefVector_3;
typedef Nef_polyhedron::Aff_transformation_3  NefAff_transformation_3;*/

using OutputMesh = mesh::PolyMesh;
using TesselationType=mesh::pe::DefaultTesselation<OutputMesh>;

/*void fill_hexagon_prism(Polyhedron& poly)
{
    double l = 5.0;
    double u = 0.5;

    std::stringstream ss;
    ss << "OFF\n16 10 0\n";
    for(double i = l; i>=-l; i=i-2*l){
        ss << 3.0*u << " " << u << " " << i << "\n";
        ss << u << " " << 3.0*u << " " << i << "\n";
        ss << -u << " " << 3.0*u << " " << i << "\n";
        ss << -3.0*u << " " << u << " " << i << "\n";

        ss << -3.0*u << " " << -u << " " << i << "\n";
        ss << -u << " " << -3.0*u << " " << i << "\n";
        ss << u << " " << -3.0*u << " " << i << "\n";
        ss << 3.0*u << " " << -u << " " << i << "\n";
    }
    // Top
    ss << "8 0 1 2 3 4 5 6 7\n";
    // Bottom
    ss << "8 15 14 13 12 11 10 9 8\n";
    // Sides
    for(int j = 0; j < 8; j++){
        int j2 = (j+1)%8;
        ss << "4 " << j << " " << j+8 << " " << j2+8 << " " << j2 << "\n";
    }
    std::cout << ss.str();
    ss >> poly;
}*/


int main( int argc, char ** argv )
{
   //! [Parameters]
   Environment env(argc, argv);
   WALBERLA_UNUSED(env);
   /*if(argc > 1 && std::strcmp(argv[1],"-write")==0){
      WALBERLA_LOG_INFO_ON_ROOT("Writing complex body");
      Polyhedron poly;
      fill_hexagon_prism(poly);
      Nef_polyhedron nefA(poly);
      Nef_polyhedron nefB(poly);
      Nef_polyhedron nefC(poly);
      NefAff_transformation_3 rotx90(1,0,0,
                      0,0,-1,
                      0,1,0,
                      1);
      NefAff_transformation_3 roty90(0,0,1,
                      0,1,0,
                      -1,0,0,
                      1);
      nefA.transform(rotx90);
      nefB.transform(roty90);
      nefC += nefA;
      nefC += nefB;
      Surface_mesh outMesh;
      CGAL::convert_nef_polyhedron_to_polygon_mesh(nefC, outMesh);
      std::ofstream output;
      output.open("ComplexBody.off");
      output << outMesh;
      output.close();
   }else{*/

      std::ifstream input;
      input.open("ComplexBody.off");
      WALBERLA_LOG_INFO_ON_ROOT("*** Reading mesh ***");
      mesh::TriangleMesh mesh;

      OpenMesh::IO::ImporterT<mesh::TriangleMesh> importer(mesh);
      OpenMesh::IO::Options opt(OpenMesh::IO::Options::Default);
      OpenMesh::IO::OFFReader().read(input, importer, opt);

      // Simulation part
      math::seedRandomGenerator( static_cast<unsigned int>(1337 * mpi::MPIManager::instance()->worldRank()) );

      real_t spacing          = real_c(11.0);
      real_t radius           = real_c(0.4);
      real_t vMax             = real_c(2.0);
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
      shared_ptr< BlockForest > forest = createBlockForest( AABB(-25,-25,-25, 25, 25, 25), // simulation domain
                                                            Vector3<uint_t>(1,1,1), // blocks in each direction
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
      cr.setGlobalLinearAcceleration( Vec3(0,0,-6) );
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
      const auto& generationDomain = AABB(-15,-15,-15,19,19,19); // simulationDomain.getExtended(-real_c(0.5) * spacing);
      //! [Planes]
      createPlane(*globalBodyStorage, 0, Vec3(1,0,0), simulationDomain.minCorner(), material );
      createPlane(*globalBodyStorage, 0, Vec3(-1,0,0), simulationDomain.maxCorner(), material );
      createPlane(*globalBodyStorage, 0, Vec3(0,1,0), simulationDomain.minCorner(), material );
      createPlane(*globalBodyStorage, 0, Vec3(0,-1,0), simulationDomain.maxCorner(), material );
      createPlane(*globalBodyStorage, 0, Vec3(0,0,1), simulationDomain.minCorner(), material );
      createPlane(*globalBodyStorage, 0, Vec3(0,0,-1), simulationDomain.maxCorner(), material );
      //! [Planes]

      /*real_t q1 = math::realRandom<real_t>(0.0, 1.0);
      real_t q2 = math::realRandom<real_t>(0.0, 1.0);
      real_t q3 = math::realRandom<real_t>(0.0, 1.0);
      real_t q4 = math::sqrt(1.0-q1*q1-q2*q2-q3*q3);*/
      //mesh::pe::TriangleMeshUnion* un = mesh::pe::createNonConvexUnion( *globalBodyStorage, *forest, storageID, 0, Vec3(), mesh);

      //! [Gas]
      uint_t numParticles = uint_c(0);
      for (auto blkIt = forest->begin(); blkIt != forest->end(); ++blkIt)
      {
         IBlock & currentBlock = *blkIt;
         for (auto it = grid_generator::SCIterator(currentBlock.getAABB().getIntersection(generationDomain), Vector3<real_t>(spacing, spacing, spacing) * real_c(0.5), spacing); it != grid_generator::SCIterator(); ++it)
         {
            mesh::pe::TriangleMeshUnion* un = mesh::pe::createNonConvexUnion( *globalBodyStorage, *forest, storageID, numParticles, *it, mesh);
            Vec3 rndVel(math::realRandom<real_t>(-vMax, vMax), math::realRandom<real_t>(-vMax, vMax), math::realRandom<real_t>(-vMax, vMax));
            if (un != nullptr) un->setLinearVel(rndVel);
            if (un != nullptr) ++numParticles;
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
   //}
   return EXIT_SUCCESS;
}
} // namespace walberla

int main( int argc, char* argv[] )
{
  return walberla::main( argc, argv );
}
