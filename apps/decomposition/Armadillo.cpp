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
#include "pe/rigidbody/BoxFactory.h"
#include <vtk/VTKOutput.h>
#include <pe/vtk/SphereVtkOutput.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>


//! [Includes]

namespace walberla {
using namespace walberla::pe;
using namespace walberla::cgalwraps;

//! [BodyTypeTuple]
typedef boost::tuple<Sphere, Plane, Box, mesh::pe::TriangleMeshUnion, mesh::pe::ConvexPolyhedron> BodyTypeTuple ;
//! [BodyTypeTuple]

using OutputMesh = mesh::PolyMesh;
using TesselationType=mesh::pe::DefaultTesselation<OutputMesh>;

// Create a hexagon prism with length 2*len, and hexagon sides of length
// facewidth and sqrt(2)*facewidth (alternating), centered at the origin.
void fill_hexagon_prism(Polyhedron& poly, double len, double facewidth)
{
    const double l = len;
    double u = facewidth;

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
    //std::cout << ss.str();
    ss >> poly;
}

/* Fill one of the four sides of a hoper
 * d_in: Inner diameter of the inlet
 * height: Height of the hopper from inlet to outlet
 * alpha: Angle of the sides
 * side_width: Width of the hoppers sides
 */
void fill_hopper_side(Polyhedron& poly, double d_in, double height, double alpha, double side_width)
{
    // Height of closed hopper.
    const double h = (d_in+2*side_width)/(2.0*tan(alpha));
    
    std::cout << "Height of total hopper: " << h << std::endl;
    std::stringstream ss;
    ss << "OFF\n6 5 0\n";
    for(double i = side_width; i>=0.0; i=i-side_width){
        ss << d_in/2.0 + side_width << " " << d_in/2.0+i << " " << height << "\n";
        ss << -d_in/2.0 - side_width << " " << d_in/2.0+i << " " << height << "\n";
        ss << 0 << " " << i-side_width << " " << height-h << "\n";
    }
    // Top
    ss << "3 0 1 2\n";
    // Bottom
    ss << "3 5 4 3\n";
    // Sides
    for(int j = 0; j < 3; j++){
        int j2 = (j+1)%3;
        ss << "4 " << j << " " << j+3 << " " << j2+3 << " " << j2 << "\n";
    }
    std::cout << ss.str();
    ss >> poly;
}

/* Fill a cube to cut of parts of the hopper, below z = 0 */
void fill_base_cube(Polyhedron& poly, double d_in, double height, double alpha, double side_width)
{
    const double lowest_z = (d_in+2*side_width)/(2.0*tan(alpha)) - height + 1.0;
    const double highest_x = (d_in/2.0)+side_width;
    std::stringstream ss;
    ss << "OFF\n8 6 0\n";
    for(double i = 0.0; i>=-lowest_z; i=i-lowest_z){
        ss << highest_x+1.0 << " " << highest_x+1.0 << " " << i << "\n";
        ss << -highest_x-1.0 << " " << highest_x+1.0 << " " << i << "\n";
        ss << -highest_x-1.0 << " " << -highest_x-1.0 << " " << i << "\n";
        ss << highest_x+1.0 << " " << -highest_x-1.0 << " " << i << "\n";
    }
    // Top
    ss << "4 0 1 2 3\n";
    // Bottom
    ss << "4 7 6 5 4 \n";
    // Sides
    for(int j = 0; j < 4; j++){
        int j2 = (j+1)%4;
        ss << "4 " << j << " " << j+4 << " " << j2+4 << " " << j2 << "\n";
    }
    std::cout << ss.str();
    ss >> poly;
}

int main( int argc, char ** argv )
{
   //! [Parameters]

   if(argc > 1 && std::strcmp(argv[1],"-write")==0){
      WALBERLA_LOG_INFO_ON_ROOT("Writing non-convex bodies");
      // Define necessary rotations
      NefAff_transformation_3 rotx90(1,0,0,
                      0,0,-1,
                      0,1,0,
                      1);
      NefAff_transformation_3 roty90(0,0,1,
                      0,1,0,
                      -1,0,0,
                      1);
      NefAff_transformation_3 rotz90(0,-1,0,
                      1,0,0,
                      0,0,1,
                      1);
                      
      WALBERLA_LOG_INFO_ON_ROOT("Generating prism assembly...");
      // Parameters for prisms
      double len = 2.0;
      double facewidth = 0.25; // Total width of prism = 3.0*facewidth
      
      Polyhedron poly;
      fill_hexagon_prism(poly, len, facewidth);
      Nef_polyhedron nefA(poly);
      Nef_polyhedron nefB(poly);
      Nef_polyhedron nefC(poly);
     
      nefA.transform(rotx90);
      nefB.transform(roty90);
      nefC += nefA;
      nefC += nefB;
      Surface_mesh outMesh;
      cgalwraps::convert_nef_polyhedron_to_polygon_mesh(nefC, outMesh);
      std::ofstream output;
      output.open("ComplexBody.off");
      output << outMesh;
      output.close();
      WALBERLA_LOG_INFO_ON_ROOT("Wrote prism assembly.");
      
      
      WALBERLA_LOG_INFO_ON_ROOT("Writing complex hopper");
      
      // Parameters for the hopper
      const double d_inlet = 44.0; // Inner Diameter of the inlet
      const double d_outlet = 20.0; // Inner Diameter of the outlet
      const double side_width = 2.0; // Width of the hopper boundary, outer diameter 
                                     //= inner diameter + 2*side_width
      const double height = 20.0; // Height of the hopper from inlet to outlet
      
      // The center of the outlet of the hopper will be at the origin. 
      // The center of the inlet is at (0, 0, height).
      // Angle of sides in degrees, alpha = 0 if d_inlet = d_outlet (infinitly steep), alpha = 90, if h = 0 (Flat)
      const double alpha = atan((d_inlet-d_outlet)/(2.0*height)); 
      std::cout << "Alpha: " << alpha << std::endl;
      
      Polyhedron polyhop;
      fill_hopper_side(polyhop, d_inlet, height, alpha, side_width);
      Nef_polyhedron hopper(Nef_polyhedron::EMPTY);
      for(int side = 0; side < 4; side++){
         Nef_polyhedron nefI(polyhop);
         for(int turnside = 0; turnside < side; turnside++){
            nefI.transform(rotz90);
         }
         hopper +=nefI;
      }
      
      //Cut of lower part
      Polyhedron pCube;
      fill_base_cube(pCube, d_inlet, height, alpha, side_width);
      Nef_polyhedron lowerHalfSpace(pCube);
      hopper -= lowerHalfSpace;
      Surface_mesh outMeshHop;
      cgalwraps::convert_nef_polyhedron_to_polygon_mesh(hopper, outMeshHop);
      output.open("Hopper.off");
      output << outMeshHop;
      output.close();
      WALBERLA_LOG_INFO_ON_ROOT("Wrote complex hopper.");
   }else{
      Environment env(argc, argv);
      WALBERLA_UNUSED(env);
      std::ifstream input, inputHopper, inputArmadillo;
      input.open("ComplexBody.off");
      inputHopper.open("Hopper.off");
      inputArmadillo.open("Armadillo.off");
      WALBERLA_LOG_INFO_ON_ROOT("*** Reading mesh ***");
      mesh::TriangleMesh mesh;
      mesh::TriangleMesh hopperMesh;
      mesh::TriangleMesh armadilloMesh;

      OpenMesh::IO::ImporterT<mesh::TriangleMesh> importer(mesh);
      OpenMesh::IO::Options opt(OpenMesh::IO::Options::Default);
      OpenMesh::IO::OFFReader().read(input, importer, opt);
      
      OpenMesh::IO::ImporterT<mesh::TriangleMesh> importerH(hopperMesh);
      OpenMesh::IO::OFFReader().read(inputHopper, importerH, opt);

      OpenMesh::IO::ImporterT<mesh::TriangleMesh> importerA(armadilloMesh);
      OpenMesh::IO::OFFReader().read(inputArmadillo, importerA, opt);

      input.close();
      inputHopper.close();
      inputArmadillo.close();

      // Scale Armadillo
      double factor = 0.15;
      for (auto v_it=armadilloMesh.vertices_begin(); v_it!=armadilloMesh.vertices_end(); ++v_it){
         armadilloMesh.set_point(*v_it, factor * armadilloMesh.point(*v_it));
      }

      // Simulation part
      math::seedRandomGenerator( static_cast<unsigned int>(1337 * mpi::MPIManager::instance()->worldRank()) );

      real_t spacing          = real_c(9.0);
      real_t vMax             = real_c(2.0);
      int    simulationSteps  = 2000;
      real_t dt               = real_c(0.01);
      //! [Parameters]

      WALBERLA_LOG_INFO_ON_ROOT("*** GLOBALBODYSTORAGE ***");
      //! [GlobalBodyStorage]
      shared_ptr<BodyStorage> globalBodyStorage = make_shared<BodyStorage>();
      //! [GlobalBodyStorage]

      WALBERLA_LOG_INFO_ON_ROOT("*** BLOCKFOREST ***");
      // create forest
      //! [BlockForest]
      shared_ptr< BlockForest > forest = createBlockForest( AABB(-25,-25, 0, 25, 25, 80), // simulation domain
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
      vtkMeshWriter->setBodyFilter([](const RigidBody& rb){ return (rb.getTypeID() == mesh::pe::TriangleMeshUnion::getStaticTypeID() || rb.getTypeID() == Box::getStaticTypeID()); });
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
      const auto& generationDomain = AABB(-17,-17, 40 ,19, 19, 60); // simulationDomain.getExtended(-real_c(0.5) * spacing);
      //! [Planes]
      createPlane(*globalBodyStorage, 0, Vec3(1,0,0), simulationDomain.minCorner(), material );
      createPlane(*globalBodyStorage, 0, Vec3(-1,0,0), simulationDomain.maxCorner(), material );
      createPlane(*globalBodyStorage, 0, Vec3(0,1,0), simulationDomain.minCorner(), material );
      createPlane(*globalBodyStorage, 0, Vec3(0,-1,0), simulationDomain.maxCorner(), material );
      createPlane(*globalBodyStorage, 0, Vec3(0,0,1), simulationDomain.minCorner(), material );
      createPlane(*globalBodyStorage, 0, Vec3(0,0,-1), simulationDomain.maxCorner(), material );
      //! [Planes]
      
      mesh::pe::TriangleMeshUnion* un = mesh::pe::createApproximateNonConvexUnion( *globalBodyStorage, *forest, storageID, 0, Vec3(0,0,70), armadilloMesh);
      Vec3 rndVel(math::realRandom<real_t>(-vMax, vMax), math::realRandom<real_t>(-vMax, vMax), math::realRandom<real_t>(-vMax, vMax));
      if (un != nullptr) un->setLinearVel(rndVel);
            
      mesh::pe::createNonConvexUnion( *globalBodyStorage, *forest, storageID, 0, Vec3(0,0,20), hopperMesh, Material::find("iron"), false, true, false);
      createBox( *globalBodyStorage, *forest, storageID, 0, Vec3(-20,0,12), Vec3(9,40,24), Material::find("iron"), false, true, false);
      createBox( *globalBodyStorage, *forest, storageID, 0, Vec3(20,0,12), Vec3(9,40,24), Material::find("iron"), false, true, false);
      //std::cout << un->isFixed() << std::endl;
      //! [Gas]
      uint_t numParticles = uint_c(0);
      for (auto blkIt = forest->begin(); blkIt != forest->end(); ++blkIt)
      {
         IBlock & currentBlock = *blkIt;
         for (auto it = grid_generator::SCIterator(currentBlock.getAABB().getIntersection(generationDomain), Vector3<real_t>(spacing, spacing, spacing) * real_c(0.5), spacing); it != grid_generator::SCIterator(); ++it)
         {
            mesh::pe::TriangleMeshUnion* particle =  mesh::pe::createNonConvexUnion( *globalBodyStorage, *forest, storageID, numParticles, *it, mesh);
            Vec3 rndVelo(math::realRandom<real_t>(-vMax, vMax), math::realRandom<real_t>(-vMax, vMax), math::realRandom<real_t>(-vMax, vMax));
            if (particle != nullptr) particle->setLinearVel(rndVelo);
            if (particle != nullptr) ++numParticles;
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
   }
   return EXIT_SUCCESS;
}
} // namespace walberla

int main( int argc, char* argv[] )
{
  return walberla::main( argc, argv );
}
