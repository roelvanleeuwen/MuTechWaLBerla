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
#include "mesh/decomposition/ConvexDecomposer.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>


//! [Includes]

namespace walberla {
using namespace walberla::pe;


//! [BodyTypeTuple]
typedef boost::tuple<Sphere, Plane, Box, mesh::pe::TriangleMeshUnion, mesh::pe::ConvexPolyhedron> BodyTypeTuple ;
//! [BodyTypeTuple]

using OutputMesh = mesh::PolyMesh;
using TesselationType=mesh::pe::DefaultTesselation<OutputMesh>;


int main( int argc, char ** argv )
{
   //! [Parameters]
  Environment env(argc, argv);
  auto cfg = env.config();
  if (cfg == NULL) WALBERLA_ABORT("No config specified!");

  const Config::BlockHandle mainConf  = cfg->getBlock( "Rhein" );
  

 

  // Simulation part
  math::seedRandomGenerator( static_cast<unsigned int>(1337 * mpi::MPIManager::instance()->worldRank()) );

  real_t spacing = mainConf.getParameter<real_t>("spacing", real_c(1.0) );
  WALBERLA_LOG_INFO_ON_ROOT("spacing: " << spacing);

  real_t dt = mainConf.getParameter<real_t>("dt", real_c(0.01) );
  WALBERLA_LOG_INFO_ON_ROOT("dt: " << dt);

  int simulationSteps = mainConf.getParameter<int>("steps", 2000 );
  WALBERLA_LOG_INFO_ON_ROOT("steps: " << simulationSteps);

  int visSpacing = mainConf.getParameter<int>("visSpacing", 10 );
  WALBERLA_LOG_INFO_ON_ROOT("visSpacing: " << visSpacing);

  real_t vMax = mainConf.getParameter<real_t>("vmax", real_c(0.5) );
  WALBERLA_LOG_INFO_ON_ROOT("vmax: " << vMax);

  size_t numStones = mainConf.getParameter<size_t>("numStones", 1 );
  WALBERLA_LOG_INFO_ON_ROOT("Number of stone types: " << numStones);

  real_t maxconcavity = mainConf.getParameter<real_t>("maxconcav", real_c(5) );
  WALBERLA_LOG_INFO_ON_ROOT("Maximum concacity: " << maxconcavity);

  // parameters of the riverbed
  real_t height = mainConf.getParameter<real_t>("rbheight", real_c(5) );
  WALBERLA_LOG_INFO_ON_ROOT("Riverbed: Height: " << height);

  real_t max_vel = mainConf.getParameter<real_t>("rbmaxvel", real_c(5) );
  WALBERLA_LOG_INFO_ON_ROOT("Riverbed: Max Flow: " << max_vel);

  real_t decline = mainConf.getParameter<real_t>("rbdecline", real_c(5) );
  WALBERLA_LOG_INFO_ON_ROOT("Riverbed: Decline: " << decline);

  real_t drag = mainConf.getParameter<real_t>("rbdrag", real_c(5) );
  WALBERLA_LOG_INFO_ON_ROOT("Riverbed: Drag: " << drag);

  //! [Parameters]
  //!
  OpenMesh::IO::Options opt(OpenMesh::IO::Options::Default);
  // Load Rhein-Stone off files
  std::vector<mesh::TriangleMesh> stones;
  for(size_t i = 0; i < numStones; i++){
     std::stringstream ss;
     ss << "0" << i << "v500.off";
     WALBERLA_LOG_INFO_ON_ROOT("Loading file: " << ss.str());
     stones.push_back(mesh::TriangleMesh());
     std::ifstream input;
     input.open(ss.str());
     OpenMesh::IO::ImporterT<mesh::TriangleMesh> importer(stones.back());
     OpenMesh::IO::OFFReader().read(input, importer, opt);
     input.close();
     double factor = 3.0;
     for (auto v_it=stones.back().vertices_begin(); v_it!=stones.back().vertices_end(); ++v_it){
       stones.back().set_point(*v_it, factor * stones.back().point(*v_it));
     }
  }

  WALBERLA_LOG_INFO_ON_ROOT("*** GLOBALBODYSTORAGE ***");
  //! [GlobalBodyStorage]
  shared_ptr<BodyStorage> globalBodyStorage = make_shared<BodyStorage>();
  //! [GlobalBodyStorage]

  WALBERLA_LOG_INFO_ON_ROOT("*** BLOCKFOREST ***");
  // create forest
  //! [BlockForest]
  shared_ptr< BlockForest > forest = createBlockForestFromConfig( mainConf );

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
  /*cr::HCSITS cr(globalBodyStorage, forest, storageID, ccdID, fcdID);
  cr.setMaxIterations( 10 );
  cr.setRelaxationModel( cr::HardContactSemiImplicitTimesteppingSolvers::ApproximateInelasticCoulombContactByDecoupling );
  cr.setRelaxationParameter( real_t(0.7) );*/
  cr::DEM cr(globalBodyStorage, forest, storageID, ccdID, fcdID);
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
  auto vtkMeshWriter = shared_ptr<mesh::pe::PeVTKMeshWriter<OutputMesh, TesselationType> >( new mesh::pe::PeVTKMeshWriter<OutputMesh, TesselationType>(forest, storageID, tesselation, std::string("MeshOutput"), uint_t(1), std::string("/local/iq72ehiq/VTK") ));
  vtkMeshWriter->setBodyFilter([](const RigidBody& rb){ return (rb.getTypeID() == mesh::pe::ConvexPolyhedron::getStaticTypeID() || rb.getTypeID() == Box::getStaticTypeID()); });
  vtkMeshWriter->addFacePropertyRank();
  shared_ptr<mesh::pe::PeVTKMeshWriter<OutputMesh, TesselationType>::FaceDataSource<uint64_t>> sidFace = make_shared<mesh::pe::SIDFaceDataSource<OutputMesh, TesselationType, uint64_t>>();
  shared_ptr<mesh::pe::PeVTKMeshWriter<OutputMesh, TesselationType>::FaceDataSource<uint64_t>> uidFace = make_shared<mesh::pe::UIDFaceDataSource<OutputMesh, TesselationType, uint64_t>>();
  vtkMeshWriter->addDataSource( sidFace );
  vtkMeshWriter->addDataSource( uidFace );

  WALBERLA_LOG_INFO_ON_ROOT("*** SETUP - START ***");
  //! [Material]
  const real_t   static_cof  ( real_c(1.2) / 2 );   // Coefficient of static friction. Note: pe doubles the input coefficient of friction for material-material contacts.
  const real_t   dynamic_cof ( static_cof ); // Coefficient of dynamic friction. Similar to static friction for low speed friction.
  MaterialID     material = createMaterial( "granular", real_t( 1.0 ), 0, static_cof, dynamic_cof, real_t( 0.5 ), 1, real_t(8.11e5), real_t(6.86e1), real_t(6.86e1) );
  //! [Material]

  auto simulationDomain = forest->getDomain();
  const auto& generationDomain = simulationDomain.getExtended(-real_c(1) * spacing);
  //! [Planes]
  //createPlane(*globalBodyStorage, 0, Vec3(1,0,0), simulationDomain.minCorner(), material );
  //createPlane(*globalBodyStorage, 0, Vec3(-1,0,0), simulationDomain.maxCorner(), material );
  createPlane(*globalBodyStorage, 0, Vec3(0,1,0), simulationDomain.minCorner(), material );
  createPlane(*globalBodyStorage, 0, Vec3(0,-1,0), simulationDomain.maxCorner(), material );
  createPlane(*globalBodyStorage, 0, Vec3(0,0,1), simulationDomain.minCorner(), material );
  createPlane(*globalBodyStorage, 0, Vec3(0,0,-1), simulationDomain.maxCorner(), material );
  //! [Planes]
  
    // Decomposition into convex parts
  std::vector<std::vector<mesh::TriangleMesh>> substones;
  for(size_t i = 0; i < (size_t)numStones; i++){
     // Decompose
     substones.push_back(mesh::ConvexDecomposer::approximateConvexDecompose(stones[i], maxconcavity));
     for(size_t part = 0; part < substones[i].size(); part++){
		Vec3 centroid = mesh::toWalberla( mesh::computeCentroid( substones[i][part]));
		mesh::translate( substones[i][part], -centroid );
	 }
  }
  
  //! [Gas]
  uint_t numParticles = uint_c(0);
  for (auto blkIt = forest->begin(); blkIt != forest->end(); ++blkIt)
  {
	 IBlock & currentBlock = *blkIt;
	 for (auto it = grid_generator::SCIterator(currentBlock.getAABB().getIntersection(generationDomain), Vector3<real_t>(spacing, spacing, spacing) * real_c(0.5), spacing); it != grid_generator::SCIterator(); ++it)
	 {
      //mesh::pe::TriangleMeshUnion* particle = createUnion<mesh::pe::PolyhedronTuple>( *globalBodyStorage, *forest, storageID, 0, Vec3());
		// Centrate parts an add them to the union
      auto stonenr = size_t(math::realRandom<real_t>(0,real_t(numStones)));
      for(size_t part = 0; part < substones[stonenr].size(); part++){
         //createConvexPolyhedron(particle, 0, (*it), substones[stonenr][part]);
         auto particle = mesh::pe::createConvexPolyhedron( *globalBodyStorage, *forest, storageID, numParticles, *it, substones[stonenr][0], material );
         Vec3 rndVel(math::realRandom<real_t>(-vMax, vMax), math::realRandom<real_t>(-vMax, vMax), math::realRandom<real_t>(-vMax, vMax));
         if (particle != nullptr) particle->setLinearVel(rndVel);
         if (particle != nullptr) ++numParticles;
      }
	}
  }
  WALBERLA_LOG_INFO_ON_ROOT("#particles created per process: " << numParticles);
  syncNextNeighbors<BodyTypeTuple>(*forest, storageID);
  //! [Gas]

  WALBERLA_LOG_INFO_ON_ROOT("*** SETUP - END ***");

  WALBERLA_LOG_INFO_ON_ROOT("*** SIMULATION - START ***");

  //! [GameLoop]
  for (int i=0; i < simulationSteps; ++i)
  {
    if( i % visSpacing == 0 )
	 {
		WALBERLA_LOG_INFO_ON_ROOT( "Timestep " << i << " / " << simulationSteps );
		vtkSphereOutput->write( true );
		vtkMeshWriter->operator()();
	 }
    // Add riverbed force.
    for (auto blockIt = forest->begin(); blockIt != forest->end(); ++blockIt)
    {
      for (auto bodyIt = LocalBodyIterator::begin(*blockIt, storageID); bodyIt != LocalBodyIterator::end(); ++bodyIt)
      {
         // my height
         real_t my_height = bodyIt->getPosition()[2]; // z coordinate
         // my velocity
         real_t my_xvel = bodyIt->getLinearVel()[0];
         real_t flow_xvel = my_height < height ? max_vel * std::exp((my_height-height)/decline) : max_vel;

         bodyIt->addForce(drag*(flow_xvel-my_xvel),0,0);
      }
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
