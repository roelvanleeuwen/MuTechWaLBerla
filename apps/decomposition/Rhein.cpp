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
  WALBERLA_UNUSED(env);
  
  OpenMesh::IO::Options opt(OpenMesh::IO::Options::Default);
  // Load Rhein-Stone off files
  std::vector<mesh::TriangleMesh> stones;
  for(int i = 0; i < 10; i++){
	  std::stringstream ss;
	  ss << "0" << i << "v500.off";
	  WALBERLA_LOG_INFO("Loading file: " << ss.str());
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
 
  // Decomposition into convex parts
  
 
  // Scale Armadillo
  /*double factor = 0.15;
  for (auto v_it=armadilloMesh.vertices_begin(); v_it!=armadilloMesh.vertices_end(); ++v_it){
	 armadilloMesh.set_point(*v_it, factor * armadilloMesh.point(*v_it));
  }*/

  // Simulation part
  math::seedRandomGenerator( static_cast<unsigned int>(1337 * mpi::MPIManager::instance()->worldRank()) );

  real_t spacing          = real_c(2.5);
  real_t vMax             = real_c(0.3);
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
  shared_ptr< BlockForest > forest = createBlockForest( AABB(-11,-11, 0, 11, 11, 80), // simulation domain
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
  const auto& generationDomain = AABB(-9,-9, 3 ,9, 9, 70); // simulationDomain.getExtended(-real_c(0.5) * spacing);
  //! [Planes]
  createPlane(*globalBodyStorage, 0, Vec3(1,0,0), simulationDomain.minCorner(), material );
  createPlane(*globalBodyStorage, 0, Vec3(-1,0,0), simulationDomain.maxCorner(), material );
  createPlane(*globalBodyStorage, 0, Vec3(0,1,0), simulationDomain.minCorner(), material );
  createPlane(*globalBodyStorage, 0, Vec3(0,-1,0), simulationDomain.maxCorner(), material );
  createPlane(*globalBodyStorage, 0, Vec3(0,0,1), simulationDomain.minCorner(), material );
  createPlane(*globalBodyStorage, 0, Vec3(0,0,-1), simulationDomain.maxCorner(), material );
  //! [Planes]
  
  std::vector<std::vector<mesh::TriangleMesh>> substones;
  for(int i = 0; i < 10; i++){
     // Decompose
     substones.push_back(mesh::ConvexDecomposer::approximateConvexDecompose(stones[i]));
     for(int part = 0; part < (int)substones[i].size(); part++){
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
		 mesh::pe::TriangleMeshUnion* particle = createUnion<mesh::pe::PolyhedronTuple>( *globalBodyStorage, *forest, storageID, 0, Vec3());
		// Centrate parts an add them to the union
		int stonenr = (int)(math::realRandom<real_t>(0,10));
		for(int part = 0; part < (int)substones[stonenr].size(); part++){
			createConvexPolyhedron(particle, 0, (*it), substones[stonenr][part]);
		}
		Vec3 rndVel(math::realRandom<real_t>(-vMax, vMax), math::realRandom<real_t>(-vMax, vMax), math::realRandom<real_t>(-vMax, vMax));
		if (particle != nullptr) particle->setLinearVel(rndVel);
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
   return EXIT_SUCCESS;
}
} // namespace walberla

int main( int argc, char* argv[] )
{
  return walberla::main( argc, argv );
}
