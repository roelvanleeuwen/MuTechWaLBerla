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
#include "core/debug/TestSubsystem.h"
#include "core/logging/Logging.h"

#include "mesh/QHull.h"
#include "mesh/TriangleMeshes.h"
#include "mesh/decomposition/ConvexDecomposer.h"

#include <OpenMesh/Core/Geometry/VectorT.hh>
#include <OpenMesh/Core/IO/reader/OFFReader.hh>
#include <OpenMesh/Core/IO/importer/ImporterT.hh>

#include "core/all.h"
#include "blockforest/StructuredBlockForest.h"
#include "blockforest/Initialization.h"
#include "domain_decomposition/all.h"


#include "pe/basic.h"
#include "pe/rigidbody/Union.h"
#include "pe/rigidbody/UnionFactory.h"
#include "pe/fcd/GenericFCD.h"
#include "pe/fcd/GJKEPACollideFunctor.h"
#include "mesh/pe/rigid_body/ConvexPolyhedron.h"
#include "mesh/pe/rigid_body/ConvexPolyhedronFactory.h"
#include <iostream>
#include <memory>
#include <tuple>

using namespace walberla;
using namespace walberla::pe;

// Typdefs for Union
using UnionType = Union<mesh::pe::ConvexPolyhedron> ;
using BodyTuple = std::tuple<UnionType, mesh::pe::ConvexPolyhedron> ;

int main( int argc, char** argv )
{
   walberla::debug::enterTestMode();
   walberla::MPIManager::instance()->initializeMPI( &argc, &argv );
   WALBERLA_LOG_INFO( "--- TESTING ARMADILLO ---");
   std::ifstream input;
   input.open("Armadillo.off");
   WALBERLA_LOG_INFO_ON_ROOT("*** Reading mesh ***");
   mesh::TriangleMesh meshArm;

   OpenMesh::IO::ImporterT<mesh::TriangleMesh> importer(meshArm);
   OpenMesh::IO::Options opt(OpenMesh::IO::Options::Default);
   OpenMesh::IO::OFFReader().read(input, importer, opt);

   std::vector<mesh::TriangleMesh> convexParts = mesh::ConvexDecomposer::approximateConvexDecompose(meshArm);
   WALBERLA_LOG_INFO("Decomposed mesh into " << convexParts.size() << " convex parts.");
   WALBERLA_CHECK_GREATER_EQUAL( convexParts.size(), 1 ); 
   
   // Check if subbodies are disjoint
   WALBERLA_LOG_INFO( "--- CHECK SUBBODIES ---");
   
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
   
   fcd::GenericFCD<BodyTuple, fcd::GJKEPACollideFunctor> testFCD;
   
   mesh::pe::TriangleMeshUnion* un = createUnion<mesh::pe::PolyhedronTuple>( *globalBodyStorage, forest->getBlockStorage(), storageID, 0, Vec3());

   // Centrate parts an add them to the union
   for(int part = 0; part < (int)convexParts.size(); part++){
      Vec3 centroid = mesh::toWalberla( mesh::computeCentroid( convexParts[part] ) );
      mesh::translate( convexParts[part], -centroid );
      createConvexPolyhedron(un, 0, centroid, convexParts[part]);
   }
   
   PossibleContacts pcs;
   for(auto subBodyA = un->begin(); subBodyA != un->end(); subBodyA++){
	   RigidBody &rba = *subBodyA;
	   for(auto subBodyB = un->begin(); subBodyB != un->end(); subBodyB++){
		   RigidBody &rbb = *subBodyB;
		   if(subBodyA != subBodyB){
			   pcs.push_back(std::pair<mesh::pe::ConvexPolyhedron*, mesh::pe::ConvexPolyhedron*>(&dynamic_cast<mesh::pe::ConvexPolyhedron&>(rba), &dynamic_cast<mesh::pe::ConvexPolyhedron&>(rbb)));
		   }
	   }
   } 
   Contacts& container = testFCD.generateContacts(pcs);
   WALBERLA_LOG_INFO("Checking " << container.size() << " intersections for deeper penetration.");
   for(Contact &c : container){
	   // Allow only small penetrations
	   WALBERLA_CHECK(c.getDistance() > -1e-4);
   }
   WALBERLA_LOG_INFO("Bodies are nearly disjoint.");
   real_t unVol =  un->getVolume();
   real_t meshVol = mesh::computeVolume( meshArm );
   real_t volumeInc = (unVol-meshVol)/meshVol;
   WALBERLA_LOG_INFO("Increase in Volume: " << volumeInc * 100.0 << "%");
   return EXIT_SUCCESS;
}
