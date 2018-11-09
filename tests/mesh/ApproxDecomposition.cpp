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
#include <OpenMesh/Core/IO/reader/OFFReader.hh>
#include <OpenMesh/Core/IO/importer/ImporterT.hh>

#include "core/all.h"
#include "blockforest/StructuredBlockForest.h"
#include "blockforest/Initialization.h"
#include "domain_decomposition/all.h"


#include "pe/basic.h"
/*#include "pe/rigidbody/Union.h"
#include "pe/rigidbody/UnionFactory.h"
#include "mesh/pe/rigid_body/ConvexPolyhedron.h"
#include "mesh/pe/rigid_body/ConvexPolyhedronFactory.h" */
#include <iostream>
#include <memory>

using namespace walberla;
using namespace walberla::pe;

int main( int argc, char** argv )
{
   walberla::debug::enterTestMode();
   walberla::MPIManager::instance()->initializeMPI( &argc, &argv );
   WALBERLA_LOG_INFO( "--- TESTING ARMADILLO ---");
   std::ifstream input;
   input.open("Armadillo.off");
   WALBERLA_LOG_INFO_ON_ROOT("*** Reading mesh ***");
   mesh::TriangleMesh mesh;

   OpenMesh::IO::ImporterT<mesh::TriangleMesh> importer(mesh);
   OpenMesh::IO::Options opt(OpenMesh::IO::Options::Default);
   OpenMesh::IO::OFFReader().read(input, importer, opt);

   std::vector<mesh::TriangleMesh> convexParts = mesh::ConvexDecomposer::approximateConvexDecompose(mesh);
   WALBERLA_LOG_INFO("Decomposed mesh into " << convexParts.size() << " convex parts.");
   return EXIT_SUCCESS;
}
