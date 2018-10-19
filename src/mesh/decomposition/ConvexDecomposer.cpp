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
//! \file ConvexDecomposer.cpp
//! \author Tobias Leemann <tobias.leemann@fau.de>
//! \brief Decomposition of non-convex meshes into convex parts for collision detection.
//
//======================================================================================================================
#include "ConvexDecomposer.h"

#include "mesh/TriangleMeshes.h"

#include <OpenMesh/Core/IO/writer/OFFWriter.hh>
#include <OpenMesh/Core/IO/reader/OFFReader.hh>
#include <OpenMesh/Core/IO/exporter/ExporterT.hh>
#include <OpenMesh/Core/IO/importer/ImporterT.hh>
#include <OpenMesh/Core/IO/Options.hh>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
#include <CGAL/boost/graph/graph_traits_PolyMesh_ArrayKernelT.h>
#include <CGAL/IO/Nef_polyhedron_iostream_3.h>
#include <CGAL/Nef_3/SNC_indexed_items.h>
#include <CGAL/convex_decomposition_3.h>
#include <CGAL/convexity_check_3.h>
#include <CGAL/OFF_to_nef_3.h>

#include <iostream>
#include <sstream>
#include <fstream>

#include <vector>

namespace walberla {
namespace mesh {


std::vector<TriangleMesh> ConvexDecomposer::convexDecompose( const TriangleMesh& mesh){

   Polyhedron poly;
   openMeshToPoly(mesh, poly);
   Nef_polyhedron nef(poly);
   //Print convexity of the input.
   Surface_mesh SMinput;
   CGAL::convert_nef_polyhedron_to_polygon_mesh(nef, SMinput);
   std::cout << "Input is convex: "  << CGAL::is_strongly_convex_3(SMinput) << std::endl;

   // Partition the polyhedron into convex parts
   CGAL::convex_decomposition_3(nef);
   std::vector<Nef_polyhedron> convex_parts;
   std::vector<TriangleMesh> convex_meshes;
   // the first volume is the outer volume, which is
   // ignored in the decomposition
   Volume_const_iterator ci = ++nef.volumes_begin();
   for( ; ci != nef.volumes_end(); ++ci) {
     if(ci->mark()) {
       Polyhedron P;
       nef.convert_inner_shell_to_polyhedron(ci->shells_begin(), P);
       convex_parts.push_back(Nef_polyhedron(P));
       TriangleMesh cmesh;
       nefToOpenMesh(convex_parts.back(), cmesh);
       convex_meshes.push_back(cmesh);
     }
   }

   std::cout << "decomposition into " << convex_parts.size() << " convex parts." << std::endl;
   if(performDecompositionTests(nef, convex_parts)){
      std::cout << "Decomposition test passed." << std::endl;
   }else{
      std::cout << "Decomposition test failed." << std::endl;
   }

   return std::vector<TriangleMesh>();

}

void ConvexDecomposer::openMeshToPoly(const TriangleMesh &mesh, Polyhedron &poly){
   std::stringstream offstream;
   //Write mesh
   OpenMesh::IO::ExporterT<TriangleMesh> exporter(mesh);
   OpenMesh::IO::Options opt(OpenMesh::IO::Options::Default);
   OpenMesh::IO::OFFWriter().write(offstream, exporter, opt);

   std::cout << offstream.str();
   // Create polyhedron from stream, which can then be used to create a nef
   offstream >> poly;
}

void ConvexDecomposer::nefToOpenMesh(const Nef_polyhedron &nef, TriangleMesh &mesh){
   std::stringstream offstream;

   Surface_mesh output;
   CGAL::convert_nef_polyhedron_to_polygon_mesh(nef, output);
   offstream << output;

   OpenMesh::IO::ImporterT<TriangleMesh> importer(mesh);
   OpenMesh::IO::Options opt(OpenMesh::IO::Options::Default);
   OpenMesh::IO::OFFReader().read(offstream, importer, opt);
}

/* Test if the decomposition was successful.
 * Criteria:
 * All parts must be convex.
 * All parts must have not common volume with each other.
 * The union of all parts has to match the input volume.
 */
bool ConvexDecomposer::performDecompositionTests(const Nef_polyhedron &input, const std::vector<Nef_polyhedron> &convex_parts){
   // Perform convexity check
   // Unite volumes
   Nef_polyhedron unitedNefs(Nef_polyhedron::EMPTY);
   for(int i = 0; i < (int)convex_parts.size(); i++){
      Surface_mesh output;
      CGAL::convert_nef_polyhedron_to_polygon_mesh(convex_parts[i], output);
      std::cout << "Output " << i << " has " << output.number_of_faces() << " faces." << std::endl;
      if(!CGAL::is_strongly_convex_3(output)){
         std::cout << "Output " << i << " is NOT convex." << std::endl;
         return false;
      }

      // Write part to file (optional)
      std::ofstream out;
      std::stringstream filename;
      filename << "outputPart" << i << ".off";
      out.open(filename.str());
      out << output;
      out.close();

      // Check if unit and new part are disjoint except for boundaries...
      if(unitedNefs.interior().intersection(convex_parts[i].interior())!=Nef_polyhedron(Nef_polyhedron::EMPTY)){
         std::cout << "Output " << i << " is NOT disjoint to the previous parts." << std::endl;
         return false;
      }
      unitedNefs += convex_parts[i];
   }
   if(unitedNefs != input){
      std::cout << "Union of all nefs does NOT match the input." << std::endl;
      return false;
   }
   return true;
}

} // mesh
} // walberla
