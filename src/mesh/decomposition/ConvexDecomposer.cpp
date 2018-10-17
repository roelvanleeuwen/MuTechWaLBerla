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


/*#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
#include <CGAL/IO/Nef_polyhedron_iostream_3.h>
#include <CGAL/Nef_3/SNC_indexed_items.h>
#include <CGAL/convex_decomposition_3.h>
#include <CGAL/convexity_check_3.h>
#include <CGAL/OFF_to_nef_3.h>*/

#include <iostream>
#include <sstream>
#include <fstream>

#include <vector>

namespace walberla {
namespace mesh {


std::vector<TriangleMesh> ConvexDecomposer::convexDecompose( const TriangleMesh& mesh){
   return std::vector<TriangleMesh>();
}



/*typedef CGAL::Exact_predicates_exact_constructions_kernel Exact_kernel;
typedef CGAL::Polyhedron_3<Exact_kernel> Polyhedron;
typedef CGAL::Surface_mesh<Exact_kernel::Point_3> Surface_mesh;
typedef CGAL::Nef_polyhedron_3<Exact_kernel> Nef_polyhedron;
typedef Nef_polyhedron::Volume_const_iterator Volume_const_iterator;*/

/* Test if the decomposition was successful.
 * Criteria:
 * All parts must be convex.
 * All parts must have not common volume with each other.
 * The union of all parts has to match the input volume.
 */
/*bool performDecompositionTests(Nef_polyhedron &input, std::vector<Nef_polyhedron> &convex_parts){
   // Perform convexity check
   // Unite volumes
   Nef_polyhedron unitedNefs(Nef_polyhedron::EMPTY);
   for(int i = 0; i < convex_parts.size(); i++){
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
   //if(unitedNefs != input){
   //   std::cout << "Union of all nefs does NOT match the input." << std::endl;
   //   return false;
   //}
   //return true;

}

void fill_cube_1(Polyhedron& poly)
{
  std::string input =
"OFF\n\
8 12 0\n\
-1 -1 -1\n\
-1 1 -1\n\
1 1 -1\n\
1 -1 -1\n\
-1 -1 1\n\
-1 1 1\n\
1 1 1\n\
1 -1 1\n\
3  0 1 3\n\
3  3 1 2\n\
3  0 4 1\n\
3  1 4 5\n\
3  3 2 7\n\
3  7 2 6\n\
3  4 0 3\n\
3  7 4 3\n\
3  6 4 7\n\
3  6 5 4\n\
3  1 5 6\n\
3  2 1 6";

  std::stringstream ss;
  ss << input;
  ss >> poly;
}

void fill_cube_1_simple(Polyhedron& poly)
{
  std::string input =
"OFF\n\
8 6 0\n\
-1 -1 1\n\
1 -1 1\n\
-1 1 1\n\
1 1 1\n\
-1 1 -1\n\
1 1 -1\n\
-1 -1 -1\n\
1 -1 -1\n\
4 0 1 3 2\n\
4 2 3 5 4\n\
4 4 5 7 6\n\
4 6 7 1 0\n\
4 1 7 5 3\n\
4 6 0 2 4";

  std::stringstream ss;
  ss << input;
  ss >> poly;
}

void fill_cube_2(Polyhedron& poly)
{
  std::string input =
"OFF\n\
8 12 0\n\
0.0 0.0 0.0\n\
0.0 1.0 0.0\n\
1.0 1.0 0.0\n\
1.0 0.0 0.0\n\
0.0 0.0 1.0\n\
0.0 1.0 1.0\n\
1.0 1.0 1.0\n\
1.0 0.0 1.0\n\
3  0 1 3\n\
3  3 1 2\n\
3  0 4 1\n\
3  1 4 5\n\
3  3 2 7\n\
3  7 2 6\n\
3  4 0 3\n\
3  7 4 3\n\
3  6 4 7\n\
3  6 5 4\n\
3  1 5 6\n\
3  2 1 6";

  std::stringstream ss;
  ss << input;
  ss >> poly;
}

int main()
{
  // create two nested cubes
  //Polyhedron cube1, cube2;
  //fill_cube_1_simple(cube1);
  //fill_cube_2(cube2);
  //Nef_polyhedron nef1(cube1);
  //Nef_polyhedron nef2(cube2);

  // compute the difference of the nested cubes
  //Nef_polyhedron nef=nef1-nef2;

  std::ifstream istream;
  istream.open("models/mushroom.off");
  //olyhedron Phed;
  //istream >> Phed;
  Nef_polyhedron nef;
  CGAL::OFF_to_nef_3(istream, nef);
  istream.close();
  //std::cout << "Read Polyhedron with : "  << Phed.size_of_facets() << " facets. " << std::endl;

  Nef_polyhedron nef(Phed);
  nef = nef.closure();

  //Print convexity of the input.
  Surface_mesh SMinput;
  CGAL::convert_nef_polyhedron_to_polygon_mesh(nef, SMinput);
  std::cout << "Input is convex: "  << CGAL::is_strongly_convex_3(SMinput) << std::endl;

  std::ofstream out2;
  std::stringstream filename2;
  filename2 << "unitedInput.off";
  out2.open(filename2.str());
  out2 << SMinput;
  out2.close();

  // Partition the polyhedron into convex parts
  CGAL::convex_decomposition_3(nef);
  std::vector<Nef_polyhedron> convex_parts;


  // the first volume is the outer volume, which is
  // ignored in the decomposition
  Volume_const_iterator ci = ++nef.volumes_begin();
  for( ; ci != nef.volumes_end(); ++ci) {
    if(ci->mark()) {
      Polyhedron P;
      nef.convert_inner_shell_to_polyhedron(ci->shells_begin(), P);
      convex_parts.push_back(Nef_polyhedron(P));
    }
  }
  std::cout << "decomposition into " << convex_parts.size() << " convex parts." << std::endl;
  if(performDecompositionTests(nef, convex_parts)){
     std::cout << "Decomposition test passed." << std::endl;
  }else{
     std::cout << "Decomposition test failed." << std::endl;
  }

   // Write part to file
   Surface_mesh output;
   CGAL::convert_nef_polyhedron_to_polygon_mesh(unitedNefs, output);
   std::ofstream out;
   std::stringstream filename;
   filename << "unitedOutput.off";
   out.open(filename.str());
   out << output;
   out.close();
}*/

} // mesh
} // walberla
