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

#include "VHACD.h"

#include "mesh/TriangleMeshes.h"
#include "core/logging/Logging.h"

#include <OpenMesh/Core/IO/writer/OFFWriter.hh>
#include <OpenMesh/Core/IO/reader/OFFReader.hh>
#include <OpenMesh/Core/IO/exporter/ExporterT.hh>
#include <OpenMesh/Core/IO/importer/ImporterT.hh>
#include <OpenMesh/Core/IO/Options.hh>

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

   // Partition the polyhedron into convex parts
   walberla::cgalwraps::convex_decomposition_3(nef);
   std::vector<Nef_polyhedron> convex_parts; // parts as nefs (for checks)
   std::vector<TriangleMesh> convex_meshes; // parts as meshes.

   // the first volume is the outer volume, which is
   // ignored in the decomposition
   Volume_const_iterator ci = ++nef.volumes_begin();
   for( ; ci != nef.volumes_end(); ++ci) {
     if(ci->mark()) {
       Polyhedron P;
       nef.convert_inner_shell_to_polyhedron(ci->shells_begin(), P);
       convex_parts.emplace_back(P);
       
       convex_meshes.emplace_back();
       nefToOpenMesh(convex_parts.back(), convex_meshes.back());
       
     }
   }

   WALBERLA_LOG_INFO( "Decomposition into " << convex_parts.size() << " convex parts.");
   WALBERLA_ASSERT(performDecompositionTests(nef, convex_parts), "Test of convex decomposition has failed.");

   return convex_meshes;

}


std::vector<TriangleMesh> ConvexDecomposer::approximateConvexDecompose( const TriangleMesh& mesh, real_t max_concavity){

   std::vector<TriangleMesh> convex_meshes; // parts as meshes.

   std::vector<double> pts;
   std::vector<uint32_t> tris;
   openMeshToVectors(mesh, pts, tris);

   // Create interface
   VHACD::IVHACD* interfaceVHACD = VHACD::CreateVHACD();
   // Set parameters
   VHACD::IVHACD::Parameters paramsVHACD;
   paramsVHACD.m_concavity = max_concavity;
   paramsVHACD.m_projectHullVertices  = false;
   bool res = interfaceVHACD->Compute(&pts[0], (unsigned int)pts.size() / 3,
       &tris[0], (unsigned int)tris.size() / 3, paramsVHACD);


   if (res) {
       // save output
      unsigned int nConvexHulls = interfaceVHACD->GetNConvexHulls();
      WALBERLA_LOG_INFO("Decomposition into " << nConvexHulls << " approximate convex parts." );
      VHACD::IVHACD::ConvexHull ch;
      for (unsigned int p = 0; p < nConvexHulls; ++p) {
         interfaceVHACD->GetConvexHull(p, ch);
         TriangleMesh cmesh;
         std::vector<double> points(ch.m_points, ch.m_points + (3*ch.m_nPoints));
         std::vector<uint32_t> triangles(ch.m_triangles, ch.m_triangles + (3*ch.m_nTriangles));
         vectorsToOpenMesh(points, triangles, cmesh);
         convex_meshes.push_back(cmesh);
      }
   } else {
      WALBERLA_LOG_INFO_ON_ROOT(" Decomposition was not successful.");
   }
   interfaceVHACD->Clean();
   interfaceVHACD->Release();
   return convex_meshes;
}

void ConvexDecomposer::openMeshToVectors(const TriangleMesh& mesh, std::vector<double> &points, std::vector<uint32_t> &triangles){
   for (auto v_it=mesh.vertices_begin(); v_it!=mesh.vertices_end(); ++v_it){
      auto pt = mesh.point(*v_it);
      points.push_back(pt[0]);
      points.push_back(pt[1]);
      points.push_back(pt[2]);
   }

   for (auto f_it=mesh.faces_begin(); f_it!=mesh.faces_end(); ++f_it){
      auto face_vert_it = mesh.cfv_ccwiter (*f_it);
      for(; face_vert_it.is_valid(); face_vert_it++){
         auto vert_hdl = *face_vert_it;
         triangles.push_back(vert_hdl.idx());
      }
   }
}

void ConvexDecomposer::vectorsToOpenMesh(const std::vector<double> &points, const std::vector<uint32_t> &triangles, TriangleMesh &mesh){

   //std::cout << "Points: " << points.size()/3 << std::endl;
   //std::cout << "Triangles: " << triangles.size()/3 << std::endl;

   std::vector<TriangleMesh::VertexHandle> handles;
   // generate vertices
   for(int i = 0; i < (int)points.size()/3; i++){
      handles.push_back(mesh.add_vertex(TriangleMesh::Point(points[3*i], points[3*i+1],  points[3*i+2])));
   }
   // Faces
   std::vector<TriangleMesh::VertexHandle>  face_vhandles;
   for(int i = 0; i < (int)triangles.size()/3; i++){
      face_vhandles.clear();
      face_vhandles.push_back(handles[triangles[3*i]]);
      face_vhandles.push_back(handles[triangles[3*i+1]]);
      face_vhandles.push_back(handles[triangles[3*i+2]]);
      mesh.add_face(face_vhandles);
   }

}

void ConvexDecomposer::openMeshToPoly(const TriangleMesh &mesh, Polyhedron &poly){
   std::stringstream offstream;
   //Write mesh
   OpenMesh::IO::ExporterT<TriangleMesh> exporter(mesh);
   OpenMesh::IO::Options opt(OpenMesh::IO::Options::Default);
   OpenMesh::IO::OFFWriter().write(offstream, exporter, opt);
   // Create polyhedron from stream, which can then be used to create a nef
   offstream >> poly;
}

void ConvexDecomposer::nefToOpenMesh(const Nef_polyhedron &nef, TriangleMesh &mesh){
   std::stringstream offstream;

   Surface_mesh output;
   walberla::cgalwraps::convert_nef_polyhedron_to_polygon_mesh(nef, output);
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
   for(size_t i = 0; i < convex_parts.size(); i++){
      Surface_mesh output;
      walberla::cgalwraps::convert_nef_polyhedron_to_polygon_mesh(convex_parts[i], output);
      // Convexity check 
      if(!walberla::cgalwraps::is_strongly_convex_3(output)){
         WALBERLA_LOG_INFO( "Output " << i << " is NOT convex." );
         return false;
      }

      // Check if unit and new part are disjoint except for boundaries...
      if(unitedNefs.interior().intersection(convex_parts[i].interior())!=Nef_polyhedron(Nef_polyhedron::EMPTY)){
         WALBERLA_LOG_INFO( "Output " << i << " is NOT disjoint to the previous parts." );
         return false;
      }
      unitedNefs += convex_parts[i];
   }
   // Check if union of all nef is the original mesh
   if(unitedNefs != input){
      WALBERLA_LOG_INFO( "Union of all nefs does NOT match the input." );
      return false;
   }
   return true;
}

} // mesh
} // walberla
