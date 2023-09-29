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
//! \file ObjectRotator.h
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================
#pragma once
#include "core/math/all.h"
#include "BoxTriangleIntersection.h"

namespace walberla
{

class GeometryOctreeNode
{
 public:
   GeometryOctreeNode(AABB meshAABB, uint_t depth, uint_t maxDepth, uint_t * triangles, uint_t numTriangles, real_t * vertices)
   : boxAABB_(meshAABB), maxDepth_(maxDepth)
   {
      if (depth == maxDepth_) {

         real_t boxHalfSize[3] = {0.5 * boxAABB_.xSize(), 0.5 * boxAABB_.ySize(), 0.5 * boxAABB_.zSize()};
         real_t boxCenter[3] = {boxAABB_.xMin() + boxHalfSize[0], boxAABB_.yMin() + boxHalfSize[1], boxAABB_.zMin() + boxHalfSize[2]};

         for (uint_t i = 0; i < numTriangles; ++i) {
            real_t triangle[3][3] ;
            for(uint_t y = 0; y < 3; ++y) {
               for(uint_t x = 0; x < 3; ++x) {
                  triangle[y][x] = vertices[3 * triangles[3*i + y] + x];
               }
            }
            if(triBoxOverlap(boxCenter, boxHalfSize, triangle)) {
               containedTriangles_.push_back(i);
            }
         }


      }
      else {
         for (uint_t i = 0; i < 8; ++i) {
            AABB halfedAABB(boxAABB_.xMin() + xOffset[i] * 0.5 * boxAABB_.xSize(),
                            boxAABB_.yMin() + yOffset[i] * 0.5 * boxAABB_.ySize(),
                            boxAABB_.zMin() + zOffset[i] * 0.5 * boxAABB_.zSize(),
                            boxAABB_.xMax() + (xOffset[i] - 1) * 0.5 * boxAABB_.xSize(),
                            boxAABB_.yMax() + (yOffset[i] - 1) * 0.5 * boxAABB_.ySize(),
                            boxAABB_.zMax() + (zOffset[i] - 1) * 0.5 * boxAABB_.zSize());
            auto childNode = make_shared<GeometryOctreeNode>(halfedAABB, depth+1, maxDepth, triangles, numTriangles, vertices);
            childNodes_.push_back(childNode);
         }
      }
   }

   AABB getBoxAABB() {
      return boxAABB_;
   }

   uint_t getMaxDepth() {
      return maxDepth_;
   }

   std::vector<shared_ptr<GeometryOctreeNode>> getChildNodes() {
      return childNodes_;
   }

   std::vector<uint_t> getContainedTriangles() {
      return containedTriangles_;
   }

   ///TODO write destructor

 private:
   AABB boxAABB_;
   uint_t maxDepth_;
   std::vector<shared_ptr<GeometryOctreeNode>> childNodes_;
   std::vector<uint_t> containedTriangles_;

   std::vector<uint_t> xOffset{0, 0, 0, 0, 1, 1, 1, 1};
   std::vector<uint_t> yOffset{0, 0, 1, 1, 0, 0, 1, 1};
   std::vector<uint_t> zOffset{0, 1, 0, 1, 0, 1, 0, 1};
};

} //namespace walberla