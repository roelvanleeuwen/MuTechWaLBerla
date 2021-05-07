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
//! \file BoundaryFromMesh.h
//! \ingroup geometry
//! \author Daniel Bauer <daniel.j.bauer@fau.de>
//
//======================================================================================================================

#pragma once

#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/StructuredBlockStorage.h"
#include "mesh/boundary/BoundaryLocation.h"
#include "mesh/boundary/BoundarySetup.h"
#include "mesh/boundary/ColorToBoundaryMapper.h"
#include "mesh_common/distance_octree/DistanceOctree.h"

#include "Initializer.h"

namespace walberla {
namespace geometry {
namespace initializer {

//**********************************************************************************************************************
/*! Initializes the flag field according to a mesh and [color -> boundary] mapping.
*/
//**********************************************************************************************************************
template<typename FlagField_T, typename Mesh_T>
class BoundaryFromMesh : public Initializer
{
public:
   BoundaryFromMesh( shared_ptr<StructuredBlockStorage> blocks,
                     BlockDataID flagFieldID,
                     shared_ptr<Mesh_T> mesh,
                     shared_ptr<mesh::DistanceOctree<Mesh_T>> distanceOctree,
                     const uint_t numGhostLayers );

   void init( shared_ptr<mesh::ColorToBoundaryMapper<Mesh_T>> colorToBoundaryMapper,
              FlagUID fluidFlagID );

   void init( BlockStorage & blockStorage, const Config::BlockHandle & blockHandle ) override;

   shared_ptr<mesh::BoundaryLocation<Mesh_T>> getBoundaryLocation() const;

protected:
   shared_ptr<StructuredBlockStorage> blocks_;
   BlockDataID flagFieldID_;
   BlockDataID flagFieldBoundarHandlingID_;

   shared_ptr<Mesh_T> mesh_;
   shared_ptr<mesh::DistanceOctree<Mesh_T>> distanceOctree_;
   mesh::BoundarySetup boundarySetup_;

   shared_ptr<mesh::BoundaryLocation<Mesh_T>> boundaryLocation_;
};

} // namespace initializer
} // namespace geometry
} // namespace walberla

#include "BoundaryFromMesh.impl.h"
