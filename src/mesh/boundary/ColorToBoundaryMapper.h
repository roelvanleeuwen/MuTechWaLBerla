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
//! \file ColorToBoundaryMapper.h
//! \ingroup mesh
//! \author Christian Godenschwager <christian.godenschwager@fau.de>
//
//======================================================================================================================

#pragma once

#include "BoundaryInfo.h"

#include "core/config/Config.h"
#include "core/DataTypes.h"
#include "core/debug/CheckFunctions.h"
#include "core/math/Vector3.h"

#include <map>

namespace walberla {
namespace mesh {

template< typename MeshType >
class ColorToBoundaryMapper
{
public:
   typedef typename MeshType::Color Color;

   ColorToBoundaryMapper( const BoundaryInfo & defaultBoundaryInfo ) : defaultBoundaryInfo_(defaultBoundaryInfo) { }

   ColorToBoundaryMapper( const Config::BlockHandle & blockHandle )
   {
      if ( not blockHandle.isValid() )
         return;

      const std::string defaultID = blockHandle.getParameter< std::string >( "default" );
      defaultBoundaryInfo_ = BoundaryInfo( boundary::BoundaryUID( defaultID) );

      Config::Blocks colorBoundaryMappings;
      blockHandle.getBlocks( colorBoundaryMappings );

      for ( const auto & mapping : colorBoundaryMappings )
      {
         const Vector3< real_t > colorRaw = mapping.getParameter< Vector3< real_t > >( "color" );
         const std::string id = mapping.getParameter< std::string >( "boundary" );

         // TODO: this conversion makes assumptions on the templated type (Color)
         const Color color{ std::round( 255 * colorRaw[0]),
                            std::round( 255 * colorRaw[1]),
                            std::round( 255 * colorRaw[2]) };
         const BoundaryInfo info{ boundary::BoundaryUID( id ) };

         set( color, info );
      }
   }

   void set( const Color & c, const BoundaryInfo & bi )
   {
      boundaryInfoMap_[c] = bi;
   }

   const BoundaryInfo & get( const Color & c ) const
   {
      auto it = boundaryInfoMap_.find(c);
      return (it == boundaryInfoMap_.end()) ? defaultBoundaryInfo_ : it->second;
   }

   shared_ptr< BoundaryLocation< MeshType > > addBoundaryInfoToMesh( MeshType & mesh ) const
   {
      WALBERLA_CHECK(mesh.has_face_colors());

      auto boundaryLocations = make_shared< BoundaryLocation< MeshType > >(mesh);

      for (auto & faceHandle : mesh.faces())
      {
         (*boundaryLocations)[faceHandle] = get(mesh.color(faceHandle));
      }

      return boundaryLocations;
   }

private:
   BoundaryInfo                    defaultBoundaryInfo_;
   std::map< Color, BoundaryInfo > boundaryInfoMap_;
};

} // namespace walberla
} // namespace mesh
