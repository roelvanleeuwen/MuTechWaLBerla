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
//! \file CellDir.h
//! \ingroup lbm
//! \author Christian Godenschwager <christian.godenschwager@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/cell/Cell.h"

namespace walberla {
namespace lbm {

struct CellDir
{
   CellDir( const Cell & _cell, const uint8_t _dir ) : cell( _cell ), dir( _dir ) {}

   Cell               cell;
   uint8_t             dir;
};



inline bool operator==( const CellDir & lhs, const CellDir & rhs )
{
   return lhs.cell == rhs.cell && lhs.dir == rhs.dir;
}



inline bool operator!=( const CellDir & lhs, const CellDir & rhs )
{
   return lhs.cell != rhs.cell || lhs.dir != rhs.dir;
}



inline bool operator<( const CellDir & lhs, const CellDir & rhs )
{
   if( lhs.cell == rhs.cell )
      return lhs.dir < rhs.dir;
   else
      return lhs.cell < rhs.cell;
}

} // namespace lbm



//namespace mpi {
//
//template< typename T,    // Element type of SendBuffer
//          typename G >   // Growth policy of SendBuffer
//mpi::GenericSendBuffer<T,G>& operator<<( mpi::GenericSendBuffer<T,G> & buf, const lbm::CellDir & cd )
//{
//   buf.addDebugMarker( "cd" );
//   buf << cd.cell << cd.dir;
//   return buf;
//}
//
//template< typename T >    // Element type  of RecvBuffer
//mpi::GenericRecvBuffer<T>& operator>>( mpi::GenericRecvBuffer<T> & buf, lbm::CellDir & cd )
//{
//   buf.readDebugMarker( "cd" );
//   buf >> cd.cell >> cd.dir;
//   return buf;
//}
//
//template<>
//struct BufferSizeTrait< lbm::CellDir >
//{
//   static const bool constantSize = true;
//   static const uint_t size = BufferSizeTrait<Cell>::size + BufferSizeTrait<stencil::Direction>::size + mpi::BUFFER_DEBUG_OVERHEAD;
//};
//
//} // namespace mpi


} // namespace walberla
