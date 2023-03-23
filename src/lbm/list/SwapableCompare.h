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
//! \file SwapableCompare.h
//! \ingroup field
//! \author Markus Holzer <markus@fau.de>
//
//======================================================================================================================

#pragma once



namespace walberla {
namespace lbm {

/****************************************************************************************************************//**
   * A comparison functor based on field size / field layout
   *
   * \ingroup field
   *
   * Can be used as Compare class for example in std::set's and std::map's
   * Template parameter FIELD_PTR should be either Field * or shared_ptr<Field>
   ********************************************************************************************************************/
template< typename FIELD_PTR >
struct SwapableCompare
{
   /// Strict weak ordering on fields, using the alloc sizes and sizes
   /// if two fields are equal in this ordering, they have equal allocSizes, equal sizes and equal layout
   /// and are therefore swap-able
   bool operator()( const FIELD_PTR& lhs, const FIELD_PTR& rhs ) const
   {
      // the field is viewed as a tuple of ( xSize,ySize,zSize,xAllocSize,yAllocSize,zAllocSize, layout)
      // and then these tuples are compared
      if ( lhs->xSize() < rhs->xSize() ) return true;
      if ( lhs->xSize() > rhs->xSize() ) return false;

      if ( lhs->xAllocSize() < rhs->xAllocSize() ) return true;
      if ( lhs->xAllocSize() > rhs->xAllocSize() ) return false;

      return ( lhs->layout() < rhs->layout() );
   }
};


} // namespace lbm
} // namespace walberla