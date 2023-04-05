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
//! \file ListVTK.h
//! \ingroup lbm
//! \author Christian Godenschwager <christian.godenschwager@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/cell/CellSet.h"
#include "vtk/BlockCellDataWriter.h"
#include "lbm/inplace_streaming/TimestepTracker.h"

namespace walberla {
namespace lbm {

template< typename List_T, typename OutputType = float >
class ListVelocityVTKWriter : public vtk::BlockCellDataWriter< OutputType, 3 >
{
public:
   ListVelocityVTKWriter( const ConstBlockDataID & listId, std::shared_ptr<lbm::TimestepTracker> & tracker, const std::string & id ) :
      vtk::BlockCellDataWriter< OutputType, 3 >( id ), listId_( listId ), list_( nullptr ), tracker_( tracker ) {}

   ListVelocityVTKWriter( const ConstBlockDataID & listId, const std::string & id ) :
      vtk::BlockCellDataWriter< OutputType, 3 >( id ), listId_( listId ), list_( nullptr ), tracker_( nullptr ) {}

protected:

   void configure() { WALBERLA_ASSERT_NOT_NULLPTR( this->block_ ) list_ = this->block_->template getData< List_T >( listId_ ); }

   OutputType evaluate( const cell_idx_t x, const cell_idx_t y, const cell_idx_t z, const cell_idx_t f )
   {
      WALBERLA_ASSERT_NOT_NULLPTR( list_ )
      Cell cell( x, y, z );
      if( !list_->isFluidCell( cell ) )
         return std::numeric_limits<OutputType>::quiet_NaN();
      size_t timestep = 0;

      if(tracker_ != nullptr) {
         timestep = tracker_->getCounter();
      }
      real_t velocity;
      if(((timestep & 1) ^ 1)) {
         velocity = ( list_->getVelocity( cell ) )[uint_c( f )];
      } else {
         velocity = ( list_->getVelocityOdd( cell ) )[uint_c( f )];
      }
      return numeric_cast< OutputType >( velocity );
   }

   const ConstBlockDataID listId_;
   const List_T * list_;
   std::shared_ptr<lbm::TimestepTracker> tracker_;

}; // class VelocityVTKWriter



template< typename List_T, typename OutputType = float >
class ListDensityVTKWriter : public vtk::BlockCellDataWriter< OutputType >
{
public:

   ListDensityVTKWriter( const ConstBlockDataID & listId, const std::string & id ) :
      vtk::BlockCellDataWriter< OutputType >( id ), listId_( listId ), list_(nullptr) {}

protected:

   void configure() { WALBERLA_ASSERT_NOT_NULLPTR( this->block_ ) list_ = this->block_->template getData< List_T >( listId_ ); }

   OutputType evaluate( const cell_idx_t x, const cell_idx_t y, const cell_idx_t z, const cell_idx_t /*f*/ )
   {
      WALBERLA_ASSERT_NOT_NULLPTR( list_ )
      Cell cell( x, y, z );
      if( !list_->isFluidCell( cell ) )
         return std::numeric_limits<OutputType>::quiet_NaN();

      return numeric_cast< OutputType >( list_->getDensity( cell ) );
   }

   const ConstBlockDataID listId_;
   const List_T * list_;

}; // class DensityVTKWriter



//template< typename List_T, typename OutputType = float >
//class ListPDFVTKWriter : public vtk::BlockCellDataWriter< OutputType, List_T::Stencil::Size >
//{
//public:
//
//   ListPDFVTKWriter( const ConstBlockDataID & listId, const std::string & id ) :
//      vtk::BlockCellDataWriter< OutputType, List_T::Stencil::Size >( id ), listId_( listId ), list_( nullptr ) {}
//
//protected:
//
//   void configure() { WALBERLA_ASSERT_NOT_NULLPTR( this->block_ ) list_ = this->block_->template getData< List_T >( listId_ ); }
//
//   OutputType evaluate( const cell_idx_t x, const cell_idx_t y, const cell_idx_t z, const cell_idx_t f )
//   {
//      WALBERLA_ASSERT_NOT_NULLPTR( list_ )
//      Cell cell( x, y, z );
//      if( !list_->isFluidCell( cell ) )
//         return std::numeric_limits<OutputType>::quiet_NaN();
//
//      return numeric_cast< OutputType >( list_->get( cell, uint_t(f) ) );
//   }
//
//   const ConstBlockDataID listId_;
//   const List_T * list_;
//
//}; // class ListPDFVTKWriter



template< typename List_T >
class ListFluidFilter {
public:

   ListFluidFilter( const ConstBlockDataID listId ) : listId_( listId ) {}

   void operator()( CellSet& filteredCells, const IBlock& block, const StructuredBlockStorage& /*storage*/, const uint_t /*ghostLayers*/ = uint_t( 0 ) ) const
   {
      const List_T* list = block.getData< List_T >( listId_ );
      WALBERLA_ASSERT_NOT_NULLPTR( list )

      filteredCells.insert( list->getFluidCells().begin(), list->getFluidCells().end() );
   }

private:

   const ConstBlockDataID listId_;

}; // class ListFluidFilter


} // namespace lbm
} // namespace walberla