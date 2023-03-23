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
//! \file CellCounters.h
//! \ingroup lbm
//! \author Christian Godenschwager <christian.godenschwager@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/DataTypes.h"
#include "core/Set.h"

#include "core/debug/CheckFunctions.h"

#include "core/mpi/Reduce.h"

#include "domain_decomposition/StructuredBlockStorage.h"

#include <numeric>

namespace walberla {
namespace lbm {

class ListCellCounter
{
public:

   ListCellCounter( const weak_ptr< StructuredBlockStorage > & blocks,
                    const Set<SUID> & requiredSelectors = Set<SUID>::emptySet(),
                    const Set<SUID> & incompatibleSelectors = Set<SUID>::emptySet() )
      : blocks_( blocks ),
        requiredSelectors_(requiredSelectors),
        incompatibleSelectors_( incompatibleSelectors )
   {
   }

   uint64_t numberOfCells() const
   {
      return numCells_;
   }

   uint64_t numberOfCells( const uint_t level ) const
   {
      return numCellsPerLevel_[level];
   }

   const std::vector< uint64_t > & numberOfCellsPerLevel() const
   {
      return numCellsPerLevel_;
   }

   void operator()()
   {
      auto blocksPtr = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR( blocksPtr, "The block structure has expired!" );

      numCellsPerLevel_.assign( uint64_c( blocksPtr->getNumberOfLevels() ), 0 );

      for( auto block = blocksPtr->begin( requiredSelectors_, incompatibleSelectors_ ); block != blocksPtr->end(); ++block )
      {
         numCellsPerLevel_[blocksPtr->getLevel( *block )] += uint64_c( blocksPtr->getNumberOfXCells( *block ) )
                                                           * uint64_c( blocksPtr->getNumberOfYCells( *block ) )
                                                           * uint64_c( blocksPtr->getNumberOfZCells( *block ) );

      }

      mpi::allReduceInplace( numCellsPerLevel_, mpi::SUM );

      numCells_ = std::accumulate( numCellsPerLevel_.begin(), numCellsPerLevel_.end(), uint64_t( 0 ) );
   }

private:
   weak_ptr< StructuredBlockStorage > blocks_;

   Set<SUID> requiredSelectors_;
   Set<SUID> incompatibleSelectors_;
   
   uint64_t numCells_;
   std::vector< uint64_t > numCellsPerLevel_;
};

template< typename List_T >
class ListFluidCellCounter
{
public:

   ListFluidCellCounter( const weak_ptr< StructuredBlockStorage > & blocks,
                         const ConstBlockDataID & listID,
                         const Set<SUID> & requiredSelectors = Set<SUID>::emptySet(),
                         const Set<SUID> & incompatibleSelectors = Set<SUID>::emptySet() )
      : blocks_( blocks ),
        listID_( listID ),
        requiredSelectors_( requiredSelectors ),
        incompatibleSelectors_( incompatibleSelectors )
   {
   }

   uint64_t numberOfCells() const
   {
      return numCells_;
   }

   uint64_t numberOfCells( const uint_t level ) const
   {
      return numCellsPerLevel_[ level ];
   }

   const std::vector< uint64_t > & numberOfCellsPerLevel() const
   {
      return numCellsPerLevel_;
   }

   void operator()()
   {
      auto blocksPtr = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR( blocksPtr, "The block structure has expired!" );

      numCellsPerLevel_.assign( uint64_c( blocksPtr->getNumberOfLevels() ), 0 );

      for( auto block = blocksPtr->begin( requiredSelectors_, incompatibleSelectors_ ); block != blocksPtr->end(); ++block )
      {
         const List_T * const list = block->template getData<List_T>( listID_ );
         numCellsPerLevel_[blocksPtr->getLevel( *block )] += list->numFluidCells();
      }

      mpi::allReduceInplace( numCellsPerLevel_, mpi::SUM );
      
      numCells_ = std::accumulate( numCellsPerLevel_.begin(), numCellsPerLevel_.end(), uint64_t( 0 ) );
   }

private:
   weak_ptr< StructuredBlockStorage > blocks_;

   ConstBlockDataID listID_;

   Set<SUID> requiredSelectors_;
   Set<SUID> incompatibleSelectors_;

   uint64_t numCells_;
   std::vector< uint64_t > numCellsPerLevel_;
};


} // namespace field
} // namespace walberla
