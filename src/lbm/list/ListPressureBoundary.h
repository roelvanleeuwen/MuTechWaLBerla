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
//! \file ListPressureBoundary.h
//! \ingroup lbm
//! \author Christian Godenschwager <christian.godenschwager@fau.de>
//
//======================================================================================================================

#pragma once

#include "List.h"

#include "core/DataTypes.h"
#include "core/Macros.h"
#include "core/cell/Cell.h"
#include "core/math/Vector3.h"

#include "lbm/lattice_model/EquilibriumDistribution.h"

#include <algorithm>
#include <vector>

namespace walberla {
namespace lbm {


template< typename List >
class ListPressureBoundary {
public:
   ListPressureBoundary() : list_( nullptr ), pdfsStartIdx_( 0 ), numPDFs_( 0 ), latticeDensity_( real_t( 1 ) ) {}

   void init( List * const list, const std::vector<Cell> & pressureCells, const real_t latticeDensity )
   {
      list_ = list;
      latticeDensity_ = latticeDensity;

      nearBoundaryCells_.clear();
      nearBoundaryVelocities_.clear();
      velocityIdx_.clear();
      antiBounceBackPDFIdx_.clear();
      fluidPressureDirection_.clear();

      // Sort boundary cells and remove duplicates
      std::vector<Cell> sortedBoundaryCells( pressureCells.begin(), pressureCells.end() );
      std::sort( sortedBoundaryCells.begin(), sortedBoundaryCells.end() );
      sortedBoundaryCells.erase( std::unique( sortedBoundaryCells.begin(), sortedBoundaryCells.end() ), sortedBoundaryCells.end() );

      // Find near boundary cells
      for( auto it = sortedBoundaryCells.begin(); it != sortedBoundaryCells.end(); ++it )
         for( auto dirIt = List::Stencil::beginNoCenter(); dirIt != List::Stencil::end(); ++dirIt )
         {
            Cell neighborCell = *it + *dirIt;
            if( list->isFluidCell( neighborCell ) )
               nearBoundaryCells_.push_back( list->getIdx( neighborCell ) );
         }

      std::sort( nearBoundaryCells_.begin(), nearBoundaryCells_.end() );
      nearBoundaryCells_.erase( std::unique( nearBoundaryCells_.begin(), nearBoundaryCells_.end() ), nearBoundaryCells_.end() );
      nearBoundaryVelocities_.resize( nearBoundaryCells_.size() );

      // setup PDFs

      std::vector<CellDir> pdfsToRegister;

      uint_t velocityIdx = 0;
      for( auto it = nearBoundaryCells_.begin(); it != nearBoundaryCells_.end(); ++it, ++velocityIdx )
      {
         Cell cell = list->getCell( *it );

         for( auto dirIt = List::Stencil::beginNoCenter(); dirIt != List::Stencil::end(); ++dirIt )
         {
            Cell neighborCell = cell + *dirIt;
            if( !std::binary_search( sortedBoundaryCells.begin(), sortedBoundaryCells.end(), neighborCell ) )
               continue;

            velocityIdx_.push_back( velocityIdx );
            antiBounceBackPDFIdx_.push_back( list->getPDFIdx( *it, /*dirIt.inverseDir()*/ *dirIt ) );
            fluidPressureDirection_.push_back( *dirIt );
            pdfsToRegister.push_back( CellDir( neighborCell, dirIt.inverseDir() ) );
         }
      }

      for( auto it = pdfsToRegister.begin(); it != pdfsToRegister.end(); ++it ){
         WALBERLA_LOG_INFO_ON_ROOT("pdfsToRegister cell  " << it->cell << "  dir  " << it->dir)
      }



      pdfsStartIdx_ = list_->registerExternalPDFs( pdfsToRegister );
      numPDFs_ = numeric_cast<typename List::index_t>( pdfsToRegister.size() );
   }
   
   void updatePDFs()
   {

      // Update velocities
      WALBERLA_ASSERT_EQUAL( nearBoundaryCells_.size(), nearBoundaryVelocities_.size() )

      {
         uint_t numCells = nearBoundaryCells_.size();
         for( uint_t i = 0; i < numCells; ++i )
         {
            nearBoundaryVelocities_[i] = list_->getVelocity( nearBoundaryCells_[i] );
         }
      }

      // UpdatePDFs
      WALBERLA_ASSERT_EQUAL( velocityIdx_.size(),            antiBounceBackPDFIdx_.size() )
      WALBERLA_ASSERT_EQUAL( fluidPressureDirection_.size(), antiBounceBackPDFIdx_.size() )

      {
         for( typename List::index_t i  = 0; i < numPDFs_; ++i )
         {
            typename List::index_t pdfIdx = pdfsStartIdx_ + i;
            list_->get( pdfIdx ) = - list_->get( antiBounceBackPDFIdx_[ i ] )
                                   + real_t( 2 ) * EquilibriumDistribution< typename List::LatticeModel >::getSymmetricPart( fluidPressureDirection_[i], nearBoundaryVelocities_[ velocityIdx_[ i ] ], latticeDensity_ );
         }
      }
   }

   bool operator==( const ListPressureBoundary & other )
   {
      return this->list_                 == other.list_
          && this->nearBoundaryCells_    == other.nearBoundaryCells_
          && this->pdfsStartIdx_         == other.pdfsStartIdx_
          && this->velocityIdx_          == other.velocityIdx_
          && this->antiBounceBackPDFIdx_ == other.antiBounceBackPDFIdx_
          && isIdentical( this->latticeDensity_, other.latticeDensity_ );
   }

   inline void toBuffer( mpi::SendBuffer & buffer ) const
   {
      buffer << nearBoundaryCells_ << pdfsStartIdx_ << numPDFs_ << velocityIdx_ << antiBounceBackPDFIdx_ << fluidPressureDirection_ << latticeDensity_;
   }

   inline void fromBuffer( List * const list, mpi::RecvBuffer & buffer )
   {
      list_ = list;

      buffer >> nearBoundaryCells_ >> pdfsStartIdx_ >> numPDFs_ >> velocityIdx_ >> antiBounceBackPDFIdx_ >> fluidPressureDirection_ >> latticeDensity_;

      nearBoundaryVelocities_.resize( nearBoundaryCells_.size(), Vector3< real_t >( std::numeric_limits<real_t>::signaling_NaN() ) );
   }


private:
   List * list_;

   std::vector< uint_t > nearBoundaryCells_;
   std::vector< Vector3< real_t > >      nearBoundaryVelocities_;
   
   typename List::index_t pdfsStartIdx_;
   typename List::index_t numPDFs_;

   std::vector< uint_t >                 velocityIdx_;
   std::vector< typename List::index_t > antiBounceBackPDFIdx_;
   std::vector< stencil::Direction >     fluidPressureDirection_;
   
   real_t  latticeDensity_;   
};



template<typename List_T>
class ListPressureBoundaryBlockDataHandling : public domain_decomposition::BlockDataHandling< ListPressureBoundary<List_T> >
{
public:
   ListPressureBoundaryBlockDataHandling( const BlockDataID & listId ) : listId_( listId ) {}

   virtual ~ListPressureBoundaryBlockDataHandling() = default;

   virtual ListPressureBoundary<List_T> * initialize( IBlock * const /*block*/ )
   {
      return new ListPressureBoundary<List_T>();
   }

   virtual void serialize( IBlock * const block, const BlockDataID & id, mpi::SendBuffer & buffer )
   {
      ListPressureBoundary<List_T> * pbh = block->getData<ListPressureBoundary<List_T>>( id );
      WALBERLA_ASSERT_NOT_NULLPTR( pbh )

      pbh->toBuffer( buffer );
   }

   virtual ListPressureBoundary<List_T> * deserialize( IBlock * const /*block*/ )
   {
      return new ListPressureBoundary<List_T>();
   }

   virtual void deserialize( IBlock * const block, const BlockDataID & id, mpi::RecvBuffer & buffer )
   {
      ListPressureBoundary<List_T> * pbh = block->getData<ListPressureBoundary<List_T>>( id );
      WALBERLA_ASSERT_NOT_NULLPTR( pbh )

      List_T * list = block->getData<List_T>( listId_ );

      pbh->fromBuffer( list, buffer );
   }

protected:
   BlockDataID listId_;
};


template<typename List_T>
BlockDataID addListPressureBoundaryToStorage( const BlockDataID & listId,
                                              const shared_ptr< StructuredBlockStorage >& bs,
                                              const std::string & identifier,
                                              const Set<SUID>& requiredSelectors = Set<SUID>::emptySet(),
                                              const Set<SUID>& incompatibleSelectors = Set<SUID>::emptySet() )
{
   return bs->addBlockData( make_shared< ListPressureBoundaryBlockDataHandling< List_T> >( listId ), identifier, requiredSelectors, incompatibleSelectors );
}


template< typename List_T >
class ListPressureBoundaryHandling
{
public:
   ListPressureBoundaryHandling( const BlockDataID pressureBoundaryHandlingId ) : pressureBoundaryHandlingId_( pressureBoundaryHandlingId ) { }
   void operator()( IBlock * const block )
   {
      ListPressureBoundary< List_T > * pbh = block->getData< ListPressureBoundary< List_T > >( pressureBoundaryHandlingId_ );
      WALBERLA_ASSERT_NOT_NULLPTR( pbh )
      pbh->updatePDFs();
   }

protected:
   BlockDataID pressureBoundaryHandlingId_;
};


} // namespace lbm
} // namespace walberla
