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
//! \file ListCommunication.h
//! \ingroup lbm
//! \author Christian Godenschwager <christian.godenschwager@fau.de>
//
//======================================================================================================================

#pragma once

#include "List.h"
#include "CellDir.h"

#include "blockforest/Block.h"
#include "blockforest/BlockNeighborhoodSection.h"
#include "blockforest/StructuredBlockForest.h"

#include "communication/UniformPackInfo.h"

#include "core/Macros.h"
#include "core/debug/Debug.h"
#include "core/mpi/BufferSystem.h"

#include "domain_decomposition/IBlock.h"

#include "stencil/AABBQuadrant.h"
#include "stencil/Directions.h"

#include <map>
#include <vector>


namespace walberla {
namespace lbm {

template< typename List_T, typename Stencil_T>
class ListPackInfo : public walberla::communication::UniformPackInfo
{
public:

//   typedef typename Stencil_T      Stencil;

   ListPackInfo( const BlockDataID & listId, const shared_ptr<StructuredBlockForest>& blockForest )
      : listId_( listId ), exchangeStructure_( true ), blockForest_( blockForest ),
        blockSize_( cell_idx_c( blockForest->getNumberOfXCellsPerBlock() ),
                    cell_idx_c( blockForest->getNumberOfYCellsPerBlock() ),
                    cell_idx_c( blockForest->getNumberOfZCellsPerBlock() ) ),
        modificationStamp_(0)
   {
   }

   ~ListPackInfo() override = default;

   bool constantDataExchange() const override { return !exchangeStructure_; }
   bool threadsafeReceiving()  const override { return !exchangeStructure_; }

   void unpackData( IBlock * receiver, stencil::Direction dir, mpi::RecvBuffer & buffer ) override;

   void communicateLocal( const IBlock * sender, IBlock * receiver, stencil::Direction dir ) override;

   void beforeStartCommunication() override { setupCommunication(); }

protected:

   void packDataImpl( const IBlock * sender, stencil::Direction dir, mpi::SendBuffer & outBuffer ) const override;

   void setupCommunication();

   static CellInterval getSendCellInterval( const Vector3<uint_t> & blockSize, const stencil::Direction dir );
   Cell mapToNeighbor( Cell cell, const stencil::Direction dir )
   {
      switch( stencil::cx[dir] )
      {
      case -1: cell.x() += blockSize_[0]; break;
      case  1: cell.x() -= blockSize_[0]; break;
      default: ;
      }

      switch( stencil::cy[dir] )
      {
      case -1: cell.y() += blockSize_[1]; break;
      case  1: cell.y() -= blockSize_[1]; break;
      default: ;
      }

      switch( stencil::cz[dir] )
      {
      case -1: cell.z() += blockSize_[2]; break;
      case  1: cell.z() -= blockSize_[2]; break;
      default: ;
      }

      return cell;
   }

   //Cell getLocalCell ( Cell cell, const IBlock & block ) { blockForest_->transformGlobalToBlockLocalCell( cell, block ); return cell; }

   const BlockDataID listId_;
   
   bool exchangeStructure_;

   weak_ptr<StructuredBlockForest> blockForest_;

   Vector3< cell_idx_t > blockSize_;

   std::map< std::pair< const IBlock*, stencil::Direction >, std::vector< typename List_T::index_t > > sendPDFs_;
   std::map< std::pair< const IBlock*, stencil::Direction >, typename List_T::index_t > startIdxs_;
   std::map< std::pair< const IBlock*, stencil::Direction >, typename List_T::index_t > numPDFs_;

   uint_t modificationStamp_;
};

struct CellInCellIntervalFilter
{
   CellInCellIntervalFilter( const CellInterval & _ci ) : ci( _ci ) { }
   bool operator()( const Cell & cell ) const { return ci.contains( cell );  }

   CellInterval ci;
};


template< typename List_T, typename Stencil_T>
CellInterval ListPackInfo< List_T, Stencil_T >::getSendCellInterval( const Vector3<uint_t> & blockSize, const stencil::Direction dir )
{
   const cell_idx_t sizeArr[] = { cell_idx_c( blockSize[0] ),
                                  cell_idx_c( blockSize[1] ),
                                  cell_idx_c( blockSize[2] ) };

   CellInterval ci;

   for( uint_t dim = 0; dim < 3; ++dim )
   {
      switch( stencil::c[dim][dir] )
      {
      case -1:
         ci.min()[dim] = 0;
         ci.max()[dim] = 0;
         break;
      case  0:
         ci.min()[dim] = 0;
         ci.max()[dim] = sizeArr[dim] - 1;
         break;
      case 1:
         ci.min()[dim] = sizeArr[dim] - 1;
         ci.max()[dim] = sizeArr[dim] - 1;
         break;
      }
   }

   return ci;
}



template< typename List_T, typename Stencil_T>
void ListPackInfo< List_T, Stencil_T>::setupCommunication()
{
   auto forest = blockForest_.lock();
   WALBERLA_CHECK_NOT_NULLPTR( forest, "Trying to execute communication for a block storage object that doesn't exist anymore" )

   if( modificationStamp_ != forest->getBlockForest().getModificationStamp() )
   {
      exchangeStructure_ = true;
   }
   modificationStamp_ = forest->getBlockForest().getModificationStamp();

   if( !exchangeStructure_ )
      return;

   WALBERLA_LOG_PROGRESS( "Setting up list communication" )

   sendPDFs_.clear();
   startIdxs_.clear();
   numPDFs_.clear();

   std::map< uint_t, uint_t > numBlocksToSend;

   const math::Vector3<uint_t> blockSize( forest->getNumberOfXCellsPerBlock(), forest->getNumberOfYCellsPerBlock(), forest->getNumberOfZCellsPerBlock() );

   for( auto senderIt = forest->begin(); senderIt != forest->end(); ++senderIt )
   {
      blockforest::Block & sender = dynamic_cast<blockforest::Block &>( *senderIt );

      for( auto dirIt = Stencil_T::beginNoCenter(); dirIt != Stencil_T::end(); ++dirIt )
      {
         auto neighborhood = sender.getNeighborhoodSection( blockforest::getBlockNeighborhoodSectionIndex( *dirIt ) );
         WALBERLA_ASSERT_LESS_EQUAL( neighborhood.size(), size_t( 1 ) )
         if( neighborhood.empty() )
            continue;
         auto * receiver = neighborhood.front();

         numBlocksToSend[receiver->getProcess()] += uint_t(1);
      }
   }

   mpi::BufferSystem bufferSystem( mpi::MPIManager::instance()->comm() );

   for( auto it = numBlocksToSend.begin(); it != numBlocksToSend.end(); ++it )
   {
      WALBERLA_LOG_DETAIL( "Packing information for " << it->second << " blocks to send to process " << it->first );
      bufferSystem.sendBuffer( it->first ) << it->second;
   }

   for( auto senderIt = forest->begin(); senderIt != forest->end(); ++senderIt )
   {
      blockforest::Block & sender = dynamic_cast<blockforest::Block &>( *senderIt );
      List_T * senderList = sender.getData< List_T >( listId_ );
      WALBERLA_ASSERT_NOT_NULLPTR( senderList )

      for( auto dirIt = Stencil_T::beginNoCenter(); dirIt != Stencil_T::end(); ++dirIt )
      {
         auto neighborhood = sender.getNeighborhoodSection( blockforest::getBlockNeighborhoodSectionIndex( *dirIt ) );
         WALBERLA_ASSERT_LESS_EQUAL( neighborhood.size(), size_t( 1 ) )
         if( neighborhood.empty() )
            continue;
         auto * receiver = neighborhood.front();

         receiver->getId().toBuffer( bufferSystem.sendBuffer( receiver->getProcess() ) );
         bufferSystem.sendBuffer( receiver->getProcess() ) << dirIt.inverseDir();

         WALBERLA_LOG_DETAIL( "Packing information for block " << receiver->getId().getID() << " in direction " << stencil::dirToString[dirIt.inverseDir()] ); 

         const CellInterval cellsToSendInterval = getSendCellInterval( blockSize, *dirIt );
         uint_t numCells = uint_c( std::count_if( senderList->getFluidCells().begin(), senderList->getFluidCells().end(), CellInCellIntervalFilter( cellsToSendInterval ) ) );
         WALBERLA_LOG_DETAIL( numCells << " cells found" ); 
         bufferSystem.sendBuffer( receiver->getProcess() ) << numCells;
         for( auto cellIt = senderList->getFluidCells().begin(); cellIt != senderList->getFluidCells().end(); ++cellIt )
         {
            if( cellsToSendInterval.contains( *cellIt ) )
            {
               bufferSystem.sendBuffer( receiver->getProcess() ) << mapToNeighbor( *cellIt, *dirIt );
            }
         }  
      }
   }

   bufferSystem.setReceiverInfoFromSendBufferState( false, false );
   WALBERLA_LOG_PROGRESS( "MPI exchange of structure data" )
   bufferSystem.sendAll();
   WALBERLA_LOG_PROGRESS( "MPI exchange of structure data finished" )

   for( auto recvBufferIt = bufferSystem.begin(); recvBufferIt != bufferSystem.end(); ++recvBufferIt )
   {
      uint_t numBlocks;
      recvBufferIt.buffer() >> numBlocks;
      WALBERLA_LOG_DETAIL( "Unpacking information from " << numBlocks << " blocks from process " << recvBufferIt.rank() );
      for( uint_t i = 0; i < numBlocks; ++i )
      {
         BlockID localBID;
         localBID.fromBuffer( recvBufferIt.buffer() );
         stencil::Direction dir;
         uint_t numCells;
         recvBufferIt.buffer() >> dir;

         WALBERLA_LOG_DETAIL( "Unpacking information for block " << localBID.getID() << " in direction " << stencil::dirToString[dir] );

         recvBufferIt.buffer() >> numCells;

         IBlock * localBlock = forest->getBlock( localBID );
         WALBERLA_ASSERT_NOT_NULLPTR( localBlock )

         std::vector<Cell> ghostCells( numCells );
         for( auto it = ghostCells.begin(); it != ghostCells.end(); ++it )
         {
            recvBufferIt.buffer() >> *it;
         }

         WALBERLA_LOG_DETAIL( ghostCells.size() << " cells found" ); 

         List_T * senderList = localBlock->template getData< List_T >( listId_ );
         WALBERLA_ASSERT_NOT_NULLPTR( senderList )

         std::vector< CellDir > externalPDFs;
         for( auto it = ghostCells.begin(); it != ghostCells.end(); ++it )
         {
            for( auto dirIt = Stencil_T::beginNoCenter(); dirIt != Stencil_T::end(); ++dirIt )
            {
               Cell neighborCell = *it + *dirIt;
               if( senderList->isFluidCell( neighborCell ) )
               {
                  externalPDFs.push_back( CellDir( *it, *dirIt ) );
               }
            }
         }

//         WALBERLA_LOG_INFO_ON_ROOT("senderList->registerExternalPDFs( externalPDFs );" << externalPDFs.size())

         startIdxs_[std::make_pair( localBlock, dir )] = senderList->registerExternalPDFs( externalPDFs );
         numPDFs_[  std::make_pair( localBlock, dir ) ] = static_cast< typename List_T::index_t >( externalPDFs.size() );

         std::sort( ghostCells.begin(), ghostCells.end() );

         auto & sendPDFsVector = sendPDFs_[ std::make_pair( localBlock, dir ) ];

         for( auto it = senderList->getFluidCells().begin(); it != senderList->getFluidCells().end(); ++it )
         {
            for( auto dirIt = Stencil_T::beginNoCenter(); dirIt != Stencil_T::end(); ++dirIt )
            {
               Cell neighborCell = *it + *dirIt;
               if( std::binary_search( ghostCells.begin(), ghostCells.end(), neighborCell ) )
               {
                  sendPDFsVector.push_back( senderList->getPDFIdx( *it, *dirIt ) );
               }
            }
         }
      }
   }

   exchangeStructure_ = false;

   WALBERLA_LOG_PROGRESS( "Setting up list communication finished" )
}


template< typename List_T, typename Stencil_T>
void ListPackInfo< List_T, Stencil_T>::unpackData( IBlock * receiver, stencil::Direction dir, mpi::RecvBuffer & buffer )
{
//   if( Stencil::idx[ dir ] >= Stencil::Size )
//      return;

   List_T * list = receiver->getData< List_T >( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( list )

   auto numPDFsIt =  numPDFs_.find( std::make_pair( receiver, dir ) );
   WALBERLA_ASSERT_UNEQUAL( numPDFsIt, numPDFs_.end() )
   
   if( numPDFsIt->second == 0 )
      return;

   auto startIdxIt = startIdxs_.find( std::make_pair( receiver, dir ) );
   WALBERLA_ASSERT_UNEQUAL( startIdxIt, startIdxs_.end() )
   
   for( typename List_T::index_t i = 0; i < numPDFsIt->second; ++i )
      buffer >> list->get( startIdxIt->second + i );
}

template< typename T, typename Index_T >
inline void indirect_memcopy( const T * const WALBERLA_RESTRICT srcBasePtr, const Index_T * const WALBERLA_RESTRICT indexes, const size_t N, T * WALBERLA_RESTRICT dstPtr )
{
   for( size_t i = 0; i < N; ++i )
   {
      dstPtr[ i ] = srcBasePtr[ indexes[ i ] ];
   }
}

template< typename List_T, typename Stencil_T>
void ListPackInfo< List_T, Stencil_T>::communicateLocal( const IBlock * sender, IBlock * receiver, stencil::Direction dir )
{
//   if( Stencil::idx[dir] >= Stencil::Size )
//      return;

   const List_T * senderList = sender->getData< List_T >( listId_ );
   List_T * receiverList = receiver->getData< List_T >( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( senderList   )
   WALBERLA_ASSERT_NOT_NULLPTR( receiverList )

   WALBERLA_ASSERT_UNEQUAL( startIdxs_.find( std::make_pair( receiver, stencil::inverseDir[dir] ) ), startIdxs_.end() )
   WALBERLA_ASSERT_UNEQUAL( numPDFs_.find  ( std::make_pair( receiver, stencil::inverseDir[dir] ) ), numPDFs_.end()   )
   WALBERLA_ASSERT_UNEQUAL( sendPDFs_.find( std::make_pair( sender, dir ) ), sendPDFs_.end() )

   const typename List_T::index_t start = startIdxs_[std::make_pair( receiver, stencil::inverseDir[dir] )];

   auto idxMapIt = sendPDFs_.find( std::make_pair( sender, dir ) );
   WALBERLA_ASSERT_UNEQUAL( idxMapIt, sendPDFs_.end() )
   const auto & idxs = idxMapIt->second;

   if( idxs.empty() )
      return;
   
   indirect_memcopy( &( senderList->get( 0 ) ), idxs.data(), idxs.size(), &( receiverList->get( start ) ) );
}


template< typename List_T, typename Stencil_T>
void ListPackInfo< List_T, Stencil_T>::packDataImpl( const IBlock * sender, stencil::Direction dir, mpi::SendBuffer & outBuffer ) const
{
//   if( Stencil::idx[dir] >= Stencil::Size )
//      return;

   const List_T * list = sender->getData< List_T >( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( list )

   auto idxMapIt = sendPDFs_.find( std::make_pair( sender, dir ) );
   WALBERLA_ASSERT_UNEQUAL( idxMapIt, sendPDFs_.end() )
   const auto & idxs = idxMapIt->second;
   
   for( const auto idx : idxs )
   {
      outBuffer << list->get( idx );
   }
}


} // namespace lbm
} // namespace walberla
