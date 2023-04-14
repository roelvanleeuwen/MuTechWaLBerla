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
//! \file SetupHybridCommunication.h
//! \author Philipp Suffa <philipp.suffa@fau.de>
//
//======================================================================================================================
#pragma once

#include "blockforest/StructuredBlockForest.h"

#include "lbm/list/CellDir.h"

#include "core/DataTypes.h"
#include "core/Macros.h"
#include "core/cell/Cell.h"
#include "core/debug/CheckFunctions.h"
#include "core/math/Vector3.h"
#include "core/logging/Logging.h"

#include "core/mpi/all.h"

#include "field/FlagField.h"

namespace walberla {


template<typename FlagField_T, typename Stencil_T>
class SetupHybridCommunication
{
 public:
   SetupHybridCommunication( weak_ptr<StructuredBlockForest> blockForest, const BlockDataID flagFieldID, const FlagUID fluidFlagUID)
      : blockForest_(blockForest), flagFieldID_(flagFieldID),  fluidFlagUID_(fluidFlagUID)
   {
      auto forest = blockForest_.lock();
      WALBERLA_CHECK_NOT_NULLPTR( forest, "Trying to execute communication for a block storage object that doesn't exist anymore" )

      WALBERLA_LOG_PROGRESS( "Setting up list communication" )

      std::map< uint_t, uint_t > numBlocksToSend;

      blockSize_ = Vector3<cell_idx_t>( cell_idx_c(forest->getNumberOfXCellsPerBlock()), cell_idx_c(forest->getNumberOfYCellsPerBlock()), cell_idx_c(forest->getNumberOfZCellsPerBlock()) );

      for( auto senderIt = forest->begin(); senderIt != forest->end(); ++senderIt )
      {
         blockforest::Block & sender = dynamic_cast<blockforest::Block &>( *senderIt );

         for (size_t f = 1; f < Stencil_T::Size; ++f)
         {
            auto neighborhood = sender.getNeighborhoodSection( blockforest::getBlockNeighborhoodSectionIndex( (stencil::Direction) f ) );
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
         auto *flagField = sender.getData<FlagField_T>(flagFieldID_);
         auto fluidFlag = flagField->getFlag( fluidFlagUID_ );

         for (size_t sendDir = 1; sendDir < Stencil_T::Size; ++sendDir)
         {
            auto neighborhood = sender.getNeighborhoodSection( blockforest::getBlockNeighborhoodSectionIndex( (stencil::Direction) sendDir ) );
            WALBERLA_ASSERT_LESS_EQUAL( neighborhood.size(), size_t( 1 ) )
            if( neighborhood.empty() )
               continue;
            auto * receiver = neighborhood.front();
            receiver->getId().toBuffer( bufferSystem.sendBuffer( receiver->getProcess() ) );
            bufferSystem.sendBuffer( receiver->getProcess() ) << stencil::inverseDir[sendDir];
            WALBERLA_LOG_DETAIL( "Packing information for block " << receiver->getId().getID() << " in direction " << stencil::dirToString[stencil::inverseDir[f]] );

            auto flagIt = flagField->beginSliceBeforeGhostLayerXYZ(Stencil_T::dir[sendDir]);
            std::vector< Cell > isBoundary;
            while( flagIt != flagField->end() )
            {
               //get send information
               Cell cell(flagIt.x(), flagIt.y(), flagIt.z());
               if (!flagField->isFlagSet(cell, fluidFlag)) {
                  isBoundary.push_back(cell);
               }
               ++flagIt;
            }

            std::sort( isBoundary.begin(), isBoundary.end() );
            bufferSystem.sendBuffer( receiver->getProcess() ) << isBoundary.size();
            for (auto boundaryCell : isBoundary) {
               bufferSystem.sendBuffer( receiver->getProcess() ) << mapToNeighbor( boundaryCell, (stencil::Direction) sendDir );
            }
         }
      }

      bufferSystem.setReceiverInfoFromSendBufferState( false, false );
      WALBERLA_LOG_PROGRESS( "MPI exchange of structure data" )
      bufferSystem.sendAll();
      WALBERLA_LOG_PROGRESS( "MPI exchange of structure data finished" )

      for( auto recvBufferIt = bufferSystem.begin(); recvBufferIt != bufferSystem.end(); ++recvBufferIt )
      {
         recvBufferIt.buffer().clear();
         continue;
      }
   }


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

 protected:
   weak_ptr<StructuredBlockForest> blockForest_;
   const BlockDataID flagFieldID_;
   const FlagUID fluidFlagUID_;
   Vector3< cell_idx_t > blockSize_;
};

} // namespace walberla