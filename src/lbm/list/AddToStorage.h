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
//! \file AddToStorage.h
//! \ingroup lbm
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#pragma once

#include "List.h"

namespace walberla {
namespace lbm {

template<typename List_T>
class ListBlockDataHandling : public domain_decomposition::BlockDataHandling< List_T >
{
 public:
   ListBlockDataHandling(const Vector3<uint64_t> split, const bool manuallyAllocateTmpPDFs = false )
      : split_(split), manuallyAllocateTmpPDFs_( manuallyAllocateTmpPDFs )
   { }

   virtual ~ListBlockDataHandling() = default;

   virtual List_T * initialize( IBlock * const /*block*/ )
   {
      return new List_T( split_, manuallyAllocateTmpPDFs_ );
   }

   virtual void serialize( IBlock * const block, const BlockDataID & id, mpi::SendBuffer & buffer )
   {
      List_T * list = block->getData<List_T>( id );
      WALBERLA_ASSERT_NOT_NULLPTR( list )

      list->toBuffer( buffer );
   }

   virtual List_T * deserialize( IBlock * const /*block*/ )
   {
      return new List_T( split_, manuallyAllocateTmpPDFs_ );
   }

   virtual void deserialize( IBlock * const block, const BlockDataID & id, mpi::RecvBuffer & buffer )
   {
      List_T * list = block->getData<List_T>( id );
      WALBERLA_ASSERT_NOT_NULLPTR( list )

      list->fromBuffer( buffer );
   }

 protected:
   const Vector3<uint64_t> split_;
   bool     manuallyAllocateTmpPDFs_;
};


template<typename List_T>
BlockDataID addListToStorage( const shared_ptr< StructuredBlockStorage >& bs,
                             const std::string & identifier,
                             const Vector3<uint64_t> split,
                             const bool manuallyAllocateTmpPDFs = false,
                             const Set<SUID>& requiredSelectors = Set<SUID>::emptySet(),
                             const Set<SUID>& incompatibleSelectors = Set<SUID>::emptySet() )
{
   return bs->addBlockData( make_shared< ListBlockDataHandling<List_T> >(split, manuallyAllocateTmpPDFs), identifier, requiredSelectors, incompatibleSelectors );
}

} // namespace lbm
} // namespace walberla