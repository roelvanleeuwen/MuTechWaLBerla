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
//! \file SparseField.h
//! \ingroup lbm
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#pragma once

#include "field/CMakeDefs.h"
#include "field/Layout.h"
#include "field/allocation/FieldAllocator.h"

#include "core/DataTypes.h"
#include "core/cell/CellInterval.h"
#include "core/debug/CheckFunctions.h"
#include "core/debug/Debug.h"

#include "domain_decomposition/StructuredBlockStorage.h"

#include <functional>
#include <memory>
#include <vector>

namespace walberla {
namespace lbm {

using field::Layout;
using field::fzyx;
using field::zyxf;


template<typename T, uint_t Alignment>
class SparseField
{
 public:

   using value_type = T;

   SparseField( uint_t xSize, uint_t fSize, const Layout & layout = fzyx)
   {
      init(xSize, fSize, layout);
   }
   void init( uint_t xSize, uint_t fSize, const Layout & layout = fzyx);
   static uint_t alignment() { return Alignment; }

   ~SparseField() = default;

   bool operator==( const SparseField<T, Alignment> & other ) const;
   bool operator!=( const SparseField<T, Alignment> & other ) const;

   inline bool hasSameAllocSize( const SparseField<T,Alignment> & other ) const;

   SparseField<T, Alignment> * cloneUninitialized() const;
   SparseField<T, Alignment> * cloneShallowCopy()   const;

   void swapDataPointers( SparseField<T, Alignment> & other );
   void swapDataPointers( SparseField<T, Alignment> * other ) { swapDataPointers( *other ); }


   //@}
   //****************************************************************************************************************


   //** Element Access **********************************************************************************************
   /*! \name Element Access */
   //@{


   //@}
   //****************************************************************************************************************


   //** Equality Checks *********************************************************************************************
   /*! \name Equality Checks */
   //@{
   //@}
   //****************************************************************************************************************


   //** Size and Layout Information *********************************************************************************
   /*! \name Size and Layout Information */
   //@{
   inline uint_t  xSize() const  { return xSize_; }
   inline uint_t  fSize() const  { return fSize_; }

   inline uint_t  xAllocSize() const  { return xAllocSize_; }
   inline uint_t  fAllocSize() const  { return fAllocSize_; }
   inline uint_t  allocSize()  const  { return allocSize_;  }

   int64_t xStride() const { return xfact_;}
   int64_t fStride() const { return ffact_; }

   inline Layout layout() const { return layout_; }


   // bool coordinatesValid( cell_idx_t x, cell_idx_t y, cell_idx_t z, cell_idx_t f ) const;
   //@}
   //****************************************************************************************************************


   //** Pointer to internal memory - use with care!        **********************************************************
   /*! \name Pointer to internal memory - use with care!  */
   //@{
   T * data()            { return values_.front(); }
   const T * data() const      { return values_.front(); }
//   T * dataAt(cell_idx_t x, cell_idx_t y, cell_idx_t z, cell_idx_t f)       { return &get(x,y,z,f); }
//   const T * dataAt(cell_idx_t x, cell_idx_t y, cell_idx_t z, cell_idx_t f) const { return &get(x,y,z,f); }

   //@}
   //****************************************************************************************************************

   inline void toBuffer  ( mpi::SendBuffer & buffer ) const;
   inline void fromBuffer( mpi::RecvBuffer & buffer );
 private:

   // SparseField & operator=( const SparseField & );
   std::vector< real_t, simd::aligned_allocator< real_t, Alignment > > values_;

   uint_t xSize_;         //!< Number of cells (excluding padded cells)
   uint_t fSize_;        //!< Values per cell

   uint_t xAllocSize_;     //!< Number of cells (including padded cells)
   uint_t fAllocSize_;    //!< Values per cell (including padded cells)

   Layout layout_;        //!< Determines in which order the values are stored

   uint_t allocSize_;     //!< The overall size of the T* (padding included)

   int64_t ffact_;        //!< Access multiplication factor for the f-dimension.
   int64_t xfact_;        //!< Access multiplication factor for the x-dimension.

}; // class SparseField


template<typename T, uint_t Alignment>
void SparseField<T, Alignment>::init( uint_t _xSize, uint_t _fSize, const Layout & l)
{
   xSize_ = _xSize;
   fSize_ = _fSize;
   layout_ = l;

   uint_t numberOfCells = xSize_ * fSize_;
   if (layout_ == fzyx )
   {
      uint_t alignedStepSize = std::max(uint_t(1), alignment() / sizeof(real_t));
      if ((numberOfCells % alignedStepSize) == 0)
         xAllocSize_ = numberOfCells;
      else
         xAllocSize_ =  (numberOfCells / alignedStepSize + uint_t(1)) * alignedStepSize;

      fAllocSize_ = _fSize;
      ffact_ = int64_t(xAllocSize_);
      xfact_ = 1;
   }
   else {
      xAllocSize_ = numberOfCells;
      fAllocSize_ = _fSize;

      xfact_ = int64_t (fAllocSize_);
      ffact_ = 1;
   }
   allocSize_ = fAllocSize_ * xAllocSize_;

   values_.assign( xAllocSize_ * fAllocSize_, std::numeric_limits<real_t>::signaling_NaN() );
}


template< typename T, uint_t Alignment>
inline void SparseField< T, Alignment >::toBuffer( mpi::SendBuffer & buffer ) const
{
   buffer << values_;
}

template< typename T, uint_t Alignment >
inline void SparseField< T, Alignment >::fromBuffer( mpi::RecvBuffer & buffer )
{
   buffer >> values_;
}


template< typename T, uint_t Alignment>
bool SparseField<T, Alignment>::operator==( const SparseField<T, Alignment> & other ) const
{
   return values_ == other.values_;
}

template< typename T, uint_t Alignment>
bool SparseField<T, Alignment>::operator!=( const SparseField<T, Alignment> & other ) const
{
   return !( *this == other );
}

//*******************************************************************************************************************
/*! True if allocation sizes of all dimensions match
    *******************************************************************************************************************/
template<typename T, uint_t Alignment>
inline bool SparseField<T, Alignment>::hasSameAllocSize( const SparseField<T, Alignment> & other ) const
{
   return xAllocSize_ == other.xAllocSize_ &&
          fAllocSize_ == other.fAllocSize_;
}


template<typename T, uint_t Alignment>
SparseField<T, Alignment> * SparseField<T, Alignment>::cloneShallowCopy() const
{
   return new SparseField<T, Alignment>(*this);
}

template <typename T, uint_t Alignment>
SparseField<T, Alignment> * SparseField<T, Alignment>::cloneUninitialized() const
{
   SparseField<T, Alignment> * res = cloneShallowCopy();
   res->allocator_->decrementReferenceCount( res->values_ );
   res->values_ = res->allocator_->allocate ( res->allocSize() );

   WALBERLA_ASSERT ( hasSameSize     ( *res ) )
   WALBERLA_ASSERT ( hasSameAllocSize( *res ) )

   return res;
}

template<typename T, uint_t Alignment>
inline void SparseField<T, Alignment>::swapDataPointers( SparseField<T, Alignment> & other)
{
   WALBERLA_ASSERT( hasSameAllocSize(other) )
   WALBERLA_ASSERT( hasSameSize(other) )
   WALBERLA_ASSERT( layout() == other.layout() )
   std::swap( values_, other.values_ );
}


template<typename SparseField_T>
class SparseFieldBlockDataHandling : public domain_decomposition::BlockDataHandling< SparseField_T >
{
 public:
   SparseFieldBlockDataHandling( const uint_t xSize, const uint_t fSize, const Layout & layout = zyxf)
      : xSize_( xSize ), fSize_( fSize ), layout_(layout)
   { }

   virtual ~SparseFieldBlockDataHandling() = default;

   virtual SparseField_T * initialize( IBlock * const /*block*/ )
   {
      return new SparseField_T( xSize_, fSize_, layout_);
   }

   virtual void serialize( IBlock * const block, const BlockDataID & id, mpi::SendBuffer & buffer )
   {
      SparseField_T * sparseField = block->getData<SparseField_T>( id );
      WALBERLA_ASSERT_NOT_NULLPTR( sparseField )

      sparseField->toBuffer( buffer );
   }

   virtual SparseField_T * deserialize( IBlock * const /*block*/ )
   {
      return new SparseField_T( xSize_, fSize_, layout_);
   }

   virtual void deserialize( IBlock * const block, const BlockDataID & id, mpi::RecvBuffer & buffer )
   {
      SparseField_T * sparseField = block->getData<SparseField_T>( id );
      WALBERLA_ASSERT_NOT_NULLPTR( sparseField )

      sparseField->fromBuffer( buffer );
   }

 protected:
   uint_t xSize_;
   uint_t fSize_;
   Layout layout_;
};


template<typename SparseField_T>
BlockDataID addSparseFieldToStorage( const shared_ptr< StructuredBlockStorage >& bs,
                                    const std::string & identifier,
                                    const uint_t xSize, const uint_t fSize,
                                    const Layout &layout = zyxf,
                                    const Set<SUID>& requiredSelectors = Set<SUID>::emptySet(),
                                    const Set<SUID>& incompatibleSelectors = Set<SUID>::emptySet() )
{
   // TODO can this be structured block data?
   // TODO Think about more than 1 block per process
   return bs->addBlockData( make_shared< SparseFieldBlockDataHandling<SparseField_T> >( xSize, fSize, layout ), identifier, requiredSelectors, incompatibleSelectors );
}

} // namespace lbm
} // namespace walberla