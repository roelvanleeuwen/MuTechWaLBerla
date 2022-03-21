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
//! \file FieldIterator.impl.h
//! \ingroup field
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================



namespace walberla {
namespace field {



//**********************************************************************************************************************
/*!\brief Constructs an iterator for the given slice of a field
 *
 * The constructed iterator goes over the specified slice of the field.
 * Iterator coordinates (f(),z(),y(),x() ) return coordinates inside that slice (they do not start at 0)
 *
 **********************************************************************************************************************/
template <typename T>
FieldIterator<T>::FieldIterator( const typename FieldIterator<T>::FieldType * field,
                                    cell_idx_t xBeg, cell_idx_t yBeg, cell_idx_t zBeg, cell_idx_t fBeg,
                                    uint_t sx, uint_t sy, uint_t sz, uint_t sf, bool forward )
   :  f_(field), xBegin_(xBeg), yBegin_(yBeg), zBegin_(zBeg), fBegin_(fBeg)
{
   if ( field->xyzSize().empty() )
   {
      linePtr_ = nullptr;
      lineEnd_ = nullptr;
      f_       = nullptr;
      return;
   }

   using NonConstT = typename std::remove_const<T>::type;

   cur_[0] = cur_[1] = cur_[2] = 0;
   if( f_->layout() == fzyx )
   {
      skips_[0] = ( f_->fAllocSize() - sf ) * uint_c( f_->ffact_ );
      skips_[1] = ( f_->zAllocSize() - sz ) * uint_c( f_->zfact_ );
      skips_[2] = ( f_->yAllocSize() - sy ) * uint_c( f_->yfact_ );
      skips_[3] = ( f_->xAllocSize() - sx ) * uint_c( f_->xfact_ );
      sizes_[0] = sf;
      sizes_[1] = sz;
      sizes_[2] = sy;
      sizes_[3] = sx;

      if ( !forward ) {
         cur_[0] = cell_idx_c( sf - 1 );
         cur_[1] = cell_idx_c( sz - 1 );
         cur_[2] = cell_idx_c( sy - 1 );
      }
   }
   else
   {
      skips_[0] = (f_->zAllocSize() - sz) * uint_c( f_->zfact_ );
      skips_[1] = (f_->yAllocSize() - sy) * uint_c( f_->yfact_ );
      skips_[2] = (f_->xAllocSize() - sx) * uint_c( f_->xfact_ );
      skips_[3] = (f_->fAllocSize() - sf) * uint_c( f_->ffact_ );
      sizes_[0] = sz;
      sizes_[1] = sy;
      sizes_[2] = sx;
      sizes_[3] = sf;

      if ( !forward ) {
         cur_[0] = cell_idx_c( sz - 1 );
         cur_[1] = cell_idx_c( sy - 1 );
         cur_[2] = cell_idx_c( sx - 1 );
      }
   }


   if ( forward )
   {
      lineBegin_ = const_cast<NonConstT *>(& f_->get(xBeg,yBeg,zBeg,fBeg) );
      linePtr_   = lineBegin_;
      lineEnd_   = linePtr_ + sizes_[3];
   }
   else
   {
      linePtr_   = const_cast<NonConstT *>(& f_->get(xBeg + cell_idx_c(sx) - 1,
                                                     yBeg + cell_idx_c(sy) - 1,
                                                     zBeg + cell_idx_c(sz) - 1,
                                                     fBeg + cell_idx_c(sf) - 1) );
      lineEnd_   = linePtr_ + 1;
      lineBegin_ = linePtr_ - sizes_[3] + 1;
   }

   initCoordinateAccessOptimizationPointers();
}





//**********************************************************************************************************************
/*!\brief Constructs an end iterator, which is represented by NULL pointers
 **********************************************************************************************************************/
template <typename T>
FieldIterator<T>::FieldIterator()
   : linePtr_(nullptr), lineEnd_(nullptr), f_(nullptr)
{
}


//**********************************************************************************************************************
/*!\brief Copy Constructor. Required for pointer member cur*_
 **********************************************************************************************************************/
template <typename T>
FieldIterator<T>::FieldIterator( const FieldIterator<T> & o )
   : lineBegin_     ( o.lineBegin_ ),
     linePtr_       ( o.linePtr_      ),
     lineEnd_       ( o.lineEnd_      ),
     f_             ( o.f_            ),
     xBegin_        ( o.xBegin_       ),
     yBegin_        ( o.yBegin_       ),
     zBegin_        ( o.zBegin_       ),
     fBegin_        ( o.fBegin_       )
{
   // no need to copy fastestCoord_, since it is updated before read
   for(int i=0; i<3; ++i)
      cur_[i] = o.cur_[i];

   for( int i=0; i<4; ++i ) {
      skips_[i] = o.skips_[i];
      sizes_[i] = o.sizes_[i];
   }

   if( f_ )
      initCoordinateAccessOptimizationPointers();
}

//**********************************************************************************************************************
/*!\brief Assignment operator. Required for pointer member cur*_
 **********************************************************************************************************************/
template <typename T>
FieldIterator<T> & FieldIterator<T>::operator= ( const FieldIterator<T> & o )
{
   if ( &o == this)
      return *this;

   lineBegin_ = o.lineBegin_;
   linePtr_   = o.linePtr_  ;
   lineEnd_   = o.lineEnd_  ;
   f_         = o.f_        ;
   xBegin_    = o.xBegin_   ;
   yBegin_    = o.yBegin_   ;
   zBegin_    = o.zBegin_   ;
   fBegin_    = o.fBegin_   ;

   for(int i=0; i<3; ++i)
      cur_[i] = o.cur_[i];

   for( int i=0; i<4; ++i ) {
      skips_[i] = o.skips_[i];
      sizes_[i] = o.sizes_[i];
   }

   if( f_ )
      initCoordinateAccessOptimizationPointers();

   return *this;
}


//**********************************************************************************************************************
/*!\brief Initializes pointers required for the optimized x(),y(),z(),f() functions
 *        See documentation of fastestCoord_, curX_, curY_, curZ_ and curF_
 **********************************************************************************************************************/
template <typename T>
void FieldIterator<T>::initCoordinateAccessOptimizationPointers( )
{
   if( f_->layout() == fzyx )
   {
      curF_ = &( cur_[0] );
      curZ_ = &( cur_[1] );
      curY_ = &( cur_[2] );
      curX_ = &( fastestCoord_ );
   }
   else
   {
      curZ_ = &( cur_[0] );
      curY_ = &( cur_[1] );
      curX_ = &( cur_[2] );
      curF_ = &( fastestCoord_ );
   }
}


//**********************************************************************************************************************
/*!\brief Increments the slower 3 coordinates, if innermost coordinate is at end
 *
 **********************************************************************************************************************/
template <typename T>
inline void FieldIterator<T>::incrementLine()
{
   WALBERLA_ASSERT_EQUAL( linePtr_, lineEnd_ )

   linePtr_ += skips_[3];
   cur_[2]++;

   if(cur_[2] == cell_idx_c(sizes_[2]) )
   {
      linePtr_ += skips_[2];
      cur_[2] = 0;
      cur_[1]++;
      if(cur_[1] == cell_idx_c(sizes_[1])  )
      {
         linePtr_ += skips_[1];
         cur_[1] = 0;
         cur_[0]++;
         if(cur_[0] == cell_idx_c(sizes_[0]) )
         {
            // iterator at end
            linePtr_ = nullptr;
            return;
         }
      }
   }

   lineEnd_   = linePtr_ + sizes_[3];
   lineBegin_ = linePtr_;
}


//**********************************************************************************************************************
/*!\brief Decrements the slower 3 coordinates, if innermost coordinate is at beginning
 *
 **********************************************************************************************************************/
template <typename T>
inline void FieldIterator<T>::decrementLine()
{
   WALBERLA_ASSERT_EQUAL( linePtr_, lineBegin_-1 )

   linePtr_ -= skips_[3];
   cur_[2]--;

   if(cur_[2] < 0 )
   {
      linePtr_ -= skips_[2];
      cur_[2] = cell_idx_c(sizes_[2])-1;
      cur_[1]--;
      if(cur_[1] < 0  )
      {
         linePtr_ -= skips_[1];
         cur_[1] = cell_idx_c(sizes_[1])-1;
         cur_[0]--;
         if(cur_[0] < 0 )
         {
            // iterator at end
            linePtr_ = nullptr;
            return;
         }
      }
   }

   lineEnd_   = linePtr_+1;
   lineBegin_ = linePtr_ - sizes_[3]+1;
}


//**********************************************************************************************************************
/*!\brief Equal operator.
 *
 * Test equality only by comparing the internal pointer
 *
 * \return true if both iterators are equal
 **********************************************************************************************************************/
template <typename T>
inline bool FieldIterator<T>::operator==( const FieldIterator<T>& it ) const
{
   return it.linePtr_ == this->linePtr_;
}



//**********************************************************************************************************************
/*!\brief Unequal operator.
 **********************************************************************************************************************/
template <typename T>
inline bool FieldIterator<T>::operator!=( const FieldIterator<T>& it ) const
{
   return it.linePtr_ != this->linePtr_;
}



//**********************************************************************************************************************
/*!\brief Neighbor access relative to current position
 * \param d Direction enumeration which defines deltas for x,y,z
 * \param cf Delta for f
 **********************************************************************************************************************/
template <typename T>
inline T & FieldIterator<T>::neighbor( stencil::Direction d, cell_idx_t cf ) const
{
   using namespace stencil;
   return neighbor(cx[d],cy[d],cz[d],cf);
}


//**********************************************************************************************************************
/*!\brief uint_t variant of above function
 * \param d Direction enumeration which defines deltas for x,y,z
 * \param cf Delta for f
 **********************************************************************************************************************/
template <typename T>
inline T & FieldIterator<T>::neighbor( stencil::Direction d, uint_t cf ) const
{
   return neighbor( d, cell_idx_c (cf) );
}


//**********************************************************************************************************************
/*!\brief Neighbor access relative to current position
 * \param cx Delta for x
 * \param cy Delta for y
 * \param cz Delta for z
 * \param cf Delta for f
 **********************************************************************************************************************/
template <typename T>
inline T & FieldIterator<T>::neighbor( cell_idx_t cx, cell_idx_t cy, cell_idx_t cz, cell_idx_t cf ) const
{
   T * res = linePtr_;

   res += cx * f_->xfact_ +
          cy * f_->yfact_ +
          cz * f_->zfact_ +
          cf * f_->ffact_;

   WALBERLA_ASSERT ( f_->addressInsideAllocedSpace( res ) )

   return *res;
}


//**********************************************************************************************************************
/*!\brief Neighbor variant that takes unsigned int as f parameter,
 *        needed since the stencil toIdx() is an unsigned int
 * \param cx Delta for x
 * \param cy Delta for y
 * \param cz Delta for z
 * \param cf Delta for f
 **********************************************************************************************************************/
template <typename T>
inline T & FieldIterator<T>::neighbor( cell_idx_t cx, cell_idx_t cy, cell_idx_t cz, uint_t cf ) const
{
   return neighbor ( cx, cy, cz, cell_idx_c( cf ) );
}



//**********************************************************************************************************************
/*!\brief For beginXYZ iterators, one often needs a specific f
 * Assumes that iterator stands at f==0
 * \param cf Delta for f
 **********************************************************************************************************************/
template <typename T>
inline  T & FieldIterator<T>::getF( cell_idx_t cf ) const
{
   WALBERLA_ASSERT_EQUAL( f(), 0 )
   WALBERLA_ASSERT_LESS( cf, cell_idx_t ( f_->fSize() ) )
   T * res = linePtr_;
   res += cf * f_->ffact_;
   return *res;
}


//**********************************************************************************************************************
/*!\brief Equivalent to neighbor(cell_idx_t) see above.
 *        Takes an uint_t instead a cell_idx_t, since stencil::toIndex() returns uint_t
 * \param cf Delta for f
 **********************************************************************************************************************/
template <typename T>
inline  T & FieldIterator<T>::getF( uint_t cf ) const
{
   return getF ( cell_idx_c ( cf ) );
}


//======================================================================================================================
//
//  PRINTING
//
//======================================================================================================================

template <typename T>
inline void FieldIterator<T>::print( std::ostream & os ) const
{
   os << "(" << x() << "," << y() << "," << z() << "/" << f() << ")";
}

template< typename T>
std::ostream & operator<< ( std::ostream & os, const FieldIterator<T> & it ) {
   it.print(os);
   return os;
}


//======================================================================================================================
//
//  COORDINATES OF CURRENT POSITIONS
//
//======================================================================================================================

/**
 * In order to get x(), y(), z(), f() function as fast a possible, no if clause for the layout was introduced.
 * Instead there are the cur[XYZF]_ members, that point to the cur_ array. The cur_ array does not store the
 * fastest coordinate, because it is implicitly stored in (linePtr_ - lineBegin_).  If it would be stored explicitly
 * there would have to be an extra update operation in operator++() , which should be as fast as possible.
 * The curX_ or curF_ pointer points to the fastestCoord_ member, which always has to be updated before curX_ or
 * curF_ is dereferenced.
 */

template <typename T>
inline cell_idx_t FieldIterator<T>::x() const
{
   fastestCoord_ = cell_idx_c(linePtr_ - lineBegin_ );
   return xBegin_ + *curX_;
}

template <typename T>
inline cell_idx_t FieldIterator<T>::y() const
{
   // no fastestCoord_ update required here, since y is never fastest coordinate
   return yBegin_ + *curY_;
}

template <typename T>
inline cell_idx_t FieldIterator<T>::z() const
{
   // no fastestCoord_ update required here, since z is never fastest coordinate
   return zBegin_ + *curZ_;
}

template <typename T>
inline cell_idx_t FieldIterator<T>::f() const
{
   fastestCoord_ = cell_idx_c(linePtr_ - lineBegin_ );
   return fBegin_ + *curF_;
}

template <typename T>
inline Cell FieldIterator<T>::cell() const
{
   fastestCoord_ = cell_idx_c( linePtr_ - lineBegin_ );
   return Cell ( xBegin_ + *curX_,
                 yBegin_ + *curY_,
                 zBegin_ + *curZ_ );
}



//======================================================================================================================
//
//  FORWARD ITERATOR
//
//======================================================================================================================


//**********************************************************************************************************************
/*!\brief Pre-increment operator.
 *
 * \return Reference to the incremented pointer iterator.
 **********************************************************************************************************************/
template <typename T>
inline ForwardFieldIterator<T>& ForwardFieldIterator<T>::operator++()
{

   ++Parent::linePtr_;
   if( Parent::linePtr_ != Parent::lineEnd_)
      return *this;

   // Iteration through line has finished - switch to next
   Parent::incrementLine();

   return *this;
}

//**********************************************************************************************************************
/*!\brief Pre-decrement operator.
 *
 * \return Reference to the decremented pointer iterator.
 **********************************************************************************************************************/
template <typename T>
inline ForwardFieldIterator<T>& ForwardFieldIterator<T>::operator--()
{

   --Parent::linePtr_;
   if( Parent::linePtr_ >= Parent::lineBegin_ )
      return *this;

   // Iteration through line has finished - switch to next
   Parent::decrementLine();

   return *this;
}

//**********************************************************************************************************************
/*!\brief Increments the second inner coordinate c2
 * Use if innermost loop is self written, which is slightly faster
 * \code
   for( const_iterator i = field.begin(); i != field.end(); i.incrOuter() )
      for (; i.testInner(); i.incrInner() )
      {}
   Instead of
  \code
   for(DoubleField::const_iterator i = field.begin(); i != field.end(); ++i) {
   }
  \endcode
 **********************************************************************************************************************/
template <typename T>
inline void ForwardFieldIterator<T>::incrOuter()
{
   // incrementing line pointer was done in "inner" iterator
   Parent::incrementLine();
}

//======================================================================================================================
//
//  REVERSE ITERATOR
//
//======================================================================================================================


//**********************************************************************************************************************
/*!\brief Pre-increment operator.
 *
 * \return Reference to the incremented pointer iterator.
 **********************************************************************************************************************/
template <typename T>
inline ReverseFieldIterator<T>& ReverseFieldIterator<T>::operator--()
{

   ++Parent::linePtr_;
   if( Parent::linePtr_ != Parent::lineEnd_)
      return *this;

   // Iteration through line has finished - switch to next
   Parent::incrementLine();

   return *this;
}


//**********************************************************************************************************************
/*!\brief Pre-decrement operator.
 *
 * \return Reference to the decremented pointer iterator.
 **********************************************************************************************************************/
template <typename T>
inline ReverseFieldIterator<T>& ReverseFieldIterator<T>::operator++()
{
   --Parent::linePtr_;

   if( Parent::linePtr_ >= Parent::lineBegin_ )
      return *this;


   // Iteration through line has finished - switch to next
   Parent::decrementLine();

   return *this;
}



} // namespace field
} // namespace walberla
