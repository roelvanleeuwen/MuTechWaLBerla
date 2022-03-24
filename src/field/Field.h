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
//! \file Field.h
//! \ingroup field
//! \author Martin Bauer <martin.bauer@fau.de>
//! \brief Four dimensional field/lattice class
//
//======================================================================================================================

#pragma once

#include "field/CMakeDefs.h"

#include "allocation/FieldAllocator.h"

#include "core/DataTypes.h"
#include "core/cell/CellInterval.h"
#include "core/debug/CheckFunctions.h"
#include "core/debug/Debug.h"

#include "domain_decomposition/StructuredBlockStorage.h"

#include "iterators/FieldIterator.h"
#include "iterators/FieldPointer.h"

#include <functional>
#include <memory>
#include <vector>


namespace walberla {
namespace field {


   //*******************************************************************************************************************
   /*!
   * A four dimensional field/lattice.
   *
   * \ingroup field
   *
   * Implemented as a vector style container using consecutive memory to
   * provide fixed time access to any member. The four coordinates are labeled x,y,z,f.
   * Two memory layouts (linearization strategies)  are offered, see Layout
   *
   *  \image html field/doc/layout.png "Two possible field layouts"
   *
   * Template Parameters:
   *   - T         type that is stored in the field
   *
   * See also \ref fieldPage
   */
   //*******************************************************************************************************************
   template<typename T, uint_t... fSize_>
   class Field {};

   template<typename T>
   class Field<T>
   {
   public:

      //** Type Definitions  *******************************************************************************************
      /*! \name Type Definitions */
      //@{
     // TODO remove when old field class is removed
      static constexpr bool OLD = false;
      using value_type = T;
      using iterator = ForwardFieldIterator<T>;
      using const_iterator =  ForwardFieldIterator<const T>;

      using reverse_iterator = ReverseFieldIterator<T>;
      using const_reverse_iterator =  ReverseFieldIterator<const T>;

      using base_iterator =  FieldIterator<T >;
      using const_base_iterator = FieldIterator<const T >;

      using Ptr = FieldPointer<Field<T>, Field<T>, T >;
      using ConstPtr =  FieldPointer<Field<T>, const Field<T>, const T >;

      using FlattenedField = typename std::conditional<VectorTrait<T>::F_SIZE!=0, Field<typename VectorTrait<T>::OutputType>, Field<T>>::type;

      //@}
      //****************************************************************************************************************



      //**Construction & Destruction************************************************************************************
      /*! \name Construction & Destruction */
      //@{


      Field( uint_t xSize, uint_t ySize, uint_t zSize, uint_t _fSize,
             const Layout & layout = zyxf,
             const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >() );
      Field( uint_t xSize, uint_t ySize, uint_t zSize, uint_t _fSize,
             const T & initValue, const Layout & layout = zyxf,
             const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >() );
      Field( uint_t xSize, uint_t ySize, uint_t zSize, uint_t _fSize,
             const std::vector<T> & fValues, const Layout & layout = zyxf,
             const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >() );
      virtual ~Field();


      void init( uint_t xSize, uint_t ySize, uint_t zSize, uint_t fSize, const Layout & layout = zyxf,
                 shared_ptr<FieldAllocator<T> > alloc = shared_ptr<FieldAllocator<T> >(),
                 uint_t innerGhostLayerSizeForAlignedAlloc = 0 );

      virtual void resize( uint_t xSize, uint_t ySize, uint_t zSize, uint_t fSize );

      virtual Field<T> * clone()              const;
      virtual Field<T> * cloneUninitialized() const;
      virtual Field<T> * cloneShallowCopy()   const;
      virtual FlattenedField * flattenedShallowCopy() const;
      //@}
      //****************************************************************************************************************


      //** Element Access **********************************************************************************************
      /*! \name Element Access */
      //@{
      inline       T & get( cell_idx_t x, cell_idx_t y, cell_idx_t z);
      inline const T & get( cell_idx_t x, cell_idx_t y, cell_idx_t z) const;
      inline       T & get( cell_idx_t x, cell_idx_t y, cell_idx_t z, cell_idx_t f );
      inline const T & get( cell_idx_t x, cell_idx_t y, cell_idx_t z, cell_idx_t f ) const;
      inline       T & get( cell_idx_t x, cell_idx_t y, cell_idx_t z, uint_t f );
      inline const T & get( cell_idx_t x, cell_idx_t y, cell_idx_t z, uint_t f ) const;
      inline       T & get( const Cell & cell );
      inline const T & get( const Cell & cell ) const;
      inline       T & get( const Cell & c, cell_idx_t f )       { return get(c[0], c[1], c[2], f); }
      inline const T & get( const Cell & c, cell_idx_t f ) const { return get(c[0], c[1], c[2], f); }
      inline       T & get( const Cell & c, uint_t f )           { return get(c[0], c[1], c[2], f); }
      inline const T & get( const Cell & c, uint_t f ) const     { return get(c[0], c[1], c[2], f); }
      inline       T & get( const base_iterator & iter );
      inline const T & get( const base_iterator & iter ) const;

      inline       T & operator()( cell_idx_t x, cell_idx_t y, cell_idx_t z)                     { return get(x,y,z);   }
      inline const T & operator()( cell_idx_t x, cell_idx_t y, cell_idx_t z) const               { return get(x,y,z);   }
      inline       T & operator()( cell_idx_t x, cell_idx_t y, cell_idx_t z, cell_idx_t f)       { return get(x,y,z,f); }
      inline const T & operator()( cell_idx_t x, cell_idx_t y, cell_idx_t z, cell_idx_t f) const { return get(x,y,z,f); }
      inline       T & operator()( cell_idx_t x, cell_idx_t y, cell_idx_t z, uint_t f)           { return get(x,y,z,f); }
      inline const T & operator()( cell_idx_t x, cell_idx_t y, cell_idx_t z, uint_t f) const     { return get(x,y,z,f); }
      inline       T & operator()( const Cell & cell )                                           { return get(cell);    }
      inline const T & operator()( const Cell & cell ) const                                     { return get(cell);    }
      inline       T & operator()( const Cell & cell, cell_idx_t f )                             { return get(cell,f);  }
      inline const T & operator()( const Cell & cell, cell_idx_t f ) const                       { return get(cell,f);  }
      inline       T & operator()( const Cell & cell, uint_t f )                                 { return get(cell,f);  }
      inline const T & operator()( const Cell & cell, uint_t f ) const                           { return get(cell,f);  }
      inline       T & operator()( const base_iterator & iter )                                  { return get(iter);    }
      inline const T & operator()( const base_iterator & iter ) const                            { return get(iter);    }

      inline       T & getNeighbor( cell_idx_t x, cell_idx_t y, cell_idx_t z, stencil::Direction d );
      inline const T & getNeighbor( cell_idx_t x, cell_idx_t y, cell_idx_t z, stencil::Direction d ) const;
      inline       T & getNeighbor( cell_idx_t x, cell_idx_t y, cell_idx_t z, uint_t f, stencil::Direction d );
      inline const T & getNeighbor( cell_idx_t x, cell_idx_t y, cell_idx_t z, uint_t f, stencil::Direction d ) const;
      inline       T & getNeighbor( const Cell & cell, stencil::Direction d );
      inline const T & getNeighbor( const Cell & cell, stencil::Direction d ) const;

      inline       T & getF(       T * const xyz0, const cell_idx_t f );
      inline const T & getF( const T * const xyz0, const cell_idx_t f ) const;
      inline       T & getF(       T * const xyz0, const uint_t f )       { return getF( xyz0, cell_idx_c(f) ); }
      inline const T & getF( const T * const xyz0, const uint_t f ) const { return getF( xyz0, cell_idx_c(f) ); }

             void set (const T & value);
             void set (const std::vector<T> & fValues);
      inline void set (const Field<T> & other );
      inline void set (const Field<T> * other ) { set( *other ); }

      void swapDataPointers( Field<T> & other );
      void swapDataPointers( Field<T> * other ) { swapDataPointers( *other ); }

      //@}
      //****************************************************************************************************************


      //** Equality Checks *********************************************************************************************
      /*! \name Equality Checks */
      //@{
      inline bool operator==      ( const Field<T> & other ) const;
      inline bool operator!=      ( const Field<T> & other ) const;
      inline bool hasSameAllocSize( const Field<T> & other ) const;
      inline bool hasSameSize     ( const Field<T> & other ) const;
      //@}
      //****************************************************************************************************************


      //** Iterators  **************************************************************************************************
      /*! \name Iterators */
      //@{
            iterator begin();
      const_iterator begin() const;

            iterator beginXYZ();
      const_iterator beginXYZ() const;

            iterator beginSlice( cell_idx_t xBeg, cell_idx_t yBeg, cell_idx_t zBeg, cell_idx_t fBeg,
                                 cell_idx_t xEnd, cell_idx_t yEnd, cell_idx_t zEnd, cell_idx_t fEnd );
      const_iterator beginSlice( cell_idx_t xBeg, cell_idx_t yBeg, cell_idx_t zBeg, cell_idx_t fBeg,
                                 cell_idx_t xEnd, cell_idx_t yEnd, cell_idx_t zEnd, cell_idx_t fEnd ) const;


            iterator beginSliceXYZ ( const CellInterval & interval, cell_idx_t f = 0 );
      const_iterator beginSliceXYZ ( const CellInterval & interval, cell_idx_t f = 0 ) const;

      inline const iterator       & end();
      inline const const_iterator & end() const;
      //@}
      //****************************************************************************************************************

      //** Reverse Iterators *******************************************************************************************
      /*! \name Reverse Iterators */
      //@{
            reverse_iterator rbegin();
      const_reverse_iterator rbegin() const;

            reverse_iterator rbeginXYZ();
      const_reverse_iterator rbeginXYZ() const;

      inline const       reverse_iterator & rend();
      inline const const_reverse_iterator & rend() const;
      //@}
      //****************************************************************************************************************



      //** Size and Layout Information *********************************************************************************
      /*! \name Size and Layout Information */
      //@{
      inline uint_t  xSize() const  { return xSize_; }
      inline uint_t  ySize() const  { return ySize_; }
      inline uint_t  zSize() const  { return zSize_; }
      inline uint_t  fSize() const  { return fSize_; }
      inline uint_t  size( uint_t coord )  const;

      inline uint_t  xAllocSize() const  { return xAllocSize_; }
      inline uint_t  yAllocSize() const  { return yAllocSize_; }
      inline uint_t  zAllocSize() const  { return zAllocSize_; }
      inline uint_t  fAllocSize() const  { return fAllocSize_; }
      inline uint_t  allocSize()  const  { return allocSize_;  }

      inline CellInterval xyzSize()      const;
      inline CellInterval xyzAllocSize() const;

      inline Layout layout() const { return layout_; }

      int64_t xStride() const { return xfact_; }
      int64_t yStride() const { return yfact_; }
      int64_t zStride() const { return zfact_; }
      int64_t fStride() const { return ffact_; }

      cell_idx_t xOff() const { return xOff_; }
      cell_idx_t yOff() const { return yOff_; }
      cell_idx_t zOff() const { return zOff_; }

      bool coordinatesValid( cell_idx_t x, cell_idx_t y, cell_idx_t z, cell_idx_t f ) const;
      //@}
      //****************************************************************************************************************


      //** Slicing  ****************************************************************************************************
      /*! \name Slicing */
      //@{
      virtual Field<T> * getSlicedField( const CellInterval & interval ) const;
      virtual void slice           ( const CellInterval & interval );
      virtual void shiftCoordinates( cell_idx_t cx, cell_idx_t cy, cell_idx_t cz );
      //@}
      //****************************************************************************************************************


      //** Monitoring  *************************************************************************************************
      /*! \name Monitoring */
      //@{
      using MonitorFunction = std::function<void (cell_idx_t, cell_idx_t, cell_idx_t, cell_idx_t, const T &)>;

      void addMonitoringFunction( const MonitorFunction & func );
      //@}
      //****************************************************************************************************************


      //** Pointer to internal memory - use with care!        **********************************************************
      /*! \name Pointer to internal memory - use with care!  */
      //@{
            T * data()            { return values_; }
      const T * data() const      { return values_; }
            T * dataAt(cell_idx_t x, cell_idx_t y, cell_idx_t z, cell_idx_t f)       { return &get(x,y,z,f); }
      const T * dataAt(cell_idx_t x, cell_idx_t y, cell_idx_t z, cell_idx_t f) const { return &get(x,y,z,f); }
            T * dataInner()       { return valuesWithOffset_; }
      const T * dataInner() const { return valuesWithOffset_; }

      shared_ptr< FieldAllocator<T> > getAllocator() const;
      //@}
      //****************************************************************************************************************


      //** Static cached end iterators *********************************************************************************
      /*! \name Static cached end iterators */
      //@{
      // As optimization, an iterator and a const_iterator are statically allocated
      // and are returned by the end() function
      static const const_iterator staticConstEnd;
      static const       iterator staticEnd;

      static const const_reverse_iterator staticConstREnd;
      static const       reverse_iterator staticREnd;
      //@}
      //****************************************************************************************************************

   protected:
      Field();

      //** Shallow Copy ************************************************************************************************
      /*! \name Shallow Copy */
      //@{
      Field(const Field & other);
      template <typename T2>
      Field(const Field<T2> & other);
      virtual uint_t referenceCount() const;


      virtual Field<T> * cloneShallowCopyInternal()   const;
      virtual FlattenedField * flattenedShallowCopyInternal() const;

      //@}
      //****************************************************************************************************************

      //** Changing Offsets ********************************************************************************************
      /*! \name Changing Offsets */
      //@{
      void setOffsets( uint_t xOffset, uint_t xSize, uint_t yOffset, uint_t ySize, uint_t zOffset, uint_t zSizes );

      shared_ptr<FieldAllocator<T> > allocator() const { return allocator_; }

      inline bool addressInsideAllocedSpace(const T * const value) const;

      void assertValidCoordinates( cell_idx_t x, cell_idx_t y, cell_idx_t z, cell_idx_t f ) const;
      //@}
      //****************************************************************************************************************

   private:

      Field & operator=( const Field & );

      T * values_;           //!< Linearized, 1-dimensional representation of the 4D data grid.
      T * valuesWithOffset_; //!< set by setOffsets(), to allow derived classes to change the offset

      cell_idx_t xOff_;      //!< Offset of the xSize() to xAllocSize()
      cell_idx_t yOff_;      //!< Offset of the ySize() to yAllocSize()
      cell_idx_t zOff_;      //!< Offset of the zSize() to zAllocSize()

      uint_t xSize_;         //!< Number of cells in x-dimension (excluding padded cells)
      uint_t ySize_;         //!< Number of cells in y-dimension (excluding padded cells)
      uint_t zSize_;         //!< Number of cells in z-dimension (excluding padded cells)
      uint_t fSize_;         //!< Number of cells in f-dimension (excluding padded cells)

      uint_t xAllocSize_;    //!< Number of cells in x-dimension (including padded cells)
      uint_t yAllocSize_;    //!< Number of cells in y-dimension (including padded cells)
      uint_t zAllocSize_;    //!< Number of cells in z-dimension (including padded cells)
      uint_t fAllocSize_;    //!< Number of cells in f-dimension (including padded cells)

      Layout layout_;        //!< Determines in which order the values are stored

      uint_t     allocSize_; //!< The overall size of the T* (padding included)
      int64_t ffact_;     //!< Access multiplication factor for the f-dimension.
      int64_t xfact_;     //!< Access multiplication factor for the x-dimension.
      int64_t yfact_;     //!< Access multiplication factor for the y-dimension.
      int64_t zfact_;     //!< Access multiplication factor for the z-dimension.

      shared_ptr<FieldAllocator<T> > allocator_; //!< Allocator for the field

      friend class FieldIterator<T>;
      friend class FieldIterator<const T>;
      template <typename T2, uint_t... fSize2>
      friend class Field;

#ifdef WALBERLA_FIELD_MONITORED_ACCESS
      std::vector<MonitorFunction> monitorFuncs_;
#endif

   }; // class Field


template<typename T, uint_t fSize_>
class Field<T, fSize_> : public Field<T> {
 public:


   static const uint_t F_SIZE = fSize_;
   static constexpr bool OLD = true;

   typedef typename std::conditional<VectorTrait<T>::F_SIZE!=0,
                                      Field<typename VectorTrait<T>::OutputType, VectorTrait<T>::F_SIZE*fSize_>,
                                      Field<T, fSize_>>::type FlattenedField;


   Field( uint_t xSize, uint_t ySize, uint_t zSize,
         const Layout & layout = zyxf,
         const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >() )
      : Field<T>::Field(xSize, ySize, zSize, fSize_, layout, alloc){}
   Field( uint_t xSize, uint_t ySize, uint_t zSize,
         const T & initValue, const Layout & layout = zyxf,
         const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >() )
      : Field<T>::Field(xSize, ySize, zSize, fSize_, initValue, layout, alloc){}
   Field( uint_t xSize, uint_t ySize, uint_t zSize,
         const std::vector<T> & fValues, const Layout & layout = zyxf,
         const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >() )
      : Field<T>::Field(xSize, ySize, zSize, fSize_, fValues, layout, alloc){}


   template<typename ...Args>
   void init(uint_t xSize, uint_t ySize, uint_t zSize, Args&&... args)
   {
      Field<T>::init(xSize, ySize, zSize, fSize_, std::forward<Args>(args)...);
   }

   virtual void resize(uint_t xSize, uint_t ySize, uint_t zSize)
   {
      Field<T>::resize(xSize, ySize, zSize, fSize_);
   }

   Field<T, fSize_>  * getSlicedField( const CellInterval & interval ) const
   {
      return static_cast<Field<T, fSize_>* > (Field<T>::getSlicedField( interval ));
   }

   virtual Field<T, fSize_>  * clone() const
   {
      return static_cast<Field<T, fSize_>* > (Field<T>::clone());
   }

   virtual Field<T, fSize_>  * cloneUninitialized() const
   {
      return static_cast<Field<T, fSize_>* > (Field<T>::cloneUninitialized());
   }

   virtual Field<T, fSize_>  * cloneShallowCopy() const
   {
      return static_cast<Field<T, fSize_>* > (Field<T>::cloneShallowCopy());
   }

   virtual FlattenedField* flattenedShallowCopy() const
   {
      return static_cast<FlattenedField* > (Field<T>::flattenedShallowCopy());
   }

 protected:
   Field()
      {
         this->values_ = nullptr;
         this->valuesWithOffset_ = nullptr;
         this->xSize_ = 0;
         this->ySize_ = 0;
         this->zSize_ = 0;
         this->xAllocSize_ = 0;
         this->yAllocSize_ = 0;
         this->zAllocSize_ = 0;
         this->fAllocSize_ = 0;
      }

   Field<T, fSize_>( const Field<T, fSize_> & other )
   {
      std::cout << "test" << std::endl;
      this->values_ = other.values_ ;
      this->valuesWithOffset_ = other.valuesWithOffset_ ;
      this->xOff_              =other.xOff_;
      this->yOff_             = other.yOff_;
      this->zOff_             = other.zOff_;
      this->xSize_            = other.xSize_ ;
      this->ySize_            = other.ySize_ ;
      this->zSize_            = other.zSize_ ;
      this->fSize_            = other.fSize_ ;
      this->xAllocSize_       =other.xAllocSize_ ;
      this->yAllocSize_       = other.yAllocSize_ ;
      this->zAllocSize_       = other.zAllocSize_ ;
      this->fAllocSize_       = other.fAllocSize_ ;
      this->layout_           =other.layout_ ;
      this->allocSize_        = other.allocSize_ ;
      this->ffact_            = other.ffact_ ;
      this->xfact_            = other.xfact_ ;
      this->yfact_            = other.yfact_ ;
      this->zfact_            = other.zfact_ ;
      this->allocator_        = other.allocator_;
      this->allocator_->incrementReferenceCount ( this->values_ );
   }

   template <typename T2, uint_t fSize2>
   Field<T, fSize_>( const Field<T2, fSize2> & other )
   {
      this->values_           = other.values_[0].data() ;
      this->valuesWithOffset_ = other.valuesWithOffset_[0].data() ;
      this->xOff_             = other.xOff_;
      this->yOff_             = other.yOff_;
      this->zOff_             = other.zOff_;
      this->xSize_            = other.xSize_ ;
      this->ySize_            = other.ySize_ ;
      this->zSize_            = other.zSize_ ;
      this->fSize_            = VectorTrait<T2>::F_SIZE * other.fSize_ ;
      this->xAllocSize_       = other.xAllocSize_ ;
      this->yAllocSize_       = other.yAllocSize_ ;
      this->zAllocSize_       = other.zAllocSize_ ;
      this->fAllocSize_       = other.fAllocSize_*fSize_/fSize2 ;
      this->layout_           = other.layout_ ;
      this->allocSize_        = other.allocSize_*fSize_/fSize2 ;
      this->ffact_            = other.ffact_ ;
      this->xfact_            = other.xfact_*cell_idx_t(fSize_/fSize2) ;
      this->yfact_            = other.yfact_*cell_idx_t(fSize_/fSize2) ;
      this->zfact_            = other.zfact_*cell_idx_t(fSize_/fSize2) ;
      this->allocator_        = std::shared_ptr<FieldAllocator<T>>(other.allocator_, reinterpret_cast<FieldAllocator<T>*>(other.allocator_.get()));
      WALBERLA_CHECK_EQUAL(this->layout_, Layout::zyxf)
      static_assert(fSize_ % fSize2 == 0, "number of field components do not match");
      static_assert(std::is_same<typename Field<T2,fSize2>::FlattenedField, Field<T,fSize_>>::value, "field types are incompatible for flattening");
      this->allocator_->incrementReferenceCount ( this->values_ );
   }

};



} // namespace field
} // namespace walberla

#include "Field.impl.h"




//======================================================================================================================
//
//  EXPORTS
//
//======================================================================================================================

namespace walberla {
   // Export field class to walberla namespace
   using field::Field;
}

