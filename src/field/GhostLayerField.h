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
//! \file GhostLayerField.h
//! \ingroup field
//! \author Martin Bauer <martin.bauer@fau.de>
//! \brief GhostLayerField class extends Field with ghost layer information
//
//======================================================================================================================

#pragma once

#include "core/DataTypes.h"
#include "core/cell/CellInterval.h"

#include "stencil/Directions.h"

#include "Field.h"
#include "GhostRegions.h"

namespace walberla
{
namespace field
{

//*******************************************************************************************************************
/*! Extends the Field with ghost-layer information
 *
 * All variants of the begin() function exist also in the "WithGhostLayer" variant
 * which iterate over the field including the ghost layers. There are also iterators that
 * go only over the ghost layer or, only over the last inner slice, which is useful when doing
 * ghost layer based communication.
 *
 * \ingroup field
 *
 * See also \ref fieldPage
 *
 */
//*******************************************************************************************************************
template<typename T, uint_t... fSize_>
class GhostLayerField : public Field<T, fSize_...>{};

template<typename T>
class GhostLayerField<T> : public Field<T>

{
 public:
   //** Type Definitions  *******************************************************************************************
   /*! \name Type Definitions */
   //@{
   static constexpr bool OLD = false;
   using value_type = typename Field< T >::value_type;

   using iterator       = typename Field< T >::iterator;
   using const_iterator = typename Field< T >::const_iterator;

   using reverse_iterator       = typename Field< T >::reverse_iterator;
   using const_reverse_iterator = typename Field< T >::const_reverse_iterator;

   using base_iterator       = typename Field< T >::base_iterator;
   using const_base_iterator = typename Field< T >::const_base_iterator;

   using Ptr      = typename Field< T >::Ptr;
   using ConstPtr = typename Field< T >::ConstPtr;

   using FlattenedField =
      typename std::conditional< VectorTrait< T >::F_SIZE != 0,
                                 GhostLayerField< typename VectorTrait< T >::OutputType >, GhostLayerField< T > >::type;
   //@}
   //****************************************************************************************************************

   //**Construction & Destruction************************************************************************************
   /*! \name Construction & Destruction */
   //@{

   GhostLayerField(uint_t xSize, uint_t ySize, uint_t zSize, uint_t _fSize, uint_t gl, const Layout& layout = zyxf,
                   const shared_ptr< FieldAllocator< T > >& alloc = shared_ptr< FieldAllocator< T > >());
   GhostLayerField(uint_t xSize, uint_t ySize, uint_t zSize, uint_t _fSize, uint_t gl, const T& initValue,
                   const Layout& layout                           = zyxf,
                   const shared_ptr< FieldAllocator< T > >& alloc = shared_ptr< FieldAllocator< T > >());
   GhostLayerField(uint_t xSize, uint_t ySize, uint_t zSize, uint_t _fSize, uint_t gl, const std::vector< T >& fValues,
                   const Layout& layout                           = zyxf,
                   const shared_ptr< FieldAllocator< T > >& alloc = shared_ptr< FieldAllocator< T > >());

   ~GhostLayerField() override = default;

   void init(uint_t xSizeWithoutGhostLayer, uint_t ySizeWithoutGhostLayer, uint_t zSizeWithoutGhostLayer, uint_t _fSize,
             uint_t nrGhostLayers, const Layout& layout = zyxf,
             const shared_ptr< FieldAllocator< T > >& alloc = shared_ptr< FieldAllocator< T > >());

   void resize(uint_t xSize, uint_t ySize, uint_t zSize, uint_t _fSize) override;
   void resize(uint_t xSize, uint_t ySize, uint_t zSize, uint_t _fSize, uint_t gl);

   using Field< T >::resize;

   inline GhostLayerField< T >* clone() const override;
   inline GhostLayerField< T >* cloneUninitialized() const override;
   inline GhostLayerField< T >* cloneShallowCopy() const override;
   inline FlattenedField* flattenedShallowCopy() const override;
   //@}
   //****************************************************************************************************************

   //** Size Information ********************************************************************************************
   /*! \name Size Information */
   //@{
   inline uint_t xSizeWithGhostLayer() const { return Field< T >::xSize() + uint_c(2) * gl_; }
   inline uint_t ySizeWithGhostLayer() const { return Field< T >::ySize() + uint_c(2) * gl_; }
   inline uint_t zSizeWithGhostLayer() const { return Field< T >::zSize() + uint_c(2) * gl_; }
   inline uint_t sizeWithGhostLayer(uint_t i) const
   {
      return i == 3 ? Field< T >::fSize() : Field< T >::size(i) + uint_c(2) * gl_;
   }
   inline uint_t nrOfGhostLayers() const { return gl_; }
   inline CellInterval xyzSizeWithGhostLayer() const;
   //@}
   //****************************************************************************************************************

   //** Element Access **********************************************************************************************
   /*! \name Element Access */
   //@{
   void setWithGhostLayer(const T& value);
   void setWithGhostLayer(const std::vector< T >& fValues);
   //@}
   //****************************************************************************************************************

   //** Iterators  **************************************************************************************************
   /*! \name Iterators */
   //@{
   iterator beginWithGhostLayer();
   const_iterator beginWithGhostLayer() const;

   iterator beginWithGhostLayer(cell_idx_t numGhostLayers);
   const_iterator beginWithGhostLayer(cell_idx_t numGhostLayers) const;

   iterator beginWithGhostLayerXYZ();
   const_iterator beginWithGhostLayerXYZ() const;

   iterator beginWithGhostLayerXYZ(cell_idx_t numGhostLayers);
   const_iterator beginWithGhostLayerXYZ(cell_idx_t numGhostLayers) const;

   iterator beginGhostLayerOnly(stencil::Direction dir, bool fullSlice = false);
   const_iterator beginGhostLayerOnly(stencil::Direction dir, bool fullSlice = false) const;

   iterator beginGhostLayerOnly(uint_t thickness, stencil::Direction dir, bool fullSlice = false);
   const_iterator beginGhostLayerOnly(uint_t thickness, stencil::Direction dir, bool fullSlice = false) const;

   iterator beginGhostLayerOnlyXYZ(stencil::Direction dir, cell_idx_t f = 0, bool fullSlice = false);
   const_iterator beginGhostLayerOnlyXYZ(stencil::Direction dir, cell_idx_t f = 0, bool fullSlice = false) const;

   iterator beginGhostLayerOnlyXYZ(uint_t thickness, stencil::Direction dir, cell_idx_t f = 0, bool fullSlice = false);
   const_iterator beginGhostLayerOnlyXYZ(uint_t thickness, stencil::Direction dir, cell_idx_t f = 0,
                                         bool fullSlice = false) const;

   iterator beginSliceBeforeGhostLayer(stencil::Direction dir, cell_idx_t thickness = 1, bool fullSlice = false);
   const_iterator beginSliceBeforeGhostLayer(stencil::Direction dir, cell_idx_t thickness = 1,
                                             bool fullSlice = false) const;

   iterator beginSliceBeforeGhostLayerXYZ(stencil::Direction dir, cell_idx_t thickness = 1, cell_idx_t f = 0,
                                          bool fullSlice = false);
   const_iterator beginSliceBeforeGhostLayerXYZ(stencil::Direction dir, cell_idx_t thickness = 1, cell_idx_t f = 0,
                                                bool fullSlice = false) const;

   void getGhostRegion(stencil::Direction dir, CellInterval& ghostAreaOut, cell_idx_t thickness,
                       bool fullSlice = false) const;
   void getSliceBeforeGhostLayer(stencil::Direction d, CellInterval& ci, cell_idx_t thickness = 1,
                                 bool fullSlice = false) const;
   bool isInInnerPart(const Cell& cell) const;
   //@}
   //****************************************************************************************************************

   //** Reverse Iterators *******************************************************************************************
   /*! \name Reverse Iterators */
   //@{
   reverse_iterator rbeginWithGhostLayer();
   const_reverse_iterator rbeginWithGhostLayer() const;

   reverse_iterator rbeginWithGhostLayerXYZ();
   const_reverse_iterator rbeginWithGhostLayerXYZ() const;
   //@}
   //****************************************************************************************************************

   //** Slicing  ****************************************************************************************************
   /*! \name Slicing */
   //@{
   GhostLayerField< T >* getSlicedField(const CellInterval& interval) const override;
   void slice(const CellInterval& interval) override;
   void shiftCoordinates(cell_idx_t cx, cell_idx_t cy, cell_idx_t cz) override;
   //@}
   //****************************************************************************************************************

 protected:
   GhostLayerField();

   uint_t gl_; ///< Number of ghost layers

   //** Shallow Copy ************************************************************************************************
   /*! \name Shallow Copy */
   //@{
   Field<T> * cloneShallowCopyInternal()   const override;
   typename Field<T>::FlattenedField * flattenedShallowCopyInternal() const override;
   GhostLayerField(const GhostLayerField<T> & other);
   template <typename T2>
   GhostLayerField<T>(const GhostLayerField<T2> & other);
   //@}
   //****************************************************************************************************************

   template <typename T2, uint_t... fSize2>
   friend class GhostLayerField;
};

#ifdef WALBERLA_CXX_COMPILER_IS_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#endif
template<typename T, uint_t fSize_>
class GhostLayerField<T, fSize_> : public Field<T, fSize_> {
 public:
   GhostLayerField(GhostLayerField<T> field)
      : GhostLayerField<T>::GhostLayerField(field)
   {}

   static const uint_t F_SIZE = fSize_;
   static constexpr bool OLD = true;

   using value_type = typename Field<T, fSize_>::value_type;

   using iterator = typename Field<T, fSize_>::iterator;
   using const_iterator = typename Field<T, fSize_>::const_iterator;

   using reverse_iterator = typename Field<T, fSize_>::reverse_iterator;
   using const_reverse_iterator = typename Field<T, fSize_>::const_reverse_iterator;

   using base_iterator = typename Field<T, fSize_>::base_iterator;
   using const_base_iterator = typename Field<T, fSize_>::const_base_iterator;

   using Ptr = typename Field<T, fSize_>::Ptr;
   using ConstPtr = typename Field<T, fSize_>::ConstPtr;


   typedef typename std::conditional<VectorTrait<T>::F_SIZE!=0,
                                      GhostLayerField<typename VectorTrait<T>::OutputType, VectorTrait<T>::F_SIZE*fSize_>,
                                      GhostLayerField<T, fSize_>>::type FlattenedField;

   GhostLayerField( uint_t xSize, uint_t ySize, uint_t zSize, uint_t gl,
                   const Layout & layout = zyxf,
                   const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >() )
   {
      init( xSize, ySize, zSize, gl, layout, alloc );
   }

   GhostLayerField( uint_t xSize, uint_t ySize, uint_t zSize, uint_t gl,
                   const T & initValue, const Layout & layout = zyxf,
                   const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >() )
   {
      init( xSize, ySize, zSize, gl, layout, alloc );
      setWithGhostLayer( initValue );

   }

   GhostLayerField( uint_t xSize, uint_t ySize, uint_t zSize, uint_t gl,
                   const std::vector<T> & fValues, const Layout & layout = zyxf,
                   const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >() )
   {
      init( xSize, ySize, zSize, gl, layout, alloc );
      setWithGhostLayer( fValues );
   }

   void init(uint_t xSizeWithoutGhostLayer,
             uint_t ySizeWithoutGhostLayer,
             uint_t zSizeWithoutGhostLayer,
             uint_t nrGhostLayers,
             const Layout & layout = zyxf,
             const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >()
   )
   {
      gl_ = nrGhostLayers;
      uint_t innerGhostLayerSize = ( layout == fzyx ) ? nrGhostLayers : uint_t(0);
      Field<T, fSize_>::init( xSizeWithoutGhostLayer + 2*nrGhostLayers ,
                               ySizeWithoutGhostLayer + 2*nrGhostLayers,
                               zSizeWithoutGhostLayer + 2*nrGhostLayers, layout, alloc,
                               innerGhostLayerSize );

      Field<T, fSize_>::setOffsets( nrGhostLayers, xSizeWithoutGhostLayer,
                                     nrGhostLayers, ySizeWithoutGhostLayer,
                                     nrGhostLayers, zSizeWithoutGhostLayer );

   }

   void resize( uint_t _xSize, uint_t _ySize, uint_t _zSize, uint_t fSize ) override
   {
      if ( _xSize == this->xSize() && _ySize == this->ySize() && _zSize == this->zSize()  )
         return;

      Field<T, fSize_>::resize( _xSize+2*gl_, _ySize+2*gl_, _zSize+2*gl_, fSize_);
      Field<T, fSize_>::setOffsets( gl_, _xSize, gl_, _ySize, gl_, _zSize );
   }

   Field<T, fSize_> * cloneShallowCopyInternal() const override
   {
      return new GhostLayerField<T,fSize_>(*this);
   }

   typename Field<T,fSize_>::FlattenedField * flattenedShallowCopyInternal() const override
   {
      return new GhostLayerField<T,fSize_>::FlattenedField(*this);
   }

   GhostLayerField<T,fSize_> * clone() const override
   {
      return reinterpret_cast<GhostLayerField<T,fSize_>* > (Field<T,fSize_>::clone() );
   }

   GhostLayerField<T,fSize_> * cloneUninitialized() const override
   {
      return reinterpret_cast<GhostLayerField<T,fSize_>* > (Field<T,fSize_>::cloneUninitialized() );
   }

   GhostLayerField<T,fSize_> * cloneShallowCopy() const override
   {
      return reinterpret_cast<GhostLayerField<T,fSize_>* > (Field<T,fSize_>::cloneShallowCopy() );
   }

   typename GhostLayerField<T,fSize_>::FlattenedField * flattenedShallowCopy() const override
   {
      return reinterpret_cast<GhostLayerField<T,fSize_>::FlattenedField* > (Field<T,fSize_>::flattenedShallowCopy() );
   }


   //*******************************************************************************************************************
   /*!\brief Sets all entries (including the ghost layer) of the field to given value
    *******************************************************************************************************************/
   void setWithGhostLayer (const T & value)
   {
#ifdef WALBERLA_CXX_COMPILER_IS_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wundefined-bool-conversion"
#endif
      // take care of proper thread<->memory assignment (first-touch allocation policy !)
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ( this,

                                                       for( uint_t f = uint_t(0); f < fSize_; ++f )
                                                          this->get(x,y,z,f) = value;

                                                       ) // WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ
#ifdef WALBERLA_CXX_COMPILER_IS_CLANG
#pragma clang diagnostic pop
#endif
   }

   //*******************************************************************************************************************
   /*!\brief Initializes the f coordinate to values from vector, in all cells including the ghost layers
    * Sets the entry (x,y,z,f) to fValues[f]
    *******************************************************************************************************************/
   void setWithGhostLayer (const std::vector<T> & fValues)
   {
      WALBERLA_ASSERT(fValues.size() == fSize_);

#ifdef WALBERLA_CXX_COMPILER_IS_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wundefined-bool-conversion"
#endif
      // take care of proper thread<->memory assignment (first-touch allocation policy !)
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ( this,

                                                       for( uint_t f = uint_t(0); f < fSize_; ++f )
                                                          this->get(x,y,z,f) = fValues[f];

                                                       ) // WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ
#ifdef WALBERLA_CXX_COMPILER_IS_CLANG
#pragma clang diagnostic pop
#endif
   }



   //** Size Information ********************************************************************************************
   /*! \name Size Information */
   //@{
   inline uint_t xSizeWithGhostLayer() const { return Field< T, fSize_ >::xSize() + uint_c(2) * gl_; }
   inline uint_t ySizeWithGhostLayer() const { return Field< T, fSize_ >::ySize() + uint_c(2) * gl_; }
   inline uint_t zSizeWithGhostLayer() const { return Field< T, fSize_ >::zSize() + uint_c(2) * gl_; }
   inline uint_t sizeWithGhostLayer(uint_t i) const
   {
      return i == 3 ? Field< T, fSize_ >::fSize() : Field< T, fSize_ >::size(i) + uint_c(2) * gl_;
   }
   inline uint_t nrOfGhostLayers() const { return gl_; }
   inline CellInterval xyzSizeWithGhostLayer() const
   {
      CellInterval ci = Field<T,fSize_>::xyzSize();
      for( uint_t i=0; i<3; ++i ) {
         ci.min()[i] -= cell_idx_c( gl_ );
         ci.max()[i] += cell_idx_c( gl_ );
      }
      return ci;
   }
   //@}
   //****************************************************************************************************************

   //===================================================================================================================
   //
   //  ITERATORS
   //
   //===================================================================================================================



   //*******************************************************************************************************************
   /*!\brief Iterator over all cells, including the ghost layers
    *
    * same as begin() , but with ghost layer
    *
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::iterator
      beginWithGhostLayer( )
   {
      return beginWithGhostLayer( cell_idx_c( gl_ ) );
   }

   inline typename GhostLayerField<T,fSize_>::iterator
      beginWithGhostLayer( cell_idx_t numGhostLayers )
   {
      WALBERLA_ASSERT_LESS_EQUAL( numGhostLayers, cell_idx_c( gl_ )  );
      const uint_t xs = Field<T,fSize_>::xSize() + 2 * uint_c( numGhostLayers );
      const uint_t ys = Field<T,fSize_>::ySize() + 2 * uint_c( numGhostLayers );
      const uint_t zs = Field<T,fSize_>::zSize() + 2 * uint_c( numGhostLayers );

      return iterator( this,
                      -numGhostLayers,-numGhostLayers,-numGhostLayers,0,
                      xs, ys, zs, fSize_ );
   }



   //*******************************************************************************************************************
   /*!\brief Returns const_iterator, see beginWithGhostLayer()
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::const_iterator
      beginWithGhostLayer( ) const
   {
      return beginWithGhostLayer( cell_idx_c( gl_ ) );
   }

   inline typename GhostLayerField<T,fSize_>::const_iterator
      beginWithGhostLayer( cell_idx_t numGhostLayers ) const
   {
      const uint_t xs = Field<T,fSize_>::xSize() + 2 * uint_c( numGhostLayers );
      const uint_t ys = Field<T,fSize_>::ySize() + 2 * uint_c( numGhostLayers );
      const uint_t zs = Field<T,fSize_>::zSize() + 2 * uint_c( numGhostLayers );

      return const_iterator(  this,
                            -numGhostLayers,-numGhostLayers,-numGhostLayers,0,
                            xs, ys, zs, fSize_ );

   }



   //*******************************************************************************************************************
   /*!\brief Iterates only over all cells including ghost layers of XYZ coordinate, f is always 0
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::iterator
      beginWithGhostLayerXYZ( )
   {
      return beginWithGhostLayerXYZ( cell_idx_c( gl_ ) );
   }

   inline typename GhostLayerField<T,fSize_>::iterator
      beginWithGhostLayerXYZ( cell_idx_t numGhostLayers )
   {
      const uint_t xs = Field<T,fSize_>::xSize() + 2 * uint_c( numGhostLayers );
      const uint_t ys = Field<T,fSize_>::ySize() + 2 * uint_c( numGhostLayers );
      const uint_t zs = Field<T,fSize_>::zSize() + 2 * uint_c( numGhostLayers );

      return iterator( this,
                      -numGhostLayers,-numGhostLayers,-numGhostLayers,0,
                      xs, ys, zs, 1 );
   }

   //*******************************************************************************************************************
   /*!\brief Const version of beginWithGhostLayerXYZ()
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::const_iterator
      beginWithGhostLayerXYZ( ) const
   {
      return beginWithGhostLayerXYZ( cell_idx_c( gl_ ) );
   }

   inline typename GhostLayerField<T,fSize_>::const_iterator
      beginWithGhostLayerXYZ( cell_idx_t numGhostLayers ) const
   {
      const uint_t xs = Field<T,fSize_>::xSize() + 2 * uint_c( numGhostLayers );
      const uint_t ys = Field<T,fSize_>::ySize() + 2 * uint_c( numGhostLayers );
      const uint_t zs = Field<T,fSize_>::zSize() + 2 * uint_c( numGhostLayers );

      return const_iterator( this,
                            -numGhostLayers,-numGhostLayers,-numGhostLayers,0,
                            xs, ys, zs, 1 );
   }

   void getGhostRegion(stencil::Direction d, CellInterval & ci, cell_idx_t thickness, bool fullSlice ) const
   {
      ci = field::getGhostRegion( *this, d, thickness, fullSlice );
   }

   void getSliceBeforeGhostLayer(stencil::Direction d, CellInterval & ci,
                                                               cell_idx_t thickness, bool fullSlice ) const
   {
      ci = field::getSliceBeforeGhostLayer( *this, d, thickness, fullSlice );
   }

   //*******************************************************************************************************************
   /*!\brief Checks if a given cell is in the inner part of the field ( not in ghost region or outside )
    *******************************************************************************************************************/
   bool isInInnerPart( const Cell & cell ) const
   {
      return !(cell[0] < 0 ||
               cell[1] < 0 ||
               cell[2] < 0 ||
               cell[0] >= cell_idx_c( this->xSize() ) ||
               cell[1] >= cell_idx_c( this->ySize() ) ||
               cell[2] >= cell_idx_c( this->zSize() ));
   }

   //*******************************************************************************************************************
   /*!\brief Iterates only over ghost layers of a given direction
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::iterator
      beginGhostLayerOnly( stencil::Direction dir, bool fullSlice = false )
   {
      CellInterval ci;
      getGhostRegion(dir,ci, cell_idx_c(gl_), fullSlice );

      return ForwardFieldIterator<T>( this,
                                               ci.xMin(),ci.yMin(), ci.zMin(), 0,
                                               ci.xSize(), ci.ySize(), ci.zSize(), fSize_ );
   }


   //*******************************************************************************************************************
   /*!\brief Const version of beginGhostLayersOnly()
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::const_iterator
      beginGhostLayerOnly( stencil::Direction dir, bool fullSlice = false ) const
   {
      CellInterval ci;
      getGhostRegion(dir,ci, cell_idx_c(gl_), fullSlice );

      return ForwardFieldIterator<const T>( this,
                                                     ci.xMin(),ci.yMin(), ci.zMin(), 0,
                                                     ci.xSize(), ci.ySize(), ci.zSize(), fSize_ );
   }


   //*******************************************************************************************************************
   /*!\brief Iterates only over specified number of ghost layers of a given direction
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::iterator
      beginGhostLayerOnly( uint_t thickness, stencil::Direction dir, bool fullSlice = false )
   {
      CellInterval ci;
      getGhostRegion( dir, ci, cell_idx_c(thickness), fullSlice );

      return ForwardFieldIterator<T>( this,
                                               ci.xMin(),ci.yMin(), ci.zMin(), 0,
                                               ci.xSize(), ci.ySize(), ci.zSize(), fSize_ );
   }


   //*******************************************************************************************************************
   /*!\brief Const version of beginGhostLayersOnly(uint_t thickness, stencil::Direction)
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::const_iterator
      beginGhostLayerOnly(  uint_t thickness, stencil::Direction dir, bool fullSlice = false ) const
   {
      CellInterval ci;
      getGhostRegion( dir, ci, cell_idx_c(thickness), fullSlice );

      return ForwardFieldIterator<const T>( this,
                                                     ci.xMin(),ci.yMin(), ci.zMin(), 0,
                                                     ci.xSize(), ci.ySize(), ci.zSize(), fSize_ );
   }



   //*******************************************************************************************************************
   /*!\brief Iterates only over ghost layers of a given direction, only over xyz coordinates, f is fixed
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::iterator
      beginGhostLayerOnlyXYZ( stencil::Direction dir, cell_idx_t f, bool fullSlice = false )
   {
      CellInterval ci;
      getGhostRegion( dir, ci, cell_idx_c(gl_) , fullSlice );

      return ForwardFieldIterator<T>( this,
                                               ci.xMin(),ci.yMin(), ci.zMin(), f,
                                               ci.xSize(), ci.ySize(), ci.zSize(), uint_c(f+1) );
   }



   //*******************************************************************************************************************
   /*!\brief Const version of beginGhostLayersOnlyXYZ()
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::const_iterator
      beginGhostLayerOnlyXYZ( stencil::Direction dir, cell_idx_t f, bool fullSlice = false ) const
   {
      CellInterval ci;
      getGhostRegion(dir, ci, cell_idx_c(gl_), fullSlice );

      return ForwardFieldIterator<const T>( this,
                                                     ci.xMin(),ci.yMin(), ci.zMin(), f,
                                                     ci.xSize(), ci.ySize(), ci.zSize(), uint_c( f + cell_idx_t(1) ) );
   }

   //*******************************************************************************************************************
   /*!\brief Iterates only over ghost layers of a given direction, only over xyz coordinates, f is fixed
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::iterator
      beginGhostLayerOnlyXYZ( uint_t thickness, stencil::Direction dir, cell_idx_t f, bool fullSlice = false )
   {
      CellInterval ci;
      getGhostRegion( dir, ci, cell_idx_c(thickness) , fullSlice );

      return ForwardFieldIterator<T>( this,
                                               ci.xMin(),ci.yMin(), ci.zMin(), f,
                                               ci.xSize(), ci.ySize(), ci.zSize(), uint_c( f + cell_idx_t(1) ) );
   }



   //*******************************************************************************************************************
   /*!\brief Const version of beginGhostLayersOnlyXYZ()
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::const_iterator
      beginGhostLayerOnlyXYZ( uint_t thickness, stencil::Direction dir, cell_idx_t f, bool fullSlice = false ) const
   {
      CellInterval ci;
      getGhostRegion(dir, ci, cell_idx_c(thickness), fullSlice );

      return ForwardFieldIterator<const T>( this,
                                                     ci.xMin(),ci.yMin(), ci.zMin(), f,
                                                     ci.xSize(), ci.ySize(), ci.zSize(), uint_c( f + cell_idx_t(1) ) );
   }


   //*******************************************************************************************************************
   /*!\brief Iterates only over the last slice before ghost layer
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::iterator
      beginSliceBeforeGhostLayer( stencil::Direction dir, cell_idx_t thickness = 1, bool fullSlice = false )
   {
      CellInterval ci;
      getSliceBeforeGhostLayer(dir, ci, thickness, fullSlice );

      return ForwardFieldIterator<T>( this,
                                               ci.xMin(),ci.yMin(), ci.zMin(), 0,
                                               ci.xSize(), ci.ySize(), ci.zSize(), fSize_ );
   }


   //*******************************************************************************************************************
   /*!\brief Const version of beginSliceBeforeGhostLayer()
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::const_iterator
      beginSliceBeforeGhostLayer( stencil::Direction dir, cell_idx_t thickness = 1, bool fullSlice = false ) const
   {
      CellInterval ci;
      getSliceBeforeGhostLayer( dir, ci, thickness, fullSlice );

      return ForwardFieldIterator<const T>( this,
                                                     ci.xMin(),ci.yMin(), ci.zMin(), 0,
                                                     ci.xSize(), ci.ySize(), ci.zSize(), fSize_ );
   }


   //*******************************************************************************************************************
   /*!\brief Iterates only over the last slice before ghost layer, only in XYZ direction, f is fixed
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::iterator
      beginSliceBeforeGhostLayerXYZ( stencil::Direction dir, cell_idx_t thickness = 1,
                                                                  cell_idx_t f = 0, bool fullSlice = false  )
   {
      CellInterval ci;
      getSliceBeforeGhostLayer(dir, ci, thickness, fullSlice );

      return ForwardFieldIterator<T>( this,
                                               ci.xMin(),ci.yMin(), ci.zMin(), f,
                                               ci.xSize(), ci.ySize(), ci.zSize(), uint_c( f + cell_idx_t(1) ) );
   }


   //*******************************************************************************************************************
   /*!\brief Const version of beginSliceBeforeGhostLayerXYZ()
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::const_iterator
      beginSliceBeforeGhostLayerXYZ( stencil::Direction dir,  cell_idx_t thickness = 1,
                                                                  cell_idx_t f = 0, bool fullSlice = false ) const
   {
      CellInterval ci;
      getSliceBeforeGhostLayer( dir, ci, thickness, fullSlice );

      return ForwardFieldIterator<const T>( this,
                                                     ci.xMin(),ci.yMin(), ci.zMin(), f,
                                                     ci.xSize(), ci.ySize(), ci.zSize(), uint_c( f + cell_idx_t(1) ) );
   }

   //*******************************************************************************************************************
   /*!\brief Returns the x/y/z Size of the field with ghost layers
    *******************************************************************************************************************/


   //===================================================================================================================
   //
   //  REVERSE ITERATORS
   //
   //===================================================================================================================



   //*******************************************************************************************************************
   /*!\brief Reverse Iterator over all cells, including the ghost layers
    *
    * same as rbegin() , but with ghost layer
    *
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::reverse_iterator
      rbeginWithGhostLayer()
   {
      const uint_t xs = Field<T,fSize_>::xSize() + 2*gl_;
      const uint_t ys = Field<T,fSize_>::ySize() + 2*gl_;
      const uint_t zs = Field<T,fSize_>::zSize() + 2*gl_;

      return reverse_iterator( this,
                              -cell_idx_c(gl_),-cell_idx_c(gl_),-cell_idx_c(gl_),0,
                              xs, ys, zs, fSize_ );
   }


   //*******************************************************************************************************************
   /*!\brief Returns const_iterator, see beginWithGhostLayer()
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::const_reverse_iterator
      rbeginWithGhostLayer() const
   {
      const uint_t xs = Field<T,fSize_>::xSize() + 2*gl_;
      const uint_t ys = Field<T,fSize_>::ySize() + 2*gl_;
      const uint_t zs = Field<T,fSize_>::zSize() + 2*gl_;

      return const_reverse_iterator(  this,
                                    -cell_idx_c(gl_),-cell_idx_c(gl_),-cell_idx_c(gl_),0,
                                    xs, ys, zs, fSize_ );

   }


   //*******************************************************************************************************************
   /*!\brief Iterates only over all cells including ghost layers of XYZ coordinate, f is always 0
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::reverse_iterator
      rbeginWithGhostLayerXYZ()
   {
      const uint_t xs = Field<T,fSize_>::xSize() + 2*gl_;
      const uint_t ys = Field<T,fSize_>::ySize() + 2*gl_;
      const uint_t zs = Field<T,fSize_>::zSize() + 2*gl_;

      return reverse_iterator( this,
                              -cell_idx_c(gl_),-cell_idx_c(gl_),-cell_idx_c(gl_),0,
                              xs, ys, zs, 1 );
   }

   //*******************************************************************************************************************
   /*!\brief Const version of beginWithGhostLayerXYZ()
    *******************************************************************************************************************/
   inline typename GhostLayerField<T,fSize_>::const_reverse_iterator
      rbeginWithGhostLayerXYZ() const
   {
      const uint_t xs = Field<T,fSize_>::xSize() + 2*gl_;
      const uint_t ys = Field<T,fSize_>::ySize() + 2*gl_;
      const uint_t zs = Field<T,fSize_>::zSize() + 2*gl_;

      return const_reverse_iterator( this,
                                    -cell_idx_c(gl_),-cell_idx_c(gl_),-cell_idx_c(gl_),0,
                                    xs, ys, zs, 1 );
   }


 protected:
   GhostLayerField( ): gl_(0){}

   uint_t gl_; ///< Number of ghost layers
   //** Shallow Copy ************************************************************************************************
   /*! \name Shallow Copy */
   //@{


   GhostLayerField(const GhostLayerField<T,fSize_> & other) : Field<T,fSize_>::Field(other), gl_( other.gl_ ){}

   template <typename T2, uint_t fSize2>
   GhostLayerField(const GhostLayerField<T2,fSize2> & other) : Field<T,fSize_>::Field(other), gl_( other.gl_ ){}


   template <typename T2, uint_t... fSize2>
   friend class GhostLayerField;

   //@}
   //****************************************************************************************************************

};
#ifdef WALBERLA_CXX_COMPILER_IS_CLANG
#pragma clang diagnostic pop
#endif


} // namespace field
} // namespace walberla

#include "GhostLayerField.impl.h"

//======================================================================================================================
//
//  EXPORTS
//
//======================================================================================================================

namespace walberla
{
// Export ghost layer field class to walberla namespace
using field::GhostLayerField;
} // namespace walberla
