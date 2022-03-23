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
class GhostLayerField<T, fSize_> : public GhostLayerField<T> {
 public:
   GhostLayerField(GhostLayerField<T> field)
      : GhostLayerField<T>::GhostLayerField(field)
   {}

   static const uint_t F_SIZE = fSize_;
   static constexpr bool OLD = true;

   typedef typename std::conditional<VectorTrait<T>::F_SIZE!=0,
                                      GhostLayerField<typename VectorTrait<T>::OutputType, VectorTrait<T>::F_SIZE*fSize_>,
                                      GhostLayerField<T, fSize_>>::type FlattenedField;

   GhostLayerField( uint_t xSize, uint_t ySize, uint_t zSize, uint_t gl,
                   const Layout & layout = zyxf,
                   const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >() )
      : GhostLayerField<T>::GhostLayerField(xSize, ySize, zSize, fSize_, gl, layout, alloc) {}

   GhostLayerField( uint_t xSize, uint_t ySize, uint_t zSize, uint_t gl,
                   const T & initValue, const Layout & layout = zyxf,
                   const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >() )
      : GhostLayerField<T>::GhostLayerField(xSize, ySize, zSize, fSize_, gl, initValue, layout, alloc) {}

   GhostLayerField( uint_t xSize, uint_t ySize, uint_t zSize, uint_t gl,
                   const std::vector<T> & fValues, const Layout & layout = zyxf,
                   const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >() )
      : GhostLayerField<T>::GhostLayerField(xSize, ySize, zSize, fSize_, gl, fValues, layout, alloc) {}



   template<typename ...Args>
   void init(uint_t xSizeWithoutGhostLayer,
             uint_t ySizeWithoutGhostLayer,
             uint_t zSizeWithoutGhostLayer,
             uint_t nrGhostLayers,
             const Layout & layout = zyxf,
             const shared_ptr<FieldAllocator<T> > &alloc = shared_ptr<FieldAllocator<T> >()
   )
   {
      GhostLayerField<T>::init(xSizeWithoutGhostLayer, ySizeWithoutGhostLayer, zSizeWithoutGhostLayer, fSize_, nrGhostLayers, layout, alloc);
   }

   template<typename ...Args>
   void resize(uint_t xSize, uint_t ySize, uint_t zSize)
   {
      GhostLayerField<T>::resize(xSize, ySize, zSize, fSize_);
   }

   template<typename ...Args>
   GhostLayerField<T, fSize_>  * clone() const
   {
      return reinterpret_cast<GhostLayerField<T, fSize_>* > (GhostLayerField<T>::clone());
   }

   template<typename ...Args>
   GhostLayerField<T, fSize_>  * cloneUninitialized() const
   {
      return reinterpret_cast<GhostLayerField<T, fSize_>* > (GhostLayerField<T>::cloneUninitialized());
   }

   template<typename ...Args>
   GhostLayerField<T, fSize_>  * cloneShallowCopy() const
   {
      return reinterpret_cast<GhostLayerField<T, fSize_>* > (GhostLayerField<T>::cloneShallowCopy());
   }

   template<typename ...Args>
   FlattenedField* flattenedShallowCopy() const
   {
      return reinterpret_cast<FlattenedField* > (GhostLayerField<T>::flattenedShallowCopy());
   }

   //** Shallow Copy ************************************************************************************************
   /*! \name Shallow Copy */
   //@{

   GhostLayerField<T, fSize_>( const GhostLayerField<T, fSize_>& other ): Field<T, fSize_>( other )
   {}


   template <typename T2, uint_t fSize2>
   GhostLayerField<T, fSize_>(const GhostLayerField<T2, fSize2> & other) : Field<T2, fSize2>( other )
   {}

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
