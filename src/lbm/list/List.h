////======================================================================================================================
////
////  This file is part of waLBerla. waLBerla is free software: you can
////  redistribute it and/or modify it under the terms of the GNU General Public
////  License as published by the Free Software Foundation, either version 3 of
////  the License, or (at your option) any later version.
////
////  waLBerla is distributed in the hope that it will be useful, but WITHOUT
////  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
////  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
////  for more details.
////
////  You should have received a copy of the GNU General Public License along
////  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
////
////! \file List.h
////! \ingroup lbm
////! \author Christian Godenschwager <christian.godenschwager@fau.de>
////
////======================================================================================================================
//
//#pragma once
//
//#include "CellDir.h"
//
//#include "core/DataTypes.h"
//#include "core/Macros.h"
//
//#include "core/cell/Cell.h"
//
//#include "core/debug/CheckFunctions.h"
//
//#include "core/math/Vector3.h"
//
//#include "core/logging/Logging.h"
//
//#include "domain_decomposition/StructuredBlockStorage.h"
//
//#include "field/GhostLayerField.h"
//
//#include "lbm/lattice_model/EquilibriumDistribution.h"
//
//#include "simd/AlignedAllocator.h"
//
//#include "stencil/Directions.h"
//
//#include <algorithm>
//#include <limits>
//#include <map>
//#include <vector>
//#include <numeric>
//
//namespace walberla {
//namespace lbm {
//
//struct CellToIdxOrdering
//{
//   bool operator()( const std::pair< Cell, uint_t > & lhs, const std::pair< Cell, uint_t > & rhs ) const
//   {
//      return lhs.first < rhs.first;
//   }
//};
//
//// TODO support different layouts -> AoS, so far only SoA supported
//template<typename Index_T, typename Stencil_T>
//class List
//{
//public:
//   typedef Index_T                        index_t;
//   typedef Stencil_T Stencil;
//
//   List(const bool manuallyAllocateTmpPDFs = false ) : manuallyAllocateTmpPDFs_( manuallyAllocateTmpPDFs ) { }
//
//   void init(const std::vector<Cell> & fluidCells);
//
//   bool isFluidCell( const Cell & cell ) const
//   {
//      return std::binary_search( cellToIdx_.begin(), cellToIdx_.end(), std::make_pair( cell, 0 ), CellToIdxOrdering() );
//   }
//
//   uint_t getIdx( const Cell & cell ) const
//   {
//      auto it = std::lower_bound( cellToIdx_.begin(), cellToIdx_.end(), std::make_pair( cell, 0 ), CellToIdxOrdering() );
//      WALBERLA_CHECK_EQUAL( it->first, cell, "Cell is no fluid cell in LBM List!" )
//      return it->second;
//   }
//
//   const Cell & getCell( const uint_t cellIdx ) const
//   {
//      WALBERLA_ASSERT_LESS( cellIdx, idxToCell_.size() )
//      return idxToCell_[ cellIdx ];
//   }
//
//   const std::vector<Cell> & getFluidCells() const { return idxToCell_; }
//
//   Index_T getPDFIdx(const uint_t idx, const uint_t f) const;
//   Index_T getPullIdx( const uint_t idx,  const uint_t d ) const;
//
//   // TODO this only works for SoA
//
//   // End: this only works for SoA
//
//   WALBERLA_FORCE_INLINE(       uint_t size( ) )       { return pdfs_.size(); }
//   WALBERLA_FORCE_INLINE(uint_t numFluidCells() const) { return numFluidCells_; }
//
//   WALBERLA_FORCE_INLINE(       uint_t xStride( ) )       { return numFluidCellsPadded_; }
//   WALBERLA_FORCE_INLINE(       uint_t fStride( ) )       { return 1; }
//
//   WALBERLA_FORCE_INLINE(       real_t * getPDFbegining( ) )       { return &pdfs_.front(); }
//   WALBERLA_FORCE_INLINE(       real_t * gettmpPDFbegining( ) )       { return &tmpPdfs_.front(); }
//   WALBERLA_FORCE_INLINE(       Index_T * getidxbeginning( ) )       {return &pullIdxs_.front(); }
//
//   WALBERLA_FORCE_INLINE(       real_t & get( const uint_t idx, const uint_t f ) )       { WALBERLA_ASSERT_LESS( getPDFIdx( idx, f ), pdfs_.size() ) return pdfs_[getPDFIdx( idx, f )]; }
//   WALBERLA_FORCE_INLINE( const real_t & get( const uint_t idx, const uint_t f ) const )       { WALBERLA_ASSERT_LESS( getPDFIdx( idx, f ), pdfs_.size() ) return pdfs_[getPDFIdx( idx, f )]; }
//
//   WALBERLA_FORCE_INLINE(       real_t & get( const Cell & cell, const uint_t f ) )      { return get( getIdx( cell ), f ); }
//   WALBERLA_FORCE_INLINE( const real_t & get( const Cell & cell, const uint_t f ) const ) { return get( getIdx( cell ), f ); }
//
//   WALBERLA_FORCE_INLINE(       real_t & getTmp( const uint_t idx, const uint_t f ) )     { WALBERLA_ASSERT_LESS( getPDFIdx( idx, f ), tmpPdfs_.size() ) return tmpPdfs_[getPDFIdx( idx, f )]; }
//   WALBERLA_FORCE_INLINE(       real_t & getTmp( const Cell & cell, const uint_t f ) )      { return getTmp( getIdx( cell ), f ); }
//
//   WALBERLA_FORCE_INLINE(       real_t & get( const Index_T pdfIdx ) )      { WALBERLA_ASSERT_LESS( pdfIdx, pdfs_.size() ) return pdfs_[pdfIdx]; }
//   WALBERLA_FORCE_INLINE( const real_t & get( const Index_T pdfIdx ) const ) { WALBERLA_ASSERT_LESS( pdfIdx, pdfs_.size() ) return pdfs_[pdfIdx]; }
//
//   WALBERLA_FORCE_INLINE(  Index_T getPullIdx( const Cell & cell, const uint_t d ) const ) { return getPullIdx( getIdx( cell ), d ); }
//   WALBERLA_FORCE_INLINE( Index_T getPDFIdx( const Cell & cell, const uint_t f ) const) { return getPDFIdx( getIdx( cell ), f ); }
//
//    Vector3<real_t> getVelocity( uint_t idx ) const;
//    Vector3<real_t> getVelocity( const Cell & cell ) const { return getVelocity( getIdx( cell ) ); }
//
//    real_t getDensity( uint_t idx ) const;
//    real_t getDensity( const Cell & cell ) const { return getDensity( getIdx( cell ) ); }
//
//   // void setEquilibrium( const real_t rho, const Vector3<real_t> velocity );
//
//   void swapTmpPdfs() { swap( pdfs_, tmpPdfs_ ); }
//
//   bool operator==( const List<Index_T, Stencil_T> & other ) const;
//
//   Index_T registerExternalPDFs( const std::vector< CellDir > & pdfs );
//
//   void enableAutomaticAllocationOfTmpPDFs() { manuallyAllocateTmpPDFs_ = false; tmpPdfs_.resize( pdfs_.size(), std::numeric_limits<real_t>::signaling_NaN() ); }
//
//   inline void toBuffer  ( mpi::SendBuffer & buffer ) const;
//   inline void fromBuffer( mpi::RecvBuffer & buffer );
//
//protected:
//
//  std::array<int8_t, 19> cx = {0,0,0,-1,1,0,0,-1,1,-1,1,0,0,-1,1,0,0,-1,1};
//  std::array<int8_t, 19> cy = {0,1,-1,0,0,0,0,1,1,-1,-1,1,-1,0,0,1,-1,0,0};
//  std::array<int8_t, 19> cz = {0,0,0,0,0,1,-1,0,0,0,0,1,1,1,1,-1,-1,-1,-1};
//
//  void setPullIdx( const uint_t idx, const uint_t d, const Index_T pdfIdx );
//   void setPullIdx( const Cell & cell, const uint_t d, const Index_T pdfIdx ) { setPullIdx( getIdx( cell ), d, pdfIdx ); }
//
//   uint_t numFluidCells_{ 0 };
//   uint_t numFluidCellsPadded_{ 0 };
//
//   // TODO: should be templated: Layout_T::alignment() did not work
//   std::vector< real_t, simd::aligned_allocator< real_t, 64 > > pdfs_;
//   std::vector< real_t, simd::aligned_allocator< real_t, 64 > > tmpPdfs_;
//
//   std::vector< Index_T, simd::aligned_allocator< Index_T, 64 > > pullIdxs_;
//
//   std::vector< Cell >                      idxToCell_;
//   std::vector< std::pair< Cell, uint_t > > cellToIdx_;
//
//   bool manuallyAllocateTmpPDFs_;
//};
//
//template< typename Index_T, typename Stencil_T >
//void List<Index_T, Stencil_T>::init( const std::vector<Cell> & fluidCells)
//{
//   // Setup conversion data structures idx <-> cartesian coordinates
//
//   idxToCell_ = fluidCells;
//   numFluidCells_ = fluidCells.size();
//   uint_t alignement = 64;
//
//   {
//      uint_t idx = 0;
//      cellToIdx_.reserve( fluidCells.size() );
//      for( const Cell & cell : fluidCells )
//      {
//         cellToIdx_.emplace_back( cell, idx++ );
//      }
//      std::sort( cellToIdx_.begin(), cellToIdx_.end(), CellToIdxOrdering() );
//   }
//
//   // Allocate Fluid PDFs
//
//   // static_assert(((ALIGNMENT & (ALIGNMENT - uint_t(1))) == 0) && (ALIGNMENT >= uint_t(1)),
//    //              "The alignment for the SoA list layout has to a power of two!");
//
//   uint_t alignedStepSize = std::max(uint_t(1), alignement / sizeof(real_t));
//   WALBERLA_LOG_DEVEL_VAR(alignedStepSize)
//   if ((fluidCells.size() % alignedStepSize) == 0)
//      numFluidCellsPadded_ =  fluidCells.size();
//   else
//      numFluidCellsPadded_ = (fluidCells.size() / alignedStepSize + uint_t(1)) * alignedStepSize;
//
//
////   WALBERLA_CHECK_LESS( Stencil_T::Size * numFluidCellsPadded_, numeric_cast<size_t>( std::numeric_limits<Index_T>::max() ),
////                        "The number of PDFs you want to initialize the PDFs list with is beyond the capacity of Index_T!" )
//
//   pdfs_.assign( Stencil_T::Size * numFluidCellsPadded_, std::numeric_limits<real_t>::signaling_NaN() );
//
//   if( !manuallyAllocateTmpPDFs_ )
//      tmpPdfs_.assign( Stencil_T::Size * numFluidCellsPadded_, std::numeric_limits<real_t>::signaling_NaN() );
//
//   pullIdxs_.resize( ( Stencil_T::Size - uint_t( 1 ) ) * numFluidCellsPadded_ );
//
////   WALBERLA_LOG_INFO_ON_ROOT("Fluid cells padded " << numFluidCellsPadded_)
////
////   WALBERLA_LOG_INFO_ON_ROOT("Pull inidzies size " << pullIdxs_.size())
////   WALBERLA_LOG_INFO_ON_ROOT("PDF inidzies size " << pdfs_.size())
//
//   // Initialize pull idxs with no-slip boundary
//
//   for( auto dirIt = Stencil_T::beginNoCenter(); dirIt != Stencil_T::end(); ++dirIt )
//   {
//      for( uint_t idx = 0; idx < numFluidCellsPadded_; ++idx )
//      {
//         setPullIdx( idx, *dirIt, getPDFIdx( idx, dirIt.inverseDir() ) );
//      }
//   }
//
////   WALBERLA_LOG_INFO_ON_ROOT("numFluidCells() " << numFluidCells())
//   // Setup neighbor indices
//   {
//      uint_t idx = 0;
//
//      for( ; idx < numFluidCells(); ++idx )
//      {
//         Cell cell = getCell( idx );
//         for( auto dirIt = Stencil_T::beginNoCenter(); dirIt != Stencil_T::end(); ++dirIt )
//         {
//            Cell neighbor = cell + *dirIt;
//            if( isFluidCell( neighbor ) )
//            {
////               WALBERLA_LOG_INFO_ON_ROOT("Nb: " << idx )
//               setPullIdx( idx, dirIt.inverseDir(), getPDFIdx( neighbor, dirIt.inverseDir() ) );
//            }
//         }
//      }
//
////      WALBERLA_LOG_INFO_ON_ROOT("start IDX: " << idx)
//
//      // make pulls for padding cells cheap
//      for( ; idx < numFluidCellsPadded_; ++idx )
//      {
//         for( auto dirIt = Stencil_T::beginNoCenter(); dirIt != Stencil_T::end(); ++dirIt )
//         {
//            setPullIdx( idx, *dirIt, getPDFIdx( idx, *dirIt ) );
//         }
//      }
//
////      WALBERLA_LOG_INFO_ON_ROOT("end IDX: " << idx)
//   }
//
////   for( int idx = 0; idx < pullIdxs_.size(); ++idx )
////   {
////      WALBERLA_LOG_INFO_ON_ROOT("entry: " << idx << "    index: " << pullIdxs_[idx])
////   }
//
//}
//
//
//template< typename Index_T, typename Stencil_T >
//Vector3<real_t> List<Index_T, Stencil_T>::getVelocity( uint_t idx ) const
//{
//   Vector3<real_t> velocity;
//
//   real_t vel0Term = get( idx, 10 ) + get( idx, 14 ) + get( idx, 18 ) + get( idx, 4 ) + get( idx, 8 );
//   real_t vel1Term = get( idx, 1 ) + get( idx, 11 ) + get( idx, 15 ) + get( idx, 7 );
//   real_t vel2Term = get( idx, 12 ) + get( idx, 13 ) + get( idx, 5 );
//
//   velocity[0] = -get( idx, 13 ) - get( idx, 17 ) - get( idx, 3 ) - get( idx, 7 ) - get( idx, 9 ) + vel0Term;
//   velocity[1] = -get( idx, 10 ) - get( idx, 12 ) - get( idx, 16 ) - get( idx, 2 ) + get( idx, 8) - get( idx, 9 ) + vel1Term;
//   velocity[2] = get( idx, 13 ) + get( idx, 14 ) - get( idx, 15 ) - get( idx, 16 ) - get( idx, 17 ) - get( idx, 8 ) - get( idx, 6 ) + vel2Term;
//
//   return velocity;
//}
//
//
//template< typename Index_T, typename Stencil_T >
//real_t List<Index_T, Stencil_T>::getDensity( uint_t idx ) const
//{
//   real_t rho = real_t( 0 );
//   for( uint_t f = 0; f < 19; ++f )
//   {
//      rho += get( idx, f );
//   }
//   rho += real_t( 1 );
//
//   return rho;
//}
//
//template< typename Index_T, typename Stencil_T >
//Index_T List<Index_T, Stencil_T>::getPullIdx( const uint_t idx, const uint_t d ) const
//{
//   // TODO: Only SoA
//   WALBERLA_ASSERT_UNEQUAL(d, 0)
//
//   const uint_t f = d - 1;
//   // TODO: think about assert
//   WALBERLA_ASSERT_LESS( f * numFluidCellsPadded_ + idx, pullIdxs_.size() )
//   return numeric_cast< Index_T >(f * numFluidCellsPadded_ + idx);
//}
//
//template< typename Index_T, typename Stencil_T >
//Index_T List<Index_T, Stencil_T>::getPDFIdx(const uint_t idx, const uint_t f) const
//{
//   // TODO: Only SoA
//   return numeric_cast< Index_T >(f * numFluidCellsPadded_ + idx);
//}
//
//template< typename Index_T, typename Stencil_T >
//void List<Index_T, Stencil_T>::setPullIdx( const uint_t idx, const uint_t d, const Index_T pdfIdx )
//{
//   // TODO: think about assert
//   WALBERLA_ASSERT_LESS( getPullIdx( idx, d ), pullIdxs_.size() )
//   // WALBERLA_LOG_INFO_ON_ROOT("getPullIdxIdx( idx, d )  "  << getPullIdxIdx( idx, d ))
//   pullIdxs_[ getPullIdx( idx, d ) ] = pdfIdx;
//}
//
//
//
//template< typename Index_T, typename Stencil_T>
//bool List<Index_T, Stencil_T>::operator==( const List<Index_T, Stencil_T> & other ) const
//{
//   return this->pdfs_      == other.pdfs_
//       && this->idxToCell_ == other.idxToCell_;
//}
//
//template< typename Index_T, typename Stencil_T >
//Index_T List< Index_T, Stencil_T>::registerExternalPDFs( const std::vector< CellDir > & externalPdfs )
//{
//   WALBERLA_CHECK_LESS( pdfs_.size() + externalPdfs.size(), numeric_cast<size_t>( std::numeric_limits<Index_T>::max() ),
//                       "The Number of PDFs you want to register as external PDFs increases the total number of stored "
//                       "PDFs beyond the capacity of Index_T!" )
//
//   const Index_T startIdx = numeric_cast<Index_T>( pdfs_.size() );
//
////   WALBERLA_LOG_INFO_ON_ROOT("boundary cells: " << externalPdfs.size())
//
//   pdfs_.resize( pdfs_.size() + externalPdfs.size(), std::numeric_limits<real_t>::signaling_NaN() );
//
//   if( !manuallyAllocateTmpPDFs_ )
//      tmpPdfs_.resize( pdfs_.size(), std::numeric_limits<real_t>::signaling_NaN() );
//
//   Index_T idx = startIdx;
//
//   for( auto it = externalPdfs.begin(); it != externalPdfs.end(); ++it )
//   {
//      Cell fluidNeighborCell = it->cell + Cell(cx[it->dir], cy[it->dir], cz[it->dir]);
//      if( isFluidCell( fluidNeighborCell ) )
//      {
////         WALBERLA_LOG_INFO_ON_ROOT("pdfsToRegister cell  " << it->cell << "  dir  " << it->dir)
//         setPullIdx( fluidNeighborCell, it->dir, idx++ );
//      }
//      else
//      {
//         WALBERLA_LOG_WARNING( "You are registering external PDF " << it->cell << " " << it->dir << " but the neighboring cell "
//                               << fluidNeighborCell << " is not a fluid cell! The external PDF is unused." )
//      }
//   }
//
////   WALBERLA_LOG_INFO_ON_ROOT("After boundary  ")
////   for( int t = 0; t < pullIdxs_.size(); ++t )
////   {
////      WALBERLA_LOG_INFO_ON_ROOT("entry: " << t << "    index: " << pullIdxs_[t])
////   }
//
//
//   WALBERLA_ASSERT_EQUAL( idx, pdfs_.size() )
//
//   return startIdx;
//}
//
//
//template< typename Index_T, typename Stencil_T >
//inline void List< Index_T, Stencil_T >::toBuffer( mpi::SendBuffer & buffer ) const
//{
//   buffer << numFluidCellsPadded_ << pdfs_ << pullIdxs_ << idxToCell_ << manuallyAllocateTmpPDFs_;
//}
//
//template< typename Index_T, typename Stencil_T >
//inline void List< Index_T, Stencil_T >::fromBuffer( mpi::RecvBuffer & buffer )
//{
//   buffer >> numFluidCellsPadded_ >> pdfs_ >> pullIdxs_ >> idxToCell_ >> manuallyAllocateTmpPDFs_;
//
//   tmpPdfs_.resize( pdfs_.size(), std::numeric_limits<real_t>::signaling_NaN() );
//
//   for( uint_t i = 0; i < idxToCell_.size(); ++i )
//      cellToIdx_.emplace_back( idxToCell_[i], i );
//
//   std::sort( cellToIdx_.begin(), cellToIdx_.end(), CellToIdxOrdering() );
//}
//
//} // namespace lbm
//} // namespace walberla
