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
//! \\file {{class_name}}.h
//! \\author pystencils
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
{% if target is equalto 'gpu' %}
#include "cuda/FieldCopy.h"
{% endif %}
#include "simd/AlignedAllocator.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <vector>
#include <array>
#include <numeric>

namespace walberla {
namespace lbmpy {

// TODO support different layouts -> AoS, so far only SoA supported
class {{class_name}}
{
 public:
   typedef {{index_type}}                       index_t;

   {{class_name}}( const Vector3<uint64_t> split, const bool manuallyAllocateTmpPDFs = false ) : {% if target is equalto 'gpu' -%}pdfsGPU_(nullptr), tmpPdfsGPU_(nullptr), omegasGPU_(nullptr), pullIdxsGPU_(nullptr), pullIdxsInnerGPU_(nullptr), pullIdxsOuterGPU_(nullptr), {%- endif %} split_(split), manuallyAllocateTmpPDFs_( manuallyAllocateTmpPDFs ) { }

   ~{{class_name}}() {
      {% if target is equalto 'gpu' -%}
      void clearGPUArrays();
      {%- endif %}
   }

   void init(const std::vector<Cell> & fluidCells);
   void syncGPU();
   void clearGPUArrays();
   void copyPDFSToCPU();

   bool isFluidCell( const Cell & cell ) const
   {
      if(cellToIdx_.find(cell) == cellToIdx_.end())
         return false;
      return true;
   }

   uint_t getIdx( const Cell & cell ) const
   {
      return cellToIdx_.at(cell);
   }

   uint_t getIdxInner( const Cell & cell ) const
   {
      return cellToIdxInner_.at(cell);
   }

   uint_t getIdxOuter( const Cell & cell ) const
   {
      return cellToIdxOuter_.at(cell);
   }

   const Cell & getCell( const uint_t cellIdx ) const
   {
      WALBERLA_ASSERT_LESS( cellIdx, idxToCell_.size() )
      return idxToCell_[ cellIdx ];
   }

   const Cell & getCellFromPDFIdx( const uint_t PDFIdx ) const
   {
      uint_t cellIdx = PDFIdx % numFluidCells_;
      WALBERLA_ASSERT_LESS( cellIdx, idxToCell_.size() )
      return idxToCell_[ cellIdx ];
   }

   uint_t getDirFromPDFIdx( const uint_t PDFIdx ) const
   {
      uint_t dir = uint_c( PDFIdx / numFluidCells_);
      WALBERLA_ASSERT_LESS( dir, {{Q}} )
      return dir;
   }

   const std::vector<Cell> & getFluidCells() const { return idxToCell_; }

   {{index_type}} getPDFIdx(const uint_t idx, const uint_t f) const;

   {{index_type}} getPullIdx( const uint_t idx,  const uint_t d ) const;
   {{index_type}} getPullIdxInner( const uint_t idx, const uint_t d ) const;
   {{index_type}} getPullIdxOuter( const uint_t idx, const uint_t d ) const;

   WALBERLA_FORCE_INLINE(       uint_t size( ) )       { return pdfs_.size(); }
   WALBERLA_FORCE_INLINE(uint_t numFluidCells() const) { return numFluidCells_; }
   WALBERLA_FORCE_INLINE(uint_t numFluidCellsInner() const) { return numFluidCellsInner_; }
   WALBERLA_FORCE_INLINE(uint_t numFluidCellsOuter() const) { return numFluidCellsOuter_; }

   WALBERLA_FORCE_INLINE(       uint_t xStride( ) )       { return numFluidCellsPadded_; }
   WALBERLA_FORCE_INLINE(       uint_t fStride( ) )       { return 1; }

   WALBERLA_FORCE_INLINE(       uint_t sizeIDX( ) )       { return pullIdxs_.size(); }
   WALBERLA_FORCE_INLINE(       uint_t sizeIDXInner( ) )       { return numFluidCellsInner_; }
   WALBERLA_FORCE_INLINE(       uint_t sizeIDXOuter( ) )       { return numFluidCellsOuter_; }

   WALBERLA_FORCE_INLINE(       real_t * getPDFbegining( ) )       { return &pdfs_.front(); }
   WALBERLA_FORCE_INLINE(       real_t * gettmpPDFbegining( ) )       { return &tmpPdfs_.front(); }
   WALBERLA_FORCE_INLINE(       {{index_type}} * getidxbeginning( ) )       {return &pullIdxs_.front(); }
   WALBERLA_FORCE_INLINE(       real_t * getomegasbegining( ) )       { return &omegas_.front(); }


   WALBERLA_FORCE_INLINE(       {{index_type}} * getidxInnerbeginning( ) )       {return &pullIdxsInner_.front(); }
   WALBERLA_FORCE_INLINE(       {{index_type}} * getidxOuterbeginning( ) )       {return &pullIdxsOuter_.front(); }

   WALBERLA_FORCE_INLINE( const real_t * getPDFbegining( ) const)       { return &pdfs_.front(); }
   WALBERLA_FORCE_INLINE( const real_t * gettmpPDFbegining( ) const)       { return &tmpPdfs_.front(); }
   WALBERLA_FORCE_INLINE( const {{index_type}} * getidxbeginning( ) const)       {return &pullIdxs_.front(); }
   WALBERLA_FORCE_INLINE( const real_t * getomegasbegining( ) const )       { return &omegas_.front(); }

   {% if target is equalto 'gpu' -%}
   WALBERLA_FORCE_INLINE(       real_t * getGPUPDFbegining( ) )       { return pdfsGPU_; }
   WALBERLA_FORCE_INLINE(       real_t * getGPUtmpPDFbegining( ) )       { return tmpPdfsGPU_; }
   WALBERLA_FORCE_INLINE(       real_t * getGPUomegasbegining( ) )       { return &omegas_.front(); }


   WALBERLA_FORCE_INLINE(       {{index_type}} * getGPUidxbeginning( ) )       {return pullIdxsGPU_; }
   WALBERLA_FORCE_INLINE(       {{index_type}} * getGPUidxInnerbeginning( ) )       {return pullIdxsInnerGPU_; }
   WALBERLA_FORCE_INLINE(       {{index_type}} * getGPUidxOuterbeginning( ) )       {return pullIdxsOuterGPU_; }

   std::pair< stencil::Direction, index_t * > getSendPDFsGPU(stencil::Direction dir) const { return *sendPDFsGPU_.find( dir ); }
   void setSendPDFsGPU(index_t * pdfs, stencil::Direction dir) {sendPDFsGPU_[dir] = pdfs; }

   //void setSendPDFsGPU(std::vector<{{index_type}}> sendPDFs, stencil::Direction dir) { sendPDFsGPU_[ dir ] = sendPDFs; }

   {%- endif %}

   WALBERLA_FORCE_INLINE(       real_t & get( const uint_t idx, const uint_t f ) )       { WALBERLA_ASSERT_LESS( getPDFIdx( idx, f ), pdfs_.size() ) return pdfs_[getPDFIdx( idx, f )]; }
   WALBERLA_FORCE_INLINE( const real_t & get( const uint_t idx, const uint_t f ) const )       { WALBERLA_ASSERT_LESS( getPDFIdx( idx, f ), pdfs_.size() ) return pdfs_[getPDFIdx( idx, f )]; }

   WALBERLA_FORCE_INLINE(       real_t & get( const Cell & cell, const uint_t f ) )      { return get( getIdx( cell ), f ); }
   WALBERLA_FORCE_INLINE( const real_t & get( const Cell & cell, const uint_t f ) const ) { return get( getIdx( cell ), f ); }

   WALBERLA_FORCE_INLINE(       real_t & getTmp( const uint_t idx, const uint_t f ) )     { WALBERLA_ASSERT_LESS( getPDFIdx( idx, f ), tmpPdfs_.size() ) return tmpPdfs_[getPDFIdx( idx, f )]; }
   WALBERLA_FORCE_INLINE(       real_t & getTmp( const Cell & cell, const uint_t f ) )      { return getTmp( getIdx( cell ), f ); }

   WALBERLA_FORCE_INLINE(       real_t & get( const {{index_type}} pdfIdx ) )      { WALBERLA_ASSERT_LESS( pdfIdx, pdfs_.size() ) return pdfs_[pdfIdx]; }
   WALBERLA_FORCE_INLINE( const real_t & get( const {{index_type}} pdfIdx ) const ) { WALBERLA_ASSERT_LESS( pdfIdx, pdfs_.size() ) return pdfs_[pdfIdx]; }

   WALBERLA_FORCE_INLINE(  {{index_type}} getPullIdx( const Cell & cell, const uint_t d ) const ) { return getPullIdx( getIdx( cell ), d ); }
   WALBERLA_FORCE_INLINE( {{index_type}} getPDFIdx( const Cell & cell, const uint_t f ) const) { return getPDFIdx( getIdx( cell ), f ); }

   WALBERLA_FORCE_INLINE(  {{index_type}} getPullIdxInner( const Cell & cell, const uint_t d ) const ) { return getPullIdxInner( getIdx( cell ), d ); }
   WALBERLA_FORCE_INLINE(  {{index_type}} getPullIdxOuter( const Cell & cell, const uint_t d ) const ) { return getPullIdxOuter( getIdx( cell ), d ); }


   std::pair< stencil::Direction, std::vector <{{index_type}}> > getSendPDFs(stencil::Direction dir) const { return *sendPDFs_.find( dir ); }
   std::pair< stencil::Direction, {{index_type}} > getStartCommIdx(stencil::Direction dir) const { return *startCommIdxs_.find( dir ); }
   std::pair< stencil::Direction, {{index_type}} > getNumCommPDFs(stencil::Direction dir) const { return *numCommPDFs_.find( dir ); }

   void setSendPDFs(std::vector<{{index_type}}> sendPDFs, stencil::Direction dir) { sendPDFs_[ dir ] = sendPDFs; }
   void setStartCommIdx({{index_type}} index, stencil::Direction dir) { startCommIdxs_[ dir ] = index; }
   void setNumCommPDFs({{index_type}} num, stencil::Direction dir) { numCommPDFs_[ dir ] = num; }

   real_t getOmega( const uint_t idx ) const;

   Vector3<real_t> getVelocity( uint_t idx ) const;
   Vector3<real_t> getVelocity( const Cell & cell ) const { return getVelocity( getIdx( cell ) ); }

   Vector3<real_t> getVelocityOdd( uint_t idx ) const;
   Vector3<real_t> getVelocityOdd( const Cell & cell ) const { return getVelocityOdd( getIdx( cell ) ); }

   real_t getDensity( uint_t idx ) const;
   real_t getDensity( const Cell & cell ) const { return getDensity( getIdx( cell ) ); }

   {% if target is equalto 'gpu' -%}
   void swapTmpPdfs() { std::swap( pdfsGPU_, tmpPdfsGPU_ ); }
   {%- else -%}
   void swapTmpPdfs() { swap( pdfs_, tmpPdfs_ ); }
   {%- endif %}

   bool operator==( const {{class_name}} & other ) const;

   {{index_type}} registerExternalPDFs( const std::vector< lbm::CellDir > & pdfs );
   {{index_type}} registerExternalPDFsDense( const std::vector< lbm::CellDir > & pdfs,  const std::vector< Cell > & boundaryCells);


   void enableAutomaticAllocationOfTmpPDFs() { manuallyAllocateTmpPDFs_ = false; tmpPdfs_.resize( pdfs_.size(), 0.0 ); }

   void toBuffer  ( mpi::SendBuffer & buffer ) const;
   void fromBuffer( mpi::RecvBuffer & buffer );

   template<typename FlagField_T>
   void fillFromFlagField(IBlock &block, ConstBlockDataID flagFieldID, FlagUID domainFlagUID)
   {
      // Setup conversion data structures idx <-> cartesian coordinates

      auto * flagField = block.getData< FlagField_T > ( flagFieldID );

      ci_ = CellInterval(cell_idx_c(split_[0]), cell_idx_c(split_[1]), cell_idx_c(split_[2]),
                         cell_idx_c(flagField->xSize()-split_[0] -1), cell_idx_c(flagField->ySize()-split_[1] -1), cell_idx_c(flagField->zSize()-split_[2] -1));

      if( !flagField->flagExists(domainFlagUID ))
         return;

      auto domainFlag = flagField->getFlag(domainFlagUID);

      for( auto it = flagField->begin(); it != flagField->end(); ++it )
      {
         if( isFlagSet(it, domainFlag) )
         {
            numFluidCells_++;
            if(ci_.contains(it.cell()))
            {
               numFluidCellsInner_++;
            }
         }
      }

      idxToCell_.reserve( numFluidCells_ );
      cellToIdx_.reserve( numFluidCells_ );

      {% if alignment > 0 %}
      uint_t alignedStepSize = std::max(uint_t(1), {{alignment}} / sizeof(real_t));
      if ((fluidCells.size() % alignedStepSize) == 0)
         numFluidCellsPadded_ =  numFluidCells_;
      else
         numFluidCellsPadded_ = (numFluidCells_ / alignedStepSize + uint_t(1)) * alignedStepSize;
      {% else -%}
      numFluidCellsPadded_ =  numFluidCells_;
      {% endif %}

      numFluidCellsOuter_ = numFluidCellsPadded_ - numFluidCellsInner_;

      WALBERLA_CHECK_LESS( {{Q}} * numFluidCellsPadded_, numeric_cast<size_t>( std::numeric_limits<{{index_type}}>::max() ),
                          "The number of PDFs you want to initialize the PDFs list with is beyond the capacity of index_type!" )

      pdfs_.assign( {{Q}} * numFluidCellsPadded_, 0.0 );

      if( !manuallyAllocateTmpPDFs_ )
         tmpPdfs_.assign( {{Q}} * numFluidCellsPadded_, 0.0 );

      pullIdxs_.resize( {{Q}} * numFluidCellsPadded_ );
      pullIdxsInner_.resize({{Q}} * numFluidCellsInner_);
      pullIdxsOuter_.resize({{Q}} * numFluidCellsOuter_);


      // Setup indices
      uint_t index = 0;
      uint_t index_inner = 0;
      uint_t index_outer = 0;
      for( auto it = flagField->begin(); it != flagField->end(); ++it )
      {
         if( !isFlagSet(it, domainFlag) )
            continue;

         idxToCell_.emplace_back(it.cell());
         cellToIdx_[it.cell()] = index;
         if(ci_.contains(it.cell()))
         {
            cellToIdxInner_[it.cell()] = index_inner;
            index_inner++;
         }
         else
         {
            cellToIdxOuter_[it.cell()] = index_outer;
            index_outer++;
         }
         index++;
      }

      for (size_t f = 0; f < {{Q}}; ++f)
      {
         index_inner = 0;
         index_outer = 0;
         for (size_t i = 0; i < numFluidCells(); ++i)
         {
            Cell cell      = getCell(i);
            Cell neighbour = cell + Cell(cx[f], cy[f], cz[f]);

            uint32_t pullIdx;
            //set pull index to neighbor, if neighbor is fluid cell, else set it to noslip
            if (isFluidCell(neighbour) && f != 0) {
               pullIdx = getPDFIdx(neighbour, inv_dir[f]);
            } else {
               pullIdx = getPDFIdx(i, f);
            }

            setPullIdx(i, inv_dir[f], pullIdx);
            if(ci_.contains(cell)) {
               setPullIdxInner(index_inner, inv_dir[f], pullIdx);
               index_inner++;
            } else {
               setPullIdxOuter(index_outer, inv_dir[f], pullIdx);
               index_outer++;
            }
         }
      }
   }




   template<typename FlagField_T, typename ScalarField_T>
   void fillOmegasFromFlagField(IBlock &block, ConstBlockDataID flagFieldID, FlagUID domainFlagUID, ConstBlockDataID omegaFieldID) {
     auto *flagField = block.getData<FlagField_T>(flagFieldID);
     auto *omegaField = block.getData<ScalarField_T>(omegaFieldID);
     auto domainFlag = flagField->getFlag(domainFlagUID);

     omegas_.reserve(numFluidCellsPadded_);
     for (auto it = flagField->begin(); it != flagField->end(); ++it) {
       if (!isFlagSet(it, domainFlag))
         continue;

       auto omega = omegaField->get(it.x(), it.y(), it.z());
       omegas_.emplace_back(omega);
     }
   }

 protected:

   std::array<int8_t, {{Q}}> cx = { {{direction_vectors['cx']}} };
   std::array<int8_t, {{Q}}> cy = { {{direction_vectors['cy']}} };
   std::array<int8_t, {{Q}}> cz = { {{direction_vectors['cz']}} };
   std::array<uint8_t, {{Q}}> inv_dir = { {{inv_dirs_vector}} };


   void setPullIdx( const uint_t idx, const uint_t d, const uint32_t pdfIdx );
   void setPullIdx( const Cell & cell, const uint_t d, const uint32_t pdfIdx ) { setPullIdx( getIdx( cell ), d, pdfIdx ); }

   void setPullIdxInner( const uint_t idx, const uint_t d, const uint32_t pdfIdx );
   void setPullIdxInner( const Cell & cell, const uint_t d, const uint32_t pdfIdx ) { setPullIdxInner( getIdxInner( cell ), d, pdfIdx ); }

   void setPullIdxOuter( const uint_t idx, const uint_t d, const uint32_t pdfIdx );
   void setPullIdxOuter( const Cell & cell, const uint_t d, const uint32_t pdfIdx ) { setPullIdxOuter( getIdxOuter( cell ), d, pdfIdx ); }

   void setOmega( const uint_t idx, const real_t omega );


   uint_t numFluidCells_{ 0 };
   uint_t numFluidCellsPadded_{ 0 };

   uint_t numFluidCellsInner_{ 0 };
   uint_t numFluidCellsOuter_{ 0 };

   {%if alignment > 0 %}
   std::vector< real_t, simd::aligned_allocator< real_t, {{alignment}}; > > pdfs_;
   std::vector< real_t, simd::aligned_allocator< real_t, {{alignment}}; > > tmpPdfs_;
   std::vector< {{index_type}}, simd::aligned_allocator< {{index_type}}, {{alignment}} > > pullIdxs_;
   {%- else -%}
   std::vector< real_t > pdfs_;
   std::vector< real_t > tmpPdfs_;
   std::vector< {{index_type}} > pullIdxs_;
   std::vector< {{index_type}} > pullIdxsInner_;
   std::vector< {{index_type}} > pullIdxsOuter_;

   std::vector< real_t > omegas_;

   {% endif %}

   {% if target is equalto 'gpu' -%}
   real_t * pdfsGPU_;
   real_t * tmpPdfsGPU_;
   real_t * omegasGPU_;


   {{index_type}} * pullIdxsGPU_;
   {{index_type}} * pullIdxsInnerGPU_;
   {{index_type}} * pullIdxsOuterGPU_;

   {%- endif %}

   std::vector< Cell >                      idxToCell_;
   std::unordered_map<Cell, uint_t>         cellToIdx_;
   std::unordered_map<Cell, uint_t>         cellToIdxInner_;
   std::unordered_map<Cell, uint_t>         cellToIdxOuter_;

   Vector3<uint64_t> split_;
   CellInterval ci_;
   bool manuallyAllocateTmpPDFs_;

   //for communication
   std::map< stencil::Direction , std::vector< index_t > > sendPDFs_;
   std::map< stencil::Direction , index_t * > sendPDFsGPU_;
   std::map< stencil::Direction , index_t > startCommIdxs_;
   std::map< stencil::Direction , index_t > numCommPDFs_;
};

template<typename FlagField_T, typename Stencil_T>
class ListCommunicationSetup
{
 public:
   ListCommunicationSetup(weak_ptr<StructuredBlockForest> blockForest, const BlockDataID listId, const BlockDataID flagFieldID = BlockDataID(), const FlagUID fluidFlagUID = "",  bool hybridComm = false, const Set<SUID> & requiredBlockSelectors = Set<SUID>::emptySet(), const Set<SUID> & incompatibleBlockSelectors = Set<SUID>::emptySet())
      :blockForest_(blockForest), listId_(listId), flagFieldID_(flagFieldID), fluidFlagUID_(fluidFlagUID) , hybridComm_(hybridComm), requiredBlockSelectors_( requiredBlockSelectors ), incompatibleBlockSelectors_( incompatibleBlockSelectors )
   {
     if (hybridComm_) {
       WALBERLA_ASSERT_UNEQUAL(flagFieldID, BlockDataID())
       WALBERLA_ASSERT_UNEQUAL(fluidFlagUID, "")
       setupHybridCommunication();
     }
     else
       setupSparseCommunication();
   }

   CellInterval getSendCellInterval( const stencil::Direction dir )
   {
      const cell_idx_t sizeArr[] = { cell_idx_c( blockSize_[0] ),
                                     cell_idx_c( blockSize_[1] ),
                                     cell_idx_c( blockSize_[2] ) };

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


   struct CellInCellIntervalFilter
   {
      CellInCellIntervalFilter( const CellInterval & _ci ) : ci( _ci ) { }
      bool operator()( const Cell & cell ) const { return ci.contains( cell );  }

      CellInterval ci;
   };

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




   void setupSparseCommunication( )
   {
      auto forest = blockForest_.lock();
      WALBERLA_CHECK_NOT_NULLPTR( forest, "Trying to execute communication for a block storage object that doesn't exist anymore" )

      WALBERLA_LOG_PROGRESS( "Setting up list communication" )

      std::map< uint_t, uint_t > numBlocksToSend;

      blockSize_ = Vector3<cell_idx_t>( cell_idx_c(forest->getNumberOfXCellsPerBlock()), cell_idx_c(forest->getNumberOfYCellsPerBlock()), cell_idx_c(forest->getNumberOfZCellsPerBlock()) );

      //get number of Blocks to send to each neighbour
      for( auto senderIt = forest->begin(); senderIt != forest->end(); ++senderIt )
      {
         blockforest::Block & sender = dynamic_cast<blockforest::Block &>( *senderIt );

         for (size_t f = 1; f < {{Q}}; ++f)
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

      //Initialize bufferSystem with number of blocks send to process x
      for( auto it = numBlocksToSend.begin(); it != numBlocksToSend.end(); ++it )
      {
         WALBERLA_LOG_DETAIL( "Packing information for " << it->second << " blocks to send to process " << it->first );
         bufferSystem.sendBuffer( it->first ) << it->second;
      }
      //send fluid cells on MPI interface to neighbor block (cells are mapped to GL of neighbor block)
      for( auto senderIt = forest->begin(); senderIt != forest->end(); ++senderIt )
      {
         blockforest::Block & sender = dynamic_cast<blockforest::Block &>( *senderIt );
         auto * senderList = sender.getData< lbmpy::{{class_name}} >( listId_ );
         WALBERLA_ASSERT_NOT_NULLPTR( senderList )

         for (size_t f = 1; f < {{Q}}; ++f)
         {
            auto neighborhood = sender.getNeighborhoodSection( blockforest::getBlockNeighborhoodSectionIndex( (stencil::Direction) f ) );
            WALBERLA_ASSERT_LESS_EQUAL( neighborhood.size(), size_t( 1 ) )
            if( neighborhood.empty() )
               continue;
            auto * receiver = neighborhood.front();

            receiver->getId().toBuffer( bufferSystem.sendBuffer( receiver->getProcess() ) );
            bufferSystem.sendBuffer( receiver->getProcess() ) << stencil::inverseDir[f];

            WALBERLA_LOG_DETAIL( "Packing information for block " << receiver->getId().getID() << " in direction " << stencil::dirToString[stencil::inverseDir[f]] );

            const CellInterval cellsToSendInterval = getSendCellInterval( (stencil::Direction) f );
            uint_t numCells = uint_c( std::count_if( senderList->getFluidCells().begin(), senderList->getFluidCells().end(), CellInCellIntervalFilter( cellsToSendInterval ) ) );
            WALBERLA_LOG_DETAIL( numCells << " cells found" );
            bufferSystem.sendBuffer( receiver->getProcess() ) << numCells;
            for( auto cellIt = senderList->getFluidCells().begin(); cellIt != senderList->getFluidCells().end(); ++cellIt )
            {
               if( cellsToSendInterval.contains( *cellIt ) )
               {
                  bufferSystem.sendBuffer( receiver->getProcess() ) << mapToNeighbor( *cellIt, (stencil::Direction) f );
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

            auto * senderList = localBlock->template getData< lbmpy::{{class_name}} >( listId_ );
            WALBERLA_ASSERT_NOT_NULLPTR( senderList )

            std::vector< lbm::CellDir > externalPDFs;
            for( auto it = ghostCells.begin(); it != ghostCells.end(); ++it )
            {
               for (size_t f = 1; f < {{Q}}; ++f)
               {
                  Cell neighborCell = *it + (stencil::Direction) f;
                  if( senderList->isFluidCell( neighborCell ) )
                  {
                     externalPDFs.push_back( lbm::CellDir( *it, (stencil::Direction) f ) );
                  }
               }
            }

            senderList->setStartCommIdx(senderList->registerExternalPDFs( externalPDFs ), dir);
            senderList->setNumCommPDFs(static_cast< {{index_type}} >( externalPDFs.size() ), dir);

            std::sort( ghostCells.begin(), ghostCells.end() );

            std::vector<{{index_type}}> sendPDFsVector;

            for( auto it = senderList->getFluidCells().begin(); it != senderList->getFluidCells().end(); ++it )
            {
               for (size_t f = 1; f < {{Q}}; ++f)
               {
                  Cell neighborCell = *it + (stencil::Direction) f;
                  if( std::binary_search( ghostCells.begin(), ghostCells.end(), neighborCell ) )
                  {
                     sendPDFsVector.push_back( senderList->getPDFIdx( *it, (stencil::Direction) f ) );
                  }
               }
            }
            senderList->setSendPDFs(sendPDFsVector, dir);

            {% if target is equalto 'gpu' -%}
            {{index_type}} * sendPDFsVectorGPU;
            cudaMalloc( &sendPDFsVectorGPU, sizeof({{index_type}}) * sendPDFsVector.size() );
            cudaMemcpy( sendPDFsVectorGPU, &sendPDFsVector[0], sizeof({{index_type}}) * sendPDFsVector.size(), cudaMemcpyHostToDevice );
            senderList->setSendPDFsGPU(sendPDFsVectorGPU, dir);
            {%- endif %}
         }
      }
      {% if target is equalto 'gpu' -%}
      for( auto blockIt = forest->begin(); blockIt != forest->end(); ++blockIt )
      {
         blockforest::Block & block = dynamic_cast<blockforest::Block &>( *blockIt );
         auto * pdfList = block.getData< lbmpy::ListLBMList >( listId_ );
         pdfList->syncGPU();
      }
      {%- endif %}

      WALBERLA_LOG_PROGRESS( "Setting up list communication finished" )
   }

   void setupHybridCommunication( )
   {
      auto forest = blockForest_.lock();
      WALBERLA_CHECK_NOT_NULLPTR( forest, "Trying to execute communication for a block storage object that doesn't exist anymore" )

      WALBERLA_LOG_PROGRESS( "Setting up list communication" )

      std::map< uint_t, uint_t > numBlocksToSend;

      blockSize_ = Vector3<cell_idx_t>( cell_idx_c(forest->getNumberOfXCellsPerBlock()), cell_idx_c(forest->getNumberOfYCellsPerBlock()), cell_idx_c(forest->getNumberOfZCellsPerBlock()) );

      //get number of Blocks to send to each neighbour
      for( auto senderIt = forest->begin(); senderIt != forest->end(); ++senderIt )
      {
         blockforest::Block & sender = dynamic_cast<blockforest::Block &>( *senderIt );

         for (size_t f = 1; f < {{Q}}; ++f)
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

      //Initialize bufferSystem with number of blocks send to process x
      for( auto it = numBlocksToSend.begin(); it != numBlocksToSend.end(); ++it )
      {
         WALBERLA_LOG_DETAIL( "Packing information for " << it->second << " blocks to send to process " << it->first );
         bufferSystem.sendBuffer( it->first ) << it->second;
      }

      //send boundary cells of beginSliceBeforeGhostLayerXYZ to neighbour sparse blocks
      for( auto senderIt = forest->begin(); senderIt != forest->end(); ++senderIt )
      {
         blockforest::Block & sender = dynamic_cast<blockforest::Block &>( *senderIt );
         auto *flagField = sender.getData<FlagField_T>(flagFieldID_);
         auto fluidFlag = flagField->getFlag( fluidFlagUID_ );

         for (size_t sendDir = 1; sendDir < {{Q}}; ++sendDir)
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

      //recieve boundary cells of sparse blocks and set only pull indices of registered PDFs, which do not pull from boundary cell
      for( auto recvBufferIt = bufferSystem.begin(); recvBufferIt != bufferSystem.end(); ++recvBufferIt )
      {
         uint_t numBlocks;
         recvBufferIt.buffer() >> numBlocks;
         WALBERLA_LOG_DETAIL( "Unpacking information from " << numBlocks << " blocks from process " << recvBufferIt.rank() );
         for( uint_t i = 0; i < numBlocks; ++i )
         {
            BlockID localBID;
            localBID.fromBuffer( recvBufferIt.buffer() );
            IBlock * localBlock = forest->getBlock( localBID );
            WALBERLA_ASSERT_NOT_NULLPTR( localBlock )

            stencil::Direction receiveDir;
            uint_t numCells;
            recvBufferIt.buffer() >> receiveDir;
            WALBERLA_LOG_DETAIL( "Unpacking information for block " << localBID.getID() << " in direction " << stencil::dirToString[receiveDir] );

            recvBufferIt.buffer() >> numCells;
            std::vector<Cell> isBoundary( numCells );
            for( auto it = isBoundary.begin(); it != isBoundary.end(); ++it )
            {
               recvBufferIt.buffer() >> *it;
            }

            if( !selectable::isSetSelected( localBlock->getState(), requiredBlockSelectors_, incompatibleBlockSelectors_ ) )
               continue;

            WALBERLA_LOG_DETAIL( isBoundary.size() << " boundary cells found" );

            auto * senderList = localBlock->template getData< lbmpy::{{class_name}} >( listId_ );
            WALBERLA_ASSERT_NOT_NULLPTR( senderList )
            auto *flagField = localBlock->getData<FlagField_T>(flagFieldID_);

            std::vector< lbm::CellDir > externalPDFs;
            std::vector< {{index_type}} > sendPDFsVector;
            auto flagIt = flagField->beginSliceBeforeGhostLayerXYZ(Stencil_T::dir[receiveDir]);
            {{index_type}} idx = numeric_cast<{{index_type}}>(senderList->size());
            while( flagIt != flagField->end() )
            {
               Cell cell(flagIt.x(), flagIt.y(), flagIt.z());
               Cell neighborCell = cell + (stencil::Direction) receiveDir;
               for(uint_t f = 0; f < Stencil_T::d_per_d_length[stencil::inverseDir[receiveDir]]; ++f)
               {
                     auto d_receive = Stencil_T::d_per_d[stencil::inverseDir[receiveDir]][f];
                     externalPDFs.push_back( lbm::CellDir( neighborCell, d_receive ) );
                     auto d_send = Stencil_T::d_per_d[receiveDir][f];
                     {{index_type}} sendIdx;
                     if ( !senderList->isFluidCell(cell) ) {
                        //if cell is boundary cell, send received pdf in GL back to neightbour to achieve (1 timestep later) NoSlip
                        sendIdx = idx + numeric_cast<{{index_type}}>(d_receive);
                     }
                     else {
                        sendIdx = senderList->getPDFIdx( cell, d_send );
                     }
                     sendPDFsVector.push_back( sendIdx );
               }
               idx+=numeric_cast<{{index_type}}>(Stencil_T::d_per_d_length[receiveDir]);
               ++flagIt;
            }
            senderList->setStartCommIdx(senderList->registerExternalPDFsDense( externalPDFs, isBoundary), receiveDir);
            senderList->setNumCommPDFs(static_cast< {{index_type}} >( externalPDFs.size() ), receiveDir);

            senderList->setSendPDFs(sendPDFsVector, (stencil::Direction) receiveDir);

            {% if target is equalto 'gpu' -%}
            {{index_type}} * sendPDFsVectorGPU;
            cudaMalloc( &sendPDFsVectorGPU, sizeof({{index_type}}) * sendPDFsVector.size() );
            cudaMemcpy( sendPDFsVectorGPU, &sendPDFsVector[0], sizeof({{index_type}}) * sendPDFsVector.size(), cudaMemcpyHostToDevice );
            senderList->setSendPDFsGPU(sendPDFsVectorGPU, receiveDir);
            {%- endif %}
         }
      }

      {% if target is equalto 'gpu' -%}
      for( auto blockIt = forest->begin(); blockIt != forest->end(); ++blockIt )
      {
         if( !selectable::isSetSelected( blockIt->getState(), requiredBlockSelectors_, incompatibleBlockSelectors_ ) )
            continue;
         blockforest::Block & block = dynamic_cast<blockforest::Block &>( *blockIt );
         auto * pdfList = block.getData< lbmpy::ListLBMList >( listId_ );
         pdfList->syncGPU();
      }
      {%- endif %}

      WALBERLA_LOG_PROGRESS( "Setting up list communication finished" )
   }

 protected:
   weak_ptr<StructuredBlockForest> blockForest_;
   const BlockDataID listId_;
   const BlockDataID flagFieldID_;
   const FlagUID fluidFlagUID_;

   Vector3< cell_idx_t > blockSize_;
   bool hybridComm_;
   Set<SUID> requiredBlockSelectors_;
   Set<SUID> incompatibleBlockSelectors_;
};

} // namespace lbmpy
} // namespace walberla


