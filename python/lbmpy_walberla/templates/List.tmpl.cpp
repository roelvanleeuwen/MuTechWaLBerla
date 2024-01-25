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
//! \\author lbmpy
//======================================================================================================================

#include "{{class_name}}.h"

namespace walberla {
namespace lbmpy {

void {{class_name}}::init( const std::vector<Cell> & fluidCells)
{
   // Setup conversion data structures idx <-> cartesian coordinates

   idxToCell_ = fluidCells;
   numFluidCells_ = fluidCells.size();

   {
      uint_t idx = 0;
      cellToIdx_.reserve( fluidCells.size() );
      for( const Cell & cell : fluidCells )
      {
         cellToIdx_[cell] = idx++;
      }
   }

   {% if alignment > 0 %}
   uint_t alignedStepSize = std::max(uint_t(1), {{alignment}} / sizeof(real_t));
   if ((fluidCells.size() % alignedStepSize) == 0)
      numFluidCellsPadded_ =  fluidCells.size();
   else
      numFluidCellsPadded_ = (fluidCells.size() / alignedStepSize + uint_t(1)) * alignedStepSize;
   {% else -%}
   numFluidCellsPadded_ =  fluidCells.size();
   {% endif %}

   WALBERLA_CHECK_LESS( {{Q}} * numFluidCellsPadded_, numeric_cast<size_t>( std::numeric_limits<{{index_type}}>::max() ),
                       "The number of PDFs you want to initialize the PDFs list with is beyond the capacity of index_type!" )

   pdfs_.assign( {{Q}} * numFluidCellsPadded_, 0.0 );

   if( !manuallyAllocateTmpPDFs_ )
      tmpPdfs_.assign( {{Q}} * numFluidCellsPadded_, 0.0 );

   pullIdxs_.resize( {{Q}} * numFluidCellsPadded_ );

   // Initialize pull idxs with no-slip boundary
   for( uint_t idx = 0; idx < numFluidCellsPadded_; ++idx )
   {
      {%- for dirIdx, dirVec, offset in stencil %}
      setPullIdx( idx, {{dirIdx}}, getPDFIdx( idx, {{inv_dir[dirIdx]}} ) );
      {%- endfor %}
   }

   // Setup neighbor indices
   {
      uint_t idx = 0;

      for( ; idx < numFluidCells(); ++idx )
      {
         Cell cell = getCell( idx );
         {%- for dirIdx, dirVec, offset in stencil %}
            if( isFluidCell( cell + Cell({{dirVec[0]}}, {{dirVec[1]}}, {{dirVec[2]}}) ) )
            {
               setPullIdx( idx, {{inv_dir[dirIdx]}}, getPDFIdx( cell + Cell({{dirVec[0]}}, {{dirVec[1]}}, {{dirVec[2]}}), {{inv_dir[dirIdx]}} ) );
            }
         {%- endfor %}
      }

      {%- if alignment > 0 %}
      // execute direct field copy on padded cells to make padded cells access cheap
      for( ; idx < numFluidCellsPadded_; ++idx )
      {
         {%- for dirIdx, dirVec, offset in stencil %}
            setPullIdx( idx, {{dirIdx}}, getPDFIdx( idx, {{dirIdx}} ) );
         {%- endfor %}
      }
      {%- endif %}

   }
   syncGPU();
}

{% if target is equalto 'gpu' -%}
void {{class_name}}::syncGPU()
{
   if (pdfsGPU_)
   {
      WALBERLA_CUDA_CHECK( cudaFree( pdfsGPU_ ) );
      pdfsGPU_ = nullptr;
   }
   if (tmpPdfsGPU_)
   {
      WALBERLA_CUDA_CHECK( cudaFree( tmpPdfsGPU_ ));
      tmpPdfsGPU_ = nullptr;
   }
   if (pullIdxsGPU_)
   {
      WALBERLA_CUDA_CHECK( cudaFree( pullIdxsGPU_ ));
      pullIdxsGPU_ = nullptr;
   }
   if (pullIdxsInnerGPU_)
   {
      WALBERLA_CUDA_CHECK( cudaFree( pullIdxsInnerGPU_ ));
      pullIdxsInnerGPU_ = nullptr;
   }
   if (pullIdxsOuterGPU_)
   {
      WALBERLA_CUDA_CHECK( cudaFree( pullIdxsOuterGPU_ ));
      pullIdxsOuterGPU_ = nullptr;
   }

   WALBERLA_CUDA_CHECK( cudaMalloc( &pdfsGPU_, sizeof(real_t) * pdfs_.size() ));
   WALBERLA_CUDA_CHECK( cudaMemcpy( pdfsGPU_, &pdfs_[0], sizeof(real_t) * pdfs_.size(), cudaMemcpyHostToDevice ));

   WALBERLA_CUDA_CHECK( cudaMalloc( &tmpPdfsGPU_, sizeof(real_t) * tmpPdfs_.size() ));
   WALBERLA_CUDA_CHECK( cudaMemcpy( tmpPdfsGPU_, &tmpPdfs_[0], sizeof(real_t) * tmpPdfs_.size(), cudaMemcpyHostToDevice ));

   WALBERLA_CUDA_CHECK( cudaMalloc( &pullIdxsGPU_, sizeof(uint32_t) * pullIdxs_.size() ));
   WALBERLA_CUDA_CHECK( cudaMemcpy( pullIdxsGPU_, &pullIdxs_[0], sizeof(uint32_t) * pullIdxs_.size(), cudaMemcpyHostToDevice ));

   WALBERLA_CUDA_CHECK( cudaMalloc( &pullIdxsInnerGPU_, sizeof(uint32_t) * pullIdxsInner_.size() ));
   WALBERLA_CUDA_CHECK( cudaMemcpy( pullIdxsInnerGPU_, &pullIdxsInner_[0], sizeof(uint32_t) * pullIdxsInner_.size(), cudaMemcpyHostToDevice ));

   WALBERLA_CUDA_CHECK( cudaMalloc( &pullIdxsOuterGPU_, sizeof(uint32_t) * pullIdxsOuter_.size() ));
   WALBERLA_CUDA_CHECK( cudaMemcpy( pullIdxsOuterGPU_, &pullIdxsOuter_[0], sizeof(uint32_t) * pullIdxsOuter_.size(), cudaMemcpyHostToDevice ));
}

void {{class_name}}::copyPDFSToCPU()
{
   WALBERLA_CUDA_CHECK( cudaMemcpy( &pdfs_[0], pdfsGPU_, sizeof(real_t) * pdfs_.size(), cudaMemcpyDeviceToHost ));
}

void {{class_name}}::clearGPUArrays()
{
   WALBERLA_CUDA_CHECK( cudaFree( pdfsGPU_ ));
   WALBERLA_CUDA_CHECK( cudaFree( tmpPdfsGPU_ ));
   WALBERLA_CUDA_CHECK( cudaFree( pullIdxsGPU_ ));
   WALBERLA_CUDA_CHECK( cudaFree( pullIdxsInnerGPU_ ));
   WALBERLA_CUDA_CHECK( cudaFree( pullIdxsOuterGPU_ ));
}
{%- else %}
void {{class_name}}::syncGPU()
{
}

void {{class_name}}::copyPDFSToCPU()
{
}

void {{class_name}}::clearGPUArrays()
{

}
{%- endif %}




real_t {{class_name}}::getOmega( const uint_t idx ) const
{
  return omegas_[ idx ];
}

void {{class_name}}::setOmega( const uint_t idx, const real_t omega )
{
    omegas_[ idx ] = omega;
}






Vector3<real_t> {{class_name}}::getVelocity( uint_t idx ) const
{
   Vector3<real_t> velocity = Vector3<real_t>(0.0);
   auto rho = getDensity(idx);

   for (size_t f = 1; f < {{Q}}; ++f)
   {
     auto pdf = get( idx, f );
     velocity[0] += numeric_cast<real_t>( cx[f] ) * pdf;
     velocity[1] += numeric_cast<real_t>( cy[f] ) * pdf;
     velocity[2] += numeric_cast<real_t>( cz[f] ) * pdf;
   }
   velocity[0]/=rho;
   velocity[1]/=rho;
   velocity[2]/=rho;

   return velocity;
}

Vector3<real_t> {{class_name}}::getVelocityOdd( uint_t idx ) const
{
   Vector3<real_t> velocity = Vector3<real_t>(0.0);
   auto rho = getDensity(idx);

   for (size_t f = 1; f < {{Q}}; ++f)
   {
     auto pdf = pdfs_[pullIdxs_[getPullIdx(idx, f)]];
     velocity[0] += numeric_cast<real_t>( cx[f] ) * pdf;
     velocity[1] += numeric_cast<real_t>( cy[f] ) * pdf;
     velocity[2] += numeric_cast<real_t>( cz[f] ) * pdf;
   }
   velocity[0]/=rho;
   velocity[1]/=rho;
   velocity[2]/=rho;

   return velocity;
}


real_t {{class_name}}::getDensity( uint_t idx ) const
{
   real_t rho = real_t( 0 );
   for( uint_t f = 0; f < {{Q}}; ++f )
   {
      rho += get( idx, f );
   }
   rho += real_t( 1 );

   return rho;
}

{{index_type}} {{class_name}}::getPullIdx( const uint_t idx, const uint_t d ) const
{
   WALBERLA_ASSERT_LESS( d * numFluidCellsPadded_ + idx, pullIdxs_.size() )
   return numeric_cast< {{index_type}} >(d * numFluidCellsPadded_ + idx);
}

uint32_t ListLBMList::getPullIdxInner( const uint_t idx, const uint_t d ) const
{
   WALBERLA_ASSERT_LESS( d * numFluidCellsInner_ + idx, pullIdxsInner_.size() )
   return numeric_cast< uint32_t >(d * numFluidCellsInner_ + idx);
}

uint32_t ListLBMList::getPullIdxOuter( const uint_t idx, const uint_t d ) const
{
   WALBERLA_ASSERT_LESS( d * numFluidCellsOuter_ + idx, pullIdxsOuter_.size() )
   return numeric_cast< uint32_t >(d * numFluidCellsOuter_ + idx);
}

{{index_type}} {{class_name}}::getPDFIdx(const uint_t idx, const uint_t f) const
{
   // TODO: Only SoA
   return numeric_cast< {{index_type}} >(f * numFluidCellsPadded_ + idx);
}

void {{class_name}}::setPullIdx( const uint_t idx, const uint_t d, const {{index_type}} pdfIdx )
{
   // TODO: think about assert
   WALBERLA_ASSERT_LESS( getPullIdx( idx, d ), pullIdxs_.size() )
   // WALBERLA_LOG_INFO_ON_ROOT("getPullIdxIdx( idx, d )  "  << getPullIdxIdx( idx, d ))
   pullIdxs_[ getPullIdx( idx, d ) ] = pdfIdx;
}

void ListLBMList::setPullIdxInner( const uint_t idx, const uint_t d, const uint32_t pdfIdx )
{
   // TODO: think about assert
   WALBERLA_ASSERT_LESS( getPullIdxInner( idx, d ), pullIdxsInner_.size() )
   // WALBERLA_LOG_INFO_ON_ROOT("getPullIdxIdx( idx, d )  "  << getPullIdxIdx( idx, d ))
   pullIdxsInner_[ getPullIdxInner( idx, d ) ] = pdfIdx;
}

void ListLBMList::setPullIdxOuter( const uint_t idx, const uint_t d, const uint32_t pdfIdx )
{
   // TODO: think about assert
   WALBERLA_ASSERT_LESS( getPullIdxOuter( idx, d ), pullIdxsOuter_.size() )
   // WALBERLA_LOG_INFO_ON_ROOT("getPullIdxIdx( idx, d )  "  << getPullIdxIdx( idx, d ))
   pullIdxsOuter_[ getPullIdxOuter( idx, d ) ] = pdfIdx;
}



bool {{class_name}}::operator==( const {{class_name}} & other ) const
{
   return this->pdfs_      == other.pdfs_
          && this->idxToCell_ == other.idxToCell_;
}

{{index_type}} {{class_name}}::registerExternalPDFs( const std::vector< lbm::CellDir > & externalPdfs )
{
   WALBERLA_CHECK_LESS( pdfs_.size() + externalPdfs.size(), numeric_cast<size_t>( std::numeric_limits<{{index_type}}>::max() ),
                       "The Number of PDFs you want to register as external PDFs increases the total number of stored "
                       "PDFs beyond the capacity of {{index_type}}!" )

   const {{index_type}} startIdx = numeric_cast<{{index_type}}>( pdfs_.size() );

   pdfs_.resize( pdfs_.size() + externalPdfs.size(), 0.0 );

   if( !manuallyAllocateTmpPDFs_ )
      tmpPdfs_.resize( pdfs_.size(), 0.0 );

   {{index_type}} idx = startIdx;

   for( auto it = externalPdfs.begin(); it != externalPdfs.end(); ++it )
   {
      Cell fluidNeighborCell = it->cell + Cell(cx[it->dir], cy[it->dir], cz[it->dir]);
      if( isFluidCell( fluidNeighborCell ) )
      {
         setPullIdx( fluidNeighborCell, it->dir, idx );
         if(ci_.contains(fluidNeighborCell))
         {
            setPullIdxInner( fluidNeighborCell, it->dir, idx );
         }
         else
         {
            setPullIdxOuter( fluidNeighborCell, it->dir, idx );
         }
      }
      idx++;
   }

   WALBERLA_ASSERT_EQUAL( idx, pdfs_.size() )

   // syncGPU();
   return startIdx;
}


{{index_type}} {{class_name}}::registerExternalPDFsDense( const std::vector< lbm::CellDir > & externalPdfs, const std::vector< Cell > & boundaryCells)
{
   WALBERLA_CHECK_LESS( pdfs_.size() + externalPdfs.size(), numeric_cast<size_t>( std::numeric_limits<{{index_type}}>::max() ),
                       "The Number of PDFs you want to register as external PDFs increases the total number of stored "
                       "PDFs beyond the capacity of {{index_type}}!" )

   const {{index_type}} startIdx = numeric_cast<{{index_type}}>( pdfs_.size() );

   pdfs_.resize( pdfs_.size() + externalPdfs.size(), 0.0 );

   if( !manuallyAllocateTmpPDFs_ )
      tmpPdfs_.resize( pdfs_.size(), 0.0 );

   {{index_type}} idx = startIdx;

   for( auto it = externalPdfs.begin(); it != externalPdfs.end(); ++it )
   {
      bool isBoundaryCell = false;
      if ( std::binary_search( boundaryCells.begin(), boundaryCells.end(), it->cell ) ) {
         //WALBERLA_LOG_INFO("Cell " << it->cell  << " is boundary cell")
         isBoundaryCell = true;
      }

      Cell fluidNeighborCell = it->cell + Cell(cx[it->dir], cy[it->dir], cz[it->dir]);
      if( isFluidCell( fluidNeighborCell ) && !isBoundaryCell)
      {
         setPullIdx( fluidNeighborCell, it->dir, idx );
         if(ci_.contains(fluidNeighborCell))
         {
            setPullIdxInner( fluidNeighborCell, it->dir, idx );
         }
         else
         {
            setPullIdxOuter( fluidNeighborCell, it->dir, idx );
         }
      }
      idx++;
   }

   WALBERLA_ASSERT_EQUAL( idx, pdfs_.size() )

   // syncGPU();
   return startIdx;
}


void {{class_name}}::toBuffer( mpi::SendBuffer & buffer ) const
{
   buffer << numFluidCellsPadded_ << pdfs_ << pullIdxs_ << idxToCell_ << manuallyAllocateTmpPDFs_;
}

void {{class_name}}::fromBuffer( mpi::RecvBuffer & buffer )
{
   buffer >> numFluidCellsPadded_ >> pdfs_ >> pullIdxs_ >> idxToCell_ >> manuallyAllocateTmpPDFs_;

   tmpPdfs_.resize( pdfs_.size(), 0.0 );

   for( uint_t i = 0; i < idxToCell_.size(); ++i )
      cellToIdx_[idxToCell_[i]] = i;
}

} // namespace lbmpy
} // namespace walberla


