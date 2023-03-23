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
//! \file CellBuffer.h
//! \ingroup lbm
//! \author Michael Hussnaetter <michael.hussnaetter@fau.de>
//
//======================================================================================================================
// TODO delete, old and works only with Intel and AMD. Only here for reference
#pragma once

#include "core/Macros.h"

#include "core/DataTypes.h"
#include "stencil/D3Q19.h"
#include "stencil/Directions.h"

#include "simd/SIMD.h"
#include "simd/AlignedAllocator.h"
#include <vector>
#include <iostream>


namespace walberla {
namespace lbm {

using namespace stencil;
using namespace simd;
template< uint_t CAPACITY, uint_t ALIGNMENT >
class CellBuffer
{
public:
   CellBuffer();

   inline uint_t                 getFillLevel()                                              { return    fillLevel_; }
   inline uint_t                 getCapacity()                                               { return    CAPACITY; }
   inline double *               getDirBegin(uint_t dir)                                     { return  &(storage_[CAPACITY *  dir]); }
   inline double *               getVelBegin(uint_t dir)                                     { return  &(storage_[CAPACITY *  (stencil::D3Q19::Size + dir)]); }
   inline double *               getCommonFeqBegin()                                         { return  &(storage_[CAPACITY *  (stencil::D3Q19::Size + stencil::D3Q19::Dimension)]); }
   inline double &               getvelocity(uint_t dimension, uint_t offset)                { return    storage_[CAPACITY * (stencil::D3Q19::Size + dimension) + offset]; }
   inline double *               getvelocityptr(uint_t dimension, uint_t offset)             { return  &(storage_[CAPACITY * (stencil::D3Q19::Size + dimension) + offset]); }
   inline double &               getfeqcommon(uint_t offset)                                 { return    storage_[CAPACITY * (stencil::D3Q19::Size + stencil::D3Q19::Dimension) + offset]; }
   inline double *               getfeqcommonptr(uint_t offset)                              { return  &(storage_[CAPACITY * (stencil::D3Q19::Size + stencil::D3Q19::Dimension) + offset]); }

   inline const double4_t        loadPdfs(uint_t dir, uint_t offset)                         { return    load_aligned(&(storage_[CAPACITY *  dir + offset])); }
   inline const double4_t        loadVelocities(uint_t dimension, uint_t offset)             { return    load_aligned(&(storage_[CAPACITY * (stencil::D3Q19::Size + dimension) + offset])); }
   inline const double4_t        loadFeqsCommon(uint_t offset)                               { return    load_aligned(&(storage_[CAPACITY * (stencil::D3Q19::Size + stencil::D3Q19::Dimension) + offset])); }

   inline void                   setFillLevel(uint_t newLevel)                               { fillLevel_ = newLevel; }
   inline void                   setMacroPos(uint_t newPos)                                  { macroPos_ = newPos; }

   inline void                   reset()                                                     { fillLevel_ = 0; macroPos_ = 0; }

   WALBERLA_FORCE_INLINE( void   calcMacroscopicValues( const double * WALBERLA_RESTRICT C_begin, double * WALBERLA_RESTRICT dC, double4_t lambda_e ) );
   WALBERLA_FORCE_INLINE( void   addPdf( const double * WALBERLA_RESTRICT begin, uint_t count, uint_t dir ) ) { memcpy( (void*) ( getDirBegin( dir ) + fillLevel_ ), begin, count * sizeof( double ) ); }


private:
   uint_t fillLevel_;
   uint_t macroPos_;
   std::vector< double, simd::aligned_allocator<double, ALIGNMENT> > storage_;
};

template< uint_t CAPACITY, uint_t ALIGNMENT >
CellBuffer<CAPACITY, ALIGNMENT>::CellBuffer() : fillLevel_( 0 ), macroPos_( 0 )
{
   storage_.assign(CAPACITY * (stencil::D3Q19::Size + stencil::D3Q19::Dimension + 1), std::numeric_limits<double>::signaling_NaN());
}

template< uint_t CAPACITY, uint_t ALIGNMENT >
void CellBuffer<CAPACITY, ALIGNMENT>::calcMacroscopicValues(const double * WALBERLA_RESTRICT C_begin, double * WALBERLA_RESTRICT dC, const double4_t lambda_e)
{

   const double * WALBERLA_RESTRICT NE_begin = &(storage_[CAPACITY * NE]);
   const double * WALBERLA_RESTRICT  N_begin = &(storage_[CAPACITY *  N]);
   const double * WALBERLA_RESTRICT NW_begin = &(storage_[CAPACITY * NW]);
   const double * WALBERLA_RESTRICT  W_begin = &(storage_[CAPACITY *  W]);
   const double * WALBERLA_RESTRICT SW_begin = &(storage_[CAPACITY * SW]);
   const double * WALBERLA_RESTRICT  S_begin = &(storage_[CAPACITY *  S]);
   const double * WALBERLA_RESTRICT SE_begin = &(storage_[CAPACITY * SE]);
   const double * WALBERLA_RESTRICT  E_begin = &(storage_[CAPACITY *  E]);
   const double * WALBERLA_RESTRICT  T_begin = &(storage_[CAPACITY *  T]);
   const double * WALBERLA_RESTRICT TE_begin = &(storage_[CAPACITY * TE]);
   const double * WALBERLA_RESTRICT TN_begin = &(storage_[CAPACITY * TN]);
   const double * WALBERLA_RESTRICT TW_begin = &(storage_[CAPACITY * TW]);
   const double * WALBERLA_RESTRICT TS_begin = &(storage_[CAPACITY * TS]);
   const double * WALBERLA_RESTRICT  B_begin = &(storage_[CAPACITY *  B]);
   const double * WALBERLA_RESTRICT BE_begin = &(storage_[CAPACITY * BE]);
   const double * WALBERLA_RESTRICT BN_begin = &(storage_[CAPACITY * BN]);
   const double * WALBERLA_RESTRICT BW_begin = &(storage_[CAPACITY * BW]);
   const double * WALBERLA_RESTRICT BS_begin = &(storage_[CAPACITY * BS]);

   double * WALBERLA_RESTRICT velX_begin = getVelBegin(0);
   double * WALBERLA_RESTRICT velY_begin = getVelBegin(1);
   double * WALBERLA_RESTRICT velZ_begin = getVelBegin(2);

   double * WALBERLA_RESTRICT commonFeq_begin = getCommonFeqBegin();

   const double4_t ONE_PT_FIVE = make_double4(1.5);
   const double4_t ONE = make_double4(1);
   const double4_t t0 = make_double4(1.0 / 3.0);
   
   for (; macroPos_ < fillLevel_ - (fillLevel_ % 4); macroPos_ += 4){

      double4_t srcNE = load_aligned(NE_begin + macroPos_);
      double4_t srcN = load_aligned(N_begin + macroPos_);
      double4_t srcNW = load_aligned(NW_begin + macroPos_);
      double4_t srcW = load_aligned(W_begin + macroPos_);
      double4_t srcSW = load_aligned(SW_begin + macroPos_);
      double4_t srcS = load_aligned(S_begin + macroPos_);
      double4_t srcSE = load_aligned(SE_begin + macroPos_);
      double4_t srcE = load_aligned(E_begin + macroPos_);
      double4_t srcT = load_aligned(T_begin + macroPos_);
      double4_t srcTE = load_aligned(TE_begin + macroPos_);
      double4_t srcTN = load_aligned(TN_begin + macroPos_);
      double4_t srcTW = load_aligned(TW_begin + macroPos_);
      double4_t srcTS = load_aligned(TS_begin + macroPos_);
      double4_t srcB = load_aligned(B_begin + macroPos_);
      double4_t srcBE = load_aligned(BE_begin + macroPos_);
      double4_t srcBN = load_aligned(BN_begin + macroPos_);
      double4_t srcBW = load_aligned(BW_begin + macroPos_);
      double4_t srcBS = load_aligned(BS_begin + macroPos_);

      const double4_t velX_trm = srcE + srcNE + srcSE + srcTE + srcBE;
      const double4_t velY_trm = srcN + srcNW + srcTN + srcBN;
      const double4_t velZ_trm = srcT + srcTS + srcTW;

      const double4_t srcC = load_aligned(C_begin + macroPos_);

      const double4_t rho = srcC + srcS + srcW + srcB + srcSW + srcBS + srcBW + velX_trm + velY_trm + velZ_trm;

      const double4_t velX_256d = velX_trm - srcW - srcNW - srcSW - srcTW - srcBW;
      const double4_t velY_256d = velY_trm + srcNE - srcS - srcSW - srcSE - srcTS - srcBS;
      const double4_t velZ_256d = velZ_trm + srcTN + srcTE - srcB - srcBN - srcBS - srcBW - srcBE;

      const double4_t feq_common_256d = rho - ONE_PT_FIVE * (velX_256d * velX_256d + velY_256d * velY_256d + velZ_256d * velZ_256d);

      store_aligned(velX_begin + macroPos_, velX_256d);
      store_aligned(velY_begin + macroPos_, velY_256d);
      store_aligned(velZ_begin + macroPos_, velZ_256d);

      store_aligned(commonFeq_begin + macroPos_, feq_common_256d);

      const double4_t dstC = srcC * (ONE - lambda_e) + lambda_e * t0 * feq_common_256d;
      store_non_temp(dC + macroPos_, dstC);

   }

}

} //namespace lbm
} //namespace walberla