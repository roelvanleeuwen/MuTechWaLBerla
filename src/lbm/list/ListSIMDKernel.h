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
//! \file ListSIMDKernel.h
//! \ingroup lbm
//! \author Michael Huﬂn‰tter <michael.hussnaetter@fau.de>
//
//======================================================================================================================
// TODO delete, old and works only with Intel and AMD. Only here for reference
#pragma once

#include "CellBuffer.h"

#include "core/Macros.h"
#include "core/math/DistributedSample.h"

#include "domain_decomposition/IBlock.h"
#include "domain_decomposition/BlockDataID.h"

#include "lbm/IntelCompilerOptimization.h"
#include "lbm/list/List.h"
#include "lbm/lattice_model/D3Q19.h"
#include "lbm/lattice_model/CollisionModel.h"

#include "simd/SIMD.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif


namespace walberla {
namespace lbm {

#define HANDLE_DIR( DirA, DirB, VelTerm, fac, t ) \
for( uint_t x = 0; x < fillLevel; x += 4 ) \
										{ \
	uint_t xx = rleBufferPos + x; \
	const double4_t vel = VelTerm; \
	const double4_t srcA = cb.loadPdfs( DirA , x ); \
	const double4_t srcB = cb.loadPdfs( DirB , x ); \
	const double4_t feqCommonQuad = cb.loadFeqsCommon( x ); \
	const double4_t  sym = lambda_e_scaled * ( srcA + srcB - fac * vel * vel - t * feqCommonQuad ); \
	const double4_t asym = lambda_d_scaled * ( srcA - srcB - THREE * t * vel ); \
	const double4_t dstA = srcA - sym - asym; \
	const double4_t dstB = srcB - sym + asym; \
	store_non_temp( d ## DirA + xx, dstA ); \
	store_non_temp( d ## DirB + xx, dstB ); \
										}

template< typename List_T, uint_t BUFFER_SIZE >
class ListSIMDSplitTRTSweep
{
   static_assert((boost::is_same< typename List_T::LatticeModel::CollisionModel::tag, collision_model::TRT_tag >::value), "Only works with TRT!");
   static_assert((boost::is_same< typename List_T::LatticeModel::Stencil, stencil::D3Q19 >::value), "Only works with D3Q19!");
   static_assert(!List_T::LatticeModel::compressible, "Only works with incompressible models!");
   static_assert((boost::is_same< typename List_T::LatticeModel::ForceModel::tag, force_model::None_tag >::value), "Only works without additional forces!");
   static_assert(List_T::LatticeModel::equilibriumAccuracyOrder == 2, "Only works for lattice models that require the equilibrium distribution to be order 2 accurate!");

public:
   ListSIMDSplitTRTSweep(const BlockDataID listId) : listId_(listId) { }
   void operator()(IBlock * const block);

protected:
   BlockDataID listId_;
   // buffer for storing all gathered pdfs and resulting macroscopic values of multiple cells
   CellBuffer<BUFFER_SIZE, 64> cb;
};

template< typename List_T, uint_t BUFFER_SIZE >
void ListSIMDSplitTRTSweep< List_T, BUFFER_SIZE >::operator()(IBlock * const block)
{
#ifdef LIKWID_PERFMON		
   LIKWID_MARKER_START("SIMD4Kernel");
#endif
   List_T * list = block->getData<List_T>(listId_);
   WALBERLA_ASSERT_NOT_NULLPTR(list);
   
   // constants
   static const double4_t PT_FIVE = make_double4(0.5);
   static const double4_t THREE = make_double4(3.0);

   const double4_t lambda_e = make_double4(list->latticeModel().collisionModel().lambda_e());
   const double4_t lambda_d = make_double4(list->latticeModel().collisionModel().lambda_d());

   // common prefactors for calculating the equilibrium parts
   // const double4_t t0 = make_double4(1.0 / 3.0);  // 1/3      for C
   const double4_t t1x2 = make_double4(1.0 / 18.0 * 2.0);  // 1/18 * 2 for N, S, W, E, T, B
   const double4_t t2x2 = make_double4(1.0 / 36.0 * 2.0);  // 1/36 * 2 else

   const double4_t inv2csq2 = make_double4(1.0 / (2.0 * (1.0 / 3.0) * (1.0 / 3.0))); //speed of sound related factor for equilibrium distribution function
   const double4_t fac1 = t1x2 * inv2csq2;
   const double4_t fac2 = t2x2 * inv2csq2;

   // relaxation parameter variables
   const double4_t lambda_e_scaled = PT_FIVE * lambda_e; // 0.5 times the usual value ...
   const double4_t lambda_d_scaled = PT_FIVE * lambda_d; // ... due to the way of calculations
   
   double * WALBERLA_RESTRICT dNE = &(list->getTmp(0, NE));
   double * WALBERLA_RESTRICT dN = &(list->getTmp(0, N));
   double * WALBERLA_RESTRICT dNW = &(list->getTmp(0, NW));
   double * WALBERLA_RESTRICT dW = &(list->getTmp(0, W));
   double * WALBERLA_RESTRICT dSW = &(list->getTmp(0, SW));
   double * WALBERLA_RESTRICT dS = &(list->getTmp(0, S));
   double * WALBERLA_RESTRICT dSE = &(list->getTmp(0, SE));
   double * WALBERLA_RESTRICT dE = &(list->getTmp(0, E));
   double * WALBERLA_RESTRICT dT = &(list->getTmp(0, T));
   double * WALBERLA_RESTRICT dTE = &(list->getTmp(0, TE));
   double * WALBERLA_RESTRICT dTN = &(list->getTmp(0, TN));
   double * WALBERLA_RESTRICT dTW = &(list->getTmp(0, TW));
   double * WALBERLA_RESTRICT dTS = &(list->getTmp(0, TS));
   double * WALBERLA_RESTRICT dB = &(list->getTmp(0, B));
   double * WALBERLA_RESTRICT dBE = &(list->getTmp(0, BE));
   double * WALBERLA_RESTRICT dBN = &(list->getTmp(0, BN));
   double * WALBERLA_RESTRICT dBW = &(list->getTmp(0, BW));
   double * WALBERLA_RESTRICT dBS = &(list->getTmp(0, BS));
   double * WALBERLA_RESTRICT dC = &(list->getTmp(0, C));

   double * WALBERLA_RESTRICT pNE = nullptr;
   double * WALBERLA_RESTRICT pN = nullptr;
   double * WALBERLA_RESTRICT pNW = nullptr;
   double * WALBERLA_RESTRICT pW = nullptr;
   double * WALBERLA_RESTRICT pSW = nullptr;
   double * WALBERLA_RESTRICT pS = nullptr;
   double * WALBERLA_RESTRICT pSE = nullptr;
   double * WALBERLA_RESTRICT pE = nullptr;
   double * WALBERLA_RESTRICT pT = nullptr;
   double * WALBERLA_RESTRICT pTE = nullptr;
   double * WALBERLA_RESTRICT pTN = nullptr;
   double * WALBERLA_RESTRICT pTW = nullptr;
   double * WALBERLA_RESTRICT pTS = nullptr;
   double * WALBERLA_RESTRICT pB = nullptr;
   double * WALBERLA_RESTRICT pBE = nullptr;
   double * WALBERLA_RESTRICT pBN = nullptr;
   double * WALBERLA_RESTRICT pBW = nullptr;
   double * WALBERLA_RESTRICT pBS = nullptr;
   double * WALBERLA_RESTRICT pC = &(list->get(0, C));

   auto rleTerminate = list->getRLEInfo().end();
   auto rleNext = list->getRLEInfo().begin();
   uint_t rleCurrentPos = *rleNext;
   ++rleNext;

   uint_t blockPos = 0;

   while (rleNext != rleTerminate)
   {
      using namespace stencil;

      // fill buffer
      cb.reset();
      uint_t rleBufferPos = rleCurrentPos;

      while (rleNext != rleTerminate)
      {

         if (blockPos == 0)
         {
            
            pNE = &( list->get( list->getPullIdx( rleCurrentPos, NE ) ) );				
            pN  = &( list->get( list->getPullIdx( rleCurrentPos, N  ) ) );				
            pNW = &( list->get( list->getPullIdx( rleCurrentPos, NW ) ) );				
            pW  = &( list->get( list->getPullIdx( rleCurrentPos, W  ) ) );				
            pSW = &( list->get( list->getPullIdx( rleCurrentPos, SW ) ) );				
            pS  = &( list->get( list->getPullIdx( rleCurrentPos, S  ) ) );				
            pSE = &( list->get( list->getPullIdx( rleCurrentPos, SE ) ) );				
            pE  = &( list->get( list->getPullIdx( rleCurrentPos, E  ) ) );				
            pT  = &( list->get( list->getPullIdx( rleCurrentPos, T  ) ) );				
            pTE = &( list->get( list->getPullIdx( rleCurrentPos, TE ) ) );				
            pTN = &( list->get( list->getPullIdx( rleCurrentPos, TN ) ) );				
            pTW = &( list->get( list->getPullIdx( rleCurrentPos, TW ) ) );				
            pTS = &( list->get( list->getPullIdx( rleCurrentPos, TS ) ) );				
            pB  = &( list->get( list->getPullIdx( rleCurrentPos, B  ) ) );				
            pBE = &( list->get( list->getPullIdx( rleCurrentPos, BE ) ) );				
            pBN = &( list->get( list->getPullIdx( rleCurrentPos, BN ) ) );				
            pBW = &( list->get( list->getPullIdx( rleCurrentPos, BW ) ) );				
            pBS = &( list->get( list->getPullIdx( rleCurrentPos, BS ) ) );
         }

         const uint_t cellsLeftInBlock = *rleNext - rleCurrentPos;
         const uint_t cellsToWrite = std::min(BUFFER_SIZE - cb.getFillLevel(), cellsLeftInBlock);
         
         WALBERLA_ASSERT(cellsToWrite < BUFFER_SIZE);
         if (cellsToWrite > 0){
            cb.addPdf( pNE + blockPos, cellsToWrite, NE );
            cb.addPdf( pN  + blockPos, cellsToWrite, N  );
            cb.addPdf( pNW + blockPos, cellsToWrite, NW );
            cb.addPdf( pW  + blockPos, cellsToWrite, W  );
            cb.addPdf( pSW + blockPos, cellsToWrite, SW );
            cb.addPdf( pS  + blockPos, cellsToWrite, S  );				
            cb.addPdf( pSE + blockPos, cellsToWrite, SE );				
            cb.addPdf( pE  + blockPos, cellsToWrite, E  );				
            cb.addPdf( pT  + blockPos, cellsToWrite, T  );				
            cb.addPdf( pTE + blockPos, cellsToWrite, TE );				
            cb.addPdf( pTN + blockPos, cellsToWrite, TN );				
            cb.addPdf( pTW + blockPos, cellsToWrite, TW );				
            cb.addPdf( pTS + blockPos, cellsToWrite, TS );				
            cb.addPdf( pB  + blockPos, cellsToWrite, B  );				
            cb.addPdf( pBE + blockPos, cellsToWrite, BE );				
            cb.addPdf( pBN + blockPos, cellsToWrite, BN );				
            cb.addPdf( pBW + blockPos, cellsToWrite, BW );				
            cb.addPdf( pBS + blockPos, cellsToWrite, BS );

            rleCurrentPos += cellsToWrite;
            blockPos += cellsToWrite;

            // process buffer
            
            cb.setFillLevel(cb.getFillLevel() + cellsToWrite);
            
            cb.calcMacroscopicValues(pC + rleBufferPos, dC + rleBufferPos, lambda_e);

            if (cellsToWrite == cellsLeftInBlock){
               blockPos = 0;
               ++rleNext;
            }
         }
         else
         {
            break;
         }
      }

      // collide and store
      uint_t fillLevel = cb.getFillLevel();

      HANDLE_DIR(NE, SW, cb.loadVelocities(0, x) + cb.loadVelocities(1, x), fac2, t2x2);
      HANDLE_DIR(SE, NW, cb.loadVelocities(0, x) - cb.loadVelocities(1, x), fac2, t2x2);
      HANDLE_DIR(TE, BW, cb.loadVelocities(0, x) + cb.loadVelocities(2, x), fac2, t2x2);
      HANDLE_DIR(BE, TW, cb.loadVelocities(0, x) - cb.loadVelocities(2, x), fac2, t2x2);
      HANDLE_DIR(TN, BS, cb.loadVelocities(1, x) + cb.loadVelocities(2, x), fac2, t2x2);
      HANDLE_DIR(BN, TS, cb.loadVelocities(1, x) - cb.loadVelocities(2, x), fac2, t2x2);
      HANDLE_DIR(E, W, cb.loadVelocities(0, x), fac1, t1x2);
      HANDLE_DIR(N, S, cb.loadVelocities(1, x), fac1, t1x2);
      HANDLE_DIR(T, B, cb.loadVelocities(2, x), fac1, t1x2);

   }

   list->swapTmpPdfs();

#ifdef LIKWID_PERFMON
   LIKWID_MARKER_STOP("SIMD4Kernel");
#endif
}
#undef HANDLE_DIR



class LoadBalancingSweepTimer
{
public:

   void clear()
   {
      for( auto it = timer_.begin(); it != timer_.end(); ++it )
         it->second.reset();
   }

   double getBlockTiming( const blockforest::BlockID & blockId ) const
   {
      auto it = timer_.find( blockId );
      WALBERLA_CHECK_UNEQUAL( it, timer_.end() );
      return it->second.total();
   }

   double getProcessTiming() const
   {
      double time = 0.0;
      for( auto it = timer_.begin(); it != timer_.end(); ++it )
         time += it->second.total();
      return time;
   }

   void startTimer( const blockforest::BlockID & blockId )
   {
      timer_[blockId].start();
   }

   void stopTimer( const blockforest::BlockID & blockId )
   {
      timer_[blockId].end();
   }

   void startTimer( const IBlockID & blockId )
   {
      startTimer( dynamic_cast<const blockforest::BlockID &>( blockId ) );
   }

   void stopTimer( const IBlockID & blockId )
   {
      stopTimer( dynamic_cast<const blockforest::BlockID &>( blockId ) );
   }

   math::DistributedSample getProcessTimingSample() const
   {
      math::DistributedSample sample;
      sample.insert( getProcessTiming() );
      sample.mpiAllGather();
      return sample;
   }

protected:
   std::map< blockforest::BlockID, WcTimer> timer_;
};



#define HANDLE_DIR( DirA, DirB, VelTerm, fac, t ) \
   for( uint_t i = 0; i < bufferPos; i += 4 ) \
            { \
      uint_t xx = bufferStartX + i; \
      const double4_t vel = VelTerm; \
      const double4_t srcA = load_aligned( buf ## DirA + i ); \
      const double4_t srcB = load_aligned( buf ## DirB + i ); \
      const double4_t feqCommonQuad = load_aligned( feqCommon + i ); \
      const double4_t  sym = lambda_e_scaled * ( srcA + srcB - fac * vel * vel - t * feqCommonQuad ); \
      const double4_t asym = lambda_d_scaled * ( srcA - srcB - THREE * t * vel ); \
      const double4_t dstA = srcA - sym - asym; \
      const double4_t dstB = srcB - sym + asym; \
      store_non_temp( d ## DirA + xx, dstA ); \
      store_non_temp( d ## DirB + xx, dstB ); \
   }

template< typename List_T, uint_t BUFFER_SIZE >
class ListSIMD2SplitTRTSweep
{
   static_assert( ( boost::is_same< typename List_T::LatticeModel::CollisionModel::tag, collision_model::TRT_tag >::value ), "Only works with TRT!" );
   static_assert( ( boost::is_same< typename List_T::LatticeModel::Stencil, stencil::D3Q19 >::value ), "Only works with D3Q19!" );
   static_assert( !List_T::LatticeModel::compressible, "Only works with incompressible models!" );
   static_assert( ( boost::is_same< typename List_T::LatticeModel::ForceModel::tag, force_model::None_tag >::value ), "Only works without additional forces!" );
   static_assert( List_T::LatticeModel::equilibriumAccuracyOrder == 2, "Only works for lattice models that require the equilibrium distribution to be order 2 accurate!" );
   static_assert( BUFFER_SIZE % 4 == 0, "BUFFER_SIZE has to be divisable by 4" );

public:
   ListSIMD2SplitTRTSweep( const BlockDataID listId )
      : listId_( listId ) {}
   void operator()( IBlock * const block );

protected:
   BlockDataID listId_;
};





template< typename List_T, uint_t BUFFER_SIZE >
void ListSIMD2SplitTRTSweep< List_T, BUFFER_SIZE >::operator()( IBlock * const block )
{
#ifdef LIKWID_PERFMON		
   LIKWID_MARKER_START( "Kernel-opt" );
#endif

   List_T * list = block->getData<List_T>( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( list );

   const uint_t numFluidCells = list->numFluidCells();

   if( numFluidCells == 0 )
   {
      return;
   }

   using namespace simd;

   // constants used during stream/collide
   static const double4_t PT_FIVE = make_double4( 0.5 );
   static const double4_t ONE = make_double4( 1.0 );
   static const double4_t ONE_PT_FIVE = make_double4( 1.5 );
   static const double4_t THREE = make_double4( 3.0 );
   
   const double4_t lambda_e = make_double4( list->latticeModel().collisionModel().lambda_e() );
   const double4_t lambda_d = make_double4( list->latticeModel().collisionModel().lambda_d() );

   // common prefactors for calculating the equilibrium parts
   const double4_t t0   = make_double4( 1.0 / 3.0        );  // 1/3      for C
   const double4_t t1x2 = make_double4( 1.0 / 18.0 * 2.0 );  // 1/18 * 2 for N, S, W, E, T, B
   const double4_t t2x2 = make_double4( 1.0 / 36.0 * 2.0 );  // 1/36 * 2 else

   const double4_t inv2csq2 = make_double4( 1.0 / ( 2.0 * ( 1.0 / 3.0 ) * ( 1.0 / 3.0 ) ) ); //speed of sound related factor for equilibrium distribution function
   const double4_t fac1     = t1x2 * inv2csq2;
   const double4_t fac2     = t2x2 * inv2csq2;

   // relaxation parameter variables
   const double4_t lambda_e_scaled = PT_FIVE * lambda_e; // 0.5 times the usual value ...
   const double4_t lambda_d_scaled = PT_FIVE * lambda_d; // ... due to the way of calculations
   
   const uint_t numQuads      = ( numFluidCells + uint_t( 3 ) ) / uint_t( 4 );

#ifdef _OPENMP
#pragma omp parallel
   {
#endif   

#ifdef _OPENMP
      const int threadId = omp_get_thread_num();
      const int numThreads = omp_get_num_threads();
#else
      const int threadId = 0;
      const int numThreads = 1;
#endif

      WALBERLA_ALIGN( 32 ) double bufNE[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufN [BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufNW[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufW [BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufSW[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufS [BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufSE[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufE [BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufT [BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufTE[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufTN[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufTW[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufTS[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufB [BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufBE[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufBN[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufBW[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double bufBS[BUFFER_SIZE];

      WALBERLA_ALIGN( 32 ) double velX[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double velY[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double velZ[BUFFER_SIZE];
      WALBERLA_ALIGN( 32 ) double feqCommon[BUFFER_SIZE];

      const uint_t startQuad = ( numQuads + uint_c( numThreads ) - uint_t( 1 ) ) / uint_c( numThreads ) * uint_c( threadId );
      const uint_t endQuad = ( numQuads + uint_c( numThreads ) - uint_t( 1 ) ) / uint_c( numThreads ) * uint_c( threadId + 1 );
      uint_t x    = std::min( startQuad * uint_t( 4 ), numFluidCells );
      uint_t xEnd = std::min( endQuad   * uint_t( 4 ), numFluidCells );
      auto beginNextRLEBlock = std::upper_bound( list->getRLEInfo().begin(), list->getRLEInfo().end() - 1, x );

      uint_t bufferPos = 0;
      uint_t bufferStartX = x;

      double * WALBERLA_RESTRICT pNE = &( list->get( list->getPullIdx( x, NE ) ) );
      double * WALBERLA_RESTRICT pN  = &( list->get( list->getPullIdx( x, N  ) ) );
      double * WALBERLA_RESTRICT pNW = &( list->get( list->getPullIdx( x, NW ) ) );
      double * WALBERLA_RESTRICT pW  = &( list->get( list->getPullIdx( x, W  ) ) );
      double * WALBERLA_RESTRICT pSW = &( list->get( list->getPullIdx( x, SW ) ) );
      double * WALBERLA_RESTRICT pS  = &( list->get( list->getPullIdx( x, S  ) ) );
      double * WALBERLA_RESTRICT pSE = &( list->get( list->getPullIdx( x, SE ) ) );
      double * WALBERLA_RESTRICT pE  = &( list->get( list->getPullIdx( x, E  ) ) );
      double * WALBERLA_RESTRICT pT  = &( list->get( list->getPullIdx( x, T  ) ) );
      double * WALBERLA_RESTRICT pTE = &( list->get( list->getPullIdx( x, TE ) ) );
      double * WALBERLA_RESTRICT pTN = &( list->get( list->getPullIdx( x, TN ) ) );
      double * WALBERLA_RESTRICT pTW = &( list->get( list->getPullIdx( x, TW ) ) );
      double * WALBERLA_RESTRICT pTS = &( list->get( list->getPullIdx( x, TS ) ) );
      double * WALBERLA_RESTRICT pB  = &( list->get( list->getPullIdx( x, B  ) ) );
      double * WALBERLA_RESTRICT pBE = &( list->get( list->getPullIdx( x, BE ) ) );
      double * WALBERLA_RESTRICT pBN = &( list->get( list->getPullIdx( x, BN ) ) );
      double * WALBERLA_RESTRICT pBW = &( list->get( list->getPullIdx( x, BW ) ) );
      double * WALBERLA_RESTRICT pBS = &( list->get( list->getPullIdx( x, BS ) ) );
      double * WALBERLA_RESTRICT pC  = &( list->get( 0, C ) );

      uint_t rlePos = 0;
      
      double * WALBERLA_RESTRICT dNE = &( list->getTmp( 0, NE ) );
      double * WALBERLA_RESTRICT dN  = &( list->getTmp( 0, N  ) );
      double * WALBERLA_RESTRICT dNW = &( list->getTmp( 0, NW ) );
      double * WALBERLA_RESTRICT dW  = &( list->getTmp( 0, W  ) );
      double * WALBERLA_RESTRICT dSW = &( list->getTmp( 0, SW ) );
      double * WALBERLA_RESTRICT dS  = &( list->getTmp( 0, S  ) );
      double * WALBERLA_RESTRICT dSE = &( list->getTmp( 0, SE ) );
      double * WALBERLA_RESTRICT dE  = &( list->getTmp( 0, E  ) );
      double * WALBERLA_RESTRICT dT  = &( list->getTmp( 0, T  ) );
      double * WALBERLA_RESTRICT dTE = &( list->getTmp( 0, TE ) );
      double * WALBERLA_RESTRICT dTN = &( list->getTmp( 0, TN ) );
      double * WALBERLA_RESTRICT dTW = &( list->getTmp( 0, TW ) );
      double * WALBERLA_RESTRICT dTS = &( list->getTmp( 0, TS ) );
      double * WALBERLA_RESTRICT dB  = &( list->getTmp( 0, B  ) );
      double * WALBERLA_RESTRICT dBE = &( list->getTmp( 0, BE ) );
      double * WALBERLA_RESTRICT dBN = &( list->getTmp( 0, BN ) );
      double * WALBERLA_RESTRICT dBW = &( list->getTmp( 0, BW ) );
      double * WALBERLA_RESTRICT dBS = &( list->getTmp( 0, BS ) );
      double * WALBERLA_RESTRICT dC  = &( list->getTmp( 0, C  ) );

      while( x < xEnd )
      {
         while( bufferPos < BUFFER_SIZE && x < xEnd )
         {
            WALBERLA_ASSERT( bufferPos % 4 == 0 );

            if( x == *beginNextRLEBlock )
            {
               pNE = &( list->get( list->getPullIdx( x, NE ) ) );
               pN  = &( list->get( list->getPullIdx( x, N  ) ) );
               pNW = &( list->get( list->getPullIdx( x, NW ) ) );
               pW  = &( list->get( list->getPullIdx( x, W  ) ) );
               pSW = &( list->get( list->getPullIdx( x, SW ) ) );
               pS  = &( list->get( list->getPullIdx( x, S  ) ) );
               pSE = &( list->get( list->getPullIdx( x, SE ) ) );
               pE  = &( list->get( list->getPullIdx( x, E  ) ) );
               pT  = &( list->get( list->getPullIdx( x, T  ) ) );
               pTE = &( list->get( list->getPullIdx( x, TE ) ) );
               pTN = &( list->get( list->getPullIdx( x, TN ) ) );
               pTW = &( list->get( list->getPullIdx( x, TW ) ) );
               pTS = &( list->get( list->getPullIdx( x, TS ) ) );
               pB  = &( list->get( list->getPullIdx( x, B  ) ) );
               pBE = &( list->get( list->getPullIdx( x, BE ) ) );
               pBN = &( list->get( list->getPullIdx( x, BN ) ) );
               pBW = &( list->get( list->getPullIdx( x, BW ) ) );
               pBS = &( list->get( list->getPullIdx( x, BS ) ) );
               ++beginNextRLEBlock;

               rlePos = 0;
            }

            const uint_t blockLength = std::min( *beginNextRLEBlock - x, BUFFER_SIZE - bufferPos );
            const uint_t iterations = blockLength / 4;
            
            const uint_t copyN = iterations * 4 * sizeof(double);

            memcpy( bufNE + bufferPos, pNE + rlePos, copyN );
            memcpy( bufN  + bufferPos, pN  + rlePos, copyN );
            memcpy( bufNW + bufferPos, pNW + rlePos, copyN );
            memcpy( bufW  + bufferPos, pW  + rlePos, copyN );
            memcpy( bufSW + bufferPos, pSW + rlePos, copyN );
            memcpy( bufS  + bufferPos, pS  + rlePos, copyN );
            memcpy( bufSE + bufferPos, pSE + rlePos, copyN );
            memcpy( bufE  + bufferPos, pE  + rlePos, copyN );
            memcpy( bufT  + bufferPos, pT  + rlePos, copyN );
            memcpy( bufTE + bufferPos, pTE + rlePos, copyN );
            memcpy( bufTN + bufferPos, pTN + rlePos, copyN );
            memcpy( bufTW + bufferPos, pTW + rlePos, copyN );
            memcpy( bufTS + bufferPos, pTS + rlePos, copyN );
            memcpy( bufB  + bufferPos, pB  + rlePos, copyN );
            memcpy( bufBE + bufferPos, pBE + rlePos, copyN );
            memcpy( bufBN + bufferPos, pBN + rlePos, copyN );
            memcpy( bufBW + bufferPos, pBW + rlePos, copyN );
            memcpy( bufBS + bufferPos, pBS + rlePos, copyN );
            
            for( uint_t i = 0; i < iterations; ++i, rlePos += 4, bufferPos += 4, x += 4 )
            {
               double4_t srcNE = load_aligned( bufNE + bufferPos );
               double4_t srcN  = load_aligned( bufN  + bufferPos );
               double4_t srcNW = load_aligned( bufNW + bufferPos );
               double4_t srcW  = load_aligned( bufW  + bufferPos );
               double4_t srcSW = load_aligned( bufSW + bufferPos );
               double4_t srcS  = load_aligned( bufS  + bufferPos );
               double4_t srcSE = load_aligned( bufSE + bufferPos );
               double4_t srcE  = load_aligned( bufE  + bufferPos );
               double4_t srcT  = load_aligned( bufT  + bufferPos );
               double4_t srcTE = load_aligned( bufTE + bufferPos );
               double4_t srcTN = load_aligned( bufTN + bufferPos );
               double4_t srcTW = load_aligned( bufTW + bufferPos );
               double4_t srcTS = load_aligned( bufTS + bufferPos );
               double4_t srcB  = load_aligned( bufB  + bufferPos );
               double4_t srcBE = load_aligned( bufBE + bufferPos );
               double4_t srcBN = load_aligned( bufBN + bufferPos );
               double4_t srcBW = load_aligned( bufBW + bufferPos );
               double4_t srcBS = load_aligned( bufBS + bufferPos );

               double4_t srcC = load_aligned( pC + x );

               const double4_t velX_trm = srcE + srcNE + srcSE + srcTE + srcBE;
               const double4_t velY_trm = srcN + srcNW + srcTN + srcBN;
               const double4_t velZ_trm = srcT + srcTS + srcTW;

               const double4_t rho = srcC + srcS + srcW + srcB + srcSW + srcBS + srcBW + velX_trm + velY_trm + velZ_trm;

               double4_t velX_256d = velX_trm - srcW - srcNW - srcSW - srcTW - srcBW;
               double4_t velY_256d = velY_trm + srcNE - srcS - srcSW - srcSE - srcTS - srcBS;
               double4_t velZ_256d = velZ_trm + srcTN + srcTE - srcB -srcBN -srcBS - srcBW - srcBE;

               double4_t feq_common_256d = rho - ONE_PT_FIVE * ( velX_256d * velX_256d + velY_256d * velY_256d + velZ_256d * velZ_256d );

               store_aligned( velX + bufferPos, velX_256d );
               store_aligned( velY + bufferPos, velY_256d );
               store_aligned( velZ + bufferPos, velZ_256d );

               store_aligned( feqCommon + bufferPos, feq_common_256d );
            }

            if( bufferPos == BUFFER_SIZE || x == xEnd )
               break;

            for( int offset = 0; offset < 4; ++offset, ++bufferPos, ++x, ++rlePos )
            {
               if( x == *beginNextRLEBlock )
               {
                  pNE = &( list->get( list->getPullIdx( x, NE ) ) );
                  pN  = &( list->get( list->getPullIdx( x, N  ) ) );
                  pNW = &( list->get( list->getPullIdx( x, NW ) ) );
                  pW  = &( list->get( list->getPullIdx( x, W  ) ) );
                  pSW = &( list->get( list->getPullIdx( x, SW ) ) );
                  pS  = &( list->get( list->getPullIdx( x, S  ) ) );
                  pSE = &( list->get( list->getPullIdx( x, SE ) ) );
                  pE  = &( list->get( list->getPullIdx( x, E  ) ) );
                  pT  = &( list->get( list->getPullIdx( x, T  ) ) );
                  pTE = &( list->get( list->getPullIdx( x, TE ) ) );
                  pTN = &( list->get( list->getPullIdx( x, TN ) ) );
                  pTW = &( list->get( list->getPullIdx( x, TW ) ) );
                  pTS = &( list->get( list->getPullIdx( x, TS ) ) );
                  pB  = &( list->get( list->getPullIdx( x, B  ) ) );
                  pBE = &( list->get( list->getPullIdx( x, BE ) ) );
                  pBN = &( list->get( list->getPullIdx( x, BN ) ) );
                  pBW = &( list->get( list->getPullIdx( x, BW ) ) );
                  pBS = &( list->get( list->getPullIdx( x, BS ) ) );
                  ++beginNextRLEBlock;

                  rlePos = 0;
               }

               bufNE[ bufferPos ] = pNE[rlePos];
               bufN [ bufferPos ] = pN [rlePos];
               bufNW[ bufferPos ] = pNW[rlePos];
               bufW [ bufferPos ] = pW [rlePos];
               bufSW[ bufferPos ] = pSW[rlePos];
               bufS [ bufferPos ] = pS [rlePos];
               bufSE[ bufferPos ] = pSE[rlePos];
               bufE [ bufferPos ] = pE [rlePos];
               bufT [ bufferPos ] = pT [rlePos];
               bufTE[ bufferPos ] = pTE[rlePos];
               bufTN[ bufferPos ] = pTN[rlePos];
               bufTW[ bufferPos ] = pTW[rlePos];
               bufTS[ bufferPos ] = pTS[rlePos];
               bufB [ bufferPos ] = pB [rlePos];
               bufBE[ bufferPos ] = pBE[rlePos];
               bufBN[ bufferPos ] = pBN[rlePos];
               bufBW[ bufferPos ] = pBW[rlePos];
               bufBS[ bufferPos ] = pBS[rlePos];

               const double velX_trm = pE[rlePos] + pNE[rlePos] + pSE[rlePos] + pTE[rlePos] + pBE[rlePos];
               const double velY_trm = pN[rlePos] + pNW[rlePos] + pTN[rlePos] + pBN[rlePos];
               const double velZ_trm = pT[rlePos] + pTS[rlePos] + pTW[rlePos];

               const double rho = pC[x] + pS[rlePos] + pW[rlePos] + pB[rlePos] + pSW[rlePos] + pBS[rlePos] + pBW[rlePos] + velX_trm + velY_trm + velZ_trm;

               velX[bufferPos] = velX_trm -  pW[rlePos] - pNW[rlePos] - pSW[rlePos] - pTW[rlePos] - pBW[rlePos];
               velY[bufferPos] = velY_trm + pNE[rlePos] -  pS[rlePos] - pSW[rlePos] - pSE[rlePos] - pTS[rlePos] - pBS[rlePos];
               velZ[bufferPos] = velZ_trm + pTN[rlePos] + pTE[rlePos] -  pB[rlePos] - pBN[rlePos] - pBS[rlePos] - pBW[rlePos] - pBE[rlePos];

               feqCommon[bufferPos] = rho - 1.5 * ( velX[bufferPos] * velX[bufferPos] + velY[bufferPos] * velY[bufferPos] + velZ[bufferPos] * velZ[bufferPos] );
            }
         } // end buffer filling


         for( uint_t i = 0; i < bufferPos; i += 4 )
         {
            uint_t xx = bufferStartX + i;
            double4_t srcC          = load_aligned( pC + xx );
            double4_t feqCommonQuad = load_aligned( feqCommon + i );

            double4_t dstC = srcC * ( ONE - lambda_e ) + lambda_e * t0 * feqCommonQuad;

            store_non_temp( dC + xx, dstC );
         }
         
         HANDLE_DIR( NE, SW, load_aligned( velX + i ) + load_aligned( velY + i ), fac2, t2x2 );      
         HANDLE_DIR( SE, NW, load_aligned( velX + i ) - load_aligned( velY + i ), fac2, t2x2 );
         HANDLE_DIR( TE, BW, load_aligned( velX + i ) + load_aligned( velZ + i ), fac2, t2x2 );
         HANDLE_DIR( BE, TW, load_aligned( velX + i ) - load_aligned( velZ + i ), fac2, t2x2 );         
         HANDLE_DIR( TN, BS, load_aligned( velY + i ) + load_aligned( velZ + i ), fac2, t2x2 );         
         HANDLE_DIR( BN, TS, load_aligned( velY + i ) - load_aligned( velZ + i ), fac2, t2x2 ); 

         HANDLE_DIR( E , W , load_aligned( velX + i ), fac1, t1x2 );         
         HANDLE_DIR( N , S , load_aligned( velY + i ), fac1, t1x2 );         
         HANDLE_DIR( T , B , load_aligned( velZ + i ), fac1, t1x2 );

         bufferPos = 0;
         bufferStartX = x;
      }

#ifdef _OPENMP
   }
#endif

   list->swapTmpPdfs();

#ifdef LIKWID_PERFMON
   LIKWID_MARKER_STOP( "Kernel-opt" );
#endif
}

#undef HANDLE_DIR


} // namespace lbm
} // namespace walberla