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
//! \file ListBGQKernel.cpp
//! \ingroup lbm
//! \author Christian Godenschwager <christian.godenschwager@fau.de>
//
//======================================================================================================================

#ifdef __bg__

#include "ListBGQKernel.h"

#include <core/debug/CheckFunctions.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace walberla {
namespace lbm {

typedef int __V4SI __attribute__((vector_size(16)));

#define __dcbf(base) \
  __asm__ volatile ("dcbf %y0" : "=Z" (*(__V4SI*) (base)) : : "memory")

#define __dcbz(base) \
  __asm__ volatile ("dcbz %y0" : "=Z" (*(__V4SI*) (base)) : : "memory")


#define HANDLE_DIR( srcPDF0, srcPDF1, fac, vel, t, dstPtr0, dstPtr1 ) \
   sym  = vec_mul( lambda_e_scaled, vec_nmsub( t, feq_common, vec_nmsub( fac, vec_mul( vel, vel ), vec_add( srcPDF0, srcPDF1 ) ) ) ); \
   asym = vec_mul( lambda_d_scaled, vec_nmsub( vec_mul( THREE, t ), vel, vec_sub( srcPDF0, srcPDF1 ) ) ); \
   dst0 = vec_sub( vec_sub( srcPDF0, sym ), asym ); \
   dst1 = vec_add( vec_sub( srcPDF1, sym ), asym ); \
   vec_sta( dst0, 0, dstPtr0 + x ); \
   vec_sta( dst1, 0, dstPtr1 + x );

#define HANDLE_DIR_DCBZ( srcPDF0, srcPDF1, fac, vel, t, dstPtr0, dstPtr1 ) \
   sym  = vec_mul( lambda_e_scaled, vec_nmsub( t, feq_common, vec_nmsub( fac, vec_mul( vel, vel ), vec_add( srcPDF0, srcPDF1 ) ) ) ); \
   asym = vec_mul( lambda_d_scaled, vec_nmsub( vec_mul( THREE, t ), vel, vec_sub( srcPDF0, srcPDF1 ) ) ); \
   dst0 = vec_sub( vec_sub( srcPDF0, sym ), asym ); \
   dst1 = vec_add( vec_sub( srcPDF1, sym ), asym ); \
   __dcbz( dstPtr0 + x ); \
   vec_sta( dst0, 0, dstPtr0 + x ); \
   __dcbz( dstPtr1 + x ); \
   vec_sta( dst1, 0, dstPtr1 + x );


#define HANDLE_DIR_DCBF( srcPDF0, srcPDF1, fac, vel, t, dstPtr0, dstPtr1 ) \
   sym  = vec_mul( lambda_e_scaled, vec_nmsub( t, feq_common, vec_nmsub( fac, vec_mul( vel, vel ), vec_add( srcPDF0, srcPDF1 ) ) ) ); \
   asym = vec_mul( lambda_d_scaled, vec_nmsub( vec_mul( THREE, t ), vel, vec_sub( srcPDF0, srcPDF1 ) ) ); \
   dst0 = vec_sub( vec_sub( srcPDF0, sym ), asym ); \
   dst1 = vec_add( vec_sub( srcPDF1, sym ), asym ); \
   vec_sta( dst0, 0, dstPtr0 + x ); \
   __dcbf( dstPtr0 + x ); \
   vec_sta( dst1, 0, dstPtr1 + x ); \
   __dcbf( dstPtr1 + x );

#define VEC_LDU( out, in_addr ) \
   tmp1 = vec_ld(0,  in_addr); \
   tmp2 = vec_ld(32, in_addr); \
   pctl = vec_lvsl(0, in_addr); \
   out = vec_perm(tmp1, tmp2, pctl);

template< typename List_T >
void ListSplitBGQSIMDTRTSweep< List_T >::operator()( IBlock * const block )
{
   bgpm::BGPM_Manager::instance()->getEventSet().start();

   List_T * list = block->getData<List_T>( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( list );

   static const vector4double PT_FIVE = vec_splats( 0.5 );
   static const vector4double ONE = vec_splats( 1 );
   static const vector4double ONE_PT_FIVE = vec_splats( 1.5 );
   static const vector4double THREE = vec_splats( 3 );

   // constants used during stream/collide

   const vector4double lambda_e = vec_splats( list->latticeModel().collisionModel().lambda_e() );
   const vector4double lambda_d = vec_splats( list->latticeModel().collisionModel().lambda_d() );

   // common prefactors for calculating the equilibrium parts
   static const vector4double t0   = vec_splats( 1.0 / 3.0        );  // 1/3      for C
   static const vector4double t1x2 = vec_splats( 1.0 / 18.0 * 2.0 );  // 1/18 * 2 for N, S, W, E, T, B
   static const vector4double t2x2 = vec_splats( 1.0 / 36.0 * 2.0 );  // 1/36 * 2 else

   static const vector4double inv2csq2 = vec_splats( 1.0 / ( 2.0 * ( 1.0 / 3.0 ) * ( 1.0 / 3.0 ) ) ); //speed of sound related factor for equilibrium distribution function
   static const vector4double fac1 = vec_mul( t1x2, inv2csq2 );
   static const vector4double fac2 = vec_mul( t2x2, inv2csq2 );

   // relaxation parameter variables
   const vector4double lambda_e_scaled = vec_mul( PT_FIVE, lambda_e ); // 0.5 times the usual value ...
   const vector4double lambda_d_scaled = vec_mul( PT_FIVE, lambda_d ); // ... due to the way of calculations

   using namespace stencil;

   double * pC  = NULL;   

   double * dNE = NULL;
   double * dSW = NULL;
   double * dSE = NULL;
   double * dNW = NULL;
   double * dTE = NULL;
   double * dBW = NULL;
   double * dBE = NULL;
   double * dTW = NULL;
   double * dTN = NULL;
   double * dBS = NULL;
   double * dBN = NULL;
   double * dTS = NULL;
   double * dN  = NULL;
   double * dS  = NULL;
   double * dE  = NULL;
   double * dW  = NULL;
   double * dT  = NULL;
   double * dB  = NULL;
   double * dC  = NULL;

   const uint_t numFluidCells = list->numFluidCells();   

   if( numFluidCells > 0 )
   {
      pC  = &( list->get( 0, C  ) );

      dNE = &( list->getTmp( 0, NE ) );
      dSW = &( list->getTmp( 0, N  ) );
      dSE = &( list->getTmp( 0, NW ) );
      dNW = &( list->getTmp( 0, W  ) );
      dTE = &( list->getTmp( 0, SW ) );
      dBW = &( list->getTmp( 0, S  ) );
      dBE = &( list->getTmp( 0, SE ) );
      dTW = &( list->getTmp( 0, E  ) );
      dTN = &( list->getTmp( 0, T  ) );
      dBS = &( list->getTmp( 0, TE ) );
      dBN = &( list->getTmp( 0, TN ) );
      dTS = &( list->getTmp( 0, TW ) );
      dN  = &( list->getTmp( 0, TS ) );
      dS  = &( list->getTmp( 0, B  ) );
      dE  = &( list->getTmp( 0, BE ) );
      dW  = &( list->getTmp( 0, BN ) );
      dT  = &( list->getTmp( 0, BW ) );
      dB  = &( list->getTmp( 0, BS ) );
      dC  = &( list->getTmp( 0, C  ) );
   }

   //#ifdef _OPENMP
   //   patterns_.resize( numeric_cast<size_t>( omp_get_max_threads() ), bgq::L1PPatternStore< real_t * >( INITIAL_PATTERN_LENGTH ) );
   //#else
   //   patterns_.resize( 1, bgq::L1PPatternStore< real_t * >( INITIAL_PATTERN_LENGTH ) );
   //#endif 

   const uint_t numCLBlocks   = ( numFluidCells + uint_t(7) ) / uint_t(8);

#ifdef _OPENMP
#pragma omp parallel
   {
#endif   

#ifdef _OPENMP
      const int threadId   = omp_get_thread_num();
      const int numThreads = omp_get_num_threads();
#else
      const int threadId   = 0;
      const int numThreads = 1;
#endif

      //patterns_[ threadId ].setActiveAndStart( pC );

      const uint_t startCLBlock = ( numCLBlocks + uint_c(numThreads) - uint_t(1) ) / uint_c(numThreads) * uint_c( threadId     );
      const uint_t endCLBlock   = ( numCLBlocks + uint_c(numThreads) - uint_t(1) ) / uint_c(numThreads) * uint_c( threadId + 1 );
      uint_t x    = std::min( startCLBlock * uint_t(8), numFluidCells );
      uint_t xEnd = std::min( endCLBlock   * uint_t(8), numFluidCells );
      auto rleBegin = std::upper_bound( list->getRLEInfo().begin(), list->getRLEInfo().end() - 1, x );

      //WALBERLA_LOG_DEVEL( "Thread       " << threadId     << '\n' <<
      //                    "numCLBlocks  " << numCLBlocks  << '\n' <<
      //                    "startCLBlock " << startCLBlock << '\n' <<
      //                    "endCLBlock   " << endCLBlock   << '\n' <<
      //                    "x            " << x            << '\n' <<
      //                    "xEnd         " << xEnd         << '\n' <<
      //                    "rleBegin     " << *rleBegin );

      vector4double srcNE;
      vector4double srcN;
      vector4double srcNW;
      vector4double srcW;
      vector4double srcSW;
      vector4double srcS;
      vector4double srcSE;
      vector4double srcE;
      vector4double srcT;
      vector4double srcTE;
      vector4double srcTN;
      vector4double srcTW;
      vector4double srcTS;
      vector4double srcB;
      vector4double srcBE;
      vector4double srcBN;
      vector4double srcBW;
      vector4double srcBS;

      double * pNE = &( list->get( list->getPullIdx( x, NE ) ) );
      double * pSW = &( list->get( list->getPullIdx( x, N  ) ) );
      double * pSE = &( list->get( list->getPullIdx( x, NW ) ) );
      double * pNW = &( list->get( list->getPullIdx( x, W  ) ) );
      double * pTE = &( list->get( list->getPullIdx( x, SW ) ) );
      double * pBW = &( list->get( list->getPullIdx( x, S  ) ) );
      double * pBE = &( list->get( list->getPullIdx( x, SE ) ) );
      double * pTW = &( list->get( list->getPullIdx( x, E  ) ) );
      double * pTN = &( list->get( list->getPullIdx( x, T  ) ) );
      double * pBS = &( list->get( list->getPullIdx( x, TE ) ) );
      double * pBN = &( list->get( list->getPullIdx( x, TN ) ) );
      double * pTS = &( list->get( list->getPullIdx( x, TW ) ) );
      double * pN  = &( list->get( list->getPullIdx( x, TS ) ) );
      double * pS  = &( list->get( list->getPullIdx( x, B  ) ) );
      double * pE  = &( list->get( list->getPullIdx( x, BE ) ) );
      double * pW  = &( list->get( list->getPullIdx( x, BN ) ) );
      double * pT  = &( list->get( list->getPullIdx( x, BW ) ) );
      double * pB  = &( list->get( list->getPullIdx( x, BS ) ) );
      uint_t rlePos = 0;

      while( x < xEnd )
      {
         const uint_t blockLength = *rleBegin - x;
         const uint_t iterations = blockLength / 4;

         for( uint_t i = 0; i < iterations; ++i, rlePos += 4 )
         {
            vector4double tmp1, tmp2, pctl;
            VEC_LDU( srcNE, pNE + rlePos );
            VEC_LDU( srcN , pN  + rlePos );
            VEC_LDU( srcNW, pNW + rlePos );
            VEC_LDU( srcW , pW  + rlePos );
            VEC_LDU( srcSW, pSW + rlePos );
            VEC_LDU( srcS , pS  + rlePos );
            VEC_LDU( srcSE, pSE + rlePos );
            VEC_LDU( srcE , pE  + rlePos );
            VEC_LDU( srcT , pT  + rlePos );
            VEC_LDU( srcTE, pTE + rlePos );
            VEC_LDU( srcTN, pTN + rlePos );
            VEC_LDU( srcTW, pTW + rlePos );
            VEC_LDU( srcTS, pTS + rlePos );
            VEC_LDU( srcB , pB  + rlePos );
            VEC_LDU( srcBE, pBE + rlePos );
            VEC_LDU( srcBN, pBN + rlePos );
            VEC_LDU( srcBW, pBW + rlePos );
            VEC_LDU( srcBS, pBS + rlePos );

            vector4double srcC = vec_lda( 0LU, pC + x );

            vector4double velX = vec_add( srcE, srcNE );
            velX = vec_add( velX, srcSE );
            velX = vec_add( velX, srcTE );
            velX = vec_add( velX, srcBE );

            vector4double velY = vec_add( srcN, srcNW );
            velY = vec_add( velY, srcTN );
            velY = vec_add( velY, srcBN );      

            vector4double velZ = vec_add( srcT, srcTS );
            velZ = vec_add( velZ, srcTW );

            vector4double rho = vec_add( srcC, srcS );
            rho = vec_add( rho, srcW  );
            rho = vec_add( rho, srcB  );
            rho = vec_add( rho, srcSW );
            rho = vec_add( rho, srcBS );
            rho = vec_add( rho, srcBW );
            rho = vec_add( rho, velX  );
            rho = vec_add( rho, velY  );
            rho = vec_add( rho, velZ  );

            velX = vec_sub( velX, srcW  );
            velX = vec_sub( velX, srcNW );
            velX = vec_sub( velX, srcSW );
            velX = vec_sub( velX, srcTW );
            velX = vec_sub( velX, srcBW );

            velY = vec_add( velY, srcNE );
            velY = vec_sub( velY, srcS  );
            velY = vec_sub( velY, srcSW );
            velY = vec_sub( velY, srcSE );
            velY = vec_sub( velY, srcTS );
            velY = vec_sub( velY, srcBS );

            velZ = vec_add( velZ, srcTN );
            velZ = vec_add( velZ, srcTE );
            velZ = vec_sub( velZ, srcB  );
            velZ = vec_sub( velZ, srcBN );
            velZ = vec_sub( velZ, srcBS );
            velZ = vec_sub( velZ, srcBW );
            velZ = vec_sub( velZ, srcBE );

            vector4double feq_common = vec_sub( rho, vec_mul( ONE_PT_FIVE, vec_madd( velX, velX, vec_madd( velY, velY, vec_mul( velZ, velZ ) ) ) ) );

            vector4double dstC = vec_madd( vec_mul( lambda_e, t0 ), feq_common, vec_mul( srcC, vec_sub( ONE, lambda_e ) ) );

            vec_sta( dstC, 0LU, dC + x );

            vector4double sym;
            vector4double asym;
            vector4double dst0;
            vector4double dst1;

            vector4double velXPY = vec_add( velX, velY ); 
            HANDLE_DIR( srcNE, srcSW, fac2, velXPY, t2x2, dNE, dSW )

            vector4double velXMY = vec_sub( velX, velY );
            HANDLE_DIR( srcSE, srcNW, fac2, velXMY, t2x2, dSE, dNW )

            vector4double velXPZ = vec_add( velX, velZ );
            HANDLE_DIR( srcTE, srcBW, fac2, velXPZ, t2x2, dTE, dBW )

            vector4double velXMZ = vec_sub( velX, velZ );
            HANDLE_DIR( srcBE, srcTW, fac2, velXMZ, t2x2, dBE, dTW )

            vector4double velYPZ = vec_add( velY, velZ );
            HANDLE_DIR( srcTN, srcBS, fac2, velYPZ, t2x2, dTN, dBS )

            vector4double velYMZ = vec_sub( velY, velZ );
            HANDLE_DIR( srcBN, srcTS, fac2, velYMZ, t2x2, dBN, dTS )

            HANDLE_DIR( srcN, srcS, fac1, velY, t1x2, dN, dS )

            HANDLE_DIR( srcE, srcW, fac1, velX, t1x2, dE, dW )

            HANDLE_DIR( srcT, srcB, fac1, velZ, t1x2, dT, dB )

            x += 4;
         }

         if( x == *rleBegin )
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
            ++rleBegin;

            rlePos = 0;

            continue;
         }

         for( int offset = 0; offset < 4; ++offset, ++rlePos )
         {
            vec_insert( pNE[rlePos], srcNE, offset );
            vec_insert( pN[rlePos] , srcN , offset ); 
            vec_insert( pNW[rlePos], srcNW, offset );
            vec_insert( pW[rlePos] , srcW , offset );
            vec_insert( pSW[rlePos], srcSW, offset );
            vec_insert( pS[rlePos] , srcS , offset );
            vec_insert( pSE[rlePos], srcSE, offset );
            vec_insert( pE[rlePos] , srcE , offset );
            vec_insert( pT[rlePos] , srcT , offset );
            vec_insert( pTE[rlePos], srcTE, offset );
            vec_insert( pTN[rlePos], srcTN, offset );
            vec_insert( pTW[rlePos], srcTW, offset );
            vec_insert( pTS[rlePos], srcTS, offset );
            vec_insert( pB[rlePos] , srcB , offset );
            vec_insert( pBE[rlePos], srcBE, offset );
            vec_insert( pBN[rlePos], srcBN, offset );
            vec_insert( pBW[rlePos], srcBW, offset );
            vec_insert( pBS[rlePos], srcBS, offset );

            uint_t xx = x + uint_c( offset );
            if( xx == *rleBegin )
            {
               pNE = &( list->get( list->getPullIdx( xx, NE ) ) );
               pN  = &( list->get( list->getPullIdx( xx, N  ) ) );
               pNW = &( list->get( list->getPullIdx( xx, NW ) ) );
               pW  = &( list->get( list->getPullIdx( xx, W  ) ) );
               pSW = &( list->get( list->getPullIdx( xx, SW ) ) );
               pS  = &( list->get( list->getPullIdx( xx, S  ) ) );
               pSE = &( list->get( list->getPullIdx( xx, SE ) ) );
               pE  = &( list->get( list->getPullIdx( xx, E  ) ) );
               pT  = &( list->get( list->getPullIdx( xx, T  ) ) );
               pTE = &( list->get( list->getPullIdx( xx, TE ) ) );
               pTN = &( list->get( list->getPullIdx( xx, TN ) ) );
               pTW = &( list->get( list->getPullIdx( xx, TW ) ) );
               pTS = &( list->get( list->getPullIdx( xx, TS ) ) );
               pB  = &( list->get( list->getPullIdx( xx, B  ) ) );
               pBE = &( list->get( list->getPullIdx( xx, BE ) ) );
               pBN = &( list->get( list->getPullIdx( xx, BN ) ) );
               pBW = &( list->get( list->getPullIdx( xx, BW ) ) );
               pBS = &( list->get( list->getPullIdx( xx, BS ) ) );
               ++rleBegin;

               rlePos = 0;
            }
         }      

         //for( uint_t offset = 0; offset < 4; ++offset, ++rlePos )
         //{
         //   srcNE[offset] = pNE[rlePos];
         //   srcN[offset]  = pN[rlePos]; 
         //   srcNW[offset] = pNW[rlePos];
         //   srcW[offset]  = pW[rlePos];
         //   srcSW[offset] = pSW[rlePos];
         //   srcS[offset]  = pS[rlePos];
         //   srcSE[offset] = pSE[rlePos];
         //   srcE[offset]  = pE[rlePos];
         //   srcT[offset]  = pT[rlePos];
         //   srcTE[offset] = pTE[rlePos];
         //   srcTN[offset] = pTN[rlePos];
         //   srcTW[offset] = pTW[rlePos];
         //   srcTS[offset] = pTS[rlePos];
         //   srcB[offset]  = pB[rlePos];
         //   srcBE[offset] = pBE[rlePos];
         //   srcBN[offset] = pBN[rlePos];
         //   srcBW[offset] = pBW[rlePos];
         //   srcBS[offset] = pBS[rlePos];
         //   
         //   uint_t xx = x + offset;
         //   if( xx == *rleBegin )
         //   {
         //      pNE = list->getPullPtr( xx, NE );
         //      pN  = list->getPullPtr( xx, N  );
         //      pNW = list->getPullPtr( xx, NW );
         //      pW  = list->getPullPtr( xx, W  );
         //      pSW = list->getPullPtr( xx, SW );
         //      pS  = list->getPullPtr( xx, S  );
         //      pSE = list->getPullPtr( xx, SE );
         //      pE  = list->getPullPtr( xx, E  );
         //      pT  = list->getPullPtr( xx, T  );
         //      pTE = list->getPullPtr( xx, TE );
         //      pTN = list->getPullPtr( xx, TN );
         //      pTW = list->getPullPtr( xx, TW );
         //      pTS = list->getPullPtr( xx, TS );
         //      pB  = list->getPullPtr( xx, B  );
         //      pBE = list->getPullPtr( xx, BE );
         //      pBN = list->getPullPtr( xx, BN );
         //      pBW = list->getPullPtr( xx, BW );
         //      pBS = list->getPullPtr( xx, BS );
         //      ++rleBegin;
         //      
         //      rlePos = 0;
         //   }
         //} 

         vector4double srcC = vec_lda( 0LU, pC + x );

         vector4double velX = vec_add( srcE, srcNE );
         velX = vec_add( velX, srcSE );
         velX = vec_add( velX, srcTE );
         velX = vec_add( velX, srcBE );

         vector4double velY = vec_add( srcN, srcNW );
         velY = vec_add( velY, srcTN );
         velY = vec_add( velY, srcBN );

         vector4double velZ = vec_add( srcT, srcTS );
         velZ = vec_add( velZ, srcTW );

         vector4double rho = vec_add( srcC, srcS );
         rho = vec_add( rho, srcW );
         rho = vec_add( rho, srcB );
         rho = vec_add( rho, srcSW );
         rho = vec_add( rho, srcBS );
         rho = vec_add( rho, srcBW );
         rho = vec_add( rho, velX );
         rho = vec_add( rho, velY );
         rho = vec_add( rho, velZ );

         velX = vec_sub( velX, srcW );
         velX = vec_sub( velX, srcNW );
         velX = vec_sub( velX, srcSW );
         velX = vec_sub( velX, srcTW );
         velX = vec_sub( velX, srcBW );

         velY = vec_add( velY, srcNE );
         velY = vec_sub( velY, srcS );
         velY = vec_sub( velY, srcSW );
         velY = vec_sub( velY, srcSE );
         velY = vec_sub( velY, srcTS );
         velY = vec_sub( velY, srcBS );

         velZ = vec_add( velZ, srcTN );
         velZ = vec_add( velZ, srcTE );
         velZ = vec_sub( velZ, srcB );
         velZ = vec_sub( velZ, srcBN );
         velZ = vec_sub( velZ, srcBS );
         velZ = vec_sub( velZ, srcBW );
         velZ = vec_sub( velZ, srcBE );

         vector4double feq_common = vec_sub( rho, vec_mul( ONE_PT_FIVE, vec_madd( velX, velX, vec_madd( velY, velY, vec_mul( velZ, velZ ) ) ) ) );

         vector4double dstC = vec_madd( vec_mul( lambda_e, t0 ), feq_common, vec_mul( srcC, vec_sub( ONE, lambda_e ) ) );

         vec_sta( dstC, 0LU, dC + x );

         vector4double sym;
         vector4double asym;
         vector4double dst0;
         vector4double dst1;

         vector4double velXPY = vec_add( velX, velY );
         HANDLE_DIR( srcNE, srcSW, fac2, velXPY, t2x2, dNE, dSW )

         vector4double velXMY = vec_sub( velX, velY );
         HANDLE_DIR( srcSE, srcNW, fac2, velXMY, t2x2, dSE, dNW )

         vector4double velXPZ = vec_add( velX, velZ );
         HANDLE_DIR( srcTE, srcBW, fac2, velXPZ, t2x2, dTE, dBW )

         vector4double velXMZ = vec_sub( velX, velZ );
         HANDLE_DIR( srcBE, srcTW, fac2, velXMZ, t2x2, dBE, dTW )

         vector4double velYPZ = vec_add( velY, velZ );
         HANDLE_DIR( srcTN, srcBS, fac2, velYPZ, t2x2, dTN, dBS )

         vector4double velYMZ = vec_sub( velY, velZ );
         HANDLE_DIR( srcBN, srcTS, fac2, velYMZ, t2x2, dBN, dTS )

         HANDLE_DIR( srcN, srcS, fac1, velY, t1x2, dN, dS )

         HANDLE_DIR( srcE, srcW, fac1, velX, t1x2, dE, dW )

         HANDLE_DIR( srcT, srcB, fac1, velZ, t1x2, dT, dB )

         x += 4;
      }

      //patterns_[ threadId ].stop();

#ifdef _OPENMP
   }
#endif

   list->swapTmpPdfs();

   bgpm::BGPM_Manager::instance()->getEventSet().stop();
}

#undef HANDLE_DIR

template class ListSplitBGQSIMDTRTSweep< lbm::List< lbm::D3Q19< lbm::collision_model::TRT, false >, lbm::LayoutFIdx<64> > >;


} // namespace lbm
} // namespace walberla

#endif // __bg__