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
//! \file ListKernel.h
//! \ingroup lbm
//! \author Christian Godenschwager <christian.godenschwager@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/Macros.h"

#include "domain_decomposition/IBlock.h"
#include "domain_decomposition/BlockDataID.h"

#include "lbm/IntelCompilerOptimization.h"


namespace walberla {
namespace lbm {


template< typename List_T >
class ListDefaultTRTSweep
{
   static_assert( ( std::is_same< typename List_T::LatticeModel::CollisionModel::tag, collision_model::TRT_tag >::value ), "Only works with TRT!" );
   static_assert( ( std::is_same< typename List_T::LatticeModel::ForceModel::tag, force_model::None_tag >::value ), "Only works without additional forces!" );
   static_assert( List_T::LatticeModel::equilibriumAccuracyOrder == 2, "Only works for lattice models that require the equilibrium distribution to be order 2 accurate!" );

public:
   ListDefaultTRTSweep( const BlockDataID listId ) : listId_( listId ) { }
   void operator()( IBlock * const block );

protected:
   BlockDataID listId_;
};


template< typename List_T >
void ListDefaultTRTSweep< List_T >::operator()( IBlock * const block )
{
   List_T * list = block->getData<List_T>( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( list )

   typedef typename List_T::Stencil Stencil;
   typedef typename List_T::LatticeModel LatticeModel;

   // stream & collide

   const real_t lambda_e = list->latticeModel().collisionModel().lambda_e();
   const real_t lambda_d = list->latticeModel().collisionModel().lambda_d();


   real_t pdfs[ Stencil::Size ];

   for( typename List_T::index_t idx = 0; idx < list->numFluidCells(); ++idx )
   {
      // stream pull & calculation of macroscopic velocities

      pdfs[ Stencil::idx[stencil::C] ] = list->get( idx, stencil::C );
      Vector3<real_t> velocity = Vector3<real_t>();
      real_t rho = pdfs[ Stencil::idx[stencil::C] ];

      for( auto d = Stencil::beginNoCenter(); d != Stencil::end(); ++d )
      {
         const auto pdf = list->get( list->getPullIdx( idx, *d ) );
         WALBERLA_ASSERT( !math::isnan( pdf ) )
         pdfs[ d.toIdx() ] = pdf;

         rho += pdf;
         velocity[0] += numeric_cast<real_t>( d.cx() ) * pdf;
         velocity[1] += numeric_cast<real_t>( d.cy() ) * pdf;
         velocity[2] += numeric_cast<real_t>( d.cz() ) * pdf;
      }

      rho += real_t( 1 );

      // collide
      for( auto d = Stencil::begin(); d != Stencil::end(); ++d )
      {
         const real_t fsym  = EquilibriumDistribution< LatticeModel >::getSymmetricPart( *d, velocity, rho );
         const real_t fasym = EquilibriumDistribution< LatticeModel >::getAsymmetricPart( *d, velocity, rho );

         const real_t f = pdfs[d.toIdx()];
         const real_t finv = pdfs[d.toInvIdx()];

         list->getTmp( idx, *d ) = f - lambda_e * ( real_t( 0.5 ) * ( f + finv ) - fsym )
                                     - lambda_d * ( real_t( 0.5 ) * ( f - finv ) - fasym );
      }
   }


   list->swapTmpPdfs();
}


template< typename List_T >
class ListTRTSweep
{
   static_assert( ( std::is_same< typename List_T::LatticeModel::CollisionModel::tag, collision_model::TRT_tag >::value ), "Only works with TRT!" );
   static_assert( ( std::is_same< typename List_T::LatticeModel::Stencil, stencil::D3Q19 >::value ), "Only works with D3Q19!" );
   static_assert( !List_T::LatticeModel::compressible, "Only works with incompressible models!" );
   static_assert( ( std::is_same< typename List_T::LatticeModel::ForceModel::tag, force_model::None_tag >::value ), "Only works without additional forces!" );
   static_assert( List_T::LatticeModel::equilibriumAccuracyOrder == 2, "Only works for lattice models that require the equilibrium distribution to be order 2 accurate!" );

public:
   ListTRTSweep( const BlockDataID listId ) : listId_( listId ) { }
   void operator()( IBlock * const block );

protected:
   BlockDataID listId_;
};

template< typename List_T >
void ListTRTSweep< List_T >::operator()( IBlock * const block )
{
   List_T * list = block->getData<List_T>( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( list )

   // constants used during stream/collide

   const real_t lambda_e = list->latticeModel().collisionModel().lambda_e();
   const real_t lambda_d = list->latticeModel().collisionModel().lambda_d();

   // common prefactors for calculating the equilibrium parts
   const real_t t0   = real_t( 1.0 ) / real_t( 3.0 );                   // 1/3      for C
   const real_t t1x2 = real_t( 1.0 ) / real_t( 18.0 ) * real_t( 2.0 );  // 1/18 * 2 for N, S, W, E, T, B
   const real_t t2x2 = real_t( 1.0 ) / real_t( 36.0 ) * real_t( 2.0 );  // 1/36 * 2 else

   const real_t inv2csq2 = real_t( 1.0 ) / ( real_t( 2.0 ) * ( real_t( 1.0 ) / real_t( 3.0 ) ) * ( real_t( 1.0 ) / real_t( 3.0 ) ) ); //speed of sound related factor for equilibrium distribution function
   const real_t fac1 = t1x2 * inv2csq2;
   const real_t fac2 = t2x2 * inv2csq2;

   // relaxation parameter variables
   const real_t lambda_e_scaled = real_t( 0.5 ) * lambda_e; // 0.5 times the usual value ...
   const real_t lambda_d_scaled = real_t( 0.5 ) * lambda_d; // ... due to the way of calculations

   // stream & collide

   const uint_t inc = list->hasIdxFLayout() ? List_T::Stencil::Size : uint_t( 1 );

   auto rleBegin = list->getRLEInfo().begin();
   auto rleEnd   = list->getRLEInfo().begin() + 1;

   while( rleEnd != list->getRLEInfo().end() )
   {
      using namespace stencil;

      const real_t * WALBERLA_RESTRICT dd_tmp_NE = &( list->get( list->getPullIdx( *rleBegin, NE ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_N  = &( list->get( list->getPullIdx( *rleBegin, N  ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_NW = &( list->get( list->getPullIdx( *rleBegin, NW ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_W  = &( list->get( list->getPullIdx( *rleBegin, W  ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_SW = &( list->get( list->getPullIdx( *rleBegin, SW ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_S  = &( list->get( list->getPullIdx( *rleBegin, S  ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_SE = &( list->get( list->getPullIdx( *rleBegin, SE ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_E  = &( list->get( list->getPullIdx( *rleBegin, E  ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_T  = &( list->get( list->getPullIdx( *rleBegin, T  ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_TE = &( list->get( list->getPullIdx( *rleBegin, TE ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_TN = &( list->get( list->getPullIdx( *rleBegin, TN ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_TW = &( list->get( list->getPullIdx( *rleBegin, TW ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_TS = &( list->get( list->getPullIdx( *rleBegin, TS ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_B  = &( list->get( list->getPullIdx( *rleBegin, B  ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_BE = &( list->get( list->getPullIdx( *rleBegin, BE ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_BN = &( list->get( list->getPullIdx( *rleBegin, BN ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_BW = &( list->get( list->getPullIdx( *rleBegin, BW ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_BS = &( list->get( list->getPullIdx( *rleBegin, BS ) ) );
      const real_t * WALBERLA_RESTRICT dd_tmp_C  = &( list->get( *rleBegin, C  ) );

      real_t * WALBERLA_RESTRICT dst_NE = &( list->getTmp( *rleBegin, NE ) );
      real_t * WALBERLA_RESTRICT dst_N  = &( list->getTmp( *rleBegin, N  ) );
      real_t * WALBERLA_RESTRICT dst_NW = &( list->getTmp( *rleBegin, NW ) );
      real_t * WALBERLA_RESTRICT dst_W  = &( list->getTmp( *rleBegin, W  ) );
      real_t * WALBERLA_RESTRICT dst_SW = &( list->getTmp( *rleBegin, SW ) );
      real_t * WALBERLA_RESTRICT dst_S  = &( list->getTmp( *rleBegin, S  ) );
      real_t * WALBERLA_RESTRICT dst_SE = &( list->getTmp( *rleBegin, SE ) );
      real_t * WALBERLA_RESTRICT dst_E  = &( list->getTmp( *rleBegin, E  ) );
      real_t * WALBERLA_RESTRICT dst_T  = &( list->getTmp( *rleBegin, T  ) );
      real_t * WALBERLA_RESTRICT dst_TE = &( list->getTmp( *rleBegin, TE ) );
      real_t * WALBERLA_RESTRICT dst_TN = &( list->getTmp( *rleBegin, TN ) );
      real_t * WALBERLA_RESTRICT dst_TW = &( list->getTmp( *rleBegin, TW ) );
      real_t * WALBERLA_RESTRICT dst_TS = &( list->getTmp( *rleBegin, TS ) );
      real_t * WALBERLA_RESTRICT dst_B  = &( list->getTmp( *rleBegin, B  ) );
      real_t * WALBERLA_RESTRICT dst_BE = &( list->getTmp( *rleBegin, BE ) );
      real_t * WALBERLA_RESTRICT dst_BN = &( list->getTmp( *rleBegin, BN ) );
      real_t * WALBERLA_RESTRICT dst_BW = &( list->getTmp( *rleBegin, BW ) );
      real_t * WALBERLA_RESTRICT dst_BS = &( list->getTmp( *rleBegin, BS ) );
      real_t * WALBERLA_RESTRICT dst_C  = &( list->getTmp( *rleBegin, C  ) );

      const uint_t length = inc * ( *rleEnd - *rleBegin );

      for( uint_t x = 0; x < length; x += inc )
      {
         const real_t velX_trm = dd_tmp_E[x] + dd_tmp_NE[x] + dd_tmp_SE[x] + dd_tmp_TE[x] + dd_tmp_BE[x];
         const real_t velY_trm = dd_tmp_N[x] + dd_tmp_NW[x] + dd_tmp_TN[x] + dd_tmp_BN[x];
         const real_t velZ_trm = dd_tmp_T[x] + dd_tmp_TS[x] + dd_tmp_TW[x];

         const real_t rho = dd_tmp_C[x] + dd_tmp_S[x] + dd_tmp_W[x] + dd_tmp_B[x] + dd_tmp_SW[x] + dd_tmp_BS[x] + dd_tmp_BW[x] + velX_trm + velY_trm + velZ_trm;

         const real_t velX = velX_trm - dd_tmp_W[x] - dd_tmp_NW[x] - dd_tmp_SW[x] - dd_tmp_TW[x] - dd_tmp_BW[x];
         const real_t velY = velY_trm + dd_tmp_NE[x] - dd_tmp_S[x] - dd_tmp_SW[x] - dd_tmp_SE[x] - dd_tmp_TS[x] - dd_tmp_BS[x];
         const real_t velZ = velZ_trm + dd_tmp_TN[x] + dd_tmp_TE[x] - dd_tmp_B[x] - dd_tmp_BN[x] - dd_tmp_BS[x] - dd_tmp_BW[x] - dd_tmp_BE[x];

         const real_t feq_common = rho - real_t( 1.5 ) * ( velX * velX + velY * velY + velZ * velZ );

         dst_C[x] = dd_tmp_C[x] * ( real_t( 1.0 ) - lambda_e ) + lambda_e * t0 * feq_common;

         const real_t velXPY = velX + velY;
         const real_t  sym_NE_SW = lambda_e_scaled * ( dd_tmp_NE[x] + dd_tmp_SW[x] - fac2 * velXPY * velXPY - t2x2 * feq_common );
         const real_t asym_NE_SW = lambda_d_scaled * ( dd_tmp_NE[x] - dd_tmp_SW[x] - real_t( 3.0 ) * t2x2 * velXPY );
         dst_NE[x] = dd_tmp_NE[x] - sym_NE_SW - asym_NE_SW;
         dst_SW[x] = dd_tmp_SW[x] - sym_NE_SW + asym_NE_SW;

         const real_t velXMY = velX - velY;
         const real_t  sym_SE_NW = lambda_e_scaled * ( dd_tmp_SE[x] + dd_tmp_NW[x] - fac2 * velXMY * velXMY - t2x2 * feq_common );
         const real_t asym_SE_NW = lambda_d_scaled * ( dd_tmp_SE[x] - dd_tmp_NW[x] - real_t( 3.0 ) * t2x2 * velXMY );
         dst_SE[x] = dd_tmp_SE[x] - sym_SE_NW - asym_SE_NW;
         dst_NW[x] = dd_tmp_NW[x] - sym_SE_NW + asym_SE_NW;

         const real_t velXPZ = velX + velZ;
         const real_t  sym_TE_BW = lambda_e_scaled * ( dd_tmp_TE[x] + dd_tmp_BW[x] - fac2 * velXPZ * velXPZ - t2x2 * feq_common );
         const real_t asym_TE_BW = lambda_d_scaled * ( dd_tmp_TE[x] - dd_tmp_BW[x] - real_t( 3.0 ) * t2x2 * velXPZ );
         dst_TE[x] = dd_tmp_TE[x] - sym_TE_BW - asym_TE_BW;
         dst_BW[x] = dd_tmp_BW[x] - sym_TE_BW + asym_TE_BW;

         const real_t velXMZ = velX - velZ;
         const real_t  sym_BE_TW = lambda_e_scaled * ( dd_tmp_BE[x] + dd_tmp_TW[x] - fac2 * velXMZ * velXMZ - t2x2 * feq_common );
         const real_t asym_BE_TW = lambda_d_scaled * ( dd_tmp_BE[x] - dd_tmp_TW[x] - real_t( 3.0 ) * t2x2 * velXMZ );
         dst_BE[x] = dd_tmp_BE[x] - sym_BE_TW - asym_BE_TW;
         dst_TW[x] = dd_tmp_TW[x] - sym_BE_TW + asym_BE_TW;

         const real_t velYPZ = velY + velZ;
         const real_t  sym_TN_BS = lambda_e_scaled * ( dd_tmp_TN[x] + dd_tmp_BS[x] - fac2 * velYPZ * velYPZ - t2x2 * feq_common );
         const real_t asym_TN_BS = lambda_d_scaled * ( dd_tmp_TN[x] - dd_tmp_BS[x] - real_t( 3.0 ) * t2x2 * velYPZ );
         dst_TN[x] = dd_tmp_TN[x] - sym_TN_BS - asym_TN_BS;
         dst_BS[x] = dd_tmp_BS[x] - sym_TN_BS + asym_TN_BS;

         const real_t velYMZ = velY - velZ;
         const real_t  sym_BN_TS = lambda_e_scaled * ( dd_tmp_BN[x] + dd_tmp_TS[x] - fac2 * velYMZ * velYMZ - t2x2 * feq_common );
         const real_t asym_BN_TS = lambda_d_scaled * ( dd_tmp_BN[x] - dd_tmp_TS[x] - real_t( 3.0 ) * t2x2 * velYMZ );
         dst_BN[x] = dd_tmp_BN[x] - sym_BN_TS - asym_BN_TS;
         dst_TS[x] = dd_tmp_TS[x] - sym_BN_TS + asym_BN_TS;

         const real_t  sym_N_S = lambda_e_scaled * ( dd_tmp_N[x] + dd_tmp_S[x] - fac1 * velY * velY - t1x2 * feq_common );
         const real_t asym_N_S = lambda_d_scaled * ( dd_tmp_N[x] - dd_tmp_S[x] - real_t( 3.0 ) * t1x2 * velY );
         dst_N[x] = dd_tmp_N[x] - sym_N_S - asym_N_S;
         dst_S[x] = dd_tmp_S[x] - sym_N_S + asym_N_S;

         const real_t  sym_E_W = lambda_e_scaled * ( dd_tmp_E[x] + dd_tmp_W[x] - fac1 * velX * velX - t1x2 * feq_common );
         const real_t asym_E_W = lambda_d_scaled * ( dd_tmp_E[x] - dd_tmp_W[x] - real_t( 3.0 ) * t1x2 * velX );
         dst_E[x] = dd_tmp_E[x] - sym_E_W - asym_E_W;
         dst_W[x] = dd_tmp_W[x] - sym_E_W + asym_E_W;

         const real_t  sym_T_B = lambda_e_scaled * ( dd_tmp_T[x] + dd_tmp_B[x] - fac1 * velZ * velZ - t1x2 * feq_common );
         const real_t asym_T_B = lambda_d_scaled * ( dd_tmp_T[x] - dd_tmp_B[x] - real_t( 3.0 ) * t1x2 * velZ );
         dst_T[x] = dd_tmp_T[x] - sym_T_B - asym_T_B;
         dst_B[x] = dd_tmp_B[x] - sym_T_B + asym_T_B;
      }
      ++rleBegin;
      ++rleEnd;
   }
   
   list->swapTmpPdfs();
}




template< typename List_T >
class ListSplitTRTSweep
{
   static_assert( ( std::is_same< typename List_T::LatticeModel::CollisionModel::tag, collision_model::TRT_tag >::value ), "Only works with TRT!" );
   static_assert( ( std::is_same< typename List_T::LatticeModel::Stencil, stencil::D3Q19 >::value ), "Only works with D3Q19!" );
   static_assert( !List_T::LatticeModel::compressible, "Only works with incompressible models!" );
   static_assert( ( std::is_same< typename List_T::LatticeModel::ForceModel::tag, force_model::None_tag >::value ), "Only works without additional forces!" );
   static_assert( List_T::LatticeModel::equilibriumAccuracyOrder == 2, "Only works for lattice models that require the equilibrium distribution to be order 2 accurate!" );

public:
   ListSplitTRTSweep( const BlockDataID listId ) : listId_( listId ) { }
   void operator()( IBlock * const block );

protected:
   BlockDataID listId_;
};

template< typename List_T >
void ListSplitTRTSweep< List_T >::operator()( IBlock * const block )
{
   List_T * list = block->getData<List_T>( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( list )

   // constants used during stream/collide

   const real_t lambda_e = list->latticeModel().collisionModel().lambda_e();
   const real_t lambda_d = list->latticeModel().collisionModel().lambda_d();

   // common prefactors for calculating the equilibrium parts
   const real_t t0 = real_t( 1.0 ) / real_t( 3.0 );                   // 1/3      for C
   const real_t t1x2 = real_t( 1.0 ) / real_t( 18.0 ) * real_t( 2.0 );  // 1/18 * 2 for N, S, W, E, T, B
   const real_t t2x2 = real_t( 1.0 ) / real_t( 36.0 ) * real_t( 2.0 );  // 1/36 * 2 else

   const real_t inv2csq2 = real_t( 1.0 ) / ( real_t( 2.0 ) * ( real_t( 1.0 ) / real_t( 3.0 ) ) * ( real_t( 1.0 ) / real_t( 3.0 ) ) ); //speed of sound related factor for equilibrium distribution function
   const real_t fac1 = t1x2 * inv2csq2;
   const real_t fac2 = t2x2 * inv2csq2;

   // relaxation parameter variables
   const real_t lambda_e_scaled = real_t( 0.5 ) * lambda_e; // 0.5 times the usual value ...
   const real_t lambda_d_scaled = real_t( 0.5 ) * lambda_d; // ... due to the way of calculations

   // stream & collide

   const uint_t maxBlockLength = list->getRLEMaxLength();

   real_t * WALBERLA_RESTRICT velX = new real_t[ maxBlockLength ];
   real_t * WALBERLA_RESTRICT velY = new real_t[ maxBlockLength ];
   real_t * WALBERLA_RESTRICT velZ = new real_t[ maxBlockLength ];

   real_t * WALBERLA_RESTRICT feq_common = new real_t[maxBlockLength];

   auto rleBegin = list->getRLEInfo().begin();
   auto rleEnd = list->getRLEInfo().begin() + 1;

   while( rleEnd != list->getRLEInfo().end() )
   {
      using namespace stencil;

      const real_t * WALBERLA_RESTRICT pNE = &( list->get( list->getPullIdx( *rleBegin, NE ) ) );
      const real_t * WALBERLA_RESTRICT pN  = &( list->get( list->getPullIdx( *rleBegin, N  ) ) );
      const real_t * WALBERLA_RESTRICT pNW = &( list->get( list->getPullIdx( *rleBegin, NW ) ) );
      const real_t * WALBERLA_RESTRICT pW  = &( list->get( list->getPullIdx( *rleBegin, W  ) ) );
      const real_t * WALBERLA_RESTRICT pSW = &( list->get( list->getPullIdx( *rleBegin, SW ) ) );
      const real_t * WALBERLA_RESTRICT pS  = &( list->get( list->getPullIdx( *rleBegin, S  ) ) );
      const real_t * WALBERLA_RESTRICT pSE = &( list->get( list->getPullIdx( *rleBegin, SE ) ) );
      const real_t * WALBERLA_RESTRICT pE  = &( list->get( list->getPullIdx( *rleBegin, E  ) ) );
      const real_t * WALBERLA_RESTRICT pT  = &( list->get( list->getPullIdx( *rleBegin, T  ) ) );
      const real_t * WALBERLA_RESTRICT pTE = &( list->get( list->getPullIdx( *rleBegin, TE ) ) );
      const real_t * WALBERLA_RESTRICT pTN = &( list->get( list->getPullIdx( *rleBegin, TN ) ) );
      const real_t * WALBERLA_RESTRICT pTW = &( list->get( list->getPullIdx( *rleBegin, TW ) ) );
      const real_t * WALBERLA_RESTRICT pTS = &( list->get( list->getPullIdx( *rleBegin, TS ) ) );
      const real_t * WALBERLA_RESTRICT pB  = &( list->get( list->getPullIdx( *rleBegin, B  ) ) );
      const real_t * WALBERLA_RESTRICT pBE = &( list->get( list->getPullIdx( *rleBegin, BE ) ) );
      const real_t * WALBERLA_RESTRICT pBN = &( list->get( list->getPullIdx( *rleBegin, BN ) ) );
      const real_t * WALBERLA_RESTRICT pBW = &( list->get( list->getPullIdx( *rleBegin, BW ) ) );
      const real_t * WALBERLA_RESTRICT pBS = &( list->get( list->getPullIdx( *rleBegin, BS ) ) );
      const real_t * WALBERLA_RESTRICT pC  = &( list->get( *rleBegin, C ) );

      real_t * WALBERLA_RESTRICT dC = &( list->getTmp( *rleBegin, C ) );

      const cell_idx_t xSize = cell_idx_c( *rleEnd - *rleBegin );

      X_LOOP
      (
         const real_t velX_trm = pE[x] + pNE[x] + pSE[x] + pTE[x] + pBE[x];
         const real_t velY_trm = pN[x] + pNW[x] + pTN[x] + pBN[x];
         const real_t velZ_trm = pT[x] + pTS[x] + pTW[x];

         const real_t rho = pC[x] + pS[x] + pW[x] + pB[x] + pSW[x] + pBS[x] + pBW[x] + velX_trm + velY_trm + velZ_trm;

         velX[x] = velX_trm - pW[x] - pNW[x] - pSW[x] - pTW[x] - pBW[x];
         velY[x] = velY_trm + pNE[x] - pS[x] - pSW[x] - pSE[x] - pTS[x] - pBS[x];
         velZ[x] = velZ_trm + pTN[x] + pTE[x] - pB[x] - pBN[x] - pBS[x] - pBW[x] - pBE[x];

         feq_common[x] = rho - real_t( 1.5 ) * ( velX[x] * velX[x] + velY[x] * velY[x] + velZ[x] * velZ[x] );

         dC[x] = pC[x] * ( real_t( 1.0 ) - lambda_e ) + lambda_e * t0 * feq_common[x];
      )

      real_t * WALBERLA_RESTRICT dNE = &( list->getTmp( *rleBegin, NE ) );
      real_t * WALBERLA_RESTRICT dSW = &( list->getTmp( *rleBegin, SW ) );

      X_LOOP
      (
         const real_t velXPY = velX[x] + velY[x];
         const real_t  sym_NE_SW = lambda_e_scaled * ( pNE[x] + pSW[x] - fac2 * velXPY * velXPY - t2x2 * feq_common[x] );
         const real_t asym_NE_SW = lambda_d_scaled * ( pNE[x] - pSW[x] - real_t(3.0) * t2x2 * velXPY );

         dNE[x] = pNE[x] - sym_NE_SW - asym_NE_SW;
         dSW[x] = pSW[x] - sym_NE_SW + asym_NE_SW;
      )

      real_t * WALBERLA_RESTRICT dSE = &( list->getTmp( *rleBegin, SE ) );
      real_t * WALBERLA_RESTRICT dNW = &( list->getTmp( *rleBegin, NW ) );

      X_LOOP
      (
         const real_t velXMY = velX[x] - velY[x];
         const real_t  sym_SE_NW = lambda_e_scaled * ( pSE[x] + pNW[x] - fac2 * velXMY * velXMY - t2x2 * feq_common[x] );
         const real_t asym_SE_NW = lambda_d_scaled * ( pSE[x] - pNW[x] - real_t(3.0) * t2x2 * velXMY );

         dSE[x] = pSE[x] - sym_SE_NW - asym_SE_NW;
         dNW[x] = pNW[x] - sym_SE_NW + asym_SE_NW;
      )

      real_t * WALBERLA_RESTRICT dTE = &( list->getTmp( *rleBegin, TE ) );
      real_t * WALBERLA_RESTRICT dBW = &( list->getTmp( *rleBegin, BW ) );

      X_LOOP
      (
         const real_t velXPZ = velX[x] + velZ[x];
         const real_t  sym_TE_BW = lambda_e_scaled * ( pTE[x] + pBW[x] - fac2 * velXPZ * velXPZ - t2x2 * feq_common[x] );
         const real_t asym_TE_BW = lambda_d_scaled * ( pTE[x] - pBW[x] - real_t(3.0) * t2x2 * velXPZ );

         dTE[x] = pTE[x] - sym_TE_BW - asym_TE_BW;
         dBW[x] = pBW[x] - sym_TE_BW + asym_TE_BW;
      )

      real_t * WALBERLA_RESTRICT dBE = &( list->getTmp( *rleBegin, BE ) );
      real_t * WALBERLA_RESTRICT dTW = &( list->getTmp( *rleBegin, TW ) );

      X_LOOP
      (
         const real_t velXMZ = velX[x] - velZ[x];
         const real_t  sym_BE_TW = lambda_e_scaled * ( pBE[x] + pTW[x] - fac2 * velXMZ * velXMZ - t2x2 * feq_common[x] );
         const real_t asym_BE_TW = lambda_d_scaled * ( pBE[x] - pTW[x] - real_t(3.0) * t2x2 * velXMZ );

         dBE[x] = pBE[x] - sym_BE_TW - asym_BE_TW;
         dTW[x] = pTW[x] - sym_BE_TW + asym_BE_TW;
      )

      real_t * WALBERLA_RESTRICT dTN = &( list->getTmp( *rleBegin, TN ) );
      real_t * WALBERLA_RESTRICT dBS = &( list->getTmp( *rleBegin, BS ) );

      X_LOOP
      (
         const real_t velYPZ = velY[x] + velZ[x];
         const real_t  sym_TN_BS = lambda_e_scaled * ( pTN[x] + pBS[x] - fac2 * velYPZ * velYPZ - t2x2 * feq_common[x] );
         const real_t asym_TN_BS = lambda_d_scaled * ( pTN[x] - pBS[x] - real_t(3.0) * t2x2 * velYPZ );

         dTN[x] = pTN[x] - sym_TN_BS - asym_TN_BS;
         dBS[x] = pBS[x] - sym_TN_BS + asym_TN_BS;
      )

      real_t * WALBERLA_RESTRICT dBN = &( list->getTmp( *rleBegin, BN ) );
      real_t * WALBERLA_RESTRICT dTS = &( list->getTmp( *rleBegin, TS ) );

      X_LOOP
      (
         const real_t velYMZ = velY[x] - velZ[x];
         const real_t  sym_BN_TS = lambda_e_scaled * ( pBN[x] + pTS[x] - fac2 * velYMZ * velYMZ - t2x2 * feq_common[x] );
         const real_t asym_BN_TS = lambda_d_scaled * ( pBN[x] - pTS[x] - real_t(3.0) * t2x2 * velYMZ );

         dBN[x] = pBN[x] - sym_BN_TS - asym_BN_TS;
         dTS[x] = pTS[x] - sym_BN_TS + asym_BN_TS;
      )

      real_t * WALBERLA_RESTRICT dN = &( list->getTmp( *rleBegin, N ) );
      real_t * WALBERLA_RESTRICT dS = &( list->getTmp( *rleBegin, S ) );

      X_LOOP
      (
         const real_t  sym_N_S = lambda_e_scaled * ( pN[x] + pS[x] - fac1 * velY[x] * velY[x] - t1x2 * feq_common[x] );
         const real_t asym_N_S = lambda_d_scaled * ( pN[x] - pS[x] - real_t(3.0) * t1x2 * velY[x] );

         dN[x] = pN[x] - sym_N_S - asym_N_S;
         dS[x] = pS[x] - sym_N_S + asym_N_S;
      )

      real_t * WALBERLA_RESTRICT dE = &( list->getTmp( *rleBegin, E ) );
      real_t * WALBERLA_RESTRICT dW = &( list->getTmp( *rleBegin, W ) );

      X_LOOP
      (
         const real_t  sym_E_W = lambda_e_scaled * ( pE[x] + pW[x] - fac1 * velX[x] * velX[x] - t1x2 * feq_common[x] );
         const real_t asym_E_W = lambda_d_scaled * ( pE[x] - pW[x] - real_t(3.0) * t1x2 * velX[x] );

         dE[x] = pE[x] - sym_E_W - asym_E_W;
         dW[x] = pW[x] - sym_E_W + asym_E_W;
      )

      real_t * WALBERLA_RESTRICT dT = &( list->getTmp( *rleBegin, T ) );
      real_t * WALBERLA_RESTRICT dB = &( list->getTmp( *rleBegin, B ) );

      X_LOOP
      (
         const real_t  sym_T_B = lambda_e_scaled * ( pT[x] + pB[x] - fac1 * velZ[x] * velZ[x] - t1x2 * feq_common[x] );
         const real_t asym_T_B = lambda_d_scaled * ( pT[x] - pB[x] - real_t(3.0) * t1x2 * velZ[x] );

         dT[x] = pT[x] - sym_T_B - asym_T_B;
         dB[x] = pB[x] - sym_T_B + asym_T_B;
      )

      ++rleBegin;
      ++rleEnd;
   }

   delete[] velX;
   delete[] velY;
   delete[] velZ;
   delete[] feq_common;

   list->swapTmpPdfs();
}

} // namespace lbm
} // namespace walberla