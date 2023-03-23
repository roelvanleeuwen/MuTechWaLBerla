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
//! \file ListBGQKernel.h
//! \ingroup lbm
//! \author Christian Godenschwager <christian.godenschwager@fau.de>
//
//======================================================================================================================
// TODO delete, old and works only with Intel and AMD. Only here for reference
#pragma once

#ifdef __bg__

#include "core/Macros.h"
#include "core/perf_analysis/BGQPerfCtr.h"
#include "core/perf_analysis/BGQL1P.h"

#include "domain_decomposition/IBlock.h"
#include "domain_decomposition/BlockDataID.h"

#include "lbm/IntelCompilerOptimization.h"
#include "lbm/list/List.h"
#include "lbm/lattice_model/D3Q19.h"
#include "lbm/lattice_model/CollisionModel.h"

#include "simd/AlignedAllocator.h"

#ifdef __IBMCPP__
#include <builtins.h>
#endif

namespace walberla {
namespace lbm {

template< typename List_T >
class ListSplitBGQSIMDTRTSweep
{
   static_assert( ( boost::is_same< typename List_T::LatticeModel::CollisionModel::tag, collision_model::TRT_tag >::value ), "Only works with TRT!" );
   static_assert( ( boost::is_same< typename List_T::LatticeModel::Stencil, stencil::D3Q19 >::value ), "Only works with D3Q19!" );
   static_assert( !List_T::LatticeModel::compressible, "Only works with incompressible models!" );
   static_assert( ( boost::is_same< typename List_T::LatticeModel::ForceModel::tag, force_model::None_tag >::value ), "Only works without additional forces!" );
   static_assert( List_T::LatticeModel::equilibriumAccuracyOrder == 2, "Only works for lattice models that require the equilibrium distribution to be order 2 accurate!" );

   static const uint_t INITIAL_PATTERN_LENGTH = 8192;
public:
   ListSplitBGQSIMDTRTSweep( const BlockDataID listId ) : listId_( listId )
   {
   }

   ~ListSplitBGQSIMDTRTSweep()
   {
   }

   void operator()( IBlock * const block );

protected:
   BlockDataID listId_;
   std::vector< bgq::L1PPatternStore< real_t * > > patterns_;
};


} // namespace lbm
} // namespace walberla

#endif // __bg__
