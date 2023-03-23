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
//! \\file UBB.cpp
//! \\author pystencils
//======================================================================================================================

#include <cmath>

#include "core/Macros.h"
#include "UBB.h"



#define FUNC_PREFIX

using namespace std;

namespace walberla {
namespace lbmpy {

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wconversion"
#endif

#ifdef __CUDACC__
#pragma push
#pragma nv_diag_suppress = declared_but_not_referenced
#endif

namespace internal_ubb_even {
static FUNC_PREFIX void ubb_even(uint8_t * RESTRICT const _data_indexVector, double * RESTRICT  _data_pdf_field, int64_t const _stride_pdf_field_0, int64_t const _stride_pdf_field_1, int64_t indexVectorSize, double u_x)
{
   
   const double weights [] = {0.29629629629629630, 0.074074074074074074, 0.074074074074074074, 0.074074074074074074, 0.074074074074074074, 0.074074074074074074, 0.074074074074074074, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.0046296296296296296, 0.0046296296296296296, 0.0046296296296296296, 0.0046296296296296296, 0.0046296296296296296, 0.0046296296296296296, 0.0046296296296296296, 0.0046296296296296296};
   
   
   
   const int32_t neighbour_offset_x [] = { 0,0,0,-1,1,0,0,-1,1,-1,1,0,0,-1,1,0,0,-1,1,1,-1,1,-1,1,-1,1,-1 }; 
   const int32_t neighbour_offset_y [] = { 0,1,-1,0,0,0,0,1,1,-1,-1,1,-1,0,0,1,-1,0,0,1,1,-1,-1,1,1,-1,-1 }; 
   const int32_t neighbour_offset_z [] = { 0,0,0,0,0,1,-1,0,0,0,0,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1 }; 
   
   for (int64_t ctr_0 = 0; ctr_0 < indexVectorSize; ctr_0 += 1)
   {
      const int64_t in = *((int64_t * )(& _data_indexVector[24*ctr_0]));
      const int64_t dir = *((int64_t * )(& _data_indexVector[24*ctr_0 + 16]));
      const double vel0Term = _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 13*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 17*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 20*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 22*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 24*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 26*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 3*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 7*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 9*_stride_pdf_field_1];
      const double vel1Term = _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 10*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 12*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 16*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 2*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 21*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 25*_stride_pdf_field_1];
      const double vel2Term = _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 15*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 18*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 23*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 6*_stride_pdf_field_1];
      const double delta_rho = vel0Term + vel1Term + vel2Term + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 11*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 14*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 19*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 4*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 5*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 8*_stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + _stride_pdf_field_1] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8]))];
      const double rho = delta_rho + 1.0;
      _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0]))] = rho*u_x*-6.0*((double)(neighbour_offset_x[dir]))*weights[dir] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + _stride_pdf_field_1*dir];
   }
}
}

namespace internal_ubb_odd {
static FUNC_PREFIX void ubb_odd(uint32_t * RESTRICT const _data_idx, uint8_t * RESTRICT const _data_indexVector, double * RESTRICT  _data_pdf_field, int64_t const _stride_idx_0, int64_t const _stride_idx_1, int64_t const _stride_pdf_field_0, int64_t const _stride_pdf_field_1, int64_t indexVectorSize, double u_x)
{
   
   const double weights [] = {0.29629629629629630, 0.074074074074074074, 0.074074074074074074, 0.074074074074074074, 0.074074074074074074, 0.074074074074074074, 0.074074074074074074, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.018518518518518519, 0.0046296296296296296, 0.0046296296296296296, 0.0046296296296296296, 0.0046296296296296296, 0.0046296296296296296, 0.0046296296296296296, 0.0046296296296296296, 0.0046296296296296296};
   
   
   
   const int32_t neighbour_offset_x [] = { 0,0,0,-1,1,0,0,-1,1,-1,1,0,0,-1,1,0,0,-1,1,1,-1,1,-1,1,-1,1,-1 }; 
   const int32_t neighbour_offset_y [] = { 0,1,-1,0,0,0,0,1,1,-1,-1,1,-1,0,0,1,-1,0,0,1,1,-1,-1,1,1,-1,-1 }; 
   const int32_t neighbour_offset_z [] = { 0,0,0,0,0,1,-1,0,0,0,0,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1 }; 
   
   for (int64_t ctr_0 = 0; ctr_0 < indexVectorSize; ctr_0 += 1)
   {
      const int64_t in = *((int64_t * )(& _data_indexVector[24*ctr_0]));
      const int64_t dir = *((int64_t * )(& _data_indexVector[24*ctr_0 + 16]));
      const double vel0Term = _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 13*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 17*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 18*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 20*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 22*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 24*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 3*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 7*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 9*_stride_idx_1]];
      const double vel1Term = _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 10*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 14*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 19*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 23*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 6*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8]))]];
      const double vel2Term = _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 11*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 12*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 21*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 4*_stride_idx_1]];
      const double delta_rho = vel0Term + vel1Term + vel2Term + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 15*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 16*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 2*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 25*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 5*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + 8*_stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + _stride_idx_1]] + _data_pdf_field[_stride_pdf_field_0*_data_idx[_stride_idx_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) - _stride_idx_1]];
      const double rho = delta_rho + 1.0;
      _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0 + 8])) + _stride_pdf_field_1*dir] = rho*u_x*-6.0*((double)(neighbour_offset_x[dir]))*weights[dir] + _data_pdf_field[_stride_pdf_field_0**((int64_t * )(& _data_indexVector[24*ctr_0]))];
   }
}
}


#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#ifdef __CUDACC__
#pragma pop
#endif


void UBB::run_impl(IBlock * block, IndexVectors::Type type, uint8_t timestep)
{
   auto * indexVectors = block->getData<IndexVectors>(indexVectorID);
   int64_t indexVectorSize = int64_c( indexVectors->indexVector(type).size() );
   if( indexVectorSize == 0)
      return;

   
   auto pointer = indexVectors->pointerCpu(type);
   

   uint8_t * _data_indexVector = reinterpret_cast<uint8_t*>(pointer);
   auto list = block->getData< lbmpy::ListLBMList >(listID);

   auto & u_x = u_x_;
   uint32_t * RESTRICT const _data_idx = list->getidxbeginning();
    double * RESTRICT  _data_pdf_field = list->getPDFbegining();
    const int64_t _stride_idx_0 = int64_t(list->fStride());
    const int64_t _stride_idx_1 = int64_t(1 * int64_t(list->xStride()));
    const int64_t _stride_pdf_field_0 = int64_t(list->fStride());
    const int64_t _stride_pdf_field_1 = int64_t(1 * int64_t(list->xStride()));
    if(((timestep & 1) ^ 1)) {
        internal_ubb_even::ubb_even(_data_indexVector, _data_pdf_field, _stride_pdf_field_0, _stride_pdf_field_1, indexVectorSize, u_x);
    } else {
        internal_ubb_odd::ubb_odd(_data_idx, _data_indexVector, _data_pdf_field, _stride_idx_0, _stride_idx_1, _stride_pdf_field_0, _stride_pdf_field_1, indexVectorSize, u_x);
    }
}

void UBB::run(IBlock * block, uint8_t timestep)
{
   run_impl(block, IndexVectors::ALL, timestep);
}

void UBB::inner(IBlock * block, uint8_t timestep)
{
   run_impl(block, IndexVectors::INNER, timestep);
}

void UBB::outer(IBlock * block, uint8_t timestep)
{
   run_impl(block, IndexVectors::OUTER, timestep);
}

} // namespace lbmpy
} // namespace walberla

