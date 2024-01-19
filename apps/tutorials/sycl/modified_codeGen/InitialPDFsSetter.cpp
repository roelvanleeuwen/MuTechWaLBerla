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
//! \\file InitialPDFsSetter.cpp
//! \\author pystencils
//======================================================================================================================

#include <cmath>

#include "core/DataTypes.h"
#include "core/Macros.h"
#include "InitialPDFsSetter.h"




#define FUNC_PREFIX

#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wfloat-equal"
#   pragma GCC diagnostic ignored "-Wshadow"
#   pragma GCC diagnostic ignored "-Wconversion"
#   pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#if ( defined WALBERLA_CXX_COMPILER_IS_INTEL )
#pragma warning push
#pragma warning( disable :  1599 )
#endif

using namespace std;

namespace walberla {
namespace pystencils {


namespace internal_initialpdfssetter_initialpdfssetter {
static FUNC_PREFIX void initialpdfssetter_initialpdfssetter(double * RESTRICT  _data_pdfs, double * RESTRICT const _data_velocity, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2, int64_t const _stride_velocity_0, int64_t const _stride_velocity_1, int64_t const _stride_velocity_2, double rho_0)
{
   const double rho = rho_0;
   const double delta_rho = rho - 1.0;
   for (int64_t ctr_1 = 0; ctr_1 < _size_pdfs_1; ctr_1 += 1)
   {
      for (int64_t ctr_0 = 0; ctr_0 < _size_pdfs_0; ctr_0 += 1)
      {
         const double u_0 = _data_velocity[_stride_velocity_0*ctr_0 + _stride_velocity_1*ctr_1];
         const double u_1 = _data_velocity[_stride_velocity_0*ctr_0 + _stride_velocity_1*ctr_1 + _stride_velocity_2];
         _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1] = delta_rho*0.44444444444444442 - 0.66666666666666663*(u_0*u_0) - 0.66666666666666663*(u_1*u_1);
         _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 + _stride_pdfs_2] = delta_rho*0.1111111111111111 + u_1*0.33333333333333331 - 0.16666666666666666*(u_0*u_0) + 0.33333333333333331*(u_1*u_1);
         _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 + 2*_stride_pdfs_2] = delta_rho*0.1111111111111111 + u_1*-0.33333333333333331 - 0.16666666666666666*(u_0*u_0) + 0.33333333333333331*(u_1*u_1);
         _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 + 3*_stride_pdfs_2] = delta_rho*0.1111111111111111 + u_0*-0.33333333333333331 - 0.16666666666666666*(u_1*u_1) + 0.33333333333333331*(u_0*u_0);
         _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 + 4*_stride_pdfs_2] = delta_rho*0.1111111111111111 + u_0*0.33333333333333331 - 0.16666666666666666*(u_1*u_1) + 0.33333333333333331*(u_0*u_0);
         _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 + 5*_stride_pdfs_2] = delta_rho*0.027777777777777776 + u_0*u_1*-0.25 + u_0*-0.083333333333333329 + u_1*0.083333333333333329 + 0.083333333333333329*(u_0*u_0) + 0.083333333333333329*(u_1*u_1);
         _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 + 6*_stride_pdfs_2] = delta_rho*0.027777777777777776 + u_0*u_1*0.25 + u_0*0.083333333333333329 + u_1*0.083333333333333329 + 0.083333333333333329*(u_0*u_0) + 0.083333333333333329*(u_1*u_1);
         _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 + 7*_stride_pdfs_2] = delta_rho*0.027777777777777776 + u_0*u_1*0.25 + u_0*-0.083333333333333329 + u_1*-0.083333333333333329 + 0.083333333333333329*(u_0*u_0) + 0.083333333333333329*(u_1*u_1);
         _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 + 8*_stride_pdfs_2] = delta_rho*0.027777777777777776 + u_0*u_1*-0.25 + u_0*0.083333333333333329 + u_1*-0.083333333333333329 + 0.083333333333333329*(u_0*u_0) + 0.083333333333333329*(u_1*u_1);
      }
   }
}
}


void InitialPDFsSetter::run(IBlock * block)
{
    auto velocity = block->getData< field::GhostLayerField<double, 2> >(velocityID);
    auto pdfs = block->getData< field::GhostLayerField<double, 9> >(pdfsID);

    auto & rho_0 = this->rho_0_;
    WALBERLA_ASSERT_GREATER_EQUAL(0, -int_c(pdfs->nrOfGhostLayers()))
    double * RESTRICT  _data_pdfs = pdfs->dataAt(0, 0, 0, 0);
    WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(0, -int_c(velocity->nrOfGhostLayers()))
    double * RESTRICT const _data_velocity = velocity->dataAt(0, 0, 0, 0);
    WALBERLA_ASSERT_EQUAL(velocity->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(pdfs->xSizeWithGhostLayer(), int64_t(int64_c(pdfs->xSize()) + 0))
    const int64_t _size_pdfs_0 = int64_t(int64_c(pdfs->xSize()) + 0);
    WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(pdfs->ySizeWithGhostLayer(), int64_t(int64_c(pdfs->ySize()) + 0))
    const int64_t _size_pdfs_1 = int64_t(int64_c(pdfs->ySize()) + 0);
    WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx)
    const int64_t _stride_pdfs_0 = int64_t(pdfs->xStride());
    const int64_t _stride_pdfs_1 = int64_t(pdfs->yStride());
    const int64_t _stride_pdfs_2 = int64_t(1 * int64_t(pdfs->fStride()));
    const int64_t _stride_velocity_0 = int64_t(velocity->xStride());
    const int64_t _stride_velocity_1 = int64_t(velocity->yStride());
    const int64_t _stride_velocity_2 = int64_t(1 * int64_t(velocity->fStride()));
    internal_initialpdfssetter_initialpdfssetter::initialpdfssetter_initialpdfssetter(_data_pdfs, _data_velocity, _size_pdfs_0, _size_pdfs_1, _stride_pdfs_0, _stride_pdfs_1, _stride_pdfs_2, _stride_velocity_0, _stride_velocity_1, _stride_velocity_2, rho_0);
    
}


void InitialPDFsSetter::runOnCellInterval(const shared_ptr<StructuredBlockStorage> & blocks, const CellInterval & globalCellInterval, cell_idx_t ghostLayers, IBlock * block)
{
    CellInterval ci = globalCellInterval;
    CellInterval blockBB = blocks->getBlockCellBB( *block);
    blockBB.expand( ghostLayers );
    ci.intersect( blockBB );
    blocks->transformGlobalToBlockLocalCellInterval( ci, *block );
    if( ci.empty() )
        return;

    auto velocity = block->getData< field::GhostLayerField<double, 2> >(velocityID);
    auto pdfs = block->getData< field::GhostLayerField<double, 9> >(pdfsID);

    auto & rho_0 = this->rho_0_;
    WALBERLA_ASSERT_GREATER_EQUAL(ci.xMin(), -int_c(pdfs->nrOfGhostLayers()))
    WALBERLA_ASSERT_GREATER_EQUAL(ci.yMin(), -int_c(pdfs->nrOfGhostLayers()))
    WALBERLA_ASSERT_GREATER_EQUAL(ci.zMin(), -int_c(pdfs->nrOfGhostLayers()))
    double * RESTRICT  _data_pdfs = pdfs->dataAt(ci.xMin(), ci.yMin(), ci.zMin(), 0);
    WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(ci.xMin(), -int_c(velocity->nrOfGhostLayers()))
    WALBERLA_ASSERT_GREATER_EQUAL(ci.yMin(), -int_c(velocity->nrOfGhostLayers()))
    WALBERLA_ASSERT_GREATER_EQUAL(ci.zMin(), -int_c(velocity->nrOfGhostLayers()))
    double * RESTRICT const _data_velocity = velocity->dataAt(ci.xMin(), ci.yMin(), ci.zMin(), 0);
    WALBERLA_ASSERT_EQUAL(velocity->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(pdfs->xSizeWithGhostLayer(), int64_t(int64_c(ci.xSize()) + 0))
    const int64_t _size_pdfs_0 = int64_t(int64_c(ci.xSize()) + 0);
    WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(pdfs->ySizeWithGhostLayer(), int64_t(int64_c(ci.ySize()) + 0))
    const int64_t _size_pdfs_1 = int64_t(int64_c(ci.ySize()) + 0);
    WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx)
    const int64_t _stride_pdfs_0 = int64_t(pdfs->xStride());
    const int64_t _stride_pdfs_1 = int64_t(pdfs->yStride());
    const int64_t _stride_pdfs_2 = int64_t(1 * int64_t(pdfs->fStride()));
    const int64_t _stride_velocity_0 = int64_t(velocity->xStride());
    const int64_t _stride_velocity_1 = int64_t(velocity->yStride());
    const int64_t _stride_velocity_2 = int64_t(1 * int64_t(velocity->fStride()));
    internal_initialpdfssetter_initialpdfssetter::initialpdfssetter_initialpdfssetter(_data_pdfs, _data_velocity, _size_pdfs_0, _size_pdfs_1, _stride_pdfs_0, _stride_pdfs_1, _stride_pdfs_2, _stride_velocity_0, _stride_velocity_1, _stride_velocity_2, rho_0);
    
}



} // namespace pystencils
} // namespace walberla


#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic pop
#endif

#if ( defined WALBERLA_CXX_COMPILER_IS_INTEL )
#pragma warning pop
#endif