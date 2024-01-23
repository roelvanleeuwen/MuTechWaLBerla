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
//! \\file SYCLTestSweep.cpp
//! \\author pystencils
//======================================================================================================================

#include <cmath>

#include "core/DataTypes.h"
#include "SYCLTestSweep.h"




#define FUNC_PREFIX __global__

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


void SYCLTestSweep::sycltestsweep_sycltestsweep(double * RESTRICT const _data_pdfs, double * RESTRICT  _data_pdfs_tmp, double * RESTRICT  _data_velocity, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2, int64_t const _stride_pdfs_tmp_0, int64_t const _stride_pdfs_tmp_1, int64_t const _stride_pdfs_tmp_2, int64_t const _stride_velocity_0, int64_t const _stride_velocity_1, int64_t const _stride_velocity_2, double omega)
{
   sycl::range global(_size_pdfs_1, _size_pdfs_0);
   try
   {
      (*syclQueue).parallel_for(global, [=](cl::sycl::item< 2 > it) {
         const int64_t ctr_0 = it.get_id(1) + 1;
         const int64_t ctr_1 = it.get_id(0) + 1;
         const double xi_3 = -_data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_0 + _stride_pdfs_1*ctr_1 + 3*_stride_pdfs_2];
         const double xi_4 = -_data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_0 + _stride_pdfs_1*ctr_1 - _stride_pdfs_1 + 5*_stride_pdfs_2];
         const double xi_5 = -_data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_0 + _stride_pdfs_1*ctr_1 + _stride_pdfs_1 + 7*_stride_pdfs_2];
         const double xi_7 = -_data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 + _stride_pdfs_1 + 2*_stride_pdfs_2];
         const double xi_8 = -_data_pdfs[_stride_pdfs_0*ctr_0 - _stride_pdfs_0 + _stride_pdfs_1*ctr_1 + _stride_pdfs_1 + 8*_stride_pdfs_2];
         const double vel0Term = _data_pdfs[_stride_pdfs_0*ctr_0 - _stride_pdfs_0 + _stride_pdfs_1*ctr_1 + 4*_stride_pdfs_2] + _data_pdfs[_stride_pdfs_0*ctr_0 - _stride_pdfs_0 + _stride_pdfs_1*ctr_1 + _stride_pdfs_1 + 8*_stride_pdfs_2] + _data_pdfs[_stride_pdfs_0*ctr_0 - _stride_pdfs_0 + _stride_pdfs_1*ctr_1 - _stride_pdfs_1 + 6*_stride_pdfs_2];
         const double xi_6 = vel0Term + xi_3 + xi_4 + xi_5;
         const double vel1Term = _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_0 + _stride_pdfs_1*ctr_1 - _stride_pdfs_1 + 5*_stride_pdfs_2] + _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 - _stride_pdfs_1 + _stride_pdfs_2];
         const double xi_9 = vel1Term + xi_5 + xi_7 + xi_8 + _data_pdfs[_stride_pdfs_0*ctr_0 - _stride_pdfs_0 + _stride_pdfs_1*ctr_1 - _stride_pdfs_1 + 6*_stride_pdfs_2];
         const double delta_rho = vel0Term + vel1Term + _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_0 + _stride_pdfs_1*ctr_1 + 3*_stride_pdfs_2] + _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_0 + _stride_pdfs_1*ctr_1 + _stride_pdfs_1 + 7*_stride_pdfs_2] + _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 + _stride_pdfs_1 + 2*_stride_pdfs_2] + _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1];
         const double u_0 = xi_6;
         const double xi_10 = u_0*u_0;
         const double xi_15 = u_0*0.33333333333333331;
         const double u_1 = xi_9;
         const double xi_11 = u_1*u_1;
         const double xi_12 = u_1*0.33333333333333331;
         const double momdensity_0 = xi_6;
         const double momdensity_1 = xi_9;
         const double u0Mu1 = u_0 - u_1;
         const double xi_17 = u0Mu1*0.083333333333333329;
         const double u0Pu1 = u_0 + u_1;
         const double xi_20 = u0Pu1*0.083333333333333329;
         const double f_eq_common = delta_rho + xi_10*-1.5 + xi_11*-1.5;
         const double xi_13 = f_eq_common*0.1111111111111111;
         const double xi_14 = xi_11*0.5 + xi_13;
         const double xi_16 = xi_10*0.5 + xi_13;
         const double xi_18 = f_eq_common*0.027777777777777776;
         const double xi_19 = xi_18 + 0.125*(u0Mu1*u0Mu1);
         const double xi_21 = xi_18 + 0.125*(u0Pu1*u0Pu1);
         _data_pdfs_tmp[_stride_pdfs_tmp_0*ctr_0 + _stride_pdfs_tmp_1*ctr_1] = omega*(f_eq_common*0.44444444444444442 - _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1]) + _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1];
         _data_pdfs_tmp[_stride_pdfs_tmp_0*ctr_0 + _stride_pdfs_tmp_1*ctr_1 + _stride_pdfs_tmp_2] = omega*(xi_12 + xi_14 - _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 - _stride_pdfs_1 + _stride_pdfs_2]) + _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 - _stride_pdfs_1 + _stride_pdfs_2];
         _data_pdfs_tmp[_stride_pdfs_tmp_0*ctr_0 + _stride_pdfs_tmp_1*ctr_1 + 2*_stride_pdfs_tmp_2] = omega*(-xi_12 + xi_14 + xi_7) + _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_1*ctr_1 + _stride_pdfs_1 + 2*_stride_pdfs_2];
         _data_pdfs_tmp[_stride_pdfs_tmp_0*ctr_0 + _stride_pdfs_tmp_1*ctr_1 + 3*_stride_pdfs_tmp_2] = omega*(-xi_15 + xi_16 + xi_3) + _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_0 + _stride_pdfs_1*ctr_1 + 3*_stride_pdfs_2];
         _data_pdfs_tmp[_stride_pdfs_tmp_0*ctr_0 + _stride_pdfs_tmp_1*ctr_1 + 4*_stride_pdfs_tmp_2] = omega*(xi_15 + xi_16 - _data_pdfs[_stride_pdfs_0*ctr_0 - _stride_pdfs_0 + _stride_pdfs_1*ctr_1 + 4*_stride_pdfs_2]) + _data_pdfs[_stride_pdfs_0*ctr_0 - _stride_pdfs_0 + _stride_pdfs_1*ctr_1 + 4*_stride_pdfs_2];
         _data_pdfs_tmp[_stride_pdfs_tmp_0*ctr_0 + _stride_pdfs_tmp_1*ctr_1 + 5*_stride_pdfs_tmp_2] = omega*(-xi_17 + xi_19 + xi_4) + _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_0 + _stride_pdfs_1*ctr_1 - _stride_pdfs_1 + 5*_stride_pdfs_2];
         _data_pdfs_tmp[_stride_pdfs_tmp_0*ctr_0 + _stride_pdfs_tmp_1*ctr_1 + 6*_stride_pdfs_tmp_2] = omega*(xi_20 + xi_21 - _data_pdfs[_stride_pdfs_0*ctr_0 - _stride_pdfs_0 + _stride_pdfs_1*ctr_1 - _stride_pdfs_1 + 6*_stride_pdfs_2]) + _data_pdfs[_stride_pdfs_0*ctr_0 - _stride_pdfs_0 + _stride_pdfs_1*ctr_1 - _stride_pdfs_1 + 6*_stride_pdfs_2];
         _data_pdfs_tmp[_stride_pdfs_tmp_0*ctr_0 + _stride_pdfs_tmp_1*ctr_1 + 7*_stride_pdfs_tmp_2] = omega*(-xi_20 + xi_21 + xi_5) + _data_pdfs[_stride_pdfs_0*ctr_0 + _stride_pdfs_0 + _stride_pdfs_1*ctr_1 + _stride_pdfs_1 + 7*_stride_pdfs_2];
         _data_pdfs_tmp[_stride_pdfs_tmp_0*ctr_0 + _stride_pdfs_tmp_1*ctr_1 + 8*_stride_pdfs_tmp_2] = omega*(xi_17 + xi_19 + xi_8) + _data_pdfs[_stride_pdfs_0*ctr_0 - _stride_pdfs_0 + _stride_pdfs_1*ctr_1 + _stride_pdfs_1 + 8*_stride_pdfs_2];
         _data_velocity[_stride_velocity_0*ctr_0 + _stride_velocity_1*ctr_1] = momdensity_0;
         _data_velocity[_stride_velocity_0*ctr_0 + _stride_velocity_1*ctr_1 + _stride_velocity_2] = momdensity_1;
      });
      (*syclQueue).wait();
   }
   catch (sycl::exception const& e)
   {
      std::cout << "In Sweep: Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
   }
}


void SYCLTestSweep::run(IBlock * block)
{
   WALBERLA_LOG_PROGRESS("Starting to run Sweep")
    auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);
    auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
    auto pdfs_tmp = block->getData< gpu::GPUField<double> >(pdfsTmpID);

    auto & omega = this->omega_;
    WALBERLA_ASSERT_GREATER_EQUAL(-1, -int_c(pdfs->nrOfGhostLayers()))
    double * RESTRICT const _data_pdfs = pdfs->dataAt(-1, -1, 0, 0);
    WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(-1, -int_c(pdfs_tmp->nrOfGhostLayers()))
    double * RESTRICT  _data_pdfs_tmp = pdfs_tmp->dataAt(-1, -1, 0, 0);
    WALBERLA_ASSERT_EQUAL(pdfs_tmp->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(-1, -int_c(velocity->nrOfGhostLayers()))
    double * RESTRICT  _data_velocity = velocity->dataAt(-1, -1, 0, 0);
    WALBERLA_ASSERT_EQUAL(velocity->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(pdfs->xSizeWithGhostLayer(), int64_t(int64_c(pdfs->xSize()) + 2))
    const int64_t _size_pdfs_0 = int64_t(int64_c(pdfs->xSize()) + 2);
    WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(pdfs->ySizeWithGhostLayer(), int64_t(int64_c(pdfs->ySize()) + 2))
    const int64_t _size_pdfs_1 = int64_t(int64_c(pdfs->ySize()) + 2);
    WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx)
    const int64_t _stride_pdfs_0 = int64_t(pdfs->xStride());
    const int64_t _stride_pdfs_1 = int64_t(pdfs->yStride());
    const int64_t _stride_pdfs_2 = int64_t(1 * int64_t(pdfs->fStride()));
    const int64_t _stride_pdfs_tmp_0 = int64_t(pdfs_tmp->xStride());
    const int64_t _stride_pdfs_tmp_1 = int64_t(pdfs_tmp->yStride());
    const int64_t _stride_pdfs_tmp_2 = int64_t(1 * int64_t(pdfs_tmp->fStride()));
    const int64_t _stride_velocity_0 = int64_t(velocity->xStride());
    const int64_t _stride_velocity_1 = int64_t(velocity->yStride());
    const int64_t _stride_velocity_2 = int64_t(1 * int64_t(velocity->fStride()));
    WALBERLA_LOG_PROGRESS("Finished setting all fields, Start kernel now")
    sycltestsweep_sycltestsweep(_data_pdfs, _data_pdfs_tmp, _data_velocity, _size_pdfs_0, _size_pdfs_1, _stride_pdfs_0, _stride_pdfs_1, _stride_pdfs_2, _stride_pdfs_tmp_0, _stride_pdfs_tmp_1, _stride_pdfs_tmp_2, _stride_velocity_0, _stride_velocity_1, _stride_velocity_2, omega);
    WALBERLA_LOG_PROGRESS("Finished kernel, Swap pointers")
    pdfs->swapDataPointers(pdfs_tmp);

}


void SYCLTestSweep::runOnCellInterval(const shared_ptr<StructuredBlockStorage> & blocks, const CellInterval & globalCellInterval, cell_idx_t ghostLayers, IBlock * block)
{
    CellInterval ci = globalCellInterval;
    CellInterval blockBB = blocks->getBlockCellBB( *block);
    blockBB.expand( ghostLayers );
    ci.intersect( blockBB );
    blocks->transformGlobalToBlockLocalCellInterval( ci, *block );
    if( ci.empty() )
        return;

    auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);
    auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
    auto pdfs_tmp = block->getData< gpu::GPUField<double> >(pdfsTmpID);


    auto & omega = this->omega_;
    WALBERLA_ASSERT_GREATER_EQUAL(ci.xMin() - 1, -int_c(pdfs->nrOfGhostLayers()))
    WALBERLA_ASSERT_GREATER_EQUAL(ci.yMin() - 1, -int_c(pdfs->nrOfGhostLayers()))
    WALBERLA_ASSERT_GREATER_EQUAL(ci.zMin() - 1, -int_c(pdfs->nrOfGhostLayers()))
    double * RESTRICT const _data_pdfs = pdfs->dataAt(ci.xMin() - 1, ci.yMin() - 1, ci.zMin() - 1, 0);
    WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(ci.xMin() - 1, -int_c(pdfs_tmp->nrOfGhostLayers()))
    WALBERLA_ASSERT_GREATER_EQUAL(ci.yMin() - 1, -int_c(pdfs_tmp->nrOfGhostLayers()))
    WALBERLA_ASSERT_GREATER_EQUAL(ci.zMin() - 1, -int_c(pdfs_tmp->nrOfGhostLayers()))
    double * RESTRICT  _data_pdfs_tmp = pdfs_tmp->dataAt(ci.xMin() - 1, ci.yMin() - 1, ci.zMin() - 1, 0);
    WALBERLA_ASSERT_EQUAL(pdfs_tmp->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(ci.xMin() - 1, -int_c(velocity->nrOfGhostLayers()))
    WALBERLA_ASSERT_GREATER_EQUAL(ci.yMin() - 1, -int_c(velocity->nrOfGhostLayers()))
    WALBERLA_ASSERT_GREATER_EQUAL(ci.zMin() - 1, -int_c(velocity->nrOfGhostLayers()))
    double * RESTRICT  _data_velocity = velocity->dataAt(ci.xMin() - 1, ci.yMin() - 1, ci.zMin() - 1, 0);
    WALBERLA_ASSERT_EQUAL(velocity->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(pdfs->xSizeWithGhostLayer(), int64_t(int64_c(ci.xSize()) + 2))
    const int64_t _size_pdfs_0 = int64_t(int64_c(ci.xSize()) + 2);
    WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx)
    WALBERLA_ASSERT_GREATER_EQUAL(pdfs->ySizeWithGhostLayer(), int64_t(int64_c(ci.ySize()) + 2))
    const int64_t _size_pdfs_1 = int64_t(int64_c(ci.ySize()) + 2);
    WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx)
    const int64_t _stride_pdfs_0 = int64_t(pdfs->xStride());
    const int64_t _stride_pdfs_1 = int64_t(pdfs->yStride());
    const int64_t _stride_pdfs_2 = int64_t(1 * int64_t(pdfs->fStride()));
    const int64_t _stride_pdfs_tmp_0 = int64_t(pdfs_tmp->xStride());
    const int64_t _stride_pdfs_tmp_1 = int64_t(pdfs_tmp->yStride());
    const int64_t _stride_pdfs_tmp_2 = int64_t(1 * int64_t(pdfs_tmp->fStride()));
    const int64_t _stride_velocity_0 = int64_t(velocity->xStride());
    const int64_t _stride_velocity_1 = int64_t(velocity->yStride());
    const int64_t _stride_velocity_2 = int64_t(1 * int64_t(velocity->fStride()));
    sycltestsweep_sycltestsweep(_data_pdfs, _data_pdfs_tmp, _data_velocity, _size_pdfs_0, _size_pdfs_1, _stride_pdfs_0, _stride_pdfs_1, _stride_pdfs_2, _stride_pdfs_tmp_0, _stride_pdfs_tmp_1, _stride_pdfs_tmp_2, _stride_velocity_0, _stride_velocity_1, _stride_velocity_2, omega);
    pdfs->swapDataPointers(pdfs_tmp);

}
} // namespace walberla


#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic pop
#endif

#if ( defined WALBERLA_CXX_COMPILER_IS_INTEL )
#pragma warning pop
#endif