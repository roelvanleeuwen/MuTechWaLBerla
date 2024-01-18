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
//! \\file SYCLTestPackInfo.h
//! \\author pystencils
//======================================================================================================================

#pragma once

#include "core/DataTypes.h"
#include "core/cell/CellInterval.h"

#include "domain_decomposition/IBlock.h"

#include "stencil/Directions.h"

#include "gpu/GPUField.h"
#include "gpu/GPUWrapper.h"
#include "gpu/communication/GeneratedGPUPackInfo.h"

#include "CL/sycl.hpp"

#define FUNC_PREFIX __global__

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

namespace walberla {
namespace pystencils {


class SYCLTestPackInfo : public ::walberla::gpu::GeneratedGPUPackInfo
{
public:
    SYCLTestPackInfo( shared_ptr<sycl::queue> syclQueue_, BlockDataID pdfsID_ )
        : syclQueue(syclQueue_), pdfsID(pdfsID_)
    {};
    virtual ~SYCLTestPackInfo() {}

    void pack  (stencil::Direction dir, unsigned char * buffer, IBlock * block) override;
    void communicateLocal  ( stencil::Direction /*dir*/, const IBlock* /* sender */, IBlock* /* receiver */) override
    {
       WALBERLA_ABORT("Local Communication not implemented yet for standard PackInfos. To run your application turn of local communication in the Communication class")
    }
    void unpack(stencil::Direction dir, unsigned char * buffer, IBlock * block) override;
    uint_t size  (stencil::Direction dir, IBlock * block) override;

    void pack_SW(double * RESTRICT  _data_buffer, double * RESTRICT const _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void pack_W(double * RESTRICT  _data_buffer, double * RESTRICT const _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void pack_NW(double * RESTRICT  _data_buffer, double * RESTRICT const _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void pack_S(double * RESTRICT  _data_buffer, double * RESTRICT const _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void pack_N(double * RESTRICT  _data_buffer, double * RESTRICT const _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void pack_SE(double * RESTRICT  _data_buffer, double * RESTRICT const _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void pack_E(double * RESTRICT  _data_buffer, double * RESTRICT const _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void pack_NE(double * RESTRICT  _data_buffer, double * RESTRICT const _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void unpack_SW(double * RESTRICT const _data_buffer, double * RESTRICT  _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void unpack_W(double * RESTRICT const _data_buffer, double * RESTRICT  _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void unpack_NW(double * RESTRICT const _data_buffer, double * RESTRICT  _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void unpack_S(double * RESTRICT const _data_buffer, double * RESTRICT  _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void unpack_N(double * RESTRICT const _data_buffer, double * RESTRICT  _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void unpack_SE(double * RESTRICT const _data_buffer, double * RESTRICT  _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void unpack_E(double * RESTRICT const _data_buffer, double * RESTRICT  _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);
    void unpack_NE(double * RESTRICT const _data_buffer, double * RESTRICT  _data_pdfs, int64_t const _size_pdfs_0, int64_t const _size_pdfs_1, int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2);


  private:
    shared_ptr<sycl::queue> syclQueue;
    BlockDataID pdfsID;
};


} // namespace pystencils
} // namespace walberla