#pragma once
#include "core/DataTypes.h"
#include "core/logging/Logging.h"

#include "gpu/FieldIndexing.h"

#include "field/SwapableCompare.h"
#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "domain_decomposition/StructuredBlockStorage.h"

#if defined(WALBERLA_BUILD_WITH_SYCL)
#include "CL/sycl.hpp"
#endif

namespace walberla {

class GameOfLifeSweepCUDA
{
 public:
#if defined(WALBERLA_BUILD_WITH_SYCL)
   GameOfLifeSweepCUDA( BlockDataID gpuFieldSrcID, BlockDataID gpuFieldDstID, shared_ptr<sycl::queue> syclQueue )
      : gpuFieldSrcID_( gpuFieldSrcID ), gpuFieldDstID_( gpuFieldDstID ), syclQueue_(syclQueue) {}
#else
   GameOfLifeSweepCUDA( BlockDataID gpuFieldSrcID, BlockDataID gpuFieldDstID )
      : gpuFieldSrcID_( gpuFieldSrcID ), gpuFieldDstID_( gpuFieldDstID ){}
#endif

   void operator() ( IBlock * block );

 private:
   BlockDataID gpuFieldSrcID_;
   BlockDataID gpuFieldDstID_;
#if defined(WALBERLA_BUILD_WITH_SYCL)
   shared_ptr<sycl::queue> syclQueue_;
#endif
};


#if defined(WALBERLA_BUILD_WITH_SYCL)
void GameOfLifeSweepCUDA::operator()(IBlock * block)
{
   auto srcCudaField = block->getData< gpu::GPUField<real_t> > ( gpuFieldSrcID_ );
   auto dstCudaField = block->getData< gpu::GPUField<real_t> > ( gpuFieldDstID_ );

   double * srcCudaFieldData = srcCudaField->dataAt(-1, -1, -1, 0);
   double * dstCudaFieldData= dstCudaField->dataAt(-1, -1, -1, 0);
   const size_t size_x = srcCudaField->xSize() ;//+ 2;
   const size_t size_y = srcCudaField->ySize() ;//+ 2;
   const size_t size_z = srcCudaField->zSize() ;//+ 2;
   const size_t stride_x = size_t(srcCudaField->xStride());
   const size_t stride_y = size_t(srcCudaField->yStride());
   const size_t stride_z = size_t(srcCudaField->zStride());

   sycl::range global(size_x, size_y, size_z);
   try
   {
      (*syclQueue_).parallel_for(global, [=](cl::sycl::item< 3 > it) {
         auto x   = it.get_id(0) + 1;
         auto y   = it.get_id(1) + 1;
         auto z   = it.get_id(2) + 1;
         auto idx = x * stride_x + y * stride_y + z * stride_z;

         // Count number of living neighbors
         /*int liveNeighbors = 0;
         if (srcCudaFieldData[idx + stride_x] > 0.5) ++liveNeighbors;
         if (srcCudaFieldData[idx - stride_x] > 0.5) ++liveNeighbors;
         if (srcCudaFieldData[idx + stride_y] > 0.5) ++liveNeighbors;
         if (srcCudaFieldData[idx - stride_y] > 0.5) ++liveNeighbors;

         if (srcCudaFieldData[idx - stride_x - stride_y] > 0.5) ++liveNeighbors;
         if (srcCudaFieldData[idx - stride_x + stride_y] > 0.5) ++liveNeighbors;
         if (srcCudaFieldData[idx + stride_x - stride_y] > 0.5) ++liveNeighbors;
         if (srcCudaFieldData[idx + stride_x + stride_y] > 0.5) ++liveNeighbors;

         // cell dies because of under- or over-population
         if (liveNeighbors < 2 || liveNeighbors > 3)
            dstCudaFieldData[idx] = 0.0;
         else if (liveNeighbors == 3) // cell comes alive
            dstCudaFieldData[idx] = 1.0;
         else
            dstCudaFieldData[idx] = srcCudaFieldData[idx];*/
      });
   }
   catch (sycl::exception const& e)
   {
      std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
   }
   (*syclQueue_).wait();

   srcCudaField->swapDataPointers( dstCudaField );
}
#else
__global__ void gameOfLifeKernel(gpu::FieldAccessor<real_t> src, gpu::FieldAccessor<real_t> dst  );
#endif

} // namespace walberla
