#pragma once

#include "blockforest/StructuredBlockForest.h"
#include "stencil/Directions.h"
#include "core/cell/CellInterval.h"
#include "core/mpi/BufferSystem.h"
#include "core/DataTypes.h"
#include "domain_decomposition/IBlock.h"
#include "cuda/communication/GeneratedGPUPackInfo.h"

#include "ListLBMList.h"
#include "stencil/D3Q19.h"

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

namespace walberla {
namespace {{namespace}} {

class {{class_name}} : public walberla::cuda::GeneratedGPUPackInfo
{
 public:
   typedef uint32_t                       index_t;

   {{class_name}}( const BlockDataID & listId)
      : listId_( listId )
   {
   }

   virtual ~{{class_name}}() = default;

   void beforeStartCommunication() override {}

   void communicateLocal( stencil::Direction dir, IBlock * sender, IBlock * receiver, gpuStream_t stream) override;

   void pack  (stencil::Direction dir, unsigned char * buffer, IBlock * block, gpuStream_t stream) override;
   void unpack(stencil::Direction dir, unsigned char * buffer, IBlock * block, gpuStream_t stream) override;
   uint_t size  (stencil::Direction dir, IBlock * block) override;

   const BlockDataID listId_;

};


} // namespace {{namespace}}
} // namespace walberla
