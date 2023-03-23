#pragma once

#include "blockforest/StructuredBlockForest.h"
#include "stencil/Directions.h"
#include "core/cell/CellInterval.h"
#include "core/mpi/BufferSystem.h"
#include "core/DataTypes.h"
#include "domain_decomposition/IBlock.h"
#include "communication/UniformPackInfo.h"

#include "ListLBMList.h"

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

namespace walberla {
namespace {{namespace}} {

class {{class_name}} : public walberla::communication::UniformPackInfo
{
 public:
   typedef uint32_t                       index_t;

   {{class_name}}( const BlockDataID & listId)
      : listId_( listId ), exchangeStructure_( true )
   {
   }

   virtual ~{{class_name}}() = default;

   bool constantDataExchange() const override { return !exchangeStructure_; }
   bool threadsafeReceiving()  const override { return !exchangeStructure_; }

   void unpackData(IBlock * receiver, stencil::Direction dir, mpi::RecvBuffer & buffer) override {
      const auto dataSize = size(dir, receiver);
      unpack(dir, buffer.skip(dataSize), receiver);
   }

   void packDataImpl(const IBlock * sender, stencil::Direction dir, mpi::SendBuffer & outBuffer) const override {
      const auto dataSize = size(dir, sender);
      pack(dir, outBuffer.forward(dataSize), const_cast<IBlock*>(sender));
   }

   void communicateLocal(const IBlock * sender, IBlock * receiver, stencil::Direction dir) override {
      //TODO: optimize by generating kernel for this case
      mpi::SendBuffer sBuffer;
      packData( sender, dir, sBuffer );
      mpi::RecvBuffer rBuffer( sBuffer );
      unpackData( receiver, stencil::inverseDir[dir], rBuffer );
   }



   void pack  (stencil::Direction dir, unsigned char * buffer, IBlock * block) const;
   void unpack(stencil::Direction dir, unsigned char * byte_buffer, IBlock * block) const;
   uint_t size  (stencil::Direction dir, const IBlock * block) const;


   const BlockDataID listId_;
   bool exchangeStructure_;

};


} // namespace {{namespace}}
} // namespace walberla
