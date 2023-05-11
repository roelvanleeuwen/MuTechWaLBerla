#pragma once
#include "stencil/Directions.h"
#include "core/cell/CellInterval.h"
#include "core/DataTypes.h"
#include "field/GhostLayerField.h"
#include "domain_decomposition/IBlock.h"
#include "communication/UniformPackInfo.h"

#include "ListLBMList.h"

#define FUNC_PREFIX

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

namespace walberla {
namespace {{namespace}} {


class {{class_name}} : public ::walberla::communication::UniformPackInfo
{
public:
    {{class_name}}( {{fused_kernel|generate_constructor_parameters(parameters_to_ignore=['buffer'])}} ,
                     const BlockDataID & listId, const Set<SUID> & sparseBlockSelectors = Set<SUID>::emptySet(),
                     const Set<SUID> & denseBlockSelectors = Set<SUID>::emptySet() )
        : {{ fused_kernel|generate_constructor_initializer_list(parameters_to_ignore=['buffer']) }}, listId_(listId), sparseBlockSelectors_( sparseBlockSelectors ), denseBlockSelectors_( denseBlockSelectors )
    {};
    virtual ~{{class_name}}() {}

   bool constantDataExchange() const { return true; } //TODO
   bool threadsafeReceiving()  const { return true; } //TODO

   void unpackData(IBlock * receiver, stencil::Direction dir, mpi::RecvBuffer & buffer) {
      auto blockState = receiver->getState();
      if (blockState == sparseBlockSelectors_) {
         const auto dataSize = sizeSparse(dir, receiver);
         unpackSparse(dir, buffer.skip(dataSize), receiver);
      } else if (blockState == denseBlockSelectors_) {
         const auto dataSize = sizeDense(dir, receiver);
         unpackDense(dir, buffer.skip(dataSize), receiver);
      } else {
        WALBERLA_ABORT("Block selector is neither Sparse nor Dense, it is " << blockState)
      }
   }

   void communicateLocal(const IBlock * sender, IBlock * receiver, stencil::Direction dir) {
       mpi::SendBuffer sBuffer;
       packData( sender, dir, sBuffer );
       mpi::RecvBuffer rBuffer( sBuffer );
       unpackData( receiver, stencil::inverseDir[dir], rBuffer );
   }

   void packDataImpl(const IBlock * sender, stencil::Direction dir, mpi::SendBuffer & outBuffer) const {
       auto blockState = sender->getState();
       if (blockState == sparseBlockSelectors_) {
          const auto dataSize = sizeSparse(dir, sender);
          packSparse(dir, outBuffer.forward(dataSize), const_cast<IBlock*>(sender));
       } else if (blockState == denseBlockSelectors_) {
          const auto dataSize = sizeDense(dir, sender);
          packDense(dir, outBuffer.forward(dataSize), const_cast<IBlock*>(sender));
       } else {
          WALBERLA_ABORT("Block selector is neither Sparse nor Dense, it is " << blockState)
       }

   }

   void packSparse  (stencil::Direction dir, unsigned char * buffer, IBlock * block) const;
   void unpackSparse(stencil::Direction dir, unsigned char * buffer, IBlock * block) const;
   uint_t sizeSparse  (stencil::Direction dir, const IBlock * block) const;

   void packDense  (stencil::Direction dir, unsigned char * buffer, IBlock * block) const;
   void unpackDense (stencil::Direction dir, unsigned char * buffer, IBlock * block) const;
   uint_t sizeDense  (stencil::Direction dir, const IBlock * block) const;

 private:
    {{fused_kernel|generate_members(parameters_to_ignore=['buffer'])|indent(4)}}
    const BlockDataID listId_;
    Set<SUID> sparseBlockSelectors_;
    Set<SUID> denseBlockSelectors_;
};


} // namespace {{namespace}}
} // namespace walberla
