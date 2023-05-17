#pragma once
#include "stencil/Directions.h"
#include "core/cell/CellInterval.h"
#include "cuda/GPUField.h"
#include "core/DataTypes.h"
#include "domain_decomposition/IBlock.h"
#include "cuda/communication/GeneratedGPUPackInfo.h"

#include "ListLBMList.h"


{% if target is equalto 'cpu' -%}
#define FUNC_PREFIX
{%- elif target is equalto 'gpu' -%}
#define FUNC_PREFIX __global__
{%- endif %}

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

namespace walberla {
namespace {{namespace}} {


class {{class_name}} : public ::walberla::cuda::GeneratedGPUPackInfo
{
 public:
   {{class_name}}( {{fused_kernel|generate_constructor_parameters(parameters_to_ignore=['buffer'])}},  const BlockDataID & listId,
                    const Set<SUID> & sparseBlockSelectors = Set<SUID>::emptySet(), const Set<SUID> & denseBlockSelectors = Set<SUID>::emptySet() )
      : {{ fused_kernel|generate_constructor_initializer_list(parameters_to_ignore=['buffer']) }}, listId_( listId ), sparseBlockSelectors_( sparseBlockSelectors ), denseBlockSelectors_( denseBlockSelectors )
        {};
   virtual ~{{class_name}}() {}


   void pack (stencil::Direction dir, unsigned char * buffer, IBlock * block, cudaStream_t stream) override;
   void unpack(stencil::Direction dir, unsigned char * buffer, IBlock * block, cudaStream_t stream) override;
   //void communicateLocal( stencil::Direction dir, IBlock * sender, IBlock * receiver, cudaStream_t stream) override;
   uint_t size (stencil::Direction dir, IBlock * block) override;


 private:
   {{fused_kernel|generate_members(parameters_to_ignore=['buffer'])|indent(4)}}
   const BlockDataID listId_;
   Set<SUID> sparseBlockSelectors_;
   Set<SUID> denseBlockSelectors_;
};


} // namespace {{namespace}}
} // namespace walberla
