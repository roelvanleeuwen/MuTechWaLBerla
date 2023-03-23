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
//! \\file {{class_name}}.h
//! \\author pystencils
//======================================================================================================================

#pragma once
#include "core/DataTypes.h"
#include "blockforest/StructuredBlockForest.h"
#include "lbm/inplace_streaming/TimestepTracker.h"

{% if target is equalto 'cpu' -%}
#include "ListLBMList.h"
{%- elif target is equalto 'gpu' -%}
#include "ListLBMList.h"
#include "cuda/ErrorChecking.h"
{% if inner_outer_split -%}
#include "cuda/ParallelStreams.h"
{%- endif %}
{%- endif %}
#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "domain_decomposition/StructuredBlockStorage.h"
#include <set>

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wunused-parameter"
#   pragma GCC diagnostic ignored "-Wreorder"
#endif

namespace walberla {
namespace {{namespace}} {

   class {{class_name}}
   {
    public:
      {{class_name}}({{kernel|generate_constructor_parameters}})
         : {{ kernel|generate_constructor_initializer_list }}
      {};
      {{ kernel| generate_destructor(class_name) |indent(4) }}

      void run( {{- ["IBlock * block", kernel.kernel_selection_parameters, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}} );

      void operator() ({{- ["IBlock * block", kernel.kernel_selection_parameters, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}})
      {
         run( {{- ["block", kernel.kernel_selection_parameters, ["stream"] if target == 'gpu' else []] | identifier_list -}} );
      }

      static std::function<void (IBlock *)> getSweep({{- ["const shared_ptr<" + class_name + "> & kernel", kernel.kernel_selection_parameters] | type_identifier_list -}})
      {
         return [ {{- [ ['kernel'], kernel.kernel_selection_parameters ] | identifier_list -}} ]
            (IBlock * b)
         { kernel->run( {{- [ ['b'], kernel.kernel_selection_parameters] | identifier_list -}} ); };
      }

      std::function<void (IBlock *)> getSweep( {{- [interface_spec.high_level_args, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}} )
      {
         return [ {{- ["this", interface_spec.high_level_args, ["stream"] if target == 'gpu' else []] | identifier_list -}} ]
            (IBlock * b)
         { this->run( {{- [ ['b'], interface_spec.mapping_codes, ["stream"] if target == 'gpu' else [] ] | identifier_list -}} ); };
      }

      void inner( {{- ["IBlock * block", kernel.kernel_selection_parameters, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}} );
      void outer( {{- ["IBlock * block", kernel.kernel_selection_parameters, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}} );

      std::function<void (IBlock *)> getInnerSweep( {{- [interface_spec.high_level_args, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}} )
      {
         return [ {{- ["this", interface_spec.high_level_args, ["stream"] if target == 'gpu' else []] | identifier_list -}} ]
            (IBlock * b)
         { this->inner( {{- [ ['b'], interface_spec.mapping_codes, ["stream"] if target == 'gpu' else [] ] | identifier_list -}} ); };
      }

      std::function<void (IBlock *)> getOuterSweep( {{- [interface_spec.high_level_args, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}} )
      {
         return [ {{- ["this", interface_spec.high_level_args, ["stream"] if target == 'gpu' else []] | identifier_list -}} ]
            (IBlock * b)
         { this->outer( {{- [ ['b'], interface_spec.mapping_codes, ["stream"] if target == 'gpu' else [] ] | identifier_list -}} ); };
      }

    private:
      {{kernel|generate_members|indent(4)}}

   };

} // namespace {{namespace}}
} // namespace walberla


#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic pop
#endif

