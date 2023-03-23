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

{% if target is equalto 'cpu' -%}
#include "lbm/list/CellDir.h"
#include "ListLBMList.h"
{%- elif target is equalto 'gpu' -%}
#include "lbm/list/CellDir.h"
#include "ListLBMList.h"
{%- endif %}
#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "blockforest/StructuredBlockForest.h"
#include "field/FlagField.h"
#include "core/debug/Debug.h"

#include <set>
#include <vector>

{%- for line in additional_headers %}
{{ line -}}
{% endfor %}

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

namespace walberla {
namespace {{namespace}} {


class {{class_name}}
{
public:
    {{StructDeclaration|indent(4)}}


    class IndexVectors
    {
    public:
        using CpuIndexVector = std::vector<{{StructName}}>;

        enum Type {
            ALL = 0,
            INNER = 1,
            OUTER = 2,
            NUM_TYPES = 3
        };

        IndexVectors() = default;
        bool operator==(IndexVectors & other) { return other.cpuVectors_ == cpuVectors_; }

        {% if target == 'gpu' -%}
        ~IndexVectors() {
            for( auto & gpuVec: gpuVectors_)
                gpuFree( gpuVec );
        }
        {% endif -%}

        CpuIndexVector & indexVector(Type t) { return cpuVectors_[t]; }
        {{StructName}} * pointerCpu(Type t)  { return &(cpuVectors_[t][0]); }

        {% if target == 'gpu' -%}
        {{StructName}} * pointerGpu(Type t)  { return gpuVectors_[t]; }
        {% endif -%}

        void syncGPU()
        {
            {% if target == 'gpu' -%}
            for( auto & gpuVec: gpuVectors_)
                gpuFree( gpuVec );
            gpuVectors_.resize( cpuVectors_.size() );

            WALBERLA_ASSERT_EQUAL(cpuVectors_.size(), NUM_TYPES);
            for(size_t i=0; i < cpuVectors_.size(); ++i )
            {
                auto & gpuVec = gpuVectors_[i];
                auto & cpuVec = cpuVectors_[i];
                gpuMalloc( &gpuVec, sizeof({{StructName}}) * cpuVec.size() );
                gpuMemcpy( gpuVec, &cpuVec[0], sizeof({{StructName}}) * cpuVec.size(), gpuMemcpyHostToDevice );
            }
            {%- endif %}
        }

    private:
        std::vector<CpuIndexVector> cpuVectors_{NUM_TYPES};

        {% if target == 'gpu' -%}
        using GpuIndexVector = {{StructName}} *;
        std::vector<GpuIndexVector> gpuVectors_;
        {%- endif %}
    };

    {{class_name}}( const shared_ptr<StructuredBlockForest> & blocks,
                   {{kernel|generate_constructor_parameters(['indexVector', 'indexVectorSize'])}}{{additional_data_handler.constructor_arguments}})
        :{{additional_data_handler.initialiser_list}} {{ kernel|generate_constructor_initializer_list(['indexVector', 'indexVectorSize']) }}
    {
        auto createIdxVector = []( IBlock * const , StructuredBlockStorage * const ) { return new IndexVectors(); };
        indexVectorID = blocks->addStructuredBlockData< IndexVectors >( createIdxVector, "IndexField_{{class_name}}");
    };

    void run (
        {{- ["IBlock * block", kernel.kernel_selection_parameters, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}}
    );

    {% if generate_functor -%}
    void operator() (
        {{- ["IBlock * block", kernel.kernel_selection_parameters, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}}
    )
    {
        run( {{- ["block", kernel.kernel_selection_parameters, ["stream"] if target == 'gpu' else []] | identifier_list -}} );
    }
    {%- endif %}

    void inner (
        {{- ["IBlock * block", kernel.kernel_selection_parameters, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}}
    );

    void outer (
        {{- ["IBlock * block", kernel.kernel_selection_parameters, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}}
    );

    std::function<void (IBlock *)> getSweep( {{- [interface_spec.high_level_args, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}} )
    {
        return [ {{- ["this", interface_spec.high_level_args, ["stream"] if target == 'gpu' else []] | identifier_list -}} ]
               (IBlock * b)
               { this->run( {{- [ ['b'], interface_spec.mapping_codes, ["stream"] if target == 'gpu' else [] ] | identifier_list -}} ); };
    }

    std::function<void (IBlock *)> getInnerSweep( {{- [interface_spec.high_level_args, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}} )
    {
        return [ {{- [ ['this'], interface_spec.high_level_args, ["stream"] if target == 'gpu' else [] ] | identifier_list -}} ]
               (IBlock * b)
               { this->inner( {{- [ ['b'], interface_spec.mapping_codes, ["stream"] if target == 'gpu' else [] ] | identifier_list -}} ); };
    }

    std::function<void (IBlock *)> getOuterSweep( {{- [interface_spec.high_level_args, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []] | type_identifier_list -}} )
    {
        return [ {{- [ ['this'], interface_spec.high_level_args, ["stream"] if target == 'gpu' else [] ] | identifier_list -}} ]
               (IBlock * b)
               { this->outer( {{- [ ['b'], interface_spec.mapping_codes, ["stream"] if target == 'gpu' else [] ] | identifier_list -}} ); };
    }

    template<typename FlagField_T>
    void fillFromFlagField( const shared_ptr<StructuredBlockForest> & blocks, ConstBlockDataID flagFieldID,
                            FlagUID boundaryFlagUID, FlagUID domainFlagUID)
    {
        for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt )
            fillFromFlagField<FlagField_T>({{additional_data_handler.additional_arguments_for_fill_function}}&*blockIt, flagFieldID, boundaryFlagUID, domainFlagUID );
    }


    template<typename FlagField_T>
    void fillFromFlagField({{additional_data_handler.additional_parameters_for_fill_function}}IBlock * block, ConstBlockDataID flagFieldID,
                            FlagUID boundaryFlagUID, FlagUID domainFlagUID )
    {
        auto * indexVectors = block->getData< IndexVectors > ( indexVectorID );
        auto & indexVectorAll = indexVectors->indexVector(IndexVectors::ALL);
        auto & indexVectorInner = indexVectors->indexVector(IndexVectors::INNER);
        auto & indexVectorOuter = indexVectors->indexVector(IndexVectors::OUTER);

        auto * flagField = block->getData< FlagField_T > ( flagFieldID );
        auto list = block->getData< lbmpy::ListLBMList >(listID);
        {{additional_data_handler.additional_field_data|indent(4)}}

        if( !(flagField->flagExists(boundaryFlagUID) && flagField->flagExists(domainFlagUID) ))
            return;

        auto boundaryFlag = flagField->getFlag(boundaryFlagUID);
        auto domainFlag = flagField->getFlag(domainFlagUID);

        auto inner = flagField->xyzSize();
        inner.expand( cell_idx_t(-1) );

        indexVectorAll.clear();
        indexVectorInner.clear();
        indexVectorOuter.clear();

        std::vector<lbm::CellDir> pdfsToRegister;
        int32_t startidx = int32_t(list->size());

        {{stencil_arrays[0]}}
        {{stencil_arrays[1]}}
        {{stencil_arrays[2]}}
        {{stencil_arrays[3]}}

        for (size_t f = 1; f < {{Q}}; ++f)
        {
           for( auto it = flagField->begin(); it != flagField->end(); ++it )
           {
              if( !isFlagSet(it, domainFlag) )
                 continue;

              if ( isFlagSet( it.neighbor(cx[f], cy[f], cz[f], 0), boundaryFlag ) )
              {
                 int32_t out = int32_t(list->getIdx( Cell(it.x(), it.y(), it.z())));
                 int32_t in = startidx;
                 startidx++;
                 auto element = IndexInfo(in, out,  int32_c(f) );
                 indexVectorAll.push_back( element );
                 pdfsToRegister.push_back(lbm::CellDir( Cell(it.x(), it.y(), it.z()) + Cell(cx[f], cy[f], cz[f]), inv_dir[f] ));

              }
           }
        }

        pdfsStartIdx_ = list->registerExternalPDFs( pdfsToRegister );
        numPDFs_ = pdfsToRegister.size();
        indexVectors->syncGPU();
    }

private:
    void run_impl(
        {{- ["IBlock * block", "IndexVectors::Type type",
             kernel.kernel_selection_parameters, ["gpuStream_t stream = nullptr"] if target == 'gpu' else []]
            | type_identifier_list -}}
   );

    BlockDataID indexVectorID;
    {{additional_data_handler.additional_member_variable|indent(4)}}
public:
  uint32_t pdfsStartIdx_;
  uint64_t numPDFs_;
  {{kernel|generate_members(('indexVector', 'indexVectorSize'))|indent(4)}}
};



} // namespace {{namespace}}
} // namespace walberla
