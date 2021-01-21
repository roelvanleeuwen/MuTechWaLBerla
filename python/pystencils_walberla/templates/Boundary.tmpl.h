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

#include "core/DataTypes.h"

{% if target is equalto 'cpu' -%}
#include "field/GhostLayerField.h"
{%- elif target is equalto 'gpu' -%}
#include "cuda/GPUField.h"
#include "cuda/FieldCopy.h"
{%- endif %}
#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "blockforest/StructuredBlockForest.h"
#include "field/FlagField.h"
#include "core/debug/Debug.h"

#include <set>
#include <vector>

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
                cudaFree( gpuVec );
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
                cudaFree( gpuVec );
            gpuVectors_.resize( cpuVectors_.size() );

            WALBERLA_ASSERT_EQUAL(cpuVectors_.size(), NUM_TYPES);
            for(size_t i=0; i < cpuVectors_.size(); ++i )
            {
                auto & gpuVec = gpuVectors_[i];
                auto & cpuVec = cpuVectors_[i];
                cudaMalloc( &gpuVec, sizeof({{StructName}}) * cpuVec.size() );
                cudaMemcpy( gpuVec, &cpuVec[0], sizeof({{StructName}}) * cpuVec.size(), cudaMemcpyHostToDevice );
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

    {% if multi_sweep %}
    {% for sweep_class_name, sweep_kernel_info in sweep_classes.items() %}
    class {{sweep_class_name}}
    {
        public:
            {{sweep_class_name}} ( {{sweep_kernel_info|generate_constructor_parameters(['indexVectorSize'])}} )
                : {{ sweep_kernel_info|generate_constructor_initializer_list(['indexVectorSize']) }} {};

            void operator() ( IBlock * block {% if target == 'gpu'%}, cudaStream_t stream = 0 {%endif%});
            void inner( IBlock * block {% if target == 'gpu'%}, cudaStream_t stream = 0 {%endif%});
            void outer( IBlock * block {% if target == 'gpu'%}, cudaStream_t stream = 0 {%endif%});
        
        private:
            void run( IBlock * block, IndexVectors::Type type{% if target == 'gpu'%}, cudaStream_t stream = 0 {%endif%});

            {{sweep_kernel_info|generate_members(['indexVectorSize'])|indent(12)}}
    };

    {{sweep_class_name}} get{{sweep_class_name}} ()
    {
        return {{sweep_class_name}} ( {{sweep_kernel_info|generate_constructor_call_arguments(['indexVectorSize'])|indent(12)}} );
    }
    {% endfor %}
    {% endif %}

    {{class_name}}( const shared_ptr<StructuredBlockForest> & blocks,
                   {{dummy_kernel_info|generate_constructor_parameters(['indexVector', 'indexVectorSize'])}}{{additional_data_handler.constructor_arguments}})
        :{{additional_data_handler.initialiser_list}} {{ dummy_kernel_info|generate_constructor_initializer_list(['indexVector', 'indexVectorSize']) }}
    {
        auto createIdxVector = []( IBlock * const , StructuredBlockStorage * const ) { return new IndexVectors(); };
        indexVectorID = blocks->addStructuredBlockData< IndexVectors >( createIdxVector, "IndexField_{{class_name}}");
    };

    {% if not multi_sweep %}

    void operator() ( IBlock * block {% if target == 'gpu'%}, cudaStream_t stream = 0 {%endif%});
    void inner( IBlock * block {% if target == 'gpu'%}, cudaStream_t stream = 0 {%endif%});
    void outer( IBlock * block {% if target == 'gpu'%}, cudaStream_t stream = 0 {%endif%});

    {% endif %}

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

        for( auto it = flagField->begin(); it != flagField->end(); ++it )
        {
            if( ! isFlagSet(it, domainFlag) )
                continue;
            {%- for dirIdx, dirVec, offset in additional_data_handler.stencil_info %}
            if ( isFlagSet( it.neighbor({{offset}} {%if dim == 3%}, 0 {%endif %}), boundaryFlag ) )
            {
                {% if inner_or_boundary -%}
                auto element = {{StructName}}(it.x(), it.y(), {%if dim == 3%} it.z(), {%endif %} {{dirIdx}} );
                {% else -%}
                auto element = {{StructName}}(it.x() + cell_idx_c({{dirVec[0]}}), it.y() + cell_idx_c({{dirVec[1]}}), {%if dim == 3%} it.z() + cell_idx_c({{dirVec[2]}}), {%endif %} {{additional_data_handler.inverse_directions[dirIdx]}} );
                {% endif -%}
                {{additional_data_handler.data_initialisation(dirIdx)|indent(16)}}
                indexVectorAll.push_back( element );
                if( inner.contains( it.x(), it.y(), it.z() ) )
                    indexVectorInner.push_back( element );
                else
                    indexVectorOuter.push_back( element );
            }
            {% endfor %}
        }
        indexVectors->syncGPU();
    }

private:
    {% if not multi_sweep %}
    void run( IBlock * block, IndexVectors::Type type{% if target == 'gpu'%}, cudaStream_t stream = 0 {%endif%});
    {% endif %}

    BlockDataID indexVectorID;
    {{additional_data_handler.additional_member_variable|indent(4)}}
public:
    {{dummy_kernel_info|generate_members(('indexVector', 'indexVectorSize'))|indent(4)}}
};



} // namespace {{namespace}}
} // namespace walberla
