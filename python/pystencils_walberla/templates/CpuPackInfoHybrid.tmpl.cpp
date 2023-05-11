#include "stencil/Directions.h"
#include "core/cell/CellInterval.h"
#include "core/DataTypes.h"
#include "{{class_name}}.h"

#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wfloat-equal"
#   pragma GCC diagnostic ignored "-Wshadow"
#   pragma GCC diagnostic ignored "-Wconversion"
#   pragma GCC diagnostic ignored "-Wunused-variable"
#endif

{% for header in headers %}
#include {{header}}
{% endfor %}

namespace walberla {
namespace {{namespace}} {

using walberla::cell::CellInterval;
using walberla::stencil::Direction;

///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////    SPARSE KERNELS   //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

{% if timestep is equalto 'Even' -%}
static void packEvenSparse(double * RESTRICT _data_buffer, const double * RESTRICT _data_pdfs, const uint32_t * RESTRICT const _data_idx, int64_t numPDFs)
{
   const double * RESTRICT _start_data_pdfs = _data_pdfs;
   for(int64_t ctr_0 = 0; ctr_0 < numPDFs; ++ctr_0)
   {
      _data_buffer[ctr_0] = _start_data_pdfs[_data_idx[ctr_0]];
   }
}
{%- elif timestep is equalto 'Odd' -%}
static void packOddSparse(double * RESTRICT _data_buffer, const double * RESTRICT _data_pdfs, const uint32_t startIDX, int64_t numPDFs)
{
   const double * RESTRICT _start_data_pdfs = _data_pdfs + startIDX;
   for(int64_t ctr_0 = 0; ctr_0 < numPDFs; ++ctr_0)
   {
      _data_buffer[ctr_0] = _start_data_pdfs[ctr_0];
   }
}
{%- endif %}


{% if timestep is equalto 'Even' -%}
static void unpackEvenSparse(double * RESTRICT const _data_buffer, double * RESTRICT _data_pdfs, int64_t startIDX, int64_t const numPDFs)
{
   double * RESTRICT _start_data_pdfs = _data_pdfs + startIDX;
   for(int64_t ctr_0 = 0; ctr_0 < numPDFs; ++ctr_0)
   {
      _start_data_pdfs[ctr_0] = _data_buffer[ctr_0];
   }
}
{%- elif timestep is equalto 'Odd' -%}
static void unpackOddSparse(double * RESTRICT const _data_buffer, double * RESTRICT _data_pdfs, const uint32_t * RESTRICT const _data_idx, int64_t const numPDFs)
{
   double * RESTRICT _start_data_pdfs = _data_pdfs;
   for(int64_t ctr_0 = 0; ctr_0 < numPDFs; ++ctr_0)
   {
      _start_data_pdfs[_data_idx[ctr_0]] = _data_buffer[ctr_0];
   }
}
{%- endif %}

void {{class_name}}::packSparse(Direction dir, unsigned char * byte_buffer, IBlock * block) const
{
   const auto * list = block->getData< lbmpy::ListLBMList >( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( list )

   double * buffer = reinterpret_cast<double*>(byte_buffer);
   const double * RESTRICT const _data_pdfs = list->getPDFbegining();

   {% if timestep is equalto 'Even' -%}
   auto sendPDFs = list->getSendPDFs(dir);
   const auto & sendPDFVector = sendPDFs.second;
   const int64_t numPDFs = sendPDFVector.size();
   const auto * idxs = &sendPDFs.second.front();

   packEvenSparse(buffer, _data_pdfs, idxs, numPDFs);

   {%- elif timestep is equalto 'Odd' -%}
   auto numPDFsIt =  list->getNumCommPDFs(dir);
   if( numPDFsIt.second == 0 )
      return;

   auto startIdxIt = list->getStartCommIdx(dir);
   uint32_t numPDFs = numPDFsIt.second;
   uint32_t startIDX = startIdxIt.second;

   packOddSparse(buffer, _data_pdfs, startIDX, numPDFs);
   {%- endif %}


}

void {{class_name}}::unpackSparse(stencil::Direction dir, unsigned char * byte_buffer, IBlock * block) const
{
   auto * list = block->getData< lbmpy::ListLBMList >( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( list )

   double * buffer = reinterpret_cast<double*>(byte_buffer);
   double * RESTRICT _data_pdf_field = list->getPDFbegining();

   {% if timestep is equalto 'Even' -%}
   auto numPDFsIt =  list->getNumCommPDFs(dir);
   if( numPDFsIt.second == 0 )
      return;

   auto startIdxIt = list->getStartCommIdx(dir);
   uint32_t numPDFs = numPDFsIt.second;
   uint32_t startIDX = startIdxIt.second;

   unpackEvenSparse(buffer, _data_pdf_field, startIDX, numPDFs);

   {%- elif timestep is equalto 'Odd' -%}
   auto sendPDFs = list->getSendPDFs(dir);
   const auto & sendPDFVector = sendPDFs.second;
   const int64_t numPDFs = sendPDFVector.size();
   const auto * idxs = &sendPDFs.second.front();

   unpackOddSparse(buffer, _data_pdf_field, idxs, numPDFs);
   {%- endif %}
}

uint_t {{class_name}}::sizeSparse(stencil::Direction dir, const IBlock * block) const
{
   auto * list = block->getData< lbmpy::ListLBMList >( listId_ );
   auto numPDFsIt = list->getNumCommPDFs(dir);
   uint32_t numPDFs = numPDFsIt.second;
   return numPDFs * sizeof( double );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////    DENSE KERNELS   ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

{% for kernel in pack_kernels.values() %}
{{kernel|generate_definition(target)}}
{% endfor %}

{% for kernel in unpack_kernels.values() %}
{{kernel|generate_definition(target)}}
{% endfor %}


void {{class_name}}::packDense(Direction dir, unsigned char * byte_buffer, IBlock * block) const
{
    {{dtype}} * buffer = reinterpret_cast<{{dtype}}*>(byte_buffer);

    {{fused_kernel|generate_block_data_to_field_extraction(parameters_to_ignore=['buffer'])|indent(4)}}
    CellInterval ci;
    {% if gl_to_inner -%}
    {{field_name}}->getGhostRegion(dir, ci, 1, false);
    {%- else -%}
    {{field_name}}->getSliceBeforeGhostLayer(dir, ci, 1, false);
    {%- endif %}

    switch( dir )
    {
        {%- for direction_set, kernel in pack_kernels.items()  %}
        {%- for dir in direction_set %}
        case stencil::{{dir}}:
        {%- endfor %}
        {
            {{kernel|generate_call(cell_interval="ci")|indent(12)}}
            break;
        }
        {% endfor %}

        default:
            WALBERLA_ASSERT(false);
    }
}


void {{class_name}}::unpackDense(Direction dir, unsigned char * byte_buffer, IBlock * block) const
{
    {{dtype}} * buffer = reinterpret_cast<{{dtype}}*>(byte_buffer);

    {{fused_kernel|generate_block_data_to_field_extraction(parameters_to_ignore=['buffer'])|indent(4)}}
    CellInterval ci;
    {% if gl_to_inner -%}
    {{field_name}}->getSliceBeforeGhostLayer(dir, ci, 1, false);
    {%- else -%}
    {{field_name}}->getGhostRegion(dir, ci, 1, false);
    {%- endif %}
    auto communciationDirection = stencil::inverseDir[dir];

    switch( communciationDirection )
    {
        {%- for direction_set, kernel in unpack_kernels.items()  %}
        {%- for dir in direction_set %}
        case stencil::{{dir}}:
        {%- endfor %}
        {
            {{kernel|generate_call(cell_interval="ci")|indent(12)}}
            break;
        }
        {% endfor %}

        default:
            WALBERLA_ASSERT(false);
    }
}


uint_t {{class_name}}::sizeDense(stencil::Direction dir, const IBlock * block) const
{
    {{fused_kernel|generate_block_data_to_field_extraction(parameters_to_ignore=['buffer'])|indent(4)}}
    CellInterval ci;
    {{field_name}}->getGhostRegion(dir, ci, 1, false);

    uint_t elementsPerCell = 0;

    switch( dir )
    {
        {%- for direction_set, elements in elements_per_cell.items()  %}
        {%- for dir in direction_set %}
        case stencil::{{dir}}:
        {%- endfor %}
            elementsPerCell = {{elements}};
            break;
        {% endfor %}
        default:
            elementsPerCell = 0;
    }
    return ci.numCells() * elementsPerCell * sizeof( {{dtype}} );
}



} // namespace {{namespace}}
} // namespace walberla