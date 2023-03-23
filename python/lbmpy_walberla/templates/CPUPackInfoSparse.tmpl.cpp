#include "{{class_name}}.h"

#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wfloat-equal"
#   pragma GCC diagnostic ignored "-Wshadow"
#   pragma GCC diagnostic ignored "-Wconversion"
#   pragma GCC diagnostic ignored "-Wunused-variable"
#endif

namespace walberla {
namespace {{namespace}} {

using walberla::cell::CellInterval;
using walberla::stencil::Direction;

{% if timestep is equalto 'Even' -%}
static void packEven(double * RESTRICT _data_buffer, const double * RESTRICT _data_pdfs, const uint32_t * RESTRICT const _data_idx, int64_t numPDFs)
{
   const double * RESTRICT _start_data_pdfs = _data_pdfs;
   for(int64_t ctr_0 = 0; ctr_0 < numPDFs; ++ctr_0)
   {
      _data_buffer[ctr_0] = _start_data_pdfs[_data_idx[ctr_0]];
   }
}
{%- elif timestep is equalto 'Odd' -%}
static void packOdd(double * RESTRICT _data_buffer, const double * RESTRICT _data_pdfs, const uint32_t startIDX, int64_t numPDFs)
{
   const double * RESTRICT _start_data_pdfs = _data_pdfs + startIDX;
   for(int64_t ctr_0 = 0; ctr_0 < numPDFs; ++ctr_0)
   {
      _data_buffer[ctr_0] = _start_data_pdfs[ctr_0];
   }
}
{%- endif %}


{% if timestep is equalto 'Even' -%}
static void unpackEven(double * RESTRICT const _data_buffer, double * RESTRICT _data_pdfs, int64_t startIDX, int64_t const numPDFs)
{
   double * RESTRICT _start_data_pdfs = _data_pdfs + startIDX;
   for(int64_t ctr_0 = 0; ctr_0 < numPDFs; ++ctr_0)
   {
      _start_data_pdfs[ctr_0] = _data_buffer[ctr_0];
   }
}
{%- elif timestep is equalto 'Odd' -%}
static void unpackOdd(double * RESTRICT const _data_buffer, double * RESTRICT _data_pdfs, const uint32_t * RESTRICT const _data_idx, int64_t const numPDFs)
{
   double * RESTRICT _start_data_pdfs = _data_pdfs;
   for(int64_t ctr_0 = 0; ctr_0 < numPDFs; ++ctr_0)
   {
      _start_data_pdfs[_data_idx[ctr_0]] = _data_buffer[ctr_0];
   }
}
{%- endif %}

void {{class_name}}::pack(Direction dir, unsigned char * byte_buffer, IBlock * block) const
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

   packEven(buffer, _data_pdfs, idxs, numPDFs);

   {%- elif timestep is equalto 'Odd' -%}
   auto numPDFsIt =  list->getNumCommPDFs(dir);
   if( numPDFsIt.second == 0 )
      return;

   auto startIdxIt = list->getStartCommIdx(dir);
   index_t numPDFs = numPDFsIt.second;
   index_t startIDX = startIdxIt.second;

   packOdd(buffer, _data_pdfs, startIDX, numPDFs);
   {%- endif %}


}

void {{class_name}}::unpack(stencil::Direction dir, unsigned char * byte_buffer, IBlock * block) const
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
   index_t numPDFs = numPDFsIt.second;
   index_t startIDX = startIdxIt.second;

   unpackEven(buffer, _data_pdf_field, startIDX, numPDFs);

   {%- elif timestep is equalto 'Odd' -%}
   auto sendPDFs = list->getSendPDFs(dir);
   const auto & sendPDFVector = sendPDFs.second;
   const int64_t numPDFs = sendPDFVector.size();
   const auto * idxs = &sendPDFs.second.front();

   unpackOdd(buffer, _data_pdf_field, idxs, numPDFs);
   {%- endif %}
}

uint_t {{class_name}}::size(stencil::Direction dir, const IBlock * block) const
{
   auto * list = block->getData< lbmpy::ListLBMList >( listId_ );
   auto numPDFsIt = list->getNumCommPDFs(dir);
   index_t numPDFs = numPDFsIt.second;
   return numPDFs * sizeof( double );
}

} // namespace {{namespace}}
} // namespace walberla