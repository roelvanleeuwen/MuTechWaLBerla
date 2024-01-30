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
static __global__ void packEven(double * RESTRICT _data_buffer, double * RESTRICT const _data_pdfs, uint32_t * RESTRICT const _data_idx, int64_t numPDFs)
{
   double * RESTRICT _start_data_pdfs = _data_pdfs;
   if (blockDim.x * blockIdx.x + threadIdx.x < numPDFs)
   {
      const int64_t ctr_0 = blockDim.x*blockIdx.x + threadIdx.x;
      _data_buffer[ctr_0] = _start_data_pdfs[_data_idx[ctr_0]];
   }
}
{%- elif timestep is equalto 'Odd' -%}
static __global__ void packOdd(double * RESTRICT _data_buffer, const double * RESTRICT _data_pdfs, const uint32_t startIDX, int64_t numPDFs)
{
   const double * RESTRICT _start_data_pdfs = _data_pdfs + startIDX;
   if (blockDim.x * blockIdx.x + threadIdx.x < numPDFs)
   {
      const int64_t ctr_0 = blockDim.x*blockIdx.x + threadIdx.x;
      _data_buffer[ctr_0] = _start_data_pdfs[ctr_0];
   }
}
{%- endif %}

{% if timestep is equalto 'Even' -%}
static __global__ void unpackEven(double * RESTRICT const _data_buffer, double * RESTRICT _data_pdfs, int64_t startIDX, int64_t const numPDFs)
{
   double * RESTRICT _start_data_pdfs = _data_pdfs + startIDX;
   if (blockDim.x * blockIdx.x + threadIdx.x < numPDFs)
   {
      const int64_t ctr_0 = blockDim.x*blockIdx.x + threadIdx.x;
      _start_data_pdfs[ctr_0] = _data_buffer[ctr_0];
   }
}
{%- elif timestep is equalto 'Odd' -%}
static __global__ void unpackOdd(double * RESTRICT const _data_buffer, double * RESTRICT _data_pdfs, const uint32_t * RESTRICT const _data_idx, int64_t const numPDFs)
{
   double * RESTRICT _start_data_pdfs = _data_pdfs;
   if (blockDim.x * blockIdx.x + threadIdx.x < numPDFs)
   {
      const int64_t ctr_0 = blockDim.x*blockIdx.x + threadIdx.x;
      _start_data_pdfs[_data_idx[ctr_0]] = _data_buffer[ctr_0];
   }
}
{%- endif %}

static __global__ void communicate(const double * RESTRICT _data_sender, double * RESTRICT _data_receiver, uint32_t * RESTRICT const _data_idx_sender, int64_t startIDXReceiver, int64_t numPDFs)
{
   double * RESTRICT _start_receiver_pdfs = _data_receiver + startIDXReceiver;
   if (blockDim.x * blockIdx.x + threadIdx.x < numPDFs)
   {
      const int64_t ctr_0 = blockDim.x*blockIdx.x + threadIdx.x;
      _start_receiver_pdfs[ctr_0] = _data_sender[_data_idx_sender[ctr_0]];
   }
}


void {{class_name}}::pack( stencil::Direction dir, unsigned char * byte_buffer, IBlock * block, gpuStream_t stream )
{
   auto * list = block->getData< lbmpy::ListLBMList >( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( list )
   double * buffer = reinterpret_cast<double*>(byte_buffer);
   double * RESTRICT const _data_pdfs = list->getGPUPDFbegining();


{% if timestep is equalto 'Even' -%}
   auto sendPDFsCPU = list->getSendPDFs(dir);
   const auto & sendPDFVector = sendPDFsCPU.second;
   const int64_t numPDFs = sendPDFVector.size();
   if(numPDFs == 0)
      return;

   auto sendPDFsGPU = list->getSendPDFsGPU(dir);
   auto idxs = sendPDFsGPU.second;

   dim3 _block(int(((256 < numPDFs) ? 256 : numPDFs)), int(1), int(1));
   dim3 _grid(int(( (numPDFs) % (((256 < numPDFs) ? 256 : numPDFs)) == 0 ? (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) : ( (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) ) +1 )), int(1), int(1));
   packEven<<<_grid, _block, 0, stream>>>(buffer, _data_pdfs, idxs, numPDFs);

{%- elif timestep is equalto 'Odd' -%}
   auto numPDFsIt =  list->getNumCommPDFs(dir);
   if( numPDFsIt.second == 0 )
      return;

   auto startIdxIt = list->getStartCommIdx(dir);
   index_t numPDFs = numPDFsIt.second;
   index_t startIDX = startIdxIt.second;

   dim3 _block(int(((256 < numPDFs) ? 256 : numPDFs)), int(1), int(1));
   dim3 _grid(int(( (numPDFs) % (((256 < numPDFs) ? 256 : numPDFs)) == 0 ? (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) : ( (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) ) +1 )), int(1), int(1));
   packOdd<<<_grid, _block, 0, stream>>>(buffer, _data_pdfs, startIDX, numPDFs);
{%- endif %}
}

void {{class_name}}::unpack( stencil::Direction dir, unsigned char * byte_buffer, IBlock * block, gpuStream_t stream )
{

   auto * list = block->getData< lbmpy::ListLBMList >( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( list )

   double * RESTRICT const buffer = reinterpret_cast<double*>(byte_buffer);
   double * RESTRICT _data_pdfs = list->getGPUPDFbegining();


{% if timestep is equalto 'Even' -%}
   auto numPDFsIt =  list->getNumCommPDFs( dir );
   if( numPDFsIt.second == 0 )
      return;

   auto startIdxIt = list->getStartCommIdx( dir );

   index_t numPDFs = numPDFsIt.second;
   index_t startIDX = startIdxIt.second;

   dim3 _block(int(((256 < numPDFs) ? 256 : numPDFs)), int(1), int(1));
   dim3 _grid(int(( (numPDFs) % (((256 < numPDFs) ? 256 : numPDFs)) == 0 ? (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) : ( (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) ) +1 )), int(1), int(1));
   unpackEven<<<_grid, _block, 0, stream>>>(buffer, _data_pdfs, startIDX, numPDFs);

{%- elif timestep is equalto 'Odd' -%}
   auto sendPDFsCPU = list->getSendPDFs(dir);
   const auto & sendPDFVector = sendPDFsCPU.second;
   const int64_t numPDFs = sendPDFVector.size();
   if(numPDFs == 0)
      return;

   auto sendPDFsGPU = list->getSendPDFsGPU(dir);
   auto idxs = sendPDFsGPU.second;

   dim3 _block(int(((256 < numPDFs) ? 256 : numPDFs)), int(1), int(1));
   dim3 _grid(int(( (numPDFs) % (((256 < numPDFs) ? 256 : numPDFs)) == 0 ? (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) : ( (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) ) +1 )), int(1), int(1));
   unpackOdd<<<_grid, _block, 0, stream>>>(buffer, _data_pdfs, idxs, numPDFs);
{%- endif %}
}

void {{class_name}}::communicateLocal( stencil::Direction dir, const IBlock * sender, IBlock * receiver, gpuStream_t stream)
{
{% if timestep is equalto 'Even' -%}

   stencil::Direction inverseDir = stencil::inverseDir[dir];

   const auto * senderList = sender->getData< lbmpy::ListLBMList >( listId_ );
   auto * receiverList = receiver->getData< lbmpy::ListLBMList >( listId_ );
   WALBERLA_ASSERT_NOT_NULLPTR( senderList   )
   WALBERLA_ASSERT_NOT_NULLPTR( receiverList )

   ////////// Sender //////////////
   auto idxMapItCPU = senderList->getSendPDFs( dir );
   const auto & indexVectorCPU = idxMapItCPU.second;
   const int64_t numPDFs = indexVectorCPU.size();
   if(numPDFs == 0)
      return;

   auto idxMapIt = senderList->getSendPDFsGPU( dir );


   ///////////// Receiver ////////////////////
   auto numPDFsIt =  receiverList->getNumCommPDFs( inverseDir );
   WALBERLA_ASSERT_EQUAL( numPDFs, numPDFsIt.second )

   auto startIdxIt = receiverList->getStartCommIdx( inverseDir );

   auto idxs = idxMapIt.second;
   const double * RESTRICT _data_sender_pdfs = senderList->getGPUPDFbegining();
   index_t ReceiverStartIDX = startIdxIt.second;
   double * RESTRICT _data_receiver_pdfs = receiverList->getGPUPDFbegining();

   dim3 _block(int(((256 < numPDFs) ? 256 : numPDFs)), int(1), int(1));
   dim3 _grid(int(( (numPDFs) % (((256 < numPDFs) ? 256 : numPDFs)) == 0 ? (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) : ( (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) ) +1 )), int(1), int(1));

   communicate<<<_grid, _block, 0, stream>>>(_data_sender_pdfs, _data_receiver_pdfs, idxs, ReceiverStartIDX, numPDFs);
   {%- elif timestep is equalto 'Odd' -%}
   WALBERLA_ABORT("Alternating Timesteps for local communication not supported")
   {%- endif %}

}

uint_t {{class_name}}::size( stencil::Direction dir, IBlock * block )
{
   auto * list = block->getData< lbmpy::ListLBMList >( listId_ );
   auto numPDFsIt = list->getNumCommPDFs( dir );
   index_t numPDFsRecieve = numPDFsIt.second;

   auto sendPDFsCPU = list->getSendPDFs(dir);
   auto & sendPDFVector = sendPDFsCPU.second;
   int64_t numPDFsSend = sendPDFVector.size();
   if(numPDFsSend != numPDFsRecieve) {
      WALBERLA_LOG_INFO("numPDFsSend " << numPDFsSend << " =! numPDFsRecieve " << numPDFsRecieve << " in dir " << stencil::dirToString[dir])
   }
   return numPDFsSend * sizeof( double );
}

} // namespace {{namespace}}
} // namespace walberla