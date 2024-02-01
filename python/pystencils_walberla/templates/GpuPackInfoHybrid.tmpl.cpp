#include "stencil/Directions.h"
#include "core/cell/CellInterval.h"
#include "gpu/GPUField.h"
#include "core/DataTypes.h"
#include "{{class_name}}.h"


{% if target is equalto 'cpu' -%}
#define FUNC_PREFIX
{%- elif target is equalto 'gpu' -%}
#define FUNC_PREFIX __global__
{%- endif %}


namespace walberla {
namespace {{namespace}} {

   using walberla::cell::CellInterval;
   using walberla::stencil::Direction;



   {% if timestep is equalto 'Even' -%}
   static __global__ void packEvenSparse(double * RESTRICT _data_buffer, double * RESTRICT const _data_pdfs, uint32_t * RESTRICT const _data_idx, int64_t numPDFs)
   {
      double * RESTRICT _start_data_pdfs = _data_pdfs;
      if (blockDim.x * blockIdx.x + threadIdx.x < numPDFs)
      {
         const int64_t ctr_0 = blockDim.x*blockIdx.x + threadIdx.x;
         _data_buffer[ctr_0] = _start_data_pdfs[_data_idx[ctr_0]];
      }
   }
   {%- elif timestep is equalto 'Odd' -%}
   static __global__ void packOddSparse(double * RESTRICT _data_buffer, const double * RESTRICT _data_pdfs, const uint32_t startIDX, int64_t numPDFs)
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
   static __global__ void unpackEvenSparse(double * RESTRICT const _data_buffer, double * RESTRICT _data_pdfs, int64_t startIDX, int64_t const numPDFs)
   {
      double * RESTRICT _start_data_pdfs = _data_pdfs + startIDX;
      if (blockDim.x * blockIdx.x + threadIdx.x < numPDFs)
      {
         const int64_t ctr_0 = blockDim.x*blockIdx.x + threadIdx.x;
         _start_data_pdfs[ctr_0] = _data_buffer[ctr_0];
      }
   }
   {%- elif timestep is equalto 'Odd' -%}
   static __global__ void unpackOddSparse(double * RESTRICT const _data_buffer, double * RESTRICT _data_pdfs, const uint32_t * RESTRICT const _data_idx, int64_t const numPDFs)
   {
      double * RESTRICT _start_data_pdfs = _data_pdfs;
      if (blockDim.x * blockIdx.x + threadIdx.x < numPDFs)
      {
         const int64_t ctr_0 = blockDim.x*blockIdx.x + threadIdx.x;
         _start_data_pdfs[_data_idx[ctr_0]] = _data_buffer[ctr_0];
      }
   }
   {%- endif %}





   {% for kernel in pack_kernels.values() %}
   {{kernel|generate_definition(target)}}
   {% endfor %}

   {% for kernel in unpack_kernels.values() %}
   {{kernel|generate_definition(target)}}
   {% endfor %}



   void {{class_name}}::pack(Direction dir, unsigned char * byte_buffer, IBlock * block, gpuStream_t stream)
   {
      {{dtype}} * buffer = reinterpret_cast<{{dtype}}*>(byte_buffer);

      auto blockState = block->getState();
      if (blockState == sparseBlockSelectors_) {

         auto * list = block->getData< lbmpy::ListLBMList >( listId_ );
         WALBERLA_ASSERT_NOT_NULLPTR( list )
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
         packEvenSparse<<<_grid, _block, 0, stream>>>(buffer, _data_pdfs, idxs, numPDFs);

         {%- elif timestep is equalto 'Odd' -%}
         auto numPDFsIt =  list->getNumCommPDFs(dir);
         if( numPDFsIt.second == 0 )
            return;

         auto startIdxIt = list->getStartCommIdx(dir);
         uint32_t numPDFs = numPDFsIt.second;
         uint32_t startIDX = startIdxIt.second;

         dim3 _block(int(((256 < numPDFs) ? 256 : numPDFs)), int(1), int(1));
         dim3 _grid(int(( (numPDFs) % (((256 < numPDFs) ? 256 : numPDFs)) == 0 ? (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) : ( (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) ) +1 )), int(1), int(1));
         packOddSparse<<<_grid, _block, 0, stream>>>(buffer, _data_pdfs, startIDX, numPDFs);
         {%- endif %}

      }
      else if (blockState == denseBlockSelectors_) {

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
               {{kernel|generate_call(cell_interval="ci", stream="stream")|indent(12)}}
               break;
            }
            {% endfor %}

         default:
            WALBERLA_ASSERT(false);
         }
      }
      else {
         WALBERLA_ABORT("Block selector is neither Sparse nor Dense, it is " << blockState)
      }
   }


   void {{class_name}}::unpack(Direction dir, unsigned char * byte_buffer, IBlock * block, gpuStream_t stream)
   {
      {{dtype}} * buffer = reinterpret_cast<{{dtype}}*>(byte_buffer);

      auto blockState = block->getState();
      if (blockState == sparseBlockSelectors_) {

         auto * list = block->getData< lbmpy::ListLBMList >( listId_ );
         WALBERLA_ASSERT_NOT_NULLPTR( list )

         double * RESTRICT _data_pdfs = list->getGPUPDFbegining();

         {% if timestep is equalto 'Even' -%}
         auto numPDFsIt =  list->getNumCommPDFs( dir );
         if( numPDFsIt.second == 0 )
            return;

         auto startIdxIt = list->getStartCommIdx( dir );

         uint32_t numPDFs = numPDFsIt.second;
         uint32_t startIDX = startIdxIt.second;

         dim3 _block(int(((256 < numPDFs) ? 256 : numPDFs)), int(1), int(1));
         dim3 _grid(int(( (numPDFs) % (((256 < numPDFs) ? 256 : numPDFs)) == 0 ? (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) : ( (int64_t)(numPDFs) / (int64_t)(((256 < numPDFs) ? 256 : numPDFs)) ) +1 )), int(1), int(1));
         unpackEvenSparse<<<_grid, _block, 0, stream>>>(buffer, _data_pdfs, startIDX, numPDFs);

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
         unpackOddSparse<<<_grid, _block, 0, stream>>>(buffer, _data_pdfs, idxs, numPDFs);
         {%- endif %}
      }
      else if (blockState == denseBlockSelectors_) {

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
               {{kernel|generate_call(cell_interval="ci", stream="stream")|indent(12)}}
               break;
            }
            {% endfor %}

         default:
            WALBERLA_ASSERT(false);
         }
      }
      else {
         WALBERLA_ABORT("Block selector is neither Sparse nor Dense, it is " << blockState)
      }
   }


   static __global__ void communicate(const double * RESTRICT _data_sender, double * RESTRICT _data_receiver, uint32_t * RESTRICT const _data_idx_sender, int64_t startIDXReceiver, int64_t numPDFs)
   {
      double * RESTRICT _start_receiver_pdfs = _data_receiver + startIDXReceiver;
      if (blockDim.x * blockIdx.x + threadIdx.x < numPDFs)
      {
         const int64_t ctr_0 = blockDim.x*blockIdx.x + threadIdx.x;
         _start_receiver_pdfs[ctr_0] = _data_sender[_data_idx_sender[ctr_0]];
      }
   }


   void {{class_name}}::communicateLocal(Direction dir, const IBlock * sender, IBlock * receiver, gpuStream_t stream) {

      auto blockStateSender = sender->getState();
      auto blockStateReceiver = receiver->getState();

      if (blockStateSender == sparseBlockSelectors_ && blockStateReceiver == sparseBlockSelectors_ )
      {
         stencil::Direction inverseDir = stencil::inverseDir[dir];

         const auto* senderList = sender->getData< lbmpy::ListLBMList >(listId_);
         auto* receiverList     = receiver->getData< lbmpy::ListLBMList >(listId_);
         WALBERLA_ASSERT_NOT_NULLPTR(senderList)
         WALBERLA_ASSERT_NOT_NULLPTR(receiverList)

         ////////// Sender //////////////
         auto idxMapItCPU           = senderList->getSendPDFs(dir);
         const auto& indexVectorCPU = idxMapItCPU.second;
         const int64_t numPDFs      = indexVectorCPU.size();
         if (numPDFs == 0) return;

         auto idxMapIt = senderList->getSendPDFsGPU(dir);

         ///////////// Receiver ////////////////////
         auto numPDFsIt = receiverList->getNumCommPDFs(inverseDir);
         WALBERLA_ASSERT_EQUAL(numPDFs, numPDFsIt.second)

         auto startIdxIt = receiverList->getStartCommIdx(inverseDir);

         auto idxs                                = idxMapIt.second;
         const double* RESTRICT _data_sender_pdfs = senderList->getGPUPDFbegining();
         auto ReceiverStartIDX                 = startIdxIt.second;
         double* RESTRICT _data_receiver_pdfs     = receiverList->getGPUPDFbegining();

         dim3 _block(int(((256 < numPDFs) ? 256 : numPDFs)), int(1), int(1));
         dim3 _grid(int(((numPDFs) % (((256 < numPDFs) ? 256 : numPDFs)) == 0 ? (int64_t) (numPDFs) / (int64_t) (((256 < numPDFs) ? 256 : numPDFs)) : ((int64_t) (numPDFs) / (int64_t) (((256 < numPDFs) ? 256 : numPDFs))) + 1)), int(1), int(1));

         communicate<<< _grid, _block, 0, stream >>>(_data_sender_pdfs, _data_receiver_pdfs, idxs, ReceiverStartIDX, numPDFs);
      } else {
         WALBERLA_ABORT("local communication not implemented by Hybrid Communication")
      }
   }



   uint_t {{class_name}}::size(stencil::Direction dir, IBlock * block)
   {
      auto blockState = block->getState();
      if (blockState == sparseBlockSelectors_) {

         auto * list = block->getData< lbmpy::ListLBMList >( listId_ );
         auto numPDFsIt = list->getNumCommPDFs( dir );
         uint32_t numPDFsRecieve = numPDFsIt.second;

         auto sendPDFsCPU = list->getSendPDFs(dir);
         auto & sendPDFVector = sendPDFsCPU.second;
         int64_t numPDFsSend = sendPDFVector.size();
         if(numPDFsSend != numPDFsRecieve) {
            WALBERLA_LOG_INFO("numPDFsSend " << numPDFsSend << " =! numPDFsRecieve " << numPDFsRecieve << " in dir " << stencil::dirToString[dir])
         }
         return numPDFsSend * sizeof( double );
      }
      else if (blockState == denseBlockSelectors_) {

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
      else {
         WALBERLA_ABORT("Block selector is neither Sparse nor Dense, it is " << blockState)
      }
   }



} // namespace {{namespace}}
} // namespace walberla