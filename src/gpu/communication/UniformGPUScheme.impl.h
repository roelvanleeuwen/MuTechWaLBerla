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
//! \file UniformGPUScheme.impl.h
//! \ingroup gpu
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

namespace walberla {
namespace gpu
{
namespace communication {


   template<typename Stencil>
   UniformGPUScheme<Stencil>::UniformGPUScheme( const weak_ptr< StructuredBlockForest >& bf,
                                                const bool sendDirectlyFromGPU,
                                                const bool useLocalCommunication,
                                                const int tag )
        : blockForest_( bf ),
          setupBeforeNextCommunication_( true ),
          communicationInProgress_( false ),
          sendFromGPU_( sendDirectlyFromGPU ),
          useLocalCommunication_(useLocalCommunication),
          bufferSystemCPU_( mpi::MPIManager::instance()->comm(), tag ),
          bufferSystemGPU_( mpi::MPIManager::instance()->comm(), tag ),
          requiredBlockSelectors_( Set<SUID>::emptySet() ),
          incompatibleBlockSelectors_( Set<SUID>::emptySet() )
   {
      WALBERLA_MPI_SECTION()
      {
// Open MPI supports compile time CUDA-aware support check
#if (defined(OPEN_MPI) && OPEN_MPI) && !(defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT)
         WALBERLA_CHECK(!sendDirectlyFromGPU)
#endif
      }
#if not defined(WALBERLA_BUILD_WITH_SYCL)
      for (uint_t i = 0; i < Stencil::Q; ++i)
         WALBERLA_GPU_CHECK(gpuStreamCreate(&streams_[i]))
#endif
   }

   template<typename Stencil>
   UniformGPUScheme<Stencil>::UniformGPUScheme( const weak_ptr< StructuredBlockForest >& bf,
                                                const Set<SUID> & requiredBlockSelectors,
                                                const Set<SUID> & incompatibleBlockSelectors,
                                                const bool sendDirectlyFromGPU,
                                                const bool useLocalCommunication,
                                                const int tag )
      : blockForest_( bf ),
        setupBeforeNextCommunication_( true ),
        communicationInProgress_( false ),
        sendFromGPU_( sendDirectlyFromGPU ),
        useLocalCommunication_(useLocalCommunication),
        bufferSystemCPU_( mpi::MPIManager::instance()->comm(), tag ),
        bufferSystemGPU_( mpi::MPIManager::instance()->comm(), tag ),
        requiredBlockSelectors_( requiredBlockSelectors ),
        incompatibleBlockSelectors_( incompatibleBlockSelectors )
   {
      WALBERLA_MPI_SECTION()
      {
#if !(defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT)
         WALBERLA_CHECK(!sendDirectlyFromGPU)
#endif
      }

#if not defined(WALBERLA_BUILD_WITH_SYCL)
      for (uint_t i = 0; i < Stencil::Q; ++i)
         WALBERLA_GPU_CHECK(gpuStreamCreate(&streams_[i]))
#endif
   }


   template<typename Stencil>
   void UniformGPUScheme<Stencil>::startCommunication( )
   {
      WALBERLA_ASSERT( !communicationInProgress_ )
      auto forest = blockForest_.lock();

#if defined(WALBERLA_BUILD_WITH_SYCL)
      auto syclQueue = forest->getSYCLQueue();
#endif

      auto currentBlockForestStamp = forest->getBlockForest().getModificationStamp();
      if( setupBeforeNextCommunication_ || currentBlockForestStamp != forestModificationStamp_ )
         setupCommunication();

      // Schedule Receives
      if( sendFromGPU_ )
         bufferSystemGPU_.scheduleReceives();
      else
         bufferSystemCPU_.scheduleReceives();


      if( !sendFromGPU_ )
         for( auto it : headers_ )
            bufferSystemGPU_.sendBuffer( it.first ).clear();

      // wait until communication dependent kernels are finished
#if not defined(WALBERLA_BUILD_WITH_SYCL)
      WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
#endif

      // Start filling send buffers
      {
         for( auto &iBlock : *forest )
         {
            auto senderBlock = dynamic_cast< Block * >( &iBlock );

            if( !selectable::isSetSelected( senderBlock->getState(), requiredBlockSelectors_, incompatibleBlockSelectors_ ) )
               continue;

            for( auto dir = Stencil::beginNoCenter(); dir != Stencil::end(); ++dir )
            {
               const auto neighborIdx = blockforest::getBlockNeighborhoodSectionIndex( *dir );

               if( senderBlock->getNeighborhoodSectionSize( neighborIdx ) == uint_t( 0 ))
                  continue;

               if( !selectable::isSetSelected( senderBlock->getNeighborState( neighborIdx, uint_t(0) ), requiredBlockSelectors_, incompatibleBlockSelectors_ ) )
                  continue;

               if( senderBlock->neighborExistsLocally( neighborIdx, uint_t(0) ) && useLocalCommunication_ )
               {
                  auto receiverBlock = dynamic_cast< Block * >( forest->getBlock( senderBlock->getNeighborId( neighborIdx, uint_t(0) )) );
                  for (auto& pi : packInfos_)
                  {
#if defined(WALBERLA_BUILD_WITH_SYCL)
                     pi->communicateLocal(*dir, senderBlock, receiverBlock);
#else
                     pi->communicateLocal(*dir, senderBlock, receiverBlock, streams_[*dir]);
#endif
                  }
               }
               else
               {
                  auto nProcess = mpi::MPIRank( senderBlock->getNeighborProcess( neighborIdx, uint_t( 0 )));

                  for( auto &pi : packInfos_ )
                  {
                     auto size = pi->size( *dir, senderBlock );
#if defined(WALBERLA_BUILD_WITH_SYCL)
                     auto gpuDataPtr = syclDeviceBuffers[nProcess];
                     WALBERLA_ASSERT_NOT_NULLPTR( gpuDataPtr )
                     pi->pack( *dir, gpuDataPtr, senderBlock);
                     (*syclQueue).wait();  //TODO necessary for sync but probably very slow here
                     if( !sendFromGPU_ )
                     {
                        auto cpuDataPtr = bufferSystemCPU_.sendBuffer( nProcess ).advanceNoResize( size );
                        WALBERLA_ASSERT_NOT_NULLPTR( cpuDataPtr )
                        (*syclQueue).memcpy(cpuDataPtr, gpuDataPtr, size);
                     }
#else
                     auto gpuDataPtr = bufferSystemGPU_.sendBuffer( nProcess ).advanceNoResize( size );
                     WALBERLA_ASSERT_NOT_NULLPTR( gpuDataPtr )
                     pi->pack( *dir, gpuDataPtr, senderBlock, streams_[*dir] );
                     if( !sendFromGPU_ )
                     {
                        auto cpuDataPtr = bufferSystemCPU_.sendBuffer( nProcess ).advanceNoResize( size );
                        WALBERLA_ASSERT_NOT_NULLPTR( cpuDataPtr )
                        WALBERLA_GPU_CHECK( gpuMemcpyAsync( cpuDataPtr, gpuDataPtr, size, gpuMemcpyDeviceToHost, streams_[*dir] ))
                     }
#endif
                  }
               }
            }
         }
      }
      // wait for packing to finish
#if defined(WALBERLA_BUILD_WITH_SYCL)
      (*syclQueue).wait();
#else
      for (uint_t i = 0; i < Stencil::Q; ++i)
      {
         WALBERLA_GPU_CHECK(gpuStreamSynchronize(streams_[i]))
      }
#endif

      if( sendFromGPU_ )
         bufferSystemGPU_.sendAll();
      else
         bufferSystemCPU_.sendAll();

      communicationInProgress_ = true;
   }


   template<typename Stencil>
   void UniformGPUScheme<Stencil>::wait()
   {
      WALBERLA_ASSERT( communicationInProgress_ )

      auto forest = blockForest_.lock();

#if defined(WALBERLA_BUILD_WITH_SYCL)
      auto syclQueue = forest->getSYCLQueue();
#endif

      if( sendFromGPU_ )
      {
         for( auto recvInfo = bufferSystemGPU_.begin(); recvInfo != bufferSystemGPU_.end(); ++recvInfo )
         {
            recvInfo.buffer().clear();
            for( auto &header : headers_[recvInfo.rank()] )
            {
               auto block = dynamic_cast< Block * >( forest->getBlock( header.blockId ));

               for( auto &pi : packInfos_ )
               {
                  auto size = pi->size( header.dir, block );
#if defined(WALBERLA_BUILD_WITH_SYCL)
                  auto gpuDataPtr = syclDeviceBuffers[recvInfo.rank()];
                  WALBERLA_ASSERT_NOT_NULLPTR( gpuDataPtr )
                  pi->unpack( stencil::inverseDir[header.dir], gpuDataPtr, block );
#else
                  auto gpuDataPtr = recvInfo.buffer().advanceNoResize( size );
                  WALBERLA_ASSERT_NOT_NULLPTR( gpuDataPtr )
                  pi->unpack( stencil::inverseDir[header.dir], gpuDataPtr, block, streams_[header.dir] );
#endif
               }
            }
         }
      }
      else
      {
         for( auto recvInfo = bufferSystemCPU_.begin(); recvInfo != bufferSystemCPU_.end(); ++recvInfo )
         {
            auto &gpuBuffer = bufferSystemGPU_.sendBuffer( recvInfo.rank());

            recvInfo.buffer().clear();
            gpuBuffer.clear();
            for( auto &header : headers_[recvInfo.rank()] ) {
               auto block = dynamic_cast< Block * >( forest->getBlock( header.blockId ));

               for( auto &pi : packInfos_ )
               {
                  auto size = pi->size( header.dir, block );
                  auto cpuDataPtr = recvInfo.buffer().advanceNoResize( size );
                  WALBERLA_ASSERT_NOT_NULLPTR( cpuDataPtr )
#if defined(WALBERLA_BUILD_WITH_SYCL)
                  auto gpuDataPtr = syclDeviceBuffers[recvInfo.rank()];
                  WALBERLA_ASSERT_NOT_NULLPTR( gpuDataPtr )
                  (*syclQueue).memcpy(gpuDataPtr, cpuDataPtr, size);
                  (*syclQueue).wait();
                  pi->unpack( stencil::inverseDir[header.dir], gpuDataPtr, block );
#else
                  auto gpuDataPtr = gpuBuffer.advanceNoResize( size );
                  WALBERLA_ASSERT_NOT_NULLPTR( gpuDataPtr )
                  WALBERLA_GPU_CHECK( gpuMemcpyAsync( gpuDataPtr, cpuDataPtr, size, gpuMemcpyHostToDevice,
                                                    streams_[header.dir] ))
                  pi->unpack( stencil::inverseDir[header.dir], gpuDataPtr, block, streams_[header.dir] );
#endif
               }
            }
         }
      }
#if defined(WALBERLA_BUILD_WITH_SYCL)
      (*syclQueue).wait();
#else
      for (uint_t i = 0; i < Stencil::Q; ++i)
      {
         WALBERLA_GPU_CHECK(gpuStreamSynchronize(streams_[i]))
      }
#endif
      communicationInProgress_ = false;
   }


   template<typename Stencil>
   void UniformGPUScheme<Stencil>::setupCommunication()
   {
      auto forest = blockForest_.lock();

#if defined(WALBERLA_BUILD_WITH_SYCL)
      auto syclQueue = forest->getSYCLQueue();
#endif

      headers_.clear();

      std::map<mpi::MPIRank, mpi::MPISize> receiverInfo; // how many bytes to send to each neighbor

      mpi::BufferSystem headerExchangeBs( mpi::MPIManager::instance()->comm(), 123 );

      for( auto &iBlock : *forest ) {
         auto block = dynamic_cast< Block * >( &iBlock );

         if( !selectable::isSetSelected( block->getState(), requiredBlockSelectors_, incompatibleBlockSelectors_ ) )
            continue;

         for( auto dir = Stencil::beginNoCenter(); dir != Stencil::end(); ++dir ) {
            // skip if block has no neighbors in this direction
            const auto neighborIdx = blockforest::getBlockNeighborhoodSectionIndex( *dir );

            if( block->getNeighborhoodSectionSize( neighborIdx ) == uint_t( 0 ))
               continue;

            WALBERLA_ASSERT( block->neighborhoodSectionHasEquallySizedBlock( neighborIdx ),
                             "Works for uniform setups only" )
            WALBERLA_ASSERT_EQUAL( block->getNeighborhoodSectionSize( neighborIdx ), uint_t( 1 ),
                                   "Works for uniform setups only" )

            const BlockID &nBlockId = block->getNeighborId( neighborIdx, uint_t( 0 ));

            if( !selectable::isSetSelected( block->getNeighborState( neighborIdx, uint_t(0) ), requiredBlockSelectors_, incompatibleBlockSelectors_ ) )
               continue;

            if( block->neighborExistsLocally( neighborIdx, uint_t(0) ) && useLocalCommunication_ )
               continue;

            auto nProcess = mpi::MPIRank( block->getNeighborProcess( neighborIdx, uint_t( 0 )));

            for( auto &pi : packInfos_ )
               receiverInfo[nProcess] += mpi::MPISize( pi->size( *dir, block ));

            auto &headerBuffer = headerExchangeBs.sendBuffer( nProcess );
            nBlockId.toBuffer( headerBuffer );
            headerBuffer << *dir;
         }
      }

      headerExchangeBs.setReceiverInfoFromSendBufferState( false, true );
      headerExchangeBs.sendAll();
      for( auto recvIter = headerExchangeBs.begin(); recvIter != headerExchangeBs.end(); ++recvIter ) {
         auto &headerVector = headers_[recvIter.rank()];
         auto &buffer = recvIter.buffer();
         while ( buffer.size()) {
            Header header;
            header.blockId.fromBuffer( buffer );
            buffer >> header.dir;
            headerVector.push_back( header );
         }
      }

      bufferSystemCPU_.setReceiverInfo( receiverInfo );
      bufferSystemGPU_.setReceiverInfo( receiverInfo );

      for( auto it : receiverInfo ) {
         bufferSystemCPU_.sendBuffer( it.first ).resize( size_t(it.second) );
#if defined(WALBERLA_BUILD_WITH_SYCL)
         syclDeviceBuffers[it.first] = cl::sycl::malloc_device<uint8_t>(it.second, *syclQueue);
#else
         bufferSystemGPU_.sendBuffer( it.first ).resize( size_t(it.second) );
#endif
      }

      forestModificationStamp_ = forest->getBlockForest().getModificationStamp();
      setupBeforeNextCommunication_ = false;
   }


   template<typename Stencil>
   void UniformGPUScheme<Stencil>::addPackInfo( const shared_ptr<GeneratedGPUPackInfo> &pi )
   {
      WALBERLA_ASSERT( !communicationInProgress_, "Cannot add pack info while communication is in progress" )
      packInfos_.push_back( pi );
      setupBeforeNextCommunication_ = true;
   }

   template< typename Stencil >
   std::function<void()> UniformGPUScheme<Stencil>::getCommunicateFunctor()
   {
      return [this]() { communicate( ); };
   }

   template< typename Stencil >
   std::function<void()> UniformGPUScheme<Stencil>::getStartCommunicateFunctor()
   {
      return [this]() { startCommunication(); };
   }

   template< typename Stencil >
   std::function<void()> UniformGPUScheme<Stencil>::getWaitFunctor()
   {
      return [this]() { wait(); };
   }

} // namespace communication
} // namespace gpu
} // namespace walberla
