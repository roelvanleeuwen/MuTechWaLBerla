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
//! \\file TestSweepCollection.h
//! \\author pystencils
//======================================================================================================================

#pragma once

#include "core/DataTypes.h"
#include "core/logging/Logging.h"
#include "core/Macros.h"

#include "gpu/GPUField.h"
#include "gpu/ParallelStreams.h"

#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "domain_decomposition/StructuredBlockStorage.h"

#include "field/SwapableCompare.h"
#include "field/GhostLayerField.h"

#include <set>
#include <cmath>



using namespace std::placeholders;

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
namespace lbm {


class TestSweepCollection
{
 public:
   enum Type { ALL = 0, INNER = 1, OUTER = 2 };

   TestSweepCollection(const shared_ptr< StructuredBlockStorage > & blocks, BlockDataID pdfsID_, BlockDataID densityID_, BlockDataID velocityID_, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, double omega, const Cell & outerWidth=Cell(1, 1, 1))
      : blocks_(blocks), pdfsID(pdfsID_), densityID(densityID_), velocityID(velocityID_), cudaBlockSize0_(cudaBlockSize0), cudaBlockSize1_(cudaBlockSize1), cudaBlockSize2_(cudaBlockSize2), omega_(omega), outerWidth_(outerWidth)
   {
      

      validInnerOuterSplit_= true;

      for (auto& iBlock : *blocks)
      {
         if (int_c(blocks->getNumberOfXCells(iBlock)) <= outerWidth_[0] * 2 || int_c(blocks->getNumberOfYCells(iBlock)) <= outerWidth_[1] * 2 || int_c(blocks->getNumberOfZCells(iBlock)) <= outerWidth_[2] * 2)
            validInnerOuterSplit_ = false;
      }
   };

   

   /*************************************************************************************
   *                Internal Function Definitions with raw Pointer
   *************************************************************************************/
   static void streamCollide (gpu::GPUField<double> * pdfs, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, double omega, uint8_t timestep, const cell_idx_t ghost_layers = 0, gpuStream_t stream = nullptr);
   static void streamCollideCellInterval (gpu::GPUField<double> * pdfs, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, double omega, uint8_t timestep, const CellInterval & ci, gpuStream_t stream = nullptr);
   
   static void collide (gpu::GPUField<double> * pdfs, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, double omega, uint8_t timestep, const cell_idx_t ghost_layers = 0, gpuStream_t stream = nullptr);
   static void collideCellInterval (gpu::GPUField<double> * pdfs, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, double omega, uint8_t timestep, const CellInterval & ci, gpuStream_t stream = nullptr);
   
   static void stream (gpu::GPUField<double> * pdfs, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, uint8_t timestep, const cell_idx_t ghost_layers = 0, gpuStream_t stream = nullptr);
   static void streamCellInterval (gpu::GPUField<double> * pdfs, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, uint8_t timestep, const CellInterval & ci, gpuStream_t stream = nullptr);
   
   static void streamOnlyNoAdvancement (gpu::GPUField<double> * pdfs, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, uint8_t timestep, const cell_idx_t ghost_layers = 0, gpuStream_t stream = nullptr);
   static void streamOnlyNoAdvancementCellInterval (gpu::GPUField<double> * pdfs, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, uint8_t timestep, const CellInterval & ci, gpuStream_t stream = nullptr);
   
   static void initialise (gpu::GPUField<double> * density, gpu::GPUField<double> * pdfs, gpu::GPUField<double> * velocity, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, uint8_t timestep, const cell_idx_t ghost_layers = 0, gpuStream_t stream = nullptr);
   static void initialiseCellInterval (gpu::GPUField<double> * density, gpu::GPUField<double> * pdfs, gpu::GPUField<double> * velocity, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, uint8_t timestep, const CellInterval & ci, gpuStream_t stream = nullptr);
   
   static void calculateMacroscopicParameters (gpu::GPUField<double> * density, gpu::GPUField<double> * pdfs, gpu::GPUField<double> * velocity, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, uint8_t timestep, const cell_idx_t ghost_layers = 0, gpuStream_t stream = nullptr);
   static void calculateMacroscopicParametersCellInterval (gpu::GPUField<double> * density, gpu::GPUField<double> * pdfs, gpu::GPUField<double> * velocity, int64_t cudaBlockSize0, int64_t cudaBlockSize1, int64_t cudaBlockSize2, uint8_t timestep, const CellInterval & ci, gpuStream_t stream = nullptr);
   

   /*************************************************************************************
   *                Function Definitions for external Usage
   *************************************************************************************/

   std::function<void (IBlock *)> streamCollide()
   {
      return [this](IBlock* block) { streamCollide(block); };
   }

   std::function<void (IBlock *)> streamCollide(Type type)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this](IBlock* block) { streamCollideInner(block); };
      case Type::OUTER:
         return [this](IBlock* block) { streamCollideOuter(block); };
      default:
         return [this](IBlock* block) { streamCollide(block); };
      }
   }

   std::function<void (IBlock *)> streamCollide(Type type, const cell_idx_t ghost_layers)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this](IBlock* block) { streamCollideInner(block); };
      case Type::OUTER:
         return [this](IBlock* block) { streamCollideOuter(block); };
      default:
         return [this, ghost_layers](IBlock* block) { streamCollide(block, ghost_layers); };
      }
   }

   std::function<void (IBlock *)> streamCollide(Type type, const cell_idx_t ghost_layers, gpuStream_t gpuStream)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this, gpuStream](IBlock* block) { streamCollideInner(block, gpuStream); };
      case Type::OUTER:
         return [this, gpuStream](IBlock* block) { streamCollideOuter(block, gpuStream); };
      default:
         return [this, ghost_layers, gpuStream](IBlock* block) { streamCollide(block, ghost_layers, gpuStream); };
      }
   }

   std::function<void (IBlock *)> streamCollide(Type type, gpuStream_t gpuStream)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this, gpuStream](IBlock* block) { streamCollideInner(block, gpuStream); };
      case Type::OUTER:
         return [this, gpuStream](IBlock* block) { streamCollideOuter(block, gpuStream); };
      default:
         return [this, gpuStream](IBlock* block) { streamCollide(block, cell_idx_c(0), gpuStream); };
      }
   }

   void streamCollide(IBlock * block)
   {
      const cell_idx_t ghost_layers = 0;
      gpuStream_t gpuStream = nullptr;

      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & omega = this->omega_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();
      streamCollide(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, omega, timestep, ghost_layers, gpuStream);
      
   }

   void streamCollide(IBlock * block, const cell_idx_t ghost_layers)
   {
      gpuStream_t gpuStream = nullptr;

      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & omega = this->omega_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();
      streamCollide(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, omega, timestep, ghost_layers, gpuStream);
      
   }

   void streamCollide(IBlock * block, const cell_idx_t ghost_layers, gpuStream_t gpuStream)
   {
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & omega = this->omega_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();
      streamCollide(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, omega, timestep, ghost_layers, gpuStream);
      
   }

   void streamCollideCellInterval(IBlock * block, const CellInterval & ci, gpuStream_t gpuStream = nullptr)
   {
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & omega = this->omega_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();
      streamCollideCellInterval(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, omega, timestep, ci, gpuStream);
      
   }

   void streamCollideInner(IBlock * block, gpuStream_t gpuStream = nullptr)
   {
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & omega = this->omega_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestepPlusOne();

      CellInterval inner = pdfs->xyzSize();
      inner.expand(Cell(-outerWidth_[0], -outerWidth_[1], -outerWidth_[2]));

      streamCollideCellInterval(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, omega, timestep, inner, gpuStream);
   }

   void streamCollideOuter(IBlock * block, gpuStream_t gpuStream = nullptr)
   {

      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & omega = this->omega_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();

      if( layers_.empty() )
      {
         CellInterval ci;

         pdfs->getSliceBeforeGhostLayer(stencil::T, ci, outerWidth_[2], false);
         layers_.push_back(ci);
         pdfs->getSliceBeforeGhostLayer(stencil::B, ci, outerWidth_[2], false);
         layers_.push_back(ci);

         pdfs->getSliceBeforeGhostLayer(stencil::N, ci, outerWidth_[1], false);
         ci.expand(Cell(0, 0, -outerWidth_[2]));
         layers_.push_back(ci);
         pdfs->getSliceBeforeGhostLayer(stencil::S, ci, outerWidth_[1], false);
         ci.expand(Cell(0, 0, -outerWidth_[2]));
         layers_.push_back(ci);

         pdfs->getSliceBeforeGhostLayer(stencil::E, ci, outerWidth_[0], false);
         ci.expand(Cell(0, -outerWidth_[1], -outerWidth_[2]));
         layers_.push_back(ci);
         pdfs->getSliceBeforeGhostLayer(stencil::W, ci, outerWidth_[0], false);
         ci.expand(Cell(0, -outerWidth_[1], -outerWidth_[2]));
         layers_.push_back(ci);
      }

      
      {
         auto parallelSection_ = parallelStreams_.parallelSection( gpuStream );
         for( auto & ci: layers_ )
         {
            parallelSection_.run([&]( auto s ) {
               streamCollideCellInterval(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, omega, timestep, ci, gpuStream);
            });
         }
      }
      

      
   }
   

   std::function<void (IBlock *)> collide()
   {
      return [this](IBlock* block) { collide(block); };
   }

   std::function<void (IBlock *)> collide(Type type)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this](IBlock* block) { collideInner(block); };
      case Type::OUTER:
         return [this](IBlock* block) { collideOuter(block); };
      default:
         return [this](IBlock* block) { collide(block); };
      }
   }

   std::function<void (IBlock *)> collide(Type type, const cell_idx_t ghost_layers)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this](IBlock* block) { collideInner(block); };
      case Type::OUTER:
         return [this](IBlock* block) { collideOuter(block); };
      default:
         return [this, ghost_layers](IBlock* block) { collide(block, ghost_layers); };
      }
   }

   std::function<void (IBlock *)> collide(Type type, const cell_idx_t ghost_layers, gpuStream_t gpuStream)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this, gpuStream](IBlock* block) { collideInner(block, gpuStream); };
      case Type::OUTER:
         return [this, gpuStream](IBlock* block) { collideOuter(block, gpuStream); };
      default:
         return [this, ghost_layers, gpuStream](IBlock* block) { collide(block, ghost_layers, gpuStream); };
      }
   }

   std::function<void (IBlock *)> collide(Type type, gpuStream_t gpuStream)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this, gpuStream](IBlock* block) { collideInner(block, gpuStream); };
      case Type::OUTER:
         return [this, gpuStream](IBlock* block) { collideOuter(block, gpuStream); };
      default:
         return [this, gpuStream](IBlock* block) { collide(block, cell_idx_c(0), gpuStream); };
      }
   }

   void collide(IBlock * block)
   {
      const cell_idx_t ghost_layers = 0;
      gpuStream_t gpuStream = nullptr;

      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & omega = this->omega_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();
      collide(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, omega, timestep, ghost_layers, gpuStream);
      
   }

   void collide(IBlock * block, const cell_idx_t ghost_layers)
   {
      gpuStream_t gpuStream = nullptr;

      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & omega = this->omega_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();
      collide(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, omega, timestep, ghost_layers, gpuStream);
      
   }

   void collide(IBlock * block, const cell_idx_t ghost_layers, gpuStream_t gpuStream)
   {
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & omega = this->omega_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();
      collide(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, omega, timestep, ghost_layers, gpuStream);
      
   }

   void collideCellInterval(IBlock * block, const CellInterval & ci, gpuStream_t gpuStream = nullptr)
   {
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & omega = this->omega_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();
      collideCellInterval(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, omega, timestep, ci, gpuStream);
      
   }

   void collideInner(IBlock * block, gpuStream_t gpuStream = nullptr)
   {
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & omega = this->omega_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestepPlusOne();

      CellInterval inner = pdfs->xyzSize();
      inner.expand(Cell(-outerWidth_[0], -outerWidth_[1], -outerWidth_[2]));

      collideCellInterval(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, omega, timestep, inner, gpuStream);
   }

   void collideOuter(IBlock * block, gpuStream_t gpuStream = nullptr)
   {

      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & omega = this->omega_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();

      if( layers_.empty() )
      {
         CellInterval ci;

         pdfs->getSliceBeforeGhostLayer(stencil::T, ci, outerWidth_[2], false);
         layers_.push_back(ci);
         pdfs->getSliceBeforeGhostLayer(stencil::B, ci, outerWidth_[2], false);
         layers_.push_back(ci);

         pdfs->getSliceBeforeGhostLayer(stencil::N, ci, outerWidth_[1], false);
         ci.expand(Cell(0, 0, -outerWidth_[2]));
         layers_.push_back(ci);
         pdfs->getSliceBeforeGhostLayer(stencil::S, ci, outerWidth_[1], false);
         ci.expand(Cell(0, 0, -outerWidth_[2]));
         layers_.push_back(ci);

         pdfs->getSliceBeforeGhostLayer(stencil::E, ci, outerWidth_[0], false);
         ci.expand(Cell(0, -outerWidth_[1], -outerWidth_[2]));
         layers_.push_back(ci);
         pdfs->getSliceBeforeGhostLayer(stencil::W, ci, outerWidth_[0], false);
         ci.expand(Cell(0, -outerWidth_[1], -outerWidth_[2]));
         layers_.push_back(ci);
      }

      
      {
         auto parallelSection_ = parallelStreams_.parallelSection( gpuStream );
         for( auto & ci: layers_ )
         {
            parallelSection_.run([&]( auto s ) {
               collideCellInterval(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, omega, timestep, ci, gpuStream);
            });
         }
      }
      

      
   }
   

   std::function<void (IBlock *)> stream()
   {
      return [this](IBlock* block) { stream(block); };
   }

   std::function<void (IBlock *)> stream(Type type)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this](IBlock* block) { streamInner(block); };
      case Type::OUTER:
         return [this](IBlock* block) { streamOuter(block); };
      default:
         return [this](IBlock* block) { stream(block); };
      }
   }

   std::function<void (IBlock *)> stream(Type type, const cell_idx_t ghost_layers)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this](IBlock* block) { streamInner(block); };
      case Type::OUTER:
         return [this](IBlock* block) { streamOuter(block); };
      default:
         return [this, ghost_layers](IBlock* block) { stream(block, ghost_layers); };
      }
   }

   std::function<void (IBlock *)> stream(Type type, const cell_idx_t ghost_layers, gpuStream_t gpuStream)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this, gpuStream](IBlock* block) { streamInner(block, gpuStream); };
      case Type::OUTER:
         return [this, gpuStream](IBlock* block) { streamOuter(block, gpuStream); };
      default:
         return [this, ghost_layers, gpuStream](IBlock* block) { stream(block, ghost_layers, gpuStream); };
      }
   }

   std::function<void (IBlock *)> stream(Type type, gpuStream_t gpuStream)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this, gpuStream](IBlock* block) { streamInner(block, gpuStream); };
      case Type::OUTER:
         return [this, gpuStream](IBlock* block) { streamOuter(block, gpuStream); };
      default:
         return [this, gpuStream](IBlock* block) { stream(block, cell_idx_c(0), gpuStream); };
      }
   }

   void stream(IBlock * block)
   {
      const cell_idx_t ghost_layers = 0;
      gpuStream_t gpuStream = nullptr;

      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();
      stream(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ghost_layers, gpuStream);
      
   }

   void stream(IBlock * block, const cell_idx_t ghost_layers)
   {
      gpuStream_t gpuStream = nullptr;

      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();
      stream(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ghost_layers, gpuStream);
      
   }

   void stream(IBlock * block, const cell_idx_t ghost_layers, gpuStream_t gpuStream)
   {
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();
      stream(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ghost_layers, gpuStream);
      
   }

   void streamCellInterval(IBlock * block, const CellInterval & ci, gpuStream_t gpuStream = nullptr)
   {
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();
      streamCellInterval(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ci, gpuStream);
      
   }

   void streamInner(IBlock * block, gpuStream_t gpuStream = nullptr)
   {
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestepPlusOne();

      CellInterval inner = pdfs->xyzSize();
      inner.expand(Cell(-outerWidth_[0], -outerWidth_[1], -outerWidth_[2]));

      streamCellInterval(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, inner, gpuStream);
   }

   void streamOuter(IBlock * block, gpuStream_t gpuStream = nullptr)
   {

      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->advanceTimestep();

      if( layers_.empty() )
      {
         CellInterval ci;

         pdfs->getSliceBeforeGhostLayer(stencil::T, ci, outerWidth_[2], false);
         layers_.push_back(ci);
         pdfs->getSliceBeforeGhostLayer(stencil::B, ci, outerWidth_[2], false);
         layers_.push_back(ci);

         pdfs->getSliceBeforeGhostLayer(stencil::N, ci, outerWidth_[1], false);
         ci.expand(Cell(0, 0, -outerWidth_[2]));
         layers_.push_back(ci);
         pdfs->getSliceBeforeGhostLayer(stencil::S, ci, outerWidth_[1], false);
         ci.expand(Cell(0, 0, -outerWidth_[2]));
         layers_.push_back(ci);

         pdfs->getSliceBeforeGhostLayer(stencil::E, ci, outerWidth_[0], false);
         ci.expand(Cell(0, -outerWidth_[1], -outerWidth_[2]));
         layers_.push_back(ci);
         pdfs->getSliceBeforeGhostLayer(stencil::W, ci, outerWidth_[0], false);
         ci.expand(Cell(0, -outerWidth_[1], -outerWidth_[2]));
         layers_.push_back(ci);
      }

      
      {
         auto parallelSection_ = parallelStreams_.parallelSection( gpuStream );
         for( auto & ci: layers_ )
         {
            parallelSection_.run([&]( auto s ) {
               streamCellInterval(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ci, gpuStream);
            });
         }
      }
      

      
   }
   

   std::function<void (IBlock *)> streamOnlyNoAdvancement()
   {
      return [this](IBlock* block) { streamOnlyNoAdvancement(block); };
   }

   std::function<void (IBlock *)> streamOnlyNoAdvancement(Type type)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this](IBlock* block) { streamOnlyNoAdvancementInner(block); };
      case Type::OUTER:
         return [this](IBlock* block) { streamOnlyNoAdvancementOuter(block); };
      default:
         return [this](IBlock* block) { streamOnlyNoAdvancement(block); };
      }
   }

   std::function<void (IBlock *)> streamOnlyNoAdvancement(Type type, const cell_idx_t ghost_layers)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this](IBlock* block) { streamOnlyNoAdvancementInner(block); };
      case Type::OUTER:
         return [this](IBlock* block) { streamOnlyNoAdvancementOuter(block); };
      default:
         return [this, ghost_layers](IBlock* block) { streamOnlyNoAdvancement(block, ghost_layers); };
      }
   }

   std::function<void (IBlock *)> streamOnlyNoAdvancement(Type type, const cell_idx_t ghost_layers, gpuStream_t gpuStream)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this, gpuStream](IBlock* block) { streamOnlyNoAdvancementInner(block, gpuStream); };
      case Type::OUTER:
         return [this, gpuStream](IBlock* block) { streamOnlyNoAdvancementOuter(block, gpuStream); };
      default:
         return [this, ghost_layers, gpuStream](IBlock* block) { streamOnlyNoAdvancement(block, ghost_layers, gpuStream); };
      }
   }

   std::function<void (IBlock *)> streamOnlyNoAdvancement(Type type, gpuStream_t gpuStream)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this, gpuStream](IBlock* block) { streamOnlyNoAdvancementInner(block, gpuStream); };
      case Type::OUTER:
         return [this, gpuStream](IBlock* block) { streamOnlyNoAdvancementOuter(block, gpuStream); };
      default:
         return [this, gpuStream](IBlock* block) { streamOnlyNoAdvancement(block, cell_idx_c(0), gpuStream); };
      }
   }

   void streamOnlyNoAdvancement(IBlock * block)
   {
      const cell_idx_t ghost_layers = 0;
      gpuStream_t gpuStream = nullptr;

      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestepPlusOne();
      streamOnlyNoAdvancement(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ghost_layers, gpuStream);
      
   }

   void streamOnlyNoAdvancement(IBlock * block, const cell_idx_t ghost_layers)
   {
      gpuStream_t gpuStream = nullptr;

      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestepPlusOne();
      streamOnlyNoAdvancement(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ghost_layers, gpuStream);
      
   }

   void streamOnlyNoAdvancement(IBlock * block, const cell_idx_t ghost_layers, gpuStream_t gpuStream)
   {
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestepPlusOne();
      streamOnlyNoAdvancement(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ghost_layers, gpuStream);
      
   }

   void streamOnlyNoAdvancementCellInterval(IBlock * block, const CellInterval & ci, gpuStream_t gpuStream = nullptr)
   {
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestepPlusOne();
      streamOnlyNoAdvancementCellInterval(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ci, gpuStream);
      
   }

   void streamOnlyNoAdvancementInner(IBlock * block, gpuStream_t gpuStream = nullptr)
   {
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestepPlusOne();

      CellInterval inner = pdfs->xyzSize();
      inner.expand(Cell(-outerWidth_[0], -outerWidth_[1], -outerWidth_[2]));

      streamOnlyNoAdvancementCellInterval(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, inner, gpuStream);
   }

   void streamOnlyNoAdvancementOuter(IBlock * block, gpuStream_t gpuStream = nullptr)
   {

      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestepPlusOne();

      if( layers_.empty() )
      {
         CellInterval ci;

         pdfs->getSliceBeforeGhostLayer(stencil::T, ci, outerWidth_[2], false);
         layers_.push_back(ci);
         pdfs->getSliceBeforeGhostLayer(stencil::B, ci, outerWidth_[2], false);
         layers_.push_back(ci);

         pdfs->getSliceBeforeGhostLayer(stencil::N, ci, outerWidth_[1], false);
         ci.expand(Cell(0, 0, -outerWidth_[2]));
         layers_.push_back(ci);
         pdfs->getSliceBeforeGhostLayer(stencil::S, ci, outerWidth_[1], false);
         ci.expand(Cell(0, 0, -outerWidth_[2]));
         layers_.push_back(ci);

         pdfs->getSliceBeforeGhostLayer(stencil::E, ci, outerWidth_[0], false);
         ci.expand(Cell(0, -outerWidth_[1], -outerWidth_[2]));
         layers_.push_back(ci);
         pdfs->getSliceBeforeGhostLayer(stencil::W, ci, outerWidth_[0], false);
         ci.expand(Cell(0, -outerWidth_[1], -outerWidth_[2]));
         layers_.push_back(ci);
      }

      
      {
         auto parallelSection_ = parallelStreams_.parallelSection( gpuStream );
         for( auto & ci: layers_ )
         {
            parallelSection_.run([&]( auto s ) {
               streamOnlyNoAdvancementCellInterval(pdfs, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ci, gpuStream);
            });
         }
      }
      

      
   }
   

   std::function<void (IBlock *)> initialise()
   {
      return [this](IBlock* block) { initialise(block); };
   }

   std::function<void (IBlock *)> initialise(Type type)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this](IBlock* block) { initialiseInner(block); };
      case Type::OUTER:
         return [this](IBlock* block) { initialiseOuter(block); };
      default:
         return [this](IBlock* block) { initialise(block); };
      }
   }

   std::function<void (IBlock *)> initialise(Type type, const cell_idx_t ghost_layers)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this](IBlock* block) { initialiseInner(block); };
      case Type::OUTER:
         return [this](IBlock* block) { initialiseOuter(block); };
      default:
         return [this, ghost_layers](IBlock* block) { initialise(block, ghost_layers); };
      }
   }

   std::function<void (IBlock *)> initialise(Type type, const cell_idx_t ghost_layers, gpuStream_t gpuStream)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this, gpuStream](IBlock* block) { initialiseInner(block, gpuStream); };
      case Type::OUTER:
         return [this, gpuStream](IBlock* block) { initialiseOuter(block, gpuStream); };
      default:
         return [this, ghost_layers, gpuStream](IBlock* block) { initialise(block, ghost_layers, gpuStream); };
      }
   }

   std::function<void (IBlock *)> initialise(Type type, gpuStream_t gpuStream)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this, gpuStream](IBlock* block) { initialiseInner(block, gpuStream); };
      case Type::OUTER:
         return [this, gpuStream](IBlock* block) { initialiseOuter(block, gpuStream); };
      default:
         return [this, gpuStream](IBlock* block) { initialise(block, cell_idx_c(0), gpuStream); };
      }
   }

   void initialise(IBlock * block)
   {
      const cell_idx_t ghost_layers = 0;
      gpuStream_t gpuStream = nullptr;

      auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
      auto density = block->getData< gpu::GPUField<double> >(densityID);
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestep();
      initialise(density, pdfs, velocity, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ghost_layers, gpuStream);
      
   }

   void initialise(IBlock * block, const cell_idx_t ghost_layers)
   {
      gpuStream_t gpuStream = nullptr;

      auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
      auto density = block->getData< gpu::GPUField<double> >(densityID);
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestep();
      initialise(density, pdfs, velocity, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ghost_layers, gpuStream);
      
   }

   void initialise(IBlock * block, const cell_idx_t ghost_layers, gpuStream_t gpuStream)
   {
      auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
      auto density = block->getData< gpu::GPUField<double> >(densityID);
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestep();
      initialise(density, pdfs, velocity, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ghost_layers, gpuStream);
      
   }

   void initialiseCellInterval(IBlock * block, const CellInterval & ci, gpuStream_t gpuStream = nullptr)
   {
      auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
      auto density = block->getData< gpu::GPUField<double> >(densityID);
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestep();
      initialiseCellInterval(density, pdfs, velocity, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ci, gpuStream);
      
   }

   void initialiseInner(IBlock * block, gpuStream_t gpuStream = nullptr)
   {
      auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
      auto density = block->getData< gpu::GPUField<double> >(densityID);
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestep();

      CellInterval inner = density->xyzSize();
      inner.expand(Cell(-outerWidth_[0], -outerWidth_[1], -outerWidth_[2]));

      initialiseCellInterval(density, pdfs, velocity, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, inner, gpuStream);
   }

   void initialiseOuter(IBlock * block, gpuStream_t gpuStream = nullptr)
   {

      auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
      auto density = block->getData< gpu::GPUField<double> >(densityID);
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestep();

      if( layers_.empty() )
      {
         CellInterval ci;

         density->getSliceBeforeGhostLayer(stencil::T, ci, outerWidth_[2], false);
         layers_.push_back(ci);
         density->getSliceBeforeGhostLayer(stencil::B, ci, outerWidth_[2], false);
         layers_.push_back(ci);

         density->getSliceBeforeGhostLayer(stencil::N, ci, outerWidth_[1], false);
         ci.expand(Cell(0, 0, -outerWidth_[2]));
         layers_.push_back(ci);
         density->getSliceBeforeGhostLayer(stencil::S, ci, outerWidth_[1], false);
         ci.expand(Cell(0, 0, -outerWidth_[2]));
         layers_.push_back(ci);

         density->getSliceBeforeGhostLayer(stencil::E, ci, outerWidth_[0], false);
         ci.expand(Cell(0, -outerWidth_[1], -outerWidth_[2]));
         layers_.push_back(ci);
         density->getSliceBeforeGhostLayer(stencil::W, ci, outerWidth_[0], false);
         ci.expand(Cell(0, -outerWidth_[1], -outerWidth_[2]));
         layers_.push_back(ci);
      }

      
      {
         auto parallelSection_ = parallelStreams_.parallelSection( gpuStream );
         for( auto & ci: layers_ )
         {
            parallelSection_.run([&]( auto s ) {
               initialiseCellInterval(density, pdfs, velocity, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ci, gpuStream);
            });
         }
      }
      

      
   }
   

   std::function<void (IBlock *)> calculateMacroscopicParameters()
   {
      return [this](IBlock* block) { calculateMacroscopicParameters(block); };
   }

   std::function<void (IBlock *)> calculateMacroscopicParameters(Type type)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this](IBlock* block) { calculateMacroscopicParametersInner(block); };
      case Type::OUTER:
         return [this](IBlock* block) { calculateMacroscopicParametersOuter(block); };
      default:
         return [this](IBlock* block) { calculateMacroscopicParameters(block); };
      }
   }

   std::function<void (IBlock *)> calculateMacroscopicParameters(Type type, const cell_idx_t ghost_layers)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this](IBlock* block) { calculateMacroscopicParametersInner(block); };
      case Type::OUTER:
         return [this](IBlock* block) { calculateMacroscopicParametersOuter(block); };
      default:
         return [this, ghost_layers](IBlock* block) { calculateMacroscopicParameters(block, ghost_layers); };
      }
   }

   std::function<void (IBlock *)> calculateMacroscopicParameters(Type type, const cell_idx_t ghost_layers, gpuStream_t gpuStream)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this, gpuStream](IBlock* block) { calculateMacroscopicParametersInner(block, gpuStream); };
      case Type::OUTER:
         return [this, gpuStream](IBlock* block) { calculateMacroscopicParametersOuter(block, gpuStream); };
      default:
         return [this, ghost_layers, gpuStream](IBlock* block) { calculateMacroscopicParameters(block, ghost_layers, gpuStream); };
      }
   }

   std::function<void (IBlock *)> calculateMacroscopicParameters(Type type, gpuStream_t gpuStream)
   {
      if (!validInnerOuterSplit_ && type != Type::ALL)
         WALBERLA_ABORT_NO_DEBUG_INFO("innerOuterSplit too large - make it smaller, increase cellsPerBlock or avoid communication hiding")

      switch (type)
      {
      case Type::INNER:
         return [this, gpuStream](IBlock* block) { calculateMacroscopicParametersInner(block, gpuStream); };
      case Type::OUTER:
         return [this, gpuStream](IBlock* block) { calculateMacroscopicParametersOuter(block, gpuStream); };
      default:
         return [this, gpuStream](IBlock* block) { calculateMacroscopicParameters(block, cell_idx_c(0), gpuStream); };
      }
   }

   void calculateMacroscopicParameters(IBlock * block)
   {
      const cell_idx_t ghost_layers = 0;
      gpuStream_t gpuStream = nullptr;

      auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
      auto density = block->getData< gpu::GPUField<double> >(densityID);
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestep();
      calculateMacroscopicParameters(density, pdfs, velocity, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ghost_layers, gpuStream);
      
   }

   void calculateMacroscopicParameters(IBlock * block, const cell_idx_t ghost_layers)
   {
      gpuStream_t gpuStream = nullptr;

      auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
      auto density = block->getData< gpu::GPUField<double> >(densityID);
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestep();
      calculateMacroscopicParameters(density, pdfs, velocity, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ghost_layers, gpuStream);
      
   }

   void calculateMacroscopicParameters(IBlock * block, const cell_idx_t ghost_layers, gpuStream_t gpuStream)
   {
      auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
      auto density = block->getData< gpu::GPUField<double> >(densityID);
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestep();
      calculateMacroscopicParameters(density, pdfs, velocity, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ghost_layers, gpuStream);
      
   }

   void calculateMacroscopicParametersCellInterval(IBlock * block, const CellInterval & ci, gpuStream_t gpuStream = nullptr)
   {
      auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
      auto density = block->getData< gpu::GPUField<double> >(densityID);
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestep();
      calculateMacroscopicParametersCellInterval(density, pdfs, velocity, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ci, gpuStream);
      
   }

   void calculateMacroscopicParametersInner(IBlock * block, gpuStream_t gpuStream = nullptr)
   {
      auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
      auto density = block->getData< gpu::GPUField<double> >(densityID);
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestep();

      CellInterval inner = density->xyzSize();
      inner.expand(Cell(-outerWidth_[0], -outerWidth_[1], -outerWidth_[2]));

      calculateMacroscopicParametersCellInterval(density, pdfs, velocity, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, inner, gpuStream);
   }

   void calculateMacroscopicParametersOuter(IBlock * block, gpuStream_t gpuStream = nullptr)
   {

      auto velocity = block->getData< gpu::GPUField<double> >(velocityID);
      auto density = block->getData< gpu::GPUField<double> >(densityID);
      auto pdfs = block->getData< gpu::GPUField<double> >(pdfsID);

      auto & cudaBlockSize0 = this->cudaBlockSize0_;
      auto & cudaBlockSize1 = this->cudaBlockSize1_;
      auto & cudaBlockSize2 = this->cudaBlockSize2_;
      uint8_t timestep = pdfs->getTimestep();

      if( layers_.empty() )
      {
         CellInterval ci;

         density->getSliceBeforeGhostLayer(stencil::T, ci, outerWidth_[2], false);
         layers_.push_back(ci);
         density->getSliceBeforeGhostLayer(stencil::B, ci, outerWidth_[2], false);
         layers_.push_back(ci);

         density->getSliceBeforeGhostLayer(stencil::N, ci, outerWidth_[1], false);
         ci.expand(Cell(0, 0, -outerWidth_[2]));
         layers_.push_back(ci);
         density->getSliceBeforeGhostLayer(stencil::S, ci, outerWidth_[1], false);
         ci.expand(Cell(0, 0, -outerWidth_[2]));
         layers_.push_back(ci);

         density->getSliceBeforeGhostLayer(stencil::E, ci, outerWidth_[0], false);
         ci.expand(Cell(0, -outerWidth_[1], -outerWidth_[2]));
         layers_.push_back(ci);
         density->getSliceBeforeGhostLayer(stencil::W, ci, outerWidth_[0], false);
         ci.expand(Cell(0, -outerWidth_[1], -outerWidth_[2]));
         layers_.push_back(ci);
      }

      
      {
         auto parallelSection_ = parallelStreams_.parallelSection( gpuStream );
         for( auto & ci: layers_ )
         {
            parallelSection_.run([&]( auto s ) {
               calculateMacroscopicParametersCellInterval(density, pdfs, velocity, cudaBlockSize0, cudaBlockSize1, cudaBlockSize2, timestep, ci, gpuStream);
            });
         }
      }
      

      
   }
   

   
   void setOuterPriority(int priority)
   {
      parallelStreams_.setStreamPriority(priority);
   }
   

 private:
   shared_ptr< StructuredBlockStorage > blocks_;
   BlockDataID pdfsID;
    BlockDataID densityID;
    BlockDataID velocityID;
    int64_t cudaBlockSize0_;
    int64_t cudaBlockSize1_;
    int64_t cudaBlockSize2_;
    double omega_;

   Cell outerWidth_;
   std::vector<CellInterval> layers_;
   bool validInnerOuterSplit_;

   gpu::ParallelStreams parallelStreams_;
   // std::map<BlockID, gpuStream_t > streams_;
};


} // namespace lbm
} // namespace walberla


#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic pop
#endif