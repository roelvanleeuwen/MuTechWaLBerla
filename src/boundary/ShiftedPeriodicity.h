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
//! \file ShiftedPeriodicity.cpp
//! \ingroup boundary
//! \author Helen Schottenhamml <helen.schottenhamml@fau.de>
//
//======================================================================================================================

#include "blockforest/Block.h"
#include "blockforest/BlockID.h"
#include "blockforest/StructuredBlockForest.h"

#include "core/DataTypes.h"
#include "core/cell/CellInterval.h"
#include "core/debug/CheckFunctions.h"
#include "core/debug/Debug.h"
#include "core/logging/Logging.h"
#include "core/math/AABBFwd.h"
#include "core/math/Vector3.h"
#include "core/mpi/MPIWrapper.h"
#include "core/mpi/MPIManager.h"

#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "domain_decomposition/IBlockID.h"
#include "domain_decomposition/MapPointToPeriodicDomain.h"

#include <array>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <mpi.h>


namespace walberla {
namespace boundary {

template<typename GhostLayerField_T, typename FlagField_T >
class ShiftedPeriodicity {

   using ValueType = typename GhostLayerField_T::value_type;
   using ShiftType = int;

 public:

   ShiftedPeriodicity( const std::weak_ptr<StructuredBlockForest> & blockForest,
                       const BlockDataID& flagFieldID, const BlockDataID & fieldID, const uint_t fieldGhostLayers,
                       const Vector3<uint_t> & boundaryNormal, const Vector3< ShiftType > & shift )
      : blockForest_(blockForest), flagFieldID_(flagFieldID),
        fieldID_( fieldID ), fieldGhostLayers_(fieldGhostLayers),
        shift_( shift[0], shift[1], shift[2] ),
        boundaryNormal_(boundaryNormal)
   {
      auto sbf = blockForest_.lock();

      WALBERLA_ASSERT_NOT_NULLPTR( sbf )

      WALBERLA_CHECK(sbf->storesUniformBlockGrid(), "Periodic shift is currently only implemented for uniform grids.")
      WALBERLA_CHECK(sbf->containsGlobalBlockInformation(), "For the periodic shift, the blockforest must be constructed to retain global information.")

      WALBERLA_CHECK_GREATER(shift.length(), 0, "Shift value must be a non-negative value.")

      uint_t shiftSum{};
      for( uint_t d = 0; d < 3; ++d ) {
         if(std::abs(shift_[d]) > 1e-8) {
            shiftDir_ = d;
            ++shiftSum;
         }
      }

      uint_t normalSum{};
      for( uint_t d = 0; d < 3; ++d ) {
         if(boundaryNormal_[d]) {
            normalDir_ = d;
            ++normalSum;
         }
      }

      // sanity checks
      WALBERLA_CHECK_EQUAL( shiftSum, 1, "Periodic shift can only be applied in one direction." )
      WALBERLA_CHECK_EQUAL( normalSum, 1, "Periodic shift can only be applied to straight, axis-aligned boundaries." )
      WALBERLA_CHECK_UNEQUAL( shiftDir_, normalDir_, "Direction of periodic shift and boundary normal must not coincide." )

      WALBERLA_CHECK( sbf->isPeriodic(shiftDir_), "Blockforest must be periodic in direction " << shiftDir_ << "!" )
      WALBERLA_CHECK( !sbf->isPeriodic(normalDir_), "Blockforest must NOT be periodic in direction " << boundaryNormal_ << "!" )

      for(uint_t d = 0; d < 3; ++d) {
         WALBERLA_CHECK_LESS(std::abs(shift[d]), sbf->getNumberOfCells(d), "Please chose a shift value whose absolute is smaller than the domain size in shift direction.")
      }
   }

   ShiftedPeriodicity( const std::weak_ptr<StructuredBlockForest> & blockForest,
                       const BlockDataID& flagFieldID, const BlockDataID & fieldID, const uint_t fieldGhostLayers,
                       const Vector3<uint_t> & boundaryNormal, const ShiftType xShift, const ShiftType yShift, const ShiftType zShift )
      : ShiftedPeriodicity(blockForest, flagFieldID, fieldID, fieldGhostLayers,
                           boundaryNormal, Vector3<ShiftType >(xShift, yShift, zShift))
   {}

   uint_t shiftDirection() const { return shiftDir_; }
   uint_t normalDirection() const { return normalDir_; }

   void operator()() {

      auto mpiInstance = mpi::MPIManager::instance();
      const auto currentRank = numeric_cast<mpi::MPIRank>(mpiInstance->rank());

      const auto sbf = blockForest_.lock();
      WALBERLA_ASSERT_NOT_NULLPTR( sbf )

     // if shift equals zero or domain size in this direction, we do not need to do anything
     if(shift_[shiftDir_] == ShiftType(sbf->getNumberOfCells(shiftDir_)) || shift_[shiftDir_] == 0)
        return;

      if(!setupPeriodicity_){
         setupPeriodicity();
         setupPeriodicity_ = true;
      }

      // set up local to avoid temporary fields; key is unique tag to ensure correct communication
      // if several messages have the same source and destination
      // thanks to the unique tag, the same buffer can be used for both local and MPI communication
      std::map<int, std::vector<ValueType>> buffer;
      std::map<BlockID, std::map<int, MPI_Request>> sendRequests;
      std::map<BlockID, std::map<int, MPI_Request>> recvRequests;

      std::vector<IBlock*> localBlocks{};
      sbf->getBlocks(localBlocks);

      /// packing and schedule sends

      for( auto block : localBlocks ) {

         const auto blockID = dynamic_cast<Block*>(block)->getId();

         WALBERLA_ASSERT_EQUAL(sendAABBs_[blockID].size(), recvAABBs_[blockID].size() )

         for(uint_t i = 0; i < sendAABBs_[blockID].size(); ++i) {

            auto & sendInfo = sendAABBs_[blockID][i];

            const auto sendRank = std::get<0>(sendInfo);
            const auto sendTag = std::get<3>(sendInfo);

            // transform AABB to CI
            const auto sendAABB = std::get<2>(sendInfo);
            const auto sendCI = globalAABBToLocalCI(sendAABB, sbf, block);

            const auto sendSize = sendCI.numCells() * GhostLayerField_T::F_SIZE;

            if (sendRank == currentRank) {
               buffer[sendTag].resize(sendSize);
               packBuffer(block, sendCI, buffer[sendTag]);
            } else {
               buffer[sendTag].resize(sendSize);
               packBuffer(block, sendCI, buffer[sendTag]);

               // schedule sends
               MPI_Isend(buffer[sendTag].data(), int_c(buffer[sendTag].size() * sizeof(ValueType)), MPI_BYTE,
                         sendRank, sendTag, mpiInstance->comm(), & sendRequests[blockID][sendTag]);

            }

         }

      }

      /// schedule receives

      for( auto block : localBlocks ) {

         const auto blockID = dynamic_cast<Block*>(block)->getId();

         WALBERLA_ASSERT_EQUAL(sendAABBs_[blockID].size(), recvAABBs_[blockID].size() )

         for(uint_t i = 0; i < recvAABBs_[blockID].size(); ++i) {

            auto & recvInfo = recvAABBs_[blockID][i];

            const auto recvRank = std::get<0>(recvInfo);
            const auto recvTag   = std::get<3>(recvInfo);
            const auto recvAABB = std::get<2>(recvInfo);
            const auto recvCI = globalAABBToLocalCI(recvAABB, sbf, block);

            // do not schedule receives for local communication
            if (recvRank != currentRank) {
               const auto recvSize = recvCI.numCells() * GhostLayerField_T::F_SIZE;
               buffer[recvTag].resize(recvSize);

               // Schedule receives
               MPI_Irecv(buffer[recvTag].data(), int_c(buffer[recvTag].size() * sizeof(ValueType)), MPI_BYTE,
                         recvRank, recvTag, mpiInstance->comm(), & recvRequests[blockID][recvTag]);

            }

         }

      }

      /// unpacking

      for( auto block : localBlocks ) {

         const auto blockID = dynamic_cast<Block*>(block)->getId();

         for(uint_t i = 0; i < recvAABBs_[blockID].size(); ++i) {

            auto & recvInfo = recvAABBs_[blockID][i];

            const auto recvRank = std::get<0>(recvInfo);
            const auto recvTag = std::get<3>(recvInfo);

            const auto recvAABB = std::get<2>(recvInfo);
            const auto recvCI = globalAABBToLocalCI(recvAABB, sbf, block);

            if(recvRank == currentRank) {
               WALBERLA_ASSERT_GREATER(buffer.count(recvTag), 0)
               unpackBuffer(block, recvCI, buffer[recvTag]);
            } else {
               MPI_Status status;
               MPI_Wait(&recvRequests[blockID][recvTag], &status);

               WALBERLA_ASSERT_GREATER(buffer.count(recvTag), 0)
               unpackBuffer(block, recvCI, buffer[recvTag]);
            }

         }

      }

      // wait for all communication to be finished
      for(auto & block : localBlocks) {
         const auto blockID = dynamic_cast<Block*>(block)->getId();

         if (sendRequests[blockID].empty())
            return;

         std::vector<MPI_Request > v;
         std::transform( sendRequests[blockID].begin(), sendRequests[blockID].end(), std::back_inserter( v ),
                        second( sendRequests[blockID] ));
         MPI_Waitall(int_c(sendRequests[blockID].size()), v.data(), MPI_STATUSES_IGNORE);
      }

   }

 private:

   void processDoubleAABB( const AABB & originalAABB, const std::shared_ptr<StructuredBlockForest> & sbf,
                           const real_t nGL, const BlockID & blockID, const int normalShift ) {

      WALBERLA_ASSERT(normalShift == -1 || normalShift == 1)

      const auto offset = ShiftType(int_c(shift_[shiftDir_]) % int_c(sbf->getNumberOfCellsPerBlock(shiftDir_)));
      Vector3<ShiftType> normalShiftVector{};
      normalShiftVector[normalDir_] = ShiftType(normalShift * int_c(sbf->getNumberOfCellsPerBlock(normalDir_)));

      // aabbs for packing
      auto sendAABB1 = originalAABB;
      auto sendAABB2 = originalAABB;
      sendAABB1.setAxisBounds(shiftDir_, sendAABB1.min(shiftDir_), sendAABB1.max(shiftDir_) - real_c(offset));
      sendAABB2.setAxisBounds(shiftDir_, sendAABB2.max(shiftDir_) - real_c(offset), sendAABB2.max(shiftDir_));

      auto sendCenter1 = sendAABB1.center();
      auto sendCenter2 = sendAABB2.center();

      sendAABB1.setAxisBounds(shiftDir_, sendAABB1.min(shiftDir_), sendAABB1.max(shiftDir_) + nGL);
      sendAABB2.setAxisBounds(shiftDir_, sendAABB2.min(shiftDir_) - nGL, sendAABB2.max(shiftDir_));

      const auto remDir = uint_t(3) - normalDir_ - shiftDir_;

      // account for ghost layers
      auto localSendAABB1 = sendAABB1;
      localSendAABB1.setAxisBounds(remDir, localSendAABB1.min(remDir) - nGL, localSendAABB1.max(remDir) + nGL);
      auto localSendAABB2 = sendAABB2;
      localSendAABB2.setAxisBounds(remDir, localSendAABB2.min(remDir) - nGL, localSendAABB2.max(remDir) + nGL);

      // aabbs for unpacking
      auto recvAABB1 = originalAABB;
      auto recvAABB2 = originalAABB;

      recvAABB1.setAxisBounds(shiftDir_, recvAABB1.min(shiftDir_), recvAABB1.min(shiftDir_) + real_c(offset));
      recvAABB2.setAxisBounds(shiftDir_, recvAABB2.min(shiftDir_) + real_c(offset), recvAABB2.max(shiftDir_));

      auto recvCenter1 = recvAABB1.center();
      auto recvCenter2 = recvAABB2.center();

      recvAABB1.setAxisBounds(shiftDir_, recvAABB1.min(shiftDir_) - nGL, recvAABB1.max(shiftDir_));
      recvAABB2.setAxisBounds(shiftDir_, recvAABB2.min(shiftDir_), recvAABB2.max(shiftDir_) + nGL);

      // receive from the interior of domain in ghost layers
      auto localRecvAABB1 = recvAABB1;
      localRecvAABB1.setAxisBounds(remDir, localRecvAABB1.min(remDir) - nGL, localRecvAABB1.max(remDir) + nGL);
      auto localRecvAABB2 = recvAABB2;
      localRecvAABB2.setAxisBounds(remDir, localRecvAABB2.min(remDir) - nGL, localRecvAABB2.max(remDir) + nGL);

      if(normalShift == 1) { // at maximum of domain -> send inner layers, receive ghost layers
         localSendAABB1.setAxisBounds(normalDir_, localSendAABB1.max(normalDir_) - nGL, localSendAABB1.max(normalDir_));
         localSendAABB2.setAxisBounds(normalDir_, localSendAABB2.max(normalDir_) - nGL, localSendAABB2.max(normalDir_));
         localRecvAABB1.setAxisBounds(normalDir_, localRecvAABB1.max(normalDir_), localRecvAABB1.max(normalDir_) + nGL);
         localRecvAABB2.setAxisBounds(normalDir_, localRecvAABB2.max(normalDir_), localRecvAABB2.max(normalDir_) + nGL);
      } else { // at minimum of domain -> send inner layers, receive ghost layers
         localSendAABB1.setAxisBounds(normalDir_, localSendAABB1.min(normalDir_), localSendAABB1.min(normalDir_) + nGL);
         localSendAABB2.setAxisBounds(normalDir_, localSendAABB2.min(normalDir_), localSendAABB2.min(normalDir_) + nGL);
         localRecvAABB1.setAxisBounds(normalDir_, localRecvAABB1.min(normalDir_) - nGL, localRecvAABB1.min(normalDir_));
         localRecvAABB2.setAxisBounds(normalDir_, localRecvAABB2.min(normalDir_) - nGL, localRecvAABB2.min(normalDir_));
      }

      WALBERLA_ASSERT_FLOAT_EQUAL(localSendAABB1.volume(), localRecvAABB2.volume())
      WALBERLA_ASSERT_FLOAT_EQUAL(localSendAABB2.volume(), localRecvAABB1.volume())

      sendCenter1 += ( shift_ + normalShiftVector);
      sendCenter2 += ( shift_ + normalShiftVector);
      recvCenter1 += (-shift_ + normalShiftVector);
      recvCenter2 += (-shift_ + normalShiftVector);

      std::array<bool, 3> virtualPeriodicity{false};
      virtualPeriodicity[normalDir_] = true;
      virtualPeriodicity[shiftDir_] = true;

      domain_decomposition::mapPointToPeriodicDomain( virtualPeriodicity, sbf->getDomain(), sendCenter1 );
      domain_decomposition::mapPointToPeriodicDomain( virtualPeriodicity, sbf->getDomain(), sendCenter2 );
      domain_decomposition::mapPointToPeriodicDomain( virtualPeriodicity, sbf->getDomain(), recvCenter1 );
      domain_decomposition::mapPointToPeriodicDomain( virtualPeriodicity, sbf->getDomain(), recvCenter2 );

      uint_t sendRank1;
      uint_t sendRank2;
      uint_t recvRank1;
      uint_t recvRank2;

      BlockID sendID1;
      BlockID sendID2;
      BlockID recvID1;
      BlockID recvID2;

      const auto blockInformation = sbf->getBlockInformation();
      WALBERLA_ASSERT(blockInformation.active())

      blockInformation.getProcess(sendRank1, sendCenter1[0], sendCenter1[1], sendCenter1[2]);
      blockInformation.getProcess(sendRank2, sendCenter2[0], sendCenter2[1], sendCenter2[2]);
      blockInformation.getProcess(recvRank1, recvCenter1[0], recvCenter1[1], recvCenter1[2]);
      blockInformation.getProcess(recvRank2, recvCenter2[0], recvCenter2[1], recvCenter2[2]);

      blockInformation.getId(sendID1, sendCenter1[0], sendCenter1[1], sendCenter1[2]);
      blockInformation.getId(sendID2, sendCenter2[0], sendCenter2[1], sendCenter2[2]);
      blockInformation.getId(recvID1, recvCenter1[0], recvCenter1[1], recvCenter1[2]);
      blockInformation.getId(recvID2, recvCenter2[0], recvCenter2[1], recvCenter2[2]);

      // define unique send / receive tags for communication

      const int atMaxTagSend = normalShift < 0 ? 0 : 1;
      const int atMaxTagRecv = normalShift < 0 ? 1 : 0;

      const int sendTag1 = ((int_c(blockID.getID()) + int_c(sendID1.getID() * numGlobalBlocks_)) * 2 + atMaxTagSend) * 2 + 0;
      const int sendTag2 = ((int_c(blockID.getID()) + int_c(sendID2.getID() * numGlobalBlocks_)) * 2 + atMaxTagSend) * 2 + 1;
      const int recvTag2 = ((int_c(recvID2.getID()) + int_c(blockID.getID() * numGlobalBlocks_)) * 2 + atMaxTagRecv) * 2 + 0;
      const int recvTag1 = ((int_c(recvID1.getID()) + int_c(blockID.getID() * numGlobalBlocks_)) * 2 + atMaxTagRecv) * 2 + 1;

      sendAABBs_[blockID].emplace_back(mpi::MPIRank(sendRank1), sendID1, localSendAABB1, sendTag1);
      sendAABBs_[blockID].emplace_back(mpi::MPIRank(sendRank2), sendID2, localSendAABB2, sendTag2);
      recvAABBs_[blockID].emplace_back(mpi::MPIRank(recvRank2), recvID2, localRecvAABB2, recvTag2);
      recvAABBs_[blockID].emplace_back(mpi::MPIRank(recvRank1), recvID1, localRecvAABB1, recvTag1);

      WALBERLA_LOG_DETAIL("blockID = " << blockID.getID() << ", normalShift = " << normalShift
                                       << "\n\tsendRank1 = " << sendRank1 << "\tsendID1 = " << sendID1.getID() << "\tsendTag1 = " << sendTag1 << "\taabb = " << localSendAABB1
                                       << "\n\tsendRank2 = " << sendRank2 << "\tsendID2 = " << sendID2.getID() << "\tsendTag2 = " << sendTag2 << "\taabb = " << localSendAABB2
                                       << "\n\trecvRank1 = " << recvRank1 << "\trecvID1 = " << recvID1.getID() << "\trecvTag1 = " << recvTag1 << "\taabb = " << localRecvAABB1
                                       << "\n\trecvRank2 = " << recvRank2 << "\trecvID2 = " << recvID2.getID() << "\trecvTag2 = " << recvTag2 << "\taabb = " << localRecvAABB2
      )
   }

   void processSingleAABB( const AABB & originalAABB, const std::shared_ptr<StructuredBlockForest> & sbf,
                           const real_t nGL, const BlockID & blockID, const int normalShift ) {

      WALBERLA_ASSERT(normalShift == -1 || normalShift == 1)

      Vector3<ShiftType> normalShiftVector{};
      normalShiftVector[normalDir_] = ShiftType(normalShift * int_c(sbf->getNumberOfCellsPerBlock(normalDir_)));

      // aabbs for packing
      auto sendAABB = originalAABB.getExtended(nGL);
      auto sendCenter = sendAABB.center();

      // aabbs for unpacking
      auto recvAABB = originalAABB.getExtended(nGL);
      auto recvCenter = recvAABB.center();

      // receive from the interior of domain in ghost layers
      auto localSendAABB = sendAABB;
      auto localRecvAABB = recvAABB;

      if(normalShift == 1) { // at maximum of domain -> send inner layers, receive ghost layers
         localSendAABB.setAxisBounds(normalDir_, localSendAABB.max(normalDir_) - 2 * nGL, localSendAABB.max(normalDir_) - nGL);
         localRecvAABB.setAxisBounds(normalDir_, localRecvAABB.max(normalDir_) - nGL, localRecvAABB.max(normalDir_));
      } else { // at minimum of domain -> send inner layers, receive ghost layers
         localSendAABB.setAxisBounds(normalDir_, localSendAABB.min(normalDir_) + nGL, localSendAABB.min(normalDir_) + 2 * nGL);
         localRecvAABB.setAxisBounds(normalDir_, localRecvAABB.min(normalDir_), localRecvAABB.min(normalDir_) + nGL);
      }

      WALBERLA_ASSERT_FLOAT_EQUAL(localSendAABB.volume(), localRecvAABB.volume())

      sendCenter += ( shift_ + normalShiftVector);
      recvCenter += (-shift_ + normalShiftVector);

      std::array<bool, 3> virtualPeriodicity{false};
      virtualPeriodicity[normalDir_] = true;
      virtualPeriodicity[shiftDir_] = true;

      domain_decomposition::mapPointToPeriodicDomain( virtualPeriodicity, sbf->getDomain(), sendCenter );
      domain_decomposition::mapPointToPeriodicDomain( virtualPeriodicity, sbf->getDomain(), recvCenter );

      uint_t sendRank;
      uint_t recvRank;

      BlockID sendID;
      BlockID recvID;

      const auto blockInformation = sbf->getBlockInformation();
      WALBERLA_ASSERT(blockInformation.active())

      blockInformation.getProcess(sendRank, sendCenter[0], sendCenter[1], sendCenter[2]);
      blockInformation.getProcess(recvRank, recvCenter[0], recvCenter[1], recvCenter[2]);

      blockInformation.getId(sendID, sendCenter[0], sendCenter[1], sendCenter[2]);
      blockInformation.getId(recvID, recvCenter[0], recvCenter[1], recvCenter[2]);

      // define unique send / receive tags for communication

      const int atMaxTagSend = normalShift < 0 ? 0 : 1;
      const int atMaxTagRecv = normalShift < 0 ? 1 : 0;

      const int sendTag = ((int_c(blockID.getID()) + int_c(sendID.getID() * numGlobalBlocks_)) * 2 + atMaxTagSend) * 2 + 0;
      const int recvTag = ((int_c(recvID.getID()) + int_c(blockID.getID() * numGlobalBlocks_)) * 2 + atMaxTagRecv) * 2 + 0;

      sendAABBs_[blockID].emplace_back(mpi::MPIRank(sendRank), sendID, localSendAABB, sendTag);
      recvAABBs_[blockID].emplace_back(mpi::MPIRank(recvRank), recvID, localRecvAABB, recvTag);

      WALBERLA_LOG_DETAIL("blockID = " << blockID.getID() << ", normalShift = " << normalShift
                                       << "\n\tsendRank = " << sendRank1 << "\tsendID = " << sendID.getID() << "\tsendTag = " << sendTag << "\taabb = " << localSendAABB
                                       << "\n\trecvRank = " << recvRank1 << "\trecvID = " << recvID.getID() << "\trecvTag = " << recvTag << "\taabb = " << localRecvAABB
      )
   }

   void setupPeriodicity() {

      const auto sbf = blockForest_.lock();
      WALBERLA_ASSERT_NOT_NULLPTR( sbf )

      auto & blockInformation = sbf->getBlockInformation();
      WALBERLA_ASSERT(blockInformation.active())
      std::vector<std::shared_ptr<IBlockID>> globalBlocks;
      blockInformation.getAllBlocks(globalBlocks);
      numGlobalBlocks_ = globalBlocks.size();

      // get blocks of current processor
      std::vector<IBlock*> localBlocks;
      sbf->getBlocks(localBlocks);

      const auto nGL = real_c(fieldGhostLayers_);

      const bool shiftWholeBlock = (shift_[shiftDir_] % ShiftType(sbf->getNumberOfCells(*localBlocks[0], shiftDir_))) == 0;

      for( auto block : localBlocks ) {

         // get minimal ghost layer region (in normal direction)
         const auto blockAABB = block->getAABB();
         const auto blockID = dynamic_cast<Block*>(block)->getId();

         const bool atMin = sbf->atDomainMinBorder(normalDir_, *block);
         const bool atMax = sbf->atDomainMaxBorder(normalDir_, *block);

         if(atMin) {
            if(shiftWholeBlock) {
               processSingleAABB(blockAABB, sbf, nGL, blockID, -1);
            } else {
               processDoubleAABB(blockAABB, sbf, nGL, blockID, -1);
            }
         }
         if(atMax)  {
            if(shiftWholeBlock) {
               processSingleAABB(blockAABB, sbf, nGL, blockID, +1);
            } else {
               processDoubleAABB(blockAABB, sbf, nGL, blockID, +1);
            }
         }

      }

   }

   void packBuffer(IBlock * const block, const CellInterval & ci,
                  std::vector<ValueType> & buffer) {

      // get field
      auto field = block->getData<GhostLayerField_T>(fieldID_);
      WALBERLA_ASSERT_NOT_NULLPTR(field)

      auto bufferIt = buffer.begin();

      // forward iterate over ci and add values to value vector
      for(auto cellIt = ci.begin(); cellIt != ci.end(); ++cellIt) {
         for(cell_idx_t f = 0; f < cell_idx_c(GhostLayerField_T::F_SIZE); ++f, ++bufferIt) {
            WALBERLA_ASSERT(field->coordinatesValid(cellIt->x(), cellIt->y(), cellIt->z(), f))
            *bufferIt = field->get(*cellIt, f);
         }
      }

   }

   void unpackBuffer(IBlock* const block, const CellInterval & ci,
                     const std::vector<ValueType> & buffer) {

      // get field
      auto field = block->getData<GhostLayerField_T>(fieldID_);
      WALBERLA_ASSERT_NOT_NULLPTR(field)

      auto bufferIt = buffer.begin();

      // forward iterate over ci and add values to value vector
      for(auto cellIt = ci.begin(); cellIt != ci.end(); ++cellIt) {
         for(cell_idx_t f = 0; f < cell_idx_c(GhostLayerField_T::F_SIZE); ++f, ++bufferIt) {
            WALBERLA_ASSERT(field->coordinatesValid(cellIt->x(), cellIt->y(), cellIt->z(), f))
            field->get(*cellIt, f) = *bufferIt;
         }
      }

   }

   Vector3<cell_idx_t> toCellVector( const Vector3<real_t> & point ) {
      return Vector3<cell_idx_t >{ cell_idx_c(point[0]), cell_idx_c(point[1]), cell_idx_c(point[2]) };
   }

   CellInterval globalAABBToLocalCI( const AABB & aabb, const std::shared_ptr<StructuredBlockForest> & sbf, IBlock * const block ) {
      auto globalCI = CellInterval{toCellVector(aabb.min()), toCellVector(aabb.max()) - Vector3<cell_idx_t>(1, 1, 1)};
      CellInterval localCI;
      sbf->transformGlobalToBlockLocalCellInterval(localCI, *block, globalCI);

      return localCI;
   }

   template< typename tPair >
   struct second_t {
      typename tPair::second_type operator()( const tPair& p ) const { return p.second; }
   };
   template< typename tMap >
   second_t< typename tMap::value_type > second( const tMap& ) { return second_t< typename tMap::value_type >(); }


   std::weak_ptr<StructuredBlockForest> blockForest_;

   const BlockDataID flagFieldID_;
   const BlockDataID fieldID_;

   const uint_t fieldGhostLayers_;

   const Vector3<ShiftType> shift_;
   uint_t shiftDir_;
   const Vector3<uint_t> boundaryNormal_;
   uint_t normalDir_;

   // for each local block, stores the ranks where to send / receive, the corresponding block IDs,
   // the local AABBs that need to be packed / unpacked, and a unique tag for communication
   std::map<BlockID, std::vector<std::tuple<mpi::MPIRank, BlockID, AABB, int>>> sendAABBs_{};
   std::map<BlockID, std::vector<std::tuple<mpi::MPIRank, BlockID, AABB, int>>> recvAABBs_{};

   bool setupPeriodicity_{false};
   uint_t numGlobalBlocks_{};

}; // class ShiftedPeriodicity

} // namespace lbm
} // namespace walberla
