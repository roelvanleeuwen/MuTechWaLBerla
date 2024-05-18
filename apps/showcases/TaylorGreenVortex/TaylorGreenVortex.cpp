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
//! \file TaylorGreenVortex.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//! \brief TaylorGreenVortex
//
//======================================================================================================================
#include "blockforest/Initialization.h"
#include "blockforest/SetupBlockForest.h"
#include "blockforest/communication/UniformBufferedScheme.h"
#include "blockforest/loadbalancing/StaticCurve.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/MemoryUsage.h"
#include "core/SharedFunctor.h"
#include "core/debug/TestSubsystem.h"
#include "core/logging/Initialization.h"
#include "core/math/Vector3.h"
#include "core/timing/RemainingTimeLogger.h"

#include "field/AddToStorage.h"
#include "field/FlagField.h"
#include "field/GhostLayerField.h"
#include "field/vtk/VTKWriter.h"

#include "geometry/InitBoundaryHandling.h"

#include "timeloop/SweepTimeloop.h"

#include "lbm_generated/communication/UniformGeneratedPdfPackInfo.h"
#include "lbm_generated/evaluation/PerformanceEvaluation.h"
#include "lbm_generated/field/AddToStorage.h"
#include "lbm_generated/field/PdfField.h"

// include the generated header file. It includes all generated classes
#include "TaylorGreenVortexHeader.h"

using namespace walberla;
using namespace std::placeholders;

using StorageSpecification_T = lbm::TaylorGreenVortexStorageSpecification;
using Stencil_T              = StorageSpecification_T::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;
using PdfField_T             = lbm_generated::PdfField< StorageSpecification_T >;
using PackInfo_T             = lbm_generated::UniformGeneratedPdfPackInfo< PdfField_T >;

using SweepCollection_T = lbm::TaylorGreenVortexSweepCollection;

using VectorField_T = GhostLayerField< real_t, StorageSpecification_T::Stencil::D >;
using ScalarField_T = GhostLayerField< real_t, 1 >;

using flag_t               = walberla::uint8_t;
using FlagField_T          = FlagField< flag_t >;
using BoundaryCollection_T = lbm::TaylorGreenVortexBoundaryCollection< FlagField_T >;

using blockforest::communication::UniformBufferedScheme;

struct IDs
{
   BlockDataID pdfField;
   BlockDataID pdfFieldTmp;
   BlockDataID velocityField;
   BlockDataID densityField;
   BlockDataID avgVelField;
   BlockDataID avgVelSqrField;
   BlockDataID avgPressureField;
   BlockDataID flagField;

   BlockDataID pdfFieldGPU;
   BlockDataID velocityFieldGPU;
   BlockDataID densityFieldGPU;
};

struct Setup
{
   Vector3< uint_t > blocks;
   Vector3< uint_t > cellsPerBlock;
   Vector3< uint_t > cells;
   Vector3< bool > periodic;
   uint_t numGhostLayers;
};

class AccuracyEvaluation
{
 public:
   AccuracyEvaluation(std::shared_ptr< StructuredBlockForest >& blocks, const IDs& ids,
                      SweepCollection_T& sweepCollection, const real_t maxLatticeVelocity, const uint_t logFrequency,
                      const bool logToStream, const bool logToFile)
      : blocks_(blocks), ids_(ids), sweepCollection_(sweepCollection), maxLatticeVelocity_(maxLatticeVelocity),
        executionCounter_(uint_c(0)), filename_("TaylorGreenVortex.txt"), normalizationFactor_(real_c(1.0))
   {
      logFrequency_  = logToStream ? logFrequency : uint_c(0);
      plotFrequency_ = logToFile ? logFrequency : uint_c(0);
   }

   void setNormalizationFactor(const real_t f) { normalizationFactor_ = f; }
   void setFilename(const std::string& filename) { filename_ = filename; }

   real_t L1() const { return L1_; }
   real_t L2() const { return L2_; }
   real_t Lmax() const { return Lmax_; }

   void operator()()
   {
      if (logFrequency_ == uint_c(0) && (plotFrequency_ == uint_c(0) || filename_.empty())) return;

      ++executionCounter_;

      const bool plot = (plotFrequency_ != uint_c(0) && (executionCounter_ - uint_c(1)) % plotFrequency_ == uint_c(0) &&
                         !filename_.empty());
      const bool log  = (logFrequency_ != uint_c(0) && (executionCounter_ - uint_c(1)) % logFrequency_ == uint_c(0));

      if (!log && !plot) return;

      const auto& domainAABB = blocks_->getDomain();

      real_t _L1(real_c(0.0));
      real_t _L2(real_c(0.0));
      real_t _Lmax(real_c(0.0));

      for (auto block = blocks_->begin(); block != blocks_->end(); ++block)
      {
         sweepCollection_.calculateMacroscopicParameters(block.get());
         const VectorField_T* velocityField = block->template getData< const VectorField_T >(ids_.velocityField);

         const auto level = blocks_->getLevel(*block);
         const real_t volumeFraction =
            (blocks_->dx(level) * blocks_->dy(level) * blocks_->dz(level)) / domainAABB.volume();

         for (auto it = velocityField->beginXYZ(); it != velocityField->end(); ++it)
         {
            Vector3< real_t > center = blocks_->getBlockLocalCellCenter(*block, Cell(it.x(), it.y(), it.z()));

            Vector3< real_t > exactVelocity(
               real_c(maxLatticeVelocity_ * (center[1] - real_c(domainAABB.yMin())) / real_c(domainAABB.ySize())),
               real_c(0.0), real_c(0.0));
            Vector3< real_t > velocity(it.getF(0), it.getF(1), it.getF(2));

            const auto error  = velocity - exactVelocity;
            const real_t diff = error.length();

            _L1 += diff * volumeFraction;
            _L2 += diff * diff * volumeFraction;
            _Lmax = std::max(_Lmax, diff);
         }
      }

      mpi::reduceInplace(_L1, mpi::SUM);
      mpi::reduceInplace(_L2, mpi::SUM);
      mpi::reduceInplace(_Lmax, mpi::MAX);
      _L2 = std::sqrt(_L2);

      L1_   = _L1;
      L2_   = _L2;
      Lmax_ = _Lmax;

      WALBERLA_ROOT_SECTION()
      {
         if (plot && executionCounter_ == uint_t(1))
         {
            std::ofstream file(filename_.c_str());
            file << "# accuracy evaluation"
                 << "# step [1], L1 [2], L2 [3], Lmax [4]" << std::endl;
            file.close();
         }

         if (log)
         {
            WALBERLA_LOG_INFO("Evaluation of accuracy:"
                              << "\n - L1:   " << L1_ << "\n - L2:   " << L2_ << "\n - Lmax: " << Lmax_);
         }

         if (plot)
         {
            std::ofstream file(filename_.c_str(), std::ofstream::out | std::ofstream::app);
            file << (executionCounter_ - uint_t(1)) << " " << L1_ << " " << L2_ << " " << Lmax_ << std::endl;
            file.close();
         }
      }
   };

 private:
   std::shared_ptr< StructuredBlockForest > blocks_;
   IDs ids_;
   SweepCollection_T& sweepCollection_;
   real_t maxLatticeVelocity_;

   uint_t executionCounter_;

   uint_t plotFrequency_;
   uint_t logFrequency_;

   std::string filename_;

   real_t normalizationFactor_;

   real_t L1_;
   real_t L2_;
   real_t Lmax_;

}; // class AccuracyEvaluation

namespace
{
void workloadMemoryAndSUIDAssignment(SetupBlockForest& forest, const memory_t memoryPerBlock)
{
   for (auto block = forest.begin(); block != forest.end(); ++block)
   {
      block->setWorkload(numeric_cast< workload_t >(uint_t(1) << block->getLevel()));
      block->setMemory(memoryPerBlock);
   }
}

shared_ptr< SetupBlockForest >createSetupBlockForest(const Setup& setup, uint_t numberOfProcesses, const memory_t memoryPerCell)
{
   shared_ptr< SetupBlockForest > forest = make_shared< SetupBlockForest >();

   const memory_t memoryPerBlock =
      numeric_cast< memory_t >((setup.cellsPerBlock[0] + uint_t(2) * setup.numGhostLayers) *
                               (setup.cellsPerBlock[1] + uint_t(2) * setup.numGhostLayers) *
                               (setup.cellsPerBlock[2] + uint_t(2) * setup.numGhostLayers)) *
      memoryPerCell;

   forest->addWorkloadMemorySUIDAssignmentFunction(
      std::bind(workloadMemoryAndSUIDAssignment, std::placeholders::_1, memoryPerBlock));

   forest->init(
      AABB(real_c(0), real_c(0), real_c(0), real_c(setup.cells[0]), real_c(setup.cells[1]), real_c(setup.cells[2])),
      setup.blocks[0], setup.blocks[1], setup.blocks[2], setup.periodic[0], setup.periodic[1], setup.periodic[2]);

   MPIManager::instance()->useWorldComm();
   forest->balanceLoad(blockforest::StaticLevelwiseCurveBalanceWeighted(), numberOfProcesses);
   // WALBERLA_LOG_INFO_ON_ROOT("SetupBlockForest created successfully:\n" << *forest);

   return forest;
}

shared_ptr< blockforest::StructuredBlockForest >createStructuredBlockForest(const Setup& setup, const memory_t memoryPerCell)
{
   // WALBERLA_LOG_INFO_ON_ROOT("Creating the block structure ...");
   shared_ptr< SetupBlockForest > sforest = createSetupBlockForest(setup, uint_c(MPIManager::instance()->numProcesses()), memoryPerCell);

   auto bf  = std::make_shared< blockforest::BlockForest >(uint_c(MPIManager::instance()->rank()), *sforest, false);
   auto sbf = std::make_shared< blockforest::StructuredBlockForest >(bf, setup.cellsPerBlock[0], setup.cellsPerBlock[1],
                                                                     setup.cellsPerBlock[2]);
   sbf->createCellBoundingBoxes();

   return sbf;
}
} // namespace

class StreamCollide
{
 public:
   StreamCollide(const shared_ptr< StructuredBlockForest > & blocks, const IDs& ids, const real_t omega) : ids_(ids)
   {

      std::map<int, int> m{{-1, 7}, {0, 0}, {1, 0}};
      _nBlocks = blocks->getNumberOfBlocks();
      blocks->getBlocks(blockVector, 0);
      for (uint_t level = 0; level < blocks->getNumberOfLevels(); level++)
      {
         const double level_scale_factor = double(uint_t(1) << level);
         const double one                = double(1.0);
         const double half               = double(0.5);

         omegaVector.push_back( double(omega / (level_scale_factor * (-omega * half + one) + omega * half)) );
      }

      for( auto it = blocks->begin(); it != blocks->end(); ++it )
      {
         auto* local = dynamic_cast< Block* >(it.get());
         for (cell_idx_t ctr_2 = -1; ctr_2 < 2; ++ctr_2)
         {
            for (cell_idx_t ctr_1 = -1; ctr_1 < 2; ++ctr_1)
            {
               for (cell_idx_t ctr_0 = -1; ctr_0 < 2; ++ctr_0)
               {
                  auto dir = stencil::vectorToDirection(ctr_0, ctr_1, ctr_2);
                  if(dir == stencil::C)
                  {
                     PdfField_T* localPdfField = local->getData< PdfField_T >(ids_.pdfField);
                     PdfField_T* localPdfFieldTmp = local->getData< PdfField_T >(ids_.pdfFieldTmp);

                     pdfs.push_back(localPdfField->dataAt(0, 0, 0, 0));
                     pdfsTmp.push_back(localPdfFieldTmp->dataAt(0, 0, 0, 0));

                     _interval = localPdfField->xyzSize();
                     _stride_pdfs_0 = int64_t(localPdfField->xStride());
                     _stride_pdfs_1 = int64_t(localPdfField->yStride());
                     _stride_pdfs_2 = int64_t(localPdfField->zStride());
                     _stride_pdfs_3 = int64_t(localPdfField->fStride());
                  }
                  else
                  {
                     const auto neighborIdx = blockforest::getBlockNeighborhoodSectionIndex(dir);
                     if (!(local->neighborhoodSectionHasEquallySizedBlock(neighborIdx))) continue;

                     const BlockID& neighborId = local->getNeighborId(neighborIdx, uint_t(0));
                     auto neighbor             = dynamic_cast< Block* >(blocks->getBlock(neighborId));

                     PdfField_T* localPdfField = neighbor->getData< PdfField_T >(ids_.pdfField);
                     PdfField_T* localPdfFieldTmp = neighbor->getData< PdfField_T >(ids_.pdfFieldTmp);

                     pdfs.push_back(localPdfField->dataAt(m[stencil::cx[dir]], m[stencil::cy[dir]], m[stencil::cz[dir]], 0));
                     pdfsTmp.push_back(localPdfFieldTmp->dataAt(m[stencil::cx[dir]], m[stencil::cy[dir]], m[stencil::cz[dir]], 0));
                  }
               }
            }
         }
      }
   }

   void streamCollide(Block * block)
   {
      auto pdfField = block->getData<PdfField_T>(ids_.pdfField);
      auto pdfFieldTmp = block->getData<PdfField_T>(ids_.pdfFieldTmp);
      CellInterval ci = pdfField->xyzSize();

      const uint_t level = block->getLevel();
      double omega = omegaVector[level];

      const double xi_0 = ((1.0) / (omega*-0.25 + 2.0));
      const double rr_0 = xi_0*(omega*-2.0 + 4.0);

      for (auto it: ci)
      {
         const cell_idx_t ctr_2 = it.z();
         const cell_idx_t ctr_1 = it.y();
         const cell_idx_t ctr_0 = it.x();
         const double f_0 = pdfField->get(ctr_0, ctr_1, ctr_2, 0);
         const double f_1 = pdfField->get(ctr_0, ctr_1 - 1, ctr_2, 1);
         const double f_2 = pdfField->get(ctr_0, ctr_1 + 1, ctr_2, 2);
         const double f_3 = pdfField->get(ctr_0 + 1, ctr_1, ctr_2, 3);
         const double f_4 = pdfField->get(ctr_0 - 1, ctr_1, ctr_2, 4);
         const double f_5 = pdfField->get(ctr_0, ctr_1, ctr_2 - 1, 5);
         const double f_6 = pdfField->get(ctr_0, ctr_1, ctr_2 + 1, 6);
         const double f_7 = pdfField->get(ctr_0 + 1, ctr_1 - 1, ctr_2, 7);
         const double f_8 = pdfField->get(ctr_0 - 1, ctr_1 - 1, ctr_2, 8);
         const double f_9 = pdfField->get(ctr_0 + 1, ctr_1 + 1, ctr_2, 9);
         const double f_10 = pdfField->get(ctr_0 - 1, ctr_1 + 1, ctr_2, 10);
         const double f_11 = pdfField->get(ctr_0, ctr_1 - 1, ctr_2 - 1, 11);
         const double f_12 = pdfField->get(ctr_0, ctr_1 + 1, ctr_2 - 1, 12);
         const double f_13 = pdfField->get(ctr_0 + 1, ctr_1, ctr_2 - 1, 13);
         const double f_14 = pdfField->get(ctr_0 - 1, ctr_1, ctr_2 - 1, 14);
         const double f_15 = pdfField->get(ctr_0, ctr_1 - 1, ctr_2 + 1, 15);
         const double f_16 = pdfField->get(ctr_0, ctr_1 + 1, ctr_2 + 1, 16);
         const double f_17 = pdfField->get(ctr_0 + 1, ctr_1, ctr_2 + 1, 17);
         const double f_18 = pdfField->get(ctr_0 - 1, ctr_1, ctr_2 + 1, 18);
         const double f_19 = pdfField->get(ctr_0 - 1, ctr_1 - 1, ctr_2 - 1, 19);
         const double f_20 = pdfField->get(ctr_0 + 1, ctr_1 - 1, ctr_2 - 1, 20);
         const double f_21 = pdfField->get(ctr_0 - 1, ctr_1 + 1, ctr_2 - 1, 21);
         const double f_22 = pdfField->get(ctr_0 + 1, ctr_1 + 1, ctr_2 - 1, 22);
         const double f_23 = pdfField->get(ctr_0 - 1, ctr_1 - 1, ctr_2 + 1, 23);
         const double f_24 = pdfField->get(ctr_0 + 1, ctr_1 - 1, ctr_2 + 1, 24);
         const double f_25 = pdfField->get(ctr_0 - 1, ctr_1 + 1, ctr_2 + 1, 25);
         const double f_26 = pdfField->get(ctr_0 + 1, ctr_1 + 1, ctr_2 + 1, 26);

         const double vel0Term = f_10 + f_14 + f_18 + f_19 + f_21 + f_23 + f_25 + f_4 + f_8;
         const double vel1Term = f_1 + f_11 + f_15 + f_20 + f_24 + f_7;
         const double vel2Term = f_12 + f_13 + f_22 + f_5;
         const double delta_rho = f_0 + f_16 + f_17 + f_2 + f_26 + f_3 + f_6 + f_9 + vel0Term + vel1Term + vel2Term;
         const double rho = delta_rho + 1.0;
         const double xi_1 = ((1.0) / (rho));
         const double u_0 = xi_1*(-f_13 - f_17 - f_20 - f_22 - f_24 - f_26 - f_3 - f_7 - f_9 + vel0Term);
         const double u_1 = xi_1*(-f_10 - f_12 - f_16 + f_19 - f_2 - f_21 - f_22 + f_23 - f_25 - f_26 + f_8 - f_9 + vel1Term);
         const double u_2 = xi_1*(f_11 + f_14 - f_15 - f_16 - f_17 - f_18 + f_19 + f_20 + f_21 - f_23 - f_24 - f_25 - f_26 - f_6 + vel2Term);
         const double u0Mu1 = u_0 - u_1;
         const double u0Pu1 = u_0 + u_1;
         const double u1Pu2 = u_1 + u_2;
         const double u1Mu2 = u_1 - u_2;
         const double u0Mu2 = u_0 - u_2;
         const double u0Pu2 = u_0 + u_2;
         const double f_eq_common = delta_rho + rho*-1.5*(u_0*u_0) + rho*-1.5*(u_1*u_1) + rho*-1.5*(u_2*u_2);

         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 0)  = f_0 + omega*(-f_0 + f_eq_common*0.29629629629629628);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 1)  = f_1 + omega*(f_1*-0.5 + f_2*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_1*u_1)) + rr_0*(f_1*-0.5 + f_2*0.5 + rho*u_1*0.22222222222222221);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 2)  = f_2 + omega*(f_1*-0.5 + f_2*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_1*u_1)) + rr_0*(f_1*0.5 + f_2*-0.5 + rho*u_1*-0.22222222222222221);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 3)  = f_3 + omega*(f_3*-0.5 + f_4*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_0*u_0)) + rr_0*(f_3*-0.5 + f_4*0.5 + rho*u_0*-0.22222222222222221);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 4)  = f_4 + omega*(f_3*-0.5 + f_4*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_0*u_0)) + rr_0*(f_3*0.5 + f_4*-0.5 + rho*u_0*0.22222222222222221);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 5)  = f_5 + omega*(f_5*-0.5 + f_6*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_2*u_2)) + rr_0*(f_5*-0.5 + f_6*0.5 + rho*u_2*0.22222222222222221);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 6)  = f_6 + omega*(f_5*-0.5 + f_6*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_2*u_2)) + rr_0*(f_5*0.5 + f_6*-0.5 + rho*u_2*-0.22222222222222221);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 7)  = f_7 + omega*(f_10*-0.5 + f_7*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Mu1*u0Mu1)) + rr_0*(f_10*0.5 + f_7*-0.5 + rho*u0Mu1*-0.055555555555555552);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 8)  = f_8 + omega*(f_8*-0.5 + f_9*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Pu1*u0Pu1)) + rr_0*(f_8*-0.5 + f_9*0.5 + rho*u0Pu1*0.055555555555555552);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 9)  = f_9 + omega*(f_8*-0.5 + f_9*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Pu1*u0Pu1)) + rr_0*(f_8*0.5 + f_9*-0.5 + rho*u0Pu1*-0.055555555555555552);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 10) = f_10 + omega*(f_10*-0.5 + f_7*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Mu1*u0Mu1)) + rr_0*(f_10*-0.5 + f_7*0.5 + rho*u0Mu1*0.055555555555555552);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 11) = f_11 + omega*(f_11*-0.5 + f_16*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u1Pu2*u1Pu2)) + rr_0*(f_11*-0.5 + f_16*0.5 + rho*u1Pu2*0.055555555555555552);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 12) = f_12 + omega*(f_12*-0.5 + f_15*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u1Mu2*u1Mu2)) + rr_0*(f_12*-0.5 + f_15*0.5 + rho*u1Mu2*-0.055555555555555552);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 13) = f_13 + omega*(f_13*-0.5 + f_18*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Mu2*u0Mu2)) + rr_0*(f_13*-0.5 + f_18*0.5 + rho*u0Mu2*-0.055555555555555552);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 14) = f_14 + omega*(f_14*-0.5 + f_17*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Pu2*u0Pu2)) + rr_0*(f_14*-0.5 + f_17*0.5 + rho*u0Pu2*0.055555555555555552);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 15) = f_15 + omega*(f_12*-0.5 + f_15*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u1Mu2*u1Mu2)) + rr_0*(f_12*0.5 + f_15*-0.5 + rho*u1Mu2*0.055555555555555552);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 16) = f_16 + omega*(f_11*-0.5 + f_16*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u1Pu2*u1Pu2)) + rr_0*(f_11*0.5 + f_16*-0.5 + rho*u1Pu2*-0.055555555555555552);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 17) = f_17 + omega*(f_14*-0.5 + f_17*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Pu2*u0Pu2)) + rr_0*(f_14*0.5 + f_17*-0.5 + rho*u0Pu2*-0.055555555555555552);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 18) = f_18 + omega*(f_13*-0.5 + f_18*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Mu2*u0Mu2)) + rr_0*(f_13*0.5 + f_18*-0.5 + rho*u0Mu2*0.055555555555555552);
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 19) = f_19 + omega*(delta_rho*-0.013888888888888888 + f_19*-0.5 + f_26*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Pu1*u0Pu1) + 0.020833333333333332*(u0Pu2*u0Pu2) + 0.020833333333333332*(u1Pu2*u1Pu2))) + rr_0*(f_19*-0.5 + f_26*0.5 + rho*(u0Pu1*0.013888888888888888 + u_2*0.013888888888888888));
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 20) = f_20 + omega*(delta_rho*-0.013888888888888888 + f_20*-0.5 + f_25*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu1*u0Mu1) + 0.020833333333333332*(u0Mu2*u0Mu2) + 0.020833333333333332*(u1Pu2*u1Pu2))) + rr_0*(f_20*-0.5 + f_25*0.5 + rho*(u0Mu1*-0.013888888888888888 + u_2*0.013888888888888888));
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 21) = f_21 + omega*(delta_rho*-0.013888888888888888 + f_21*-0.5 + f_24*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu1*u0Mu1) + 0.020833333333333332*(u0Pu2*u0Pu2) + 0.020833333333333332*(u1Mu2*u1Mu2))) + rr_0*(f_21*-0.5 + f_24*0.5 + rho*(u0Mu1*0.013888888888888888 + u_2*0.013888888888888888));
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 22) = f_22 + omega*(delta_rho*-0.013888888888888888 + f_22*-0.5 + f_23*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu2*u0Mu2) + 0.020833333333333332*(u0Pu1*u0Pu1) + 0.020833333333333332*(u1Mu2*u1Mu2))) + rr_0*(f_22*-0.5 + f_23*0.5 + rho*(u0Pu1*-0.013888888888888888 + u_2*0.013888888888888888));
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 23) = f_23 + omega*(delta_rho*-0.013888888888888888 + f_22*-0.5 + f_23*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu2*u0Mu2) + 0.020833333333333332*(u0Pu1*u0Pu1) + 0.020833333333333332*(u1Mu2*u1Mu2))) + rr_0*(f_22*0.5 + f_23*-0.5 + rho*(u0Pu1*0.013888888888888888 + u_2*-0.013888888888888888));
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 24) = f_24 + omega*(delta_rho*-0.013888888888888888 + f_21*-0.5 + f_24*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu1*u0Mu1) + 0.020833333333333332*(u0Pu2*u0Pu2) + 0.020833333333333332*(u1Mu2*u1Mu2))) + rr_0*(f_21*0.5 + f_24*-0.5 + rho*(u0Mu1*-0.013888888888888888 + u_2*-0.013888888888888888));
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 25) = f_25 + omega*(delta_rho*-0.013888888888888888 + f_20*-0.5 + f_25*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu1*u0Mu1) + 0.020833333333333332*(u0Mu2*u0Mu2) + 0.020833333333333332*(u1Pu2*u1Pu2))) + rr_0*(f_20*0.5 + f_25*-0.5 + rho*(u0Mu1*0.013888888888888888 + u_2*-0.013888888888888888));
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 26) = f_26 + omega*(delta_rho*-0.013888888888888888 + f_19*-0.5 + f_26*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Pu1*u0Pu1) + 0.020833333333333332*(u0Pu2*u0Pu2) + 0.020833333333333332*(u1Pu2*u1Pu2))) + rr_0*(f_19*0.5 + f_26*-0.5 + rho*(u0Pu1*-0.013888888888888888 + u_2*-0.013888888888888888));
      }
      pdfField->swapDataPointers(pdfFieldTmp);
   }


   void streamCollideEven()
   {
      double omega = omegaVector[0];
      const double xi_0 = ((1.0) / (omega*-0.25 + 2.0));
      const double rr_0 = xi_0*(omega*-2.0 + 4.0);

      // std::map<int, int> m{{-1, 0}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 0}};
      std::map<int, int> m{{-1, 0}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 0}};


      for(uint_t i = 0; i < _nBlocks; ++i)
      {
         for (auto it: _interval)
         {
            const cell_idx_t ctr_2 = it.z();
            const cell_idx_t ctr_1 = it.y();
            const cell_idx_t ctr_0 = it.x();

            const uint_t index_C = uint_c(27) * i + uint_c( (0 + 1) * 9 + (0 + 1) * 3 + 0 + 1 );
            const uint_t index_N = uint_c(27) * i + uint_c( (0 + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + 0 + 1 );
            const uint_t index_S = uint_c(27) * i + uint_c( (0 + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + 0 + 1 );
            const uint_t index_W = uint_c(27) * i + uint_c( (0 + 1) * 9 + (0 + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_E = uint_c(27) * i + uint_c( (0 + 1) * 9 + (0 + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_T = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + (0 + 1) * 3 + 0 + 1 );
            const uint_t index_B = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + (0 + 1) * 3 + 0 + 1 );
            const uint_t index_NW = uint_c(27) * i + uint_c( (0 + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_NE = uint_c(27) * i + uint_c( (0 + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_SW = uint_c(27) * i + uint_c( (0 + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_SE = uint_c(27) * i + uint_c( (0 + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_TN = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + 0 + 1 );
            const uint_t index_TS = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + 0 + 1 );
            const uint_t index_TW = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + (0 + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_TE = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + (0 + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_BN = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + 0 + 1 );
            const uint_t index_BS = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + 0 + 1 );
            const uint_t index_BW = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + (0 + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_BE = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + (0 + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_TNE = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_TNW = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_TSE = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_TSW = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_BNE = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_BNW = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_BSE = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_BSW = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
//
//            WALBERLA_LOG_DEVEL_VAR(ctr_1)
//            WALBERLA_LOG_DEVEL_VAR(((ctr_1 -1 < 0 ? -1 : 0) + 1))
//            WALBERLA_LOG_DEVEL_VAR(index_N)

            const double f_0 = pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 0*_stride_pdfs_3];
            const double f_1 = pdfs[index_N][ctr_0*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 1*_stride_pdfs_3];
            const double f_2 = pdfs[index_S][ctr_0*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 2*_stride_pdfs_3];
            const double f_3 = pdfs[index_W][m[ctr_0 + 1]*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 3*_stride_pdfs_3];
            const double f_4 = pdfs[index_E][m[ctr_0 - 1]*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 4*_stride_pdfs_3];
            const double f_5 = pdfs[index_T][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 5*_stride_pdfs_3];
            const double f_6 = pdfs[index_B][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 6*_stride_pdfs_3];
            const double f_7 = pdfs[index_NW][m[ctr_0 + 1]*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 7*_stride_pdfs_3];
            const double f_8 = pdfs[index_NE][m[ctr_0 - 1]*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 8*_stride_pdfs_3];
            const double f_9 = pdfs[index_SW][m[ctr_0 + 1]*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 9*_stride_pdfs_3];
            const double f_10 = pdfs[index_SE][m[ctr_0 - 1]*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 10*_stride_pdfs_3];
            const double f_11 = pdfs[index_TN][ctr_0*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 11*_stride_pdfs_3];
            const double f_12 = pdfs[index_TS][ctr_0*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 12*_stride_pdfs_3];
            const double f_13 = pdfs[index_TW][m[ctr_0 + 1]*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 13*_stride_pdfs_3];
            const double f_14 = pdfs[index_TE][m[ctr_0 - 1]*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 14*_stride_pdfs_3];
            const double f_15 = pdfs[index_BN][ctr_0*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 15*_stride_pdfs_3];
            const double f_16 = pdfs[index_BS][ctr_0*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 16*_stride_pdfs_3];
            const double f_17 = pdfs[index_BW][m[ctr_0 + 1]*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 17*_stride_pdfs_3];
            const double f_18 = pdfs[index_BE][m[ctr_0 - 1]*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 18*_stride_pdfs_3];
            const double f_19 = pdfs[index_TNE][m[ctr_0 - 1]*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 19*_stride_pdfs_3];
            const double f_20 = pdfs[index_TNW][m[ctr_0 + 1]*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 20*_stride_pdfs_3];
            const double f_21 = pdfs[index_TSE][m[ctr_0 - 1]*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 21*_stride_pdfs_3];
            const double f_22 = pdfs[index_TSW][m[ctr_0 + 1]*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 22*_stride_pdfs_3];
            const double f_23 = pdfs[index_BNE][m[ctr_0 - 1]*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 23*_stride_pdfs_3];
            const double f_24 = pdfs[index_BNW][m[ctr_0 + 1]*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 24*_stride_pdfs_3];
            const double f_25 = pdfs[index_BSE][m[ctr_0 - 1]*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 25*_stride_pdfs_3];
            const double f_26 = pdfs[index_BSW][m[ctr_0 + 1]*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 26*_stride_pdfs_3];

            const double vel0Term = f_10 + f_14 + f_18 + f_19 + f_21 + f_23 + f_25 + f_4 + f_8;
            const double vel1Term = f_1 + f_11 + f_15 + f_20 + f_24 + f_7;
            const double vel2Term = f_12 + f_13 + f_22 + f_5;
            const double delta_rho = f_0 + f_16 + f_17 + f_2 + f_26 + f_3 + f_6 + f_9 + vel0Term + vel1Term + vel2Term;
            const double rho = delta_rho + 1.0;
            const double xi_1 = ((1.0) / (rho));
            const double u_0 = xi_1*(-f_13 - f_17 - f_20 - f_22 - f_24 - f_26 - f_3 - f_7 - f_9 + vel0Term);
            const double u_1 = xi_1*(-f_10 - f_12 - f_16 + f_19 - f_2 - f_21 - f_22 + f_23 - f_25 - f_26 + f_8 - f_9 + vel1Term);
            const double u_2 = xi_1*(f_11 + f_14 - f_15 - f_16 - f_17 - f_18 + f_19 + f_20 + f_21 - f_23 - f_24 - f_25 - f_26 - f_6 + vel2Term);
            const double u0Mu1 = u_0 - u_1;
            const double u0Pu1 = u_0 + u_1;
            const double u1Pu2 = u_1 + u_2;
            const double u1Mu2 = u_1 - u_2;
            const double u0Mu2 = u_0 - u_2;
            const double u0Pu2 = u_0 + u_2;
            const double f_eq_common = delta_rho + rho*-1.5*(u_0*u_0) + rho*-1.5*(u_1*u_1) + rho*-1.5*(u_2*u_2);

            const double r_0  = f_0 + omega*(-f_0 + f_eq_common*0.29629629629629628);
            const double r_1  = f_1 + omega*(f_1*-0.5 + f_2*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_1*u_1)) + rr_0*(f_1*-0.5 + f_2*0.5 + rho*u_1*0.22222222222222221);
            const double r_2  = f_2 + omega*(f_1*-0.5 + f_2*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_1*u_1)) + rr_0*(f_1*0.5 + f_2*-0.5 + rho*u_1*-0.22222222222222221);
            const double r_3  = f_3 + omega*(f_3*-0.5 + f_4*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_0*u_0)) + rr_0*(f_3*-0.5 + f_4*0.5 + rho*u_0*-0.22222222222222221);
            const double r_4  = f_4 + omega*(f_3*-0.5 + f_4*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_0*u_0)) + rr_0*(f_3*0.5 + f_4*-0.5 + rho*u_0*0.22222222222222221);
            const double r_5  = f_5 + omega*(f_5*-0.5 + f_6*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_2*u_2)) + rr_0*(f_5*-0.5 + f_6*0.5 + rho*u_2*0.22222222222222221);
            const double r_6  = f_6 + omega*(f_5*-0.5 + f_6*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_2*u_2)) + rr_0*(f_5*0.5 + f_6*-0.5 + rho*u_2*-0.22222222222222221);
            const double r_7  = f_7 + omega*(f_10*-0.5 + f_7*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Mu1*u0Mu1)) + rr_0*(f_10*0.5 + f_7*-0.5 + rho*u0Mu1*-0.055555555555555552);
            const double r_8  = f_8 + omega*(f_8*-0.5 + f_9*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Pu1*u0Pu1)) + rr_0*(f_8*-0.5 + f_9*0.5 + rho*u0Pu1*0.055555555555555552);
            const double r_9  = f_9 + omega*(f_8*-0.5 + f_9*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Pu1*u0Pu1)) + rr_0*(f_8*0.5 + f_9*-0.5 + rho*u0Pu1*-0.055555555555555552);
            const double r_10 = f_10 + omega*(f_10*-0.5 + f_7*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Mu1*u0Mu1)) + rr_0*(f_10*-0.5 + f_7*0.5 + rho*u0Mu1*0.055555555555555552);
            const double r_11 = f_11 + omega*(f_11*-0.5 + f_16*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u1Pu2*u1Pu2)) + rr_0*(f_11*-0.5 + f_16*0.5 + rho*u1Pu2*0.055555555555555552);
            const double r_12 = f_12 + omega*(f_12*-0.5 + f_15*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u1Mu2*u1Mu2)) + rr_0*(f_12*-0.5 + f_15*0.5 + rho*u1Mu2*-0.055555555555555552);
            const double r_13 = f_13 + omega*(f_13*-0.5 + f_18*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Mu2*u0Mu2)) + rr_0*(f_13*-0.5 + f_18*0.5 + rho*u0Mu2*-0.055555555555555552);
            const double r_14 = f_14 + omega*(f_14*-0.5 + f_17*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Pu2*u0Pu2)) + rr_0*(f_14*-0.5 + f_17*0.5 + rho*u0Pu2*0.055555555555555552);
            const double r_15 = f_15 + omega*(f_12*-0.5 + f_15*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u1Mu2*u1Mu2)) + rr_0*(f_12*0.5 + f_15*-0.5 + rho*u1Mu2*0.055555555555555552);
            const double r_16 = f_16 + omega*(f_11*-0.5 + f_16*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u1Pu2*u1Pu2)) + rr_0*(f_11*0.5 + f_16*-0.5 + rho*u1Pu2*-0.055555555555555552);
            const double r_17 = f_17 + omega*(f_14*-0.5 + f_17*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Pu2*u0Pu2)) + rr_0*(f_14*0.5 + f_17*-0.5 + rho*u0Pu2*-0.055555555555555552);
            const double r_18 = f_18 + omega*(f_13*-0.5 + f_18*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Mu2*u0Mu2)) + rr_0*(f_13*0.5 + f_18*-0.5 + rho*u0Mu2*0.055555555555555552);
            const double r_19 = f_19 + omega*(delta_rho*-0.013888888888888888 + f_19*-0.5 + f_26*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Pu1*u0Pu1) + 0.020833333333333332*(u0Pu2*u0Pu2) + 0.020833333333333332*(u1Pu2*u1Pu2))) + rr_0*(f_19*-0.5 + f_26*0.5 + rho*(u0Pu1*0.013888888888888888 + u_2*0.013888888888888888));
            const double r_20 = f_20 + omega*(delta_rho*-0.013888888888888888 + f_20*-0.5 + f_25*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu1*u0Mu1) + 0.020833333333333332*(u0Mu2*u0Mu2) + 0.020833333333333332*(u1Pu2*u1Pu2))) + rr_0*(f_20*-0.5 + f_25*0.5 + rho*(u0Mu1*-0.013888888888888888 + u_2*0.013888888888888888));
            const double r_21 = f_21 + omega*(delta_rho*-0.013888888888888888 + f_21*-0.5 + f_24*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu1*u0Mu1) + 0.020833333333333332*(u0Pu2*u0Pu2) + 0.020833333333333332*(u1Mu2*u1Mu2))) + rr_0*(f_21*-0.5 + f_24*0.5 + rho*(u0Mu1*0.013888888888888888 + u_2*0.013888888888888888));
            const double r_22 = f_22 + omega*(delta_rho*-0.013888888888888888 + f_22*-0.5 + f_23*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu2*u0Mu2) + 0.020833333333333332*(u0Pu1*u0Pu1) + 0.020833333333333332*(u1Mu2*u1Mu2))) + rr_0*(f_22*-0.5 + f_23*0.5 + rho*(u0Pu1*-0.013888888888888888 + u_2*0.013888888888888888));
            const double r_23 = f_23 + omega*(delta_rho*-0.013888888888888888 + f_22*-0.5 + f_23*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu2*u0Mu2) + 0.020833333333333332*(u0Pu1*u0Pu1) + 0.020833333333333332*(u1Mu2*u1Mu2))) + rr_0*(f_22*0.5 + f_23*-0.5 + rho*(u0Pu1*0.013888888888888888 + u_2*-0.013888888888888888));
            const double r_24 = f_24 + omega*(delta_rho*-0.013888888888888888 + f_21*-0.5 + f_24*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu1*u0Mu1) + 0.020833333333333332*(u0Pu2*u0Pu2) + 0.020833333333333332*(u1Mu2*u1Mu2))) + rr_0*(f_21*0.5 + f_24*-0.5 + rho*(u0Mu1*-0.013888888888888888 + u_2*-0.013888888888888888));
            const double r_25 = f_25 + omega*(delta_rho*-0.013888888888888888 + f_20*-0.5 + f_25*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu1*u0Mu1) + 0.020833333333333332*(u0Mu2*u0Mu2) + 0.020833333333333332*(u1Pu2*u1Pu2))) + rr_0*(f_20*0.5 + f_25*-0.5 + rho*(u0Mu1*0.013888888888888888 + u_2*-0.013888888888888888));
            const double r_26 = f_26 + omega*(delta_rho*-0.013888888888888888 + f_19*-0.5 + f_26*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Pu1*u0Pu1) + 0.020833333333333332*(u0Pu2*u0Pu2) + 0.020833333333333332*(u1Pu2*u1Pu2))) + rr_0*(f_19*0.5 + f_26*-0.5 + rho*(u0Pu1*-0.013888888888888888 + u_2*-0.013888888888888888));

            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 0*_stride_pdfs_3] = r_0;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 1*_stride_pdfs_3] = r_1;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 2*_stride_pdfs_3] = r_2;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 3*_stride_pdfs_3] = r_3;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 4*_stride_pdfs_3] = r_4;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 5*_stride_pdfs_3] = r_5;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 6*_stride_pdfs_3] = r_6;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 7*_stride_pdfs_3] = r_7;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 8*_stride_pdfs_3] = r_8;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 9*_stride_pdfs_3] = r_9;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 10*_stride_pdfs_3] = r_10;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 11*_stride_pdfs_3] = r_11;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 12*_stride_pdfs_3] = r_12;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 13*_stride_pdfs_3] = r_13;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 14*_stride_pdfs_3] = r_14;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 15*_stride_pdfs_3] = r_15;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 16*_stride_pdfs_3] = r_16;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 17*_stride_pdfs_3] = r_17;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 18*_stride_pdfs_3] = r_18;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 19*_stride_pdfs_3] = r_19;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 20*_stride_pdfs_3] = r_20;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 21*_stride_pdfs_3] = r_21;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 22*_stride_pdfs_3] = r_22;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 23*_stride_pdfs_3] = r_23;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 24*_stride_pdfs_3] = r_24;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 25*_stride_pdfs_3] = r_25;
            pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 26*_stride_pdfs_3] = r_26;
         }
      }
   }

   void streamCollideOdd()
   {
      double omega = omegaVector[0];
      const double xi_0 = ((1.0) / (omega*-0.25 + 2.0));
      const double rr_0 = xi_0*(omega*-2.0 + 4.0);

      // std::map<int, int> m{{-1, 0}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 0}};
      std::map<int, int> m{{-1, 0}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 0}};

      for(uint_t i = 0; i < _nBlocks; ++i)
      {
         for (auto it: _interval)
         {
            const cell_idx_t ctr_2 = it.z();
            const cell_idx_t ctr_1 = it.y();
            const cell_idx_t ctr_0 = it.x();

            const uint_t index_C = uint_c(27) * i + uint_c( (0 + 1) * 9 + (0 + 1) * 3 + 0 + 1 );
            const uint_t index_N = uint_c(27) * i + uint_c( (0 + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + 0 + 1 );
            const uint_t index_S = uint_c(27) * i + uint_c( (0 + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + 0 + 1 );
            const uint_t index_W = uint_c(27) * i + uint_c( (0 + 1) * 9 + (0 + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_E = uint_c(27) * i + uint_c( (0 + 1) * 9 + (0 + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_T = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + (0 + 1) * 3 + 0 + 1 );
            const uint_t index_B = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + (0 + 1) * 3 + 0 + 1 );
            const uint_t index_NW = uint_c(27) * i + uint_c( (0 + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_NE = uint_c(27) * i + uint_c( (0 + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_SW = uint_c(27) * i + uint_c( (0 + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_SE = uint_c(27) * i + uint_c( (0 + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_TN = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + 0 + 1 );
            const uint_t index_TS = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + 0 + 1 );
            const uint_t index_TW = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + (0 + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_TE = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + (0 + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_BN = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + 0 + 1 );
            const uint_t index_BS = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + 0 + 1 );
            const uint_t index_BW = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + (0 + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_BE = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + (0 + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_TNE = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_TNW = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_TSE = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_TSW = uint_c(27) * i + uint_c( ((ctr_2 -1 < 0 ? -1 : 0) + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_BNE = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_BNW = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + ((ctr_1 -1 < 0 ? -1 : 0) + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );
            const uint_t index_BSE = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + (ctr_0 -1 < 0 ? -1 : 0) + 1 );
            const uint_t index_BSW = uint_c(27) * i + uint_c( ((ctr_2 +1 > 7 ? 1 : 0) + 1) * 9 + ((ctr_1 +1 > 7 ? 1 : 0) + 1) * 3 + (ctr_0 +1 > 7 ? 1 : 0) + 1 );


            const double f_0 = pdfsTmp[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 0*_stride_pdfs_3];
            const double f_1 = pdfsTmp[index_N][ctr_0*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 1*_stride_pdfs_3];
            const double f_2 = pdfsTmp[index_S][ctr_0*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 2*_stride_pdfs_3];
            const double f_3 = pdfsTmp[index_W][m[ctr_0 + 1]*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 3*_stride_pdfs_3];
            const double f_4 = pdfsTmp[index_E][m[ctr_0 - 1]*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 4*_stride_pdfs_3];
            const double f_5 = pdfsTmp[index_T][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 5*_stride_pdfs_3];
            const double f_6 = pdfsTmp[index_B][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 6*_stride_pdfs_3];
            const double f_7 = pdfsTmp[index_NW][m[ctr_0 + 1]*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 7*_stride_pdfs_3];
            const double f_8 = pdfsTmp[index_NE][m[ctr_0 - 1]*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 8*_stride_pdfs_3];
            const double f_9 = pdfsTmp[index_SW][m[ctr_0 + 1]*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 9*_stride_pdfs_3];
            const double f_10 = pdfsTmp[index_SE][m[ctr_0 - 1]*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 10*_stride_pdfs_3];
            const double f_11 = pdfsTmp[index_TN][ctr_0*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 11*_stride_pdfs_3];
            const double f_12 = pdfsTmp[index_TS][ctr_0*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 12*_stride_pdfs_3];
            const double f_13 = pdfsTmp[index_TW][m[ctr_0 + 1]*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 13*_stride_pdfs_3];
            const double f_14 = pdfsTmp[index_TE][m[ctr_0 - 1]*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 14*_stride_pdfs_3];
            const double f_15 = pdfsTmp[index_BN][ctr_0*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 15*_stride_pdfs_3];
            const double f_16 = pdfsTmp[index_BS][ctr_0*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 16*_stride_pdfs_3];
            const double f_17 = pdfsTmp[index_BW][m[ctr_0 + 1]*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 17*_stride_pdfs_3];
            const double f_18 = pdfsTmp[index_BE][m[ctr_0 - 1]*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 18*_stride_pdfs_3];
            const double f_19 = pdfsTmp[index_TNE][m[ctr_0 - 1]*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 19*_stride_pdfs_3];
            const double f_20 = pdfsTmp[index_TNW][m[ctr_0 + 1]*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 20*_stride_pdfs_3];
            const double f_21 = pdfsTmp[index_TSE][m[ctr_0 - 1]*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 21*_stride_pdfs_3];
            const double f_22 = pdfsTmp[index_TSW][m[ctr_0 + 1]*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + m[ctr_2 - 1]*_stride_pdfs_2 + 22*_stride_pdfs_3];
            const double f_23 = pdfsTmp[index_BNE][m[ctr_0 - 1]*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 23*_stride_pdfs_3];
            const double f_24 = pdfsTmp[index_BNW][m[ctr_0 + 1]*_stride_pdfs_0 + m[ctr_1 - 1]*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 24*_stride_pdfs_3];
            const double f_25 = pdfsTmp[index_BSE][m[ctr_0 - 1]*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 25*_stride_pdfs_3];
            const double f_26 = pdfsTmp[index_BSW][m[ctr_0 + 1]*_stride_pdfs_0 + m[ctr_1 + 1]*_stride_pdfs_1 + m[ctr_2 + 1]*_stride_pdfs_2 + 26*_stride_pdfs_3];

            const double vel0Term = f_10 + f_14 + f_18 + f_19 + f_21 + f_23 + f_25 + f_4 + f_8;
            const double vel1Term = f_1 + f_11 + f_15 + f_20 + f_24 + f_7;
            const double vel2Term = f_12 + f_13 + f_22 + f_5;
            const double delta_rho = f_0 + f_16 + f_17 + f_2 + f_26 + f_3 + f_6 + f_9 + vel0Term + vel1Term + vel2Term;
            const double rho = delta_rho + 1.0;
            const double xi_1 = ((1.0) / (rho));
            const double u_0 = xi_1*(-f_13 - f_17 - f_20 - f_22 - f_24 - f_26 - f_3 - f_7 - f_9 + vel0Term);
            const double u_1 = xi_1*(-f_10 - f_12 - f_16 + f_19 - f_2 - f_21 - f_22 + f_23 - f_25 - f_26 + f_8 - f_9 + vel1Term);
            const double u_2 = xi_1*(f_11 + f_14 - f_15 - f_16 - f_17 - f_18 + f_19 + f_20 + f_21 - f_23 - f_24 - f_25 - f_26 - f_6 + vel2Term);
            const double u0Mu1 = u_0 - u_1;
            const double u0Pu1 = u_0 + u_1;
            const double u1Pu2 = u_1 + u_2;
            const double u1Mu2 = u_1 - u_2;
            const double u0Mu2 = u_0 - u_2;
            const double u0Pu2 = u_0 + u_2;
            const double f_eq_common = delta_rho + rho*-1.5*(u_0*u_0) + rho*-1.5*(u_1*u_1) + rho*-1.5*(u_2*u_2);

            const double r_0  = f_0 + omega*(-f_0 + f_eq_common*0.29629629629629628);
            const double r_1  = f_1 + omega*(f_1*-0.5 + f_2*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_1*u_1)) + rr_0*(f_1*-0.5 + f_2*0.5 + rho*u_1*0.22222222222222221);
            const double r_2  = f_2 + omega*(f_1*-0.5 + f_2*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_1*u_1)) + rr_0*(f_1*0.5 + f_2*-0.5 + rho*u_1*-0.22222222222222221);
            const double r_3  = f_3 + omega*(f_3*-0.5 + f_4*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_0*u_0)) + rr_0*(f_3*-0.5 + f_4*0.5 + rho*u_0*-0.22222222222222221);
            const double r_4  = f_4 + omega*(f_3*-0.5 + f_4*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_0*u_0)) + rr_0*(f_3*0.5 + f_4*-0.5 + rho*u_0*0.22222222222222221);
            const double r_5  = f_5 + omega*(f_5*-0.5 + f_6*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_2*u_2)) + rr_0*(f_5*-0.5 + f_6*0.5 + rho*u_2*0.22222222222222221);
            const double r_6  = f_6 + omega*(f_5*-0.5 + f_6*-0.5 + f_eq_common*0.07407407407407407 + rho*0.33333333333333331*(u_2*u_2)) + rr_0*(f_5*0.5 + f_6*-0.5 + rho*u_2*-0.22222222222222221);
            const double r_7  = f_7 + omega*(f_10*-0.5 + f_7*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Mu1*u0Mu1)) + rr_0*(f_10*0.5 + f_7*-0.5 + rho*u0Mu1*-0.055555555555555552);
            const double r_8  = f_8 + omega*(f_8*-0.5 + f_9*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Pu1*u0Pu1)) + rr_0*(f_8*-0.5 + f_9*0.5 + rho*u0Pu1*0.055555555555555552);
            const double r_9  = f_9 + omega*(f_8*-0.5 + f_9*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Pu1*u0Pu1)) + rr_0*(f_8*0.5 + f_9*-0.5 + rho*u0Pu1*-0.055555555555555552);
            const double r_10 = f_10 + omega*(f_10*-0.5 + f_7*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Mu1*u0Mu1)) + rr_0*(f_10*-0.5 + f_7*0.5 + rho*u0Mu1*0.055555555555555552);
            const double r_11 = f_11 + omega*(f_11*-0.5 + f_16*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u1Pu2*u1Pu2)) + rr_0*(f_11*-0.5 + f_16*0.5 + rho*u1Pu2*0.055555555555555552);
            const double r_12 = f_12 + omega*(f_12*-0.5 + f_15*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u1Mu2*u1Mu2)) + rr_0*(f_12*-0.5 + f_15*0.5 + rho*u1Mu2*-0.055555555555555552);
            const double r_13 = f_13 + omega*(f_13*-0.5 + f_18*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Mu2*u0Mu2)) + rr_0*(f_13*-0.5 + f_18*0.5 + rho*u0Mu2*-0.055555555555555552);
            const double r_14 = f_14 + omega*(f_14*-0.5 + f_17*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Pu2*u0Pu2)) + rr_0*(f_14*-0.5 + f_17*0.5 + rho*u0Pu2*0.055555555555555552);
            const double r_15 = f_15 + omega*(f_12*-0.5 + f_15*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u1Mu2*u1Mu2)) + rr_0*(f_12*0.5 + f_15*-0.5 + rho*u1Mu2*0.055555555555555552);
            const double r_16 = f_16 + omega*(f_11*-0.5 + f_16*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u1Pu2*u1Pu2)) + rr_0*(f_11*0.5 + f_16*-0.5 + rho*u1Pu2*-0.055555555555555552);
            const double r_17 = f_17 + omega*(f_14*-0.5 + f_17*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Pu2*u0Pu2)) + rr_0*(f_14*0.5 + f_17*-0.5 + rho*u0Pu2*-0.055555555555555552);
            const double r_18 = f_18 + omega*(f_13*-0.5 + f_18*-0.5 + f_eq_common*0.018518518518518517 + rho*0.083333333333333329*(u0Mu2*u0Mu2)) + rr_0*(f_13*0.5 + f_18*-0.5 + rho*u0Mu2*0.055555555555555552);
            const double r_19 = f_19 + omega*(delta_rho*-0.013888888888888888 + f_19*-0.5 + f_26*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Pu1*u0Pu1) + 0.020833333333333332*(u0Pu2*u0Pu2) + 0.020833333333333332*(u1Pu2*u1Pu2))) + rr_0*(f_19*-0.5 + f_26*0.5 + rho*(u0Pu1*0.013888888888888888 + u_2*0.013888888888888888));
            const double r_20 = f_20 + omega*(delta_rho*-0.013888888888888888 + f_20*-0.5 + f_25*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu1*u0Mu1) + 0.020833333333333332*(u0Mu2*u0Mu2) + 0.020833333333333332*(u1Pu2*u1Pu2))) + rr_0*(f_20*-0.5 + f_25*0.5 + rho*(u0Mu1*-0.013888888888888888 + u_2*0.013888888888888888));
            const double r_21 = f_21 + omega*(delta_rho*-0.013888888888888888 + f_21*-0.5 + f_24*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu1*u0Mu1) + 0.020833333333333332*(u0Pu2*u0Pu2) + 0.020833333333333332*(u1Mu2*u1Mu2))) + rr_0*(f_21*-0.5 + f_24*0.5 + rho*(u0Mu1*0.013888888888888888 + u_2*0.013888888888888888));
            const double r_22 = f_22 + omega*(delta_rho*-0.013888888888888888 + f_22*-0.5 + f_23*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu2*u0Mu2) + 0.020833333333333332*(u0Pu1*u0Pu1) + 0.020833333333333332*(u1Mu2*u1Mu2))) + rr_0*(f_22*-0.5 + f_23*0.5 + rho*(u0Pu1*-0.013888888888888888 + u_2*0.013888888888888888));
            const double r_23 = f_23 + omega*(delta_rho*-0.013888888888888888 + f_22*-0.5 + f_23*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu2*u0Mu2) + 0.020833333333333332*(u0Pu1*u0Pu1) + 0.020833333333333332*(u1Mu2*u1Mu2))) + rr_0*(f_22*0.5 + f_23*-0.5 + rho*(u0Pu1*0.013888888888888888 + u_2*-0.013888888888888888));
            const double r_24 = f_24 + omega*(delta_rho*-0.013888888888888888 + f_21*-0.5 + f_24*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu1*u0Mu1) + 0.020833333333333332*(u0Pu2*u0Pu2) + 0.020833333333333332*(u1Mu2*u1Mu2))) + rr_0*(f_21*0.5 + f_24*-0.5 + rho*(u0Mu1*-0.013888888888888888 + u_2*-0.013888888888888888));
            const double r_25 = f_25 + omega*(delta_rho*-0.013888888888888888 + f_20*-0.5 + f_25*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Mu1*u0Mu1) + 0.020833333333333332*(u0Mu2*u0Mu2) + 0.020833333333333332*(u1Pu2*u1Pu2))) + rr_0*(f_20*0.5 + f_25*-0.5 + rho*(u0Mu1*0.013888888888888888 + u_2*-0.013888888888888888));
            const double r_26 = f_26 + omega*(delta_rho*-0.013888888888888888 + f_19*-0.5 + f_26*-0.5 + f_eq_common*0.018518518518518517 + rho*(0.020833333333333332*(u0Pu1*u0Pu1) + 0.020833333333333332*(u0Pu2*u0Pu2) + 0.020833333333333332*(u1Pu2*u1Pu2))) + rr_0*(f_19*0.5 + f_26*-0.5 + rho*(u0Pu1*-0.013888888888888888 + u_2*-0.013888888888888888));

            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 0*_stride_pdfs_3] = r_0;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 1*_stride_pdfs_3] = r_1;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 2*_stride_pdfs_3] = r_2;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 3*_stride_pdfs_3] = r_3;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 4*_stride_pdfs_3] = r_4;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 5*_stride_pdfs_3] = r_5;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 6*_stride_pdfs_3] = r_6;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 7*_stride_pdfs_3] = r_7;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 8*_stride_pdfs_3] = r_8;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 9*_stride_pdfs_3] = r_9;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 10*_stride_pdfs_3] = r_10;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 11*_stride_pdfs_3] = r_11;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 12*_stride_pdfs_3] = r_12;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 13*_stride_pdfs_3] = r_13;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 14*_stride_pdfs_3] = r_14;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 15*_stride_pdfs_3] = r_15;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 16*_stride_pdfs_3] = r_16;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 17*_stride_pdfs_3] = r_17;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 18*_stride_pdfs_3] = r_18;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 19*_stride_pdfs_3] = r_19;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 20*_stride_pdfs_3] = r_20;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 21*_stride_pdfs_3] = r_21;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 22*_stride_pdfs_3] = r_22;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 23*_stride_pdfs_3] = r_23;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 24*_stride_pdfs_3] = r_24;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 25*_stride_pdfs_3] = r_25;
            pdfs[index_C][ctr_0*_stride_pdfs_0 + ctr_1*_stride_pdfs_1 + ctr_2*_stride_pdfs_2 + 26*_stride_pdfs_3] = r_26;
         }
      }
   }
 private:
   IDs ids_;
   std::vector<Block*> blockVector;
   std::vector<real_t> omegaVector;
   std::vector<real_t *> pdfs;
   std::vector<real_t *> pdfsTmp;
   CellInterval _interval;
   uint_t _nBlocks;
   int64_t _stride_pdfs_0;
   int64_t _stride_pdfs_1;
   int64_t _stride_pdfs_2;
   int64_t _stride_pdfs_3;
};

class Timestep
{
 public:
   Timestep(std::shared_ptr< StructuredBlockForest >& blocks, const IDs& ids,
            StreamCollide & streamCollide, std::shared_ptr< PackInfo_T >& packInfo, UniformBufferedScheme<CommunicationStencil_T>& communication)
      : blocks_(blocks), ids_(ids), streamCollide_(streamCollide), packInfo_(packInfo), communication_(communication)
   {

   }

   void commLocal(const uint_t level)
   {
      for( auto it = blocks_->begin(); it != blocks_->end(); ++it )
      {
         if(blocks_->getLevel(*it.get()) != level)
            continue;

         auto* sender = dynamic_cast< Block* >(it.get());
         for( auto dir = Stencil_T::beginNoCenter(); dir != Stencil_T::end(); ++dir )
         {
            const auto neighborIdx = blockforest::getBlockNeighborhoodSectionIndex(*dir);
            if (!(sender->neighborhoodSectionHasEquallySizedBlock(neighborIdx)))
               continue;

            const BlockID & receiverId = sender->getNeighborId( neighborIdx, uint_t(0) );
            auto receiver = dynamic_cast< Block * >( blocks_->getBlock(receiverId) );

            const PdfField_T * srcField = sender->getData< PdfField_T >( ids_.pdfField );
            PdfField_T * dstField = receiver->getData< PdfField_T >( ids_.pdfField );

            CellInterval srcRegion;
            CellInterval dstRegion;
            cell_idx_t gls = 1;
            srcField->getSliceBeforeGhostLayer(*dir, srcRegion, gls, false);
            dstField->getGhostRegion(stencil::inverseDir[*dir], dstRegion, gls, false);
            WALBERLA_ASSERT_EQUAL(srcRegion.xSize(), dstRegion.xSize())
            WALBERLA_ASSERT_EQUAL(srcRegion.ySize(), dstRegion.ySize())
            WALBERLA_ASSERT_EQUAL(srcRegion.zSize(), dstRegion.zSize())

            //WALBERLA_LOG_DEVEL_VAR(srcRegion)
            //WALBERLA_LOG_DEVEL_VAR(dstRegion)

            auto srcIter = srcRegion.begin();
            auto dstIter = dstRegion.begin();

            while (srcIter != srcRegion.end())
            {
               for( uint_t f = 0; f < Stencil_T::d_per_d_length[*dir]; ++f )
               {
                  dstField->get(*dstIter, Stencil_T::idx[ Stencil_T::d_per_d[*dir][f] ]) = srcField->get(*srcIter, Stencil_T::idx[ Stencil_T::d_per_d[*dir][f] ]);
               }
               ++srcIter;
               ++dstIter;
            }
            WALBERLA_ASSERT( srcIter == srcRegion.end() )
            WALBERLA_ASSERT( dstIter == dstRegion.end() )
         }
      }
   }

   void swap()
   {
      for( auto it = blocks_->begin(); it != blocks_->end(); ++it )
      {
         PdfField_T * srcField = it->getData< PdfField_T >( ids_.pdfField );
         PdfField_T * dstField = it->getData< PdfField_T >( ids_.pdfFieldTmp );
         srcField->swapDataPointers(dstField);
      }
   }

   void timestep()
   {
//      for( auto it = blocks_->begin(); it != blocks_->end(); ++it )
//      {
//         auto block = dynamic_cast< Block* >(it.get());
//         commLocal(0);
//         streamCollide_.streamCollide(block);
//         // streamCollide_(block);
//      }
// commLocal(0);
      if(((timestepCounter_) &1) ^ 1)
      {
         streamCollide_.streamCollideEven();
         swap();
      }
      else
      {
         streamCollide_.streamCollideOdd();
         swap();
      }
      timestepCounter_ = (timestepCounter_ + 1) & 1;
   }

   void operator()(){ timestep(); };

 private:
   std::shared_ptr< StructuredBlockForest > blocks_;
   IDs ids_;
   StreamCollide & streamCollide_;
   std::shared_ptr< PackInfo_T > packInfo_;
   UniformBufferedScheme<CommunicationStencil_T> communication_;
   uint8_t timestepCounter_{0};
};

int main(int argc, char** argv)
{
   mpi::Environment env(argc, argv);
   shared_ptr< Config > config = make_shared< Config >();
   config->readParameterFile(argv[1]);
   logging::configureLogging(config);

   // read parameters
   auto parameters           = config->getOneBlock("Parameters");
   auto loggingParameters    = config->getOneBlock("Logging");
   auto domainParameters     = config->getOneBlock("DomainSetup");
   // auto EvaluationParameters = config->getOneBlock("Evaluation");

   const real_t omega              = parameters.getParameter< real_t >("omega");
   const real_t maxLatticeVelocity = parameters.getParameter< real_t >("maxLatticeVelocity");
   const uint_t timesteps          = parameters.getParameter< uint_t >("timesteps") + uint_c(1);

   Setup setup;
   setup.blocks        = domainParameters.getParameter< Vector3< uint_t > >("blocks");
   setup.cellsPerBlock = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
   setup.cells = Vector3< uint_t >(setup.blocks[0] * setup.cellsPerBlock[0], setup.blocks[1] * setup.cellsPerBlock[1],
                                   setup.blocks[2] * setup.cellsPerBlock[2]);
   setup.periodic       = domainParameters.getParameter< Vector3< bool > >("periodic");
   setup.numGhostLayers = uint_c(1);

   const uint_t valuesPerCell = (uint_c(2) * Stencil_T::Q + VectorField_T ::F_SIZE + uint_c(2) * ScalarField_T::F_SIZE);
   const uint_t sizePerValue  = sizeof(PdfField_T::value_type);
   const memory_t memoryPerCell = memory_t(valuesPerCell * sizePerValue + uint_c(1));

   bool writeSetupForestAndReturn = loggingParameters.getParameter< bool >("writeSetupForestAndReturn", false);
   if (uint_c(MPIManager::instance()->numProcesses()) > 1) writeSetupForestAndReturn = false;

   if (writeSetupForestAndReturn)
   {
      std::string sbffile = "TaylorGreenVortex.bfs";

      std::ostringstream infoString;
      infoString << "You have selected the option of just creating the block structure (= domain decomposition) and "
                    "saving the result to file\n"
                    "by specifying the output file name \'"
                 << sbffile << "\' AND also specifying \'saveToFile\'.\n";

      if (MPIManager::instance()->numProcesses() > 1)
         WALBERLA_ABORT(infoString.str() << "In this mode you need to start " << argv[0] << " with just one process!")

      WALBERLA_LOG_INFO_ON_ROOT(infoString.str() << "Creating the block structure ...")

      const uint_t numberProcesses = domainParameters.getParameter< uint_t >("numberProcesses");

      shared_ptr< SetupBlockForest > sforest = createSetupBlockForest(setup, numberProcesses, memoryPerCell);
      sforest->writeVTKOutput("domain_decomposition");
      sforest->saveToFile(sbffile.c_str());

      logging::Logging::printFooterOnStream();
      return EXIT_SUCCESS;
   }

   auto blocks = createStructuredBlockForest(setup, memoryPerCell);

   IDs ids;
   const StorageSpecification_T StorageSpec = StorageSpecification_T();
   ids.pdfField      = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, setup.numGhostLayers, field::fzyx);
   ids.pdfFieldTmp   = lbm_generated::addPdfFieldToStorage(blocks, "pdfs tmp", StorageSpec, setup.numGhostLayers, field::fzyx);
   ids.velocityField = field::addToStorage< VectorField_T >(blocks, "vel", real_c(0.0), field::fzyx, setup.numGhostLayers);
   ids.densityField = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, setup.numGhostLayers);
   ids.flagField    = field::addFlagFieldToStorage< FlagField_T >(blocks, "Boundary Flag Field", uint_c(3));

   const cell_idx_t domainHalf = cell_idx_c(blocks->getDomain().yMax() / real_c(2.0));
   for (auto& block : *blocks)
   {
      auto velocityField = block.getData<VectorField_T>(ids.velocityField);
      for (cell_idx_t ctr_2 = 0; ctr_2 < velocityField->zSize(); ++ctr_2)
      {
         for(cell_idx_t ctr_1 = 0; ctr_1 < velocityField->ySize(); ++ctr_1)
         {
            for (cell_idx_t ctr_0 = 0; ctr_0 < velocityField->xSize(); ++ctr_0)
            {
               Cell domainCell(ctr_0, ctr_1, ctr_2);
               blocks->transformBlockLocalToGlobalCell(domainCell, block);
               if((domainCell.y() + real_c(0.5)) > domainHalf)
               {
                  velocityField->get(ctr_0, ctr_1, ctr_2, 0) = maxLatticeVelocity;
               }
               else
               {
                  velocityField->get(ctr_0, ctr_1, ctr_2, 0) = -maxLatticeVelocity;
               }
            }
         }
      }
   }

   SweepCollection_T sweepCollection(blocks, ids.pdfField, ids.densityField, ids.velocityField, omega);
   for (auto& block : *blocks)
   {
      sweepCollection.initialise(&block, cell_idx_c(setup.numGhostLayers - uint_c(1)));
   }

   for (auto& block : *blocks)
   {
      if(blocks->getLevel(block) == 0)
      {
         sweepCollection.streamCollide(&block);
      }
   }

   const FlagUID fluidFlagUID("Fluid");
   auto boundariesConfig = config->getBlock("Boundaries");
   geometry::initBoundaryHandling< FlagField_T >(*blocks, ids.flagField, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, ids.flagField, fluidFlagUID, cell_idx_c(0));

   // BoundaryCollection_T boundaryCollection(blocks, ids.flagField, ids.pdfField, fluidFlagUID, maxLatticeVelocity);

   WALBERLA_LOG_INFO_ON_ROOT("Setting up communication...")

   std::shared_ptr< PackInfo_T > packInfo = std::make_shared<lbm_generated::UniformGeneratedPdfPackInfo< PdfField_T >>(ids.pdfField);
   UniformBufferedScheme< Stencil_T > communication(blocks);
   communication.addPackInfo(packInfo);

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   auto VTKWriter                 = config->getOneBlock("VTKWriter");
   const uint_t vtkWriteFrequency = VTKWriter.getParameter< uint_t >("vtkWriteFrequency");
   const bool writeVelocity       = VTKWriter.getParameter< bool >("velocity");
   const bool writeDensity        = VTKWriter.getParameter< bool >("density");
   // const bool writeAverageFields  = VTKWriter.getParameter< bool >("averageFields", false);
   const bool writeFlag = VTKWriter.getParameter< bool >("flag");
   // const bool writeOnlySlice      = VTKWriter.getParameter< bool >("writeOnlySlice", true);
   const bool amrFileFormat     = VTKWriter.getParameter< bool >("amrFileFormat", false);
   const bool oneFilePerProcess = VTKWriter.getParameter< bool >("oneFilePerProcess", false);

   if (vtkWriteFrequency > 0)
   {
      auto vtkOutput =
         vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_TaylorGreenVortex", "simulation_step",
                                        false, true, true, false, 0, amrFileFormat, oneFilePerProcess);

      vtkOutput->addBeforeFunction([&]() {
         for (auto& block : *blocks)
         {
            sweepCollection.calculateMacroscopicParameters(&block);
         }
      });

      if (writeVelocity)
      {
         auto velWriter = make_shared< field::VTKWriter< VectorField_T, float32 > >(ids.velocityField, "velocity");
         vtkOutput->addCellDataWriter(velWriter);
      }
      if (writeDensity)
      {
         auto densityWriter = make_shared< field::VTKWriter< ScalarField_T, float32 > >(ids.densityField, "density");
         vtkOutput->addCellDataWriter(densityWriter);
      }
      if (writeFlag)
      {
         auto flagWriter = make_shared< field::VTKWriter< FlagField_T > >(ids.flagField, "flag");
         vtkOutput->addCellDataWriter(flagWriter);
      }
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   }

   StreamCollide streamCollide(blocks, ids, omega);
   Timestep timeloopFunction(blocks, ids, streamCollide, packInfo, communication);
   timeloop.addFuncBeforeTimeStep(timeloopFunction, "Refinement Cycle");


   // timeloop.addFuncBeforeTimeStep(streamCollide.getTimestepFunction(), "Refinement Cycle");
   // timeloop.addFuncBeforeTimeStep(streamCollide.swapPDFs(), "Swap Pointer");

   const real_t remainingTimeLoggerFrequency = loggingParameters.getParameter< real_t >("remainingTimeLoggerFrequency", 3.0); // in seconds
   if (uint_c(remainingTimeLoggerFrequency) > 0)
   {
      timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency), "remaining time logger");
   }

//   const uint_t evaluationCheckFrequency = EvaluationParameters.getParameter< uint_t >("evaluationCheckFrequency");
//   const bool evaluationLogToStream      = EvaluationParameters.getParameter< bool >("logToStream");
//   const bool evaluationLogToFile        = EvaluationParameters.getParameter< bool >("logToFile");
//   const std::string evaluationFilename  = EvaluationParameters.getParameter< std::string >("filename");

   // std::shared_ptr< AccuracyEvaluation > evaluation = std::make_shared< AccuracyEvaluation >(blocks, ids, sweepCollection, maxLatticeVelocity, evaluationCheckFrequency, evaluationLogToStream, evaluationLogToFile);
   // timeloop.addFuncBeforeTimeStep(SharedFunctor< AccuracyEvaluation >(evaluation), "evaluation");

   //////////////////////
   /// RUN SIMULATION ///
   //////////////////////
   const lbm_generated::PerformanceEvaluation< FlagField_T > performance(blocks, ids.flagField, fluidFlagUID);
   field::CellCounter< FlagField_T > fluidCells(blocks, ids.flagField, fluidFlagUID);
   fluidCells();

   WALBERLA_LOG_INFO_ON_ROOT("Blocks created: " << blocks->getNumberOfBlocks())
   WALBERLA_LOG_INFO_ON_ROOT("Starting Simulation")
   WcTimingPool timeloopTiming;
   WcTimer simTimer;

   WALBERLA_MPI_BARRIER()
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif

   simTimer.start();
   timeloop.run(timeloopTiming);
#if defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)
   WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
   WALBERLA_GPU_CHECK(gpuPeekAtLastError())
#endif
   WALBERLA_MPI_BARRIER()
   simTimer.end();

   WALBERLA_LOG_INFO_ON_ROOT("Simulation finished")
   real_t time = simTimer.max();
   WALBERLA_MPI_SECTION() { walberla::mpi::reduceInplace(time, walberla::mpi::MAX); }
   performance.logResultOnRoot(timesteps, time);

   const auto reducedTimeloopTiming = timeloopTiming.getReduced();
   WALBERLA_LOG_RESULT_ON_ROOT("Time loop timing:\n" << *reducedTimeloopTiming)

   printResidentMemoryStatistics();

   return EXIT_SUCCESS;
}
