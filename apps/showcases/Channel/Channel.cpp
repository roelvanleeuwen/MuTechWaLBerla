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
//! \file Channel.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//! \brief Couette flow
//
//======================================================================================================================
#include "blockforest/Initialization.h"
#include "blockforest/SetupBlockForest.h"
#include "blockforest/communication/NonUniformBufferedScheme.h"
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

#include "lbm_generated/communication/NonuniformGeneratedPdfPackInfo.h"
#include "lbm_generated/evaluation/PerformanceEvaluation.h"
#include "lbm_generated/field/AddToStorage.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/refinement/BasicRecursiveTimeStep.h"

// include the generated header file. It includes all generated classes
#include "ChannelHeader.h"

using namespace walberla;
using namespace std::placeholders;

using StorageSpecification_T = lbm::ChannelStorageSpecification;
using Stencil_T              = StorageSpecification_T::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;
using PdfField_T             = lbm_generated::PdfField< StorageSpecification_T >;
using PackInfo_T             = lbm_generated::NonuniformGeneratedPdfPackInfo< PdfField_T >;

using SweepCollection_T = lbm::ChannelSweepCollection;

using VectorField_T = GhostLayerField< real_t, StorageSpecification_T::Stencil::D >;
using ScalarField_T = GhostLayerField< real_t, 1 >;

using flag_t               = walberla::uint8_t;
using FlagField_T          = FlagField< flag_t >;
using BoundaryCollection_T = lbm::ChannelBoundaryCollection< FlagField_T >;

using Timestep_T = lbm_generated::BasicRecursiveTimeStep< PdfField_T, SweepCollection_T, BoundaryCollection_T >;

using blockforest::communication::NonUniformBufferedScheme;

static const real_t c0o1 = real_c(0.0) / real_c(1.0);
static const real_t c1o1 = real_c(1.0) / real_c(1.0);
static const real_t c2o1 = real_c(2.0) / real_c(1.0);
static const real_t c3o1 = real_c(3.0) / real_c(1.0);
static const real_t c6o1 = real_c(6.0) / real_c(1.0);
static const real_t c8o1 = real_c(8.0) / real_c(1.0);
static const real_t c9o1 = real_c(9.0) / real_c(1.0);
static const real_t c18o1 = real_c(18.0) / real_c(1.0);
static const real_t c36o1 = real_c(36.0) / real_c(1.0);

static const real_t c1o2 = real_c(1.0) / real_c(2.0);
static const real_t c3o2 = real_c(3.0) / real_c(2.0);
static const real_t c9o2 = real_c(9.0) / real_c(2.0);

static const real_t c9o4 = real_c(9.0) / real_c(4.0);

static const real_t c1o3 = real_c(1.0) / real_c(3.0);
static const real_t c2o3 = real_c(2.0) / real_c(3.0);
static const real_t c2o9 = real_c(2.0) / real_c(9.0);
static const real_t c4o9 = real_c(4.0) / real_c(9.0);

static const real_t c1o4 = real_c(1.0) / real_c(4.0);
static const real_t c1o6 = real_c(1.0) / real_c(6.0);
static const real_t c1o8 = real_c(1.0) / real_c(8.0);
static const real_t c1o9 = real_c(1.0) / real_c(9.0);
static const real_t c1o16 = real_c(1.0) / real_c(16.0);
static const real_t c1o18 = real_c(1.0) / real_c(18.0);
static const real_t c1o27 = real_c(1.0) / real_c(27.0);
static const real_t c1o36 = real_c(1.0) / real_c(36.0);
static const real_t c1o64 = real_c(1.0) / real_c(64.0);

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
        executionCounter_(uint_c(0)), filename_("Channel.txt"), normalizationFactor_(real_c(1.0))
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

shared_ptr< SetupBlockForest >
   createSetupBlockForest(const blockforest::RefinementSelectionFunctions& refinementSelectionFunctions,
                          const Setup& setup, uint_t numberOfProcesses, const memory_t memoryPerCell)
{
   shared_ptr< SetupBlockForest > forest = make_shared< SetupBlockForest >();

   const memory_t memoryPerBlock =
      numeric_cast< memory_t >((setup.cellsPerBlock[0] + uint_t(2) * setup.numGhostLayers) *
                               (setup.cellsPerBlock[1] + uint_t(2) * setup.numGhostLayers) *
                               (setup.cellsPerBlock[2] + uint_t(2) * setup.numGhostLayers)) *
      memoryPerCell;

   forest->addRefinementSelectionFunction(refinementSelectionFunctions);
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

shared_ptr< blockforest::StructuredBlockForest >
   createStructuredBlockForest(const blockforest::RefinementSelectionFunctions& refinementSelectionFunctions,
                               const Setup& setup, const memory_t memoryPerCell)
{
   // WALBERLA_LOG_INFO_ON_ROOT("Creating the block structure ...");
   shared_ptr< SetupBlockForest > sforest = createSetupBlockForest(
      refinementSelectionFunctions, setup, uint_c(MPIManager::instance()->numProcesses()), memoryPerCell);

   auto bf  = std::make_shared< blockforest::BlockForest >(uint_c(MPIManager::instance()->rank()), *sforest, false);
   auto sbf = std::make_shared< blockforest::StructuredBlockForest >(bf, setup.cellsPerBlock[0], setup.cellsPerBlock[1],
                                                                     setup.cellsPerBlock[2]);
   sbf->createCellBoundingBoxes();

   return sbf;
}
} // namespace

class RefinementSelection
{
 public:
   RefinementSelection(const uint_t level) : level_(level) {}

   void operator()(SetupBlockForest& forest)
   {
      const AABB& domain = forest.getDomain();
      const real_t ySize = (domain.ySize() / real_t(12)) * real_c(0.99);

      const AABB intersectDomain(domain.xMin(), domain.yMax() - ySize, domain.zMin(), domain.xMax(), domain.yMax(),
                                 domain.zMax());

      for (auto& block : forest)
      {
         const AABB& aabb = block.getAABB();
         if (intersectDomain.intersects(aabb))
         {
            if (block.getLevel() < level_) block.setMarker(true);
         }
      }
   }

 private:
   uint_t level_;
}; // class RefinementSelection

class StreamCollide
{
 public:
   StreamCollide(const shared_ptr< StructuredBlockForest > & blocks, const IDs& ids, const real_t omega) : ids_(ids)
   {
      for (uint_t level = 0; level < blocks->getNumberOfLevels(); level++)
      {
         const double level_scale_factor = double(uint_t(1) << level);
         const double one                = double(1.0);
         const double half               = double(0.5);

         omegaVector.push_back( double(omega / (level_scale_factor * (-omega * half + one) + omega * half)) );
      }
   }

   void operator()(Block * block)
   {
      auto pdfField = block->getData<PdfField_T>(ids_.pdfField);
      auto pdfFieldTmp = block->getData<PdfField_T>(ids_.pdfFieldTmp);

      CellInterval interval =pdfField->xyzSize();
      streamCollide(block, interval);
      pdfField->swapDataPointers(pdfFieldTmp);
   }

   void streamCollide(Block * block, const CellInterval & ci)
   {
      auto pdfField = block->getData<PdfField_T>(ids_.pdfField);
      auto pdfFieldTmp = block->getData<PdfField_T>(ids_.pdfFieldTmp);

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
   }

   void stream(Block * block, const CellInterval & ci)
   {
      auto pdfField = block->getData<PdfField_T>(ids_.pdfField);
      auto pdfFieldTmp = block->getData<PdfField_T>(ids_.pdfFieldTmp);

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

         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 0)  = f_0;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 1)  = f_1;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 2)  = f_2;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 3)  = f_3;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 4)  = f_4;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 5)  = f_5;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 6)  = f_6;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 7)  = f_7;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 8)  = f_8;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 9)  = f_9;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 10) = f_10;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 11) = f_11;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 12) = f_12;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 13) = f_13;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 14) = f_14;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 15) = f_15;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 16) = f_16;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 17) = f_17;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 18) = f_18;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 19) = f_19;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 20) = f_20;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 21) = f_21;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 22) = f_22;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 23) = f_23;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 24) = f_24;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 25) = f_25;
         pdfFieldTmp->get(ctr_0, ctr_1, ctr_2, 26) = f_26;
      }
   }

 private:
   IDs ids_;
   std::vector<real_t> omegaVector;
};

struct InterpolationCoefficients
{
   real_t a000, a100, a010, a001, a200, a020, a002, a110, a101, a011;
   real_t b000, b100, b010, b001, b200, b020, b002, b110, b101, b011;
   real_t c000, c100, c010, c001, c200, c020, c002, c110, c101, c011;
   real_t d000, d100, d010, d001, d110, d101, d011;
   real_t a111, b111, c111, d111;
   real_t LaplaceRho;
};


struct MomentsOnSourceNode
{
   real_t drho;
   real_t velocityX;
   real_t velocityY;
   real_t velocityZ;
   real_t kxyFromfcNEQ;
   real_t kyzFromfcNEQ;
   real_t kxzFromfcNEQ;
   real_t kxxMyyFromfcNEQ;
   real_t kxxMzzFromfcNEQ;
   bool dirty{true};

   void calculate(const real_t *const f, const real_t omega)
   {
      const real_t fP00 = f[4];
      const real_t fM00 = f[3];
      const real_t f0P0 = f[1];
      const real_t f0M0 = f[2];
      const real_t f00P = f[5];
      const real_t f00M = f[6];
      const real_t fPP0 = f[8];
      const real_t fMM0 = f[9];
      const real_t fPM0 = f[10];
      const real_t fMP0 = f[7];
      const real_t fP0P = f[14];
      const real_t fM0M = f[17];
      const real_t fP0M = f[18];
      const real_t fM0P = f[13];
      const real_t f0PP = f[11];
      const real_t f0MM = f[16];
      const real_t f0PM = f[15];
      const real_t f0MP = f[12];
      const real_t fPPP = f[19];
      const real_t fMPP = f[20];
      const real_t fPMP = f[21];
      const real_t fMMP = f[22];
      const real_t fPPM = f[23];
      const real_t fMPM = f[24];
      const real_t fPMM = f[25];
      const real_t fMMM = f[26];

      const real_t vel0Term = f[10] + f[14] + f[18] + f[19] + f[21] + f[23] + f[25] + f[4] + f[8];
      const real_t vel1Term = f[1] + f[11] + f[15] + f[20] + f[24] + f[7];
      const real_t vel2Term = f[12] + f[13] + f[22] + f[5];
      this->drho = f[0] + f[16] + f[17] + f[2] + f[26] + f[3] + f[6] + f[9] + vel0Term + vel1Term + vel2Term;
      const real_t rho = this->drho + 1.0;
      const real_t oneOverRho = ((1.0) / (rho));
      this->velocityX = oneOverRho*(-f[13] - f[17] - f[20] - f[22] - f[24] - f[26] - f[3] - f[7] - f[9] + vel0Term);
      this->velocityY = oneOverRho*(-f[10] - f[12] - f[16] + f[19] - f[2] - f[21] - f[22] + f[23] - f[25] - f[26] + f[8] - f[9] + vel1Term);
      this->velocityZ = oneOverRho*(f[11] + f[14] - f[15] - f[16] - f[17] - f[18] + f[19] + f[20] + f[21] - f[23] - f[24] - f[25] - f[26] - f[6] + vel2Term);

      ////////////////////////////////////////////////////////////////////////////////////
      //! - Calculate second order moments for interpolation
      //!
      // example: kxxMzz: moment, second derivative in x direction minus the second derivative in z direction

      this->kxyFromfcNEQ = -c3o1 * omega *
                           ((fMM0 + fMMM + fMMP - fMP0 - fMPM - fMPP - fPM0 - fPMM - fPMP + fPP0 + fPPM + fPPP) *
                               oneOverRho - ((this->velocityX * this->velocityY)));


      this->kyzFromfcNEQ = -c3o1 * omega *
                           ((f0MM + fPMM + fMMM - f0MP - fPMP - fMMP - f0PM - fPPM - fMPM + f0PP + fPPP + fMPP) *
                               oneOverRho -
                            ((this->velocityY * this->velocityZ)));
      this->kxzFromfcNEQ = -c3o1 * omega *
                           ((fM0M + fMMM + fMPM - fM0P - fMMP - fMPP - fP0M - fPMM - fPPM + fP0P + fPMP + fPPP) *
                               oneOverRho -
                            ((this->velocityX * this->velocityZ)));
      this->kxxMyyFromfcNEQ = -c3o2 * omega *
                              ((fM0M + fM00 + fM0P - f0MM - f0M0 - f0MP - f0PM - f0P0 - f0PP + fP0M + fP00 + fP0P) *
                                  oneOverRho -
                               ((this->velocityX * this->velocityX - this->velocityY * this->velocityY)));
      this->kxxMzzFromfcNEQ = -c3o2 * omega *
                              ((fMM0 + fM00 + fMP0 - f0MM - f0MP - f00M - f00P - f0PM - f0PP + fPM0 + fP00 + fPP0) *
                                  oneOverRho -
                               ((this->velocityX * this->velocityX - this->velocityZ * this->velocityZ)));
      dirty = false;
   }

};


class MomentsOnSourceNodeSet
{
 private:
   MomentsOnSourceNode momentsPPP;
   MomentsOnSourceNode momentsMPP;
   MomentsOnSourceNode momentsPMP;
   MomentsOnSourceNode momentsMMP;
   MomentsOnSourceNode momentsPPM;
   MomentsOnSourceNode momentsMPM;
   MomentsOnSourceNode momentsPMM;
   MomentsOnSourceNode momentsMMM;

 public:
   void calculatePPP(const real_t *const f, const real_t omega)
   {
      momentsPPP.calculate(f, omega);
   }

    void calculateMPP(const real_t *const f, const real_t omega)
   {
      momentsMPP.calculate(f, omega);
   }

   void calculatePMP(const real_t *const f, const real_t omega)
   {
      momentsPMP.calculate(f, omega);
   }

   void calculateMMP(const real_t *const f, const real_t omega)
   {
      momentsMMP.calculate(f, omega);
   }

   void calculatePPM(const real_t *const f, const real_t omega)
   {
      momentsPPM.calculate(f, omega);
   }

   void calculateMPM(const real_t *const f, const real_t omega)
   {
      momentsMPM.calculate(f, omega);
   }

   void calculatePMM(const real_t *const f, const real_t omega)
   {
      momentsPMM.calculate(f, omega);
   }

   void calculateMMM(const real_t *const f, const real_t omega)
   {
      momentsMMM.calculate(f, omega);
   }

   void check()
   {
      WALBERLA_CHECK(!momentsPPP.dirty)
      WALBERLA_CHECK(!momentsMPP.dirty)
      WALBERLA_CHECK(!momentsPMP.dirty)
      WALBERLA_CHECK(!momentsMMP.dirty)
      WALBERLA_CHECK(!momentsPPM.dirty)
      WALBERLA_CHECK(!momentsMPM.dirty)
      WALBERLA_CHECK(!momentsPMM.dirty)
      WALBERLA_CHECK(!momentsMMM.dirty)
   }

   void calculateCoefficients(InterpolationCoefficients &coefficients) const
   {
      real_t& a000 = coefficients.a000;
      real_t& b000 = coefficients.b000;
      real_t& c000 = coefficients.c000;
      real_t& d000 = coefficients.d000;

      real_t& a100 = coefficients.a100;
      real_t& b100 = coefficients.b100;
      real_t& c100 = coefficients.c100;
      real_t& d100 = coefficients.d100;

      real_t& a010 = coefficients.a010;
      real_t& b010 = coefficients.b010;
      real_t& c010 = coefficients.c010;
      real_t& d010 = coefficients.d010;

      real_t& a001 = coefficients.a001;
      real_t& b001 = coefficients.b001;
      real_t& c001 = coefficients.c001;
      real_t& d001 = coefficients.d001;

      real_t& d110 = coefficients.d110, &d101 = coefficients.d101, &d011 = coefficients.d011;

      real_t& a200 = coefficients.a200, &a020 = coefficients.a020, &a002 = coefficients.a002;
      real_t& b200 = coefficients.b200, &b020 = coefficients.b020, &b002 = coefficients.b002;
      real_t& c200 = coefficients.c200, &c020 = coefficients.c020, &c002 = coefficients.c002;

      real_t& a110 = coefficients.a110, &a101 = coefficients.a101, &a011 = coefficients.a011;
      real_t& b110 = coefficients.b110, &b101 = coefficients.b101, &b011 = coefficients.b011;
      real_t& c110 = coefficients.c110, &c101 = coefficients.c101, &c011 = coefficients.c011;

      real_t &a111 = coefficients.a111, &b111 = coefficients.b111, &c111 = coefficients.c111, &d111 = coefficients.d111;

      real_t &LaplaceRho = coefficients.LaplaceRho;

      const real_t drhoPPP = momentsPPP.drho, vx1PPP = momentsPPP.velocityX, vx2PPP = momentsPPP.velocityY, vx3PPP = momentsPPP.velocityZ;
      const real_t drhoMPP = momentsMPP.drho, vx1MPP = momentsMPP.velocityX, vx2MPP = momentsMPP.velocityY, vx3MPP = momentsMPP.velocityZ;
      const real_t drhoPMP = momentsPMP.drho, vx1PMP = momentsPMP.velocityX, vx2PMP = momentsPMP.velocityY, vx3PMP = momentsPMP.velocityZ;
      const real_t drhoMMP = momentsMMP.drho, vx1MMP = momentsMMP.velocityX, vx2MMP = momentsMMP.velocityY, vx3MMP = momentsMMP.velocityZ;
      const real_t drhoPPM = momentsPPM.drho, vx1PPM = momentsPPM.velocityX, vx2PPM = momentsPPM.velocityY, vx3PPM = momentsPPM.velocityZ;
      const real_t drhoMPM = momentsMPM.drho, vx1MPM = momentsMPM.velocityX, vx2MPM = momentsMPM.velocityY, vx3MPM = momentsMPM.velocityZ;
      const real_t drhoPMM = momentsPMM.drho, vx1PMM = momentsPMM.velocityX, vx2PMM = momentsPMM.velocityY, vx3PMM = momentsPMM.velocityZ;
      const real_t drhoMMM = momentsMMM.drho, vx1MMM = momentsMMM.velocityX, vx2MMM = momentsMMM.velocityY, vx3MMM = momentsMMM.velocityZ;

      // second order moments at the source nodes
      const real_t kxyFromfcNEQPPP = momentsPPP.kxyFromfcNEQ, kyzFromfcNEQPPP = momentsPPP.kyzFromfcNEQ, kxzFromfcNEQPPP = momentsPPP.kxzFromfcNEQ, kxxMyyFromfcNEQPPP = momentsPPP.kxxMyyFromfcNEQ, kxxMzzFromfcNEQPPP = momentsPPP.kxxMzzFromfcNEQ;
      const real_t kxyFromfcNEQMPP = momentsMPP.kxyFromfcNEQ, kyzFromfcNEQMPP = momentsMPP.kyzFromfcNEQ, kxzFromfcNEQMPP = momentsMPP.kxzFromfcNEQ, kxxMyyFromfcNEQMPP = momentsMPP.kxxMyyFromfcNEQ, kxxMzzFromfcNEQMPP = momentsMPP.kxxMzzFromfcNEQ;
      const real_t kxyFromfcNEQPMP = momentsPMP.kxyFromfcNEQ, kyzFromfcNEQPMP = momentsPMP.kyzFromfcNEQ, kxzFromfcNEQPMP = momentsPMP.kxzFromfcNEQ, kxxMyyFromfcNEQPMP = momentsPMP.kxxMyyFromfcNEQ, kxxMzzFromfcNEQPMP = momentsPMP.kxxMzzFromfcNEQ;
      const real_t kxyFromfcNEQMMP = momentsMMP.kxyFromfcNEQ, kyzFromfcNEQMMP = momentsMMP.kyzFromfcNEQ, kxzFromfcNEQMMP = momentsMMP.kxzFromfcNEQ, kxxMyyFromfcNEQMMP = momentsMMP.kxxMyyFromfcNEQ, kxxMzzFromfcNEQMMP = momentsMMP.kxxMzzFromfcNEQ;
      const real_t kxyFromfcNEQPPM = momentsPPM.kxyFromfcNEQ, kyzFromfcNEQPPM = momentsPPM.kyzFromfcNEQ, kxzFromfcNEQPPM = momentsPPM.kxzFromfcNEQ, kxxMyyFromfcNEQPPM = momentsPPM.kxxMyyFromfcNEQ, kxxMzzFromfcNEQPPM = momentsPPM.kxxMzzFromfcNEQ;
      const real_t kxyFromfcNEQMPM = momentsMPM.kxyFromfcNEQ, kyzFromfcNEQMPM = momentsMPM.kyzFromfcNEQ, kxzFromfcNEQMPM = momentsMPM.kxzFromfcNEQ, kxxMyyFromfcNEQMPM = momentsMPM.kxxMyyFromfcNEQ, kxxMzzFromfcNEQMPM = momentsMPM.kxxMzzFromfcNEQ;
      const real_t kxyFromfcNEQPMM = momentsPMM.kxyFromfcNEQ, kyzFromfcNEQPMM = momentsPMM.kyzFromfcNEQ, kxzFromfcNEQPMM = momentsPMM.kxzFromfcNEQ, kxxMyyFromfcNEQPMM = momentsPMM.kxxMyyFromfcNEQ, kxxMzzFromfcNEQPMM = momentsPMM.kxxMzzFromfcNEQ;
      const real_t kxyFromfcNEQMMM = momentsMMM.kxyFromfcNEQ, kyzFromfcNEQMMM = momentsMMM.kyzFromfcNEQ, kxzFromfcNEQMMM = momentsMMM.kxzFromfcNEQ, kxxMyyFromfcNEQMMM = momentsMMM.kxxMyyFromfcNEQ, kxxMzzFromfcNEQMMM = momentsMMM.kxxMzzFromfcNEQ;

      a000 = c1o64 * (
                        c2o1 * (
                                  ((kxyFromfcNEQMMM - kxyFromfcNEQPPP) + (kxyFromfcNEQMMP - kxyFromfcNEQPPM)) + ((kxyFromfcNEQPMM - kxyFromfcNEQMPP) + (kxyFromfcNEQPMP - kxyFromfcNEQMPM)) +
                                  ((kxzFromfcNEQMMM - kxzFromfcNEQPPP) + (kxzFromfcNEQPPM - kxzFromfcNEQMMP)) + ((kxzFromfcNEQPMM - kxzFromfcNEQMPP) + (kxzFromfcNEQMPM - kxzFromfcNEQPMP)) +
                                  ((vx2PPP + vx2MMM) + (vx2PPM + vx2MMP)) - ((vx2MPP + vx2PMM) + (vx2MPM + vx2PMP)) +
                                  ((vx3PPP + vx3MMM) - (vx3PPM + vx3MMP)) + ((vx3PMP + vx3MPM) - (vx3MPP + vx3PMM))) +
                        c8o1 * (((vx1PPP + vx1MMM) + (vx1PPM + vx1MMP)) + ((vx1MPP + vx1PMM) + (vx1PMP + vx1MPM))) +
                        ((kxxMyyFromfcNEQMMM - kxxMyyFromfcNEQPPP) + (kxxMyyFromfcNEQMMP - kxxMyyFromfcNEQPPM)) +
                        ((kxxMyyFromfcNEQMPP - kxxMyyFromfcNEQPMM) + (kxxMyyFromfcNEQMPM - kxxMyyFromfcNEQPMP)) +
                        ((kxxMzzFromfcNEQMMM - kxxMzzFromfcNEQPPP) + (kxxMzzFromfcNEQMMP - kxxMzzFromfcNEQPPM)) +
                        ((kxxMzzFromfcNEQMPP - kxxMzzFromfcNEQPMM) + (kxxMzzFromfcNEQMPM - kxxMzzFromfcNEQPMP)));
      b000 = c1o64 * (
                        c2o1 * (
                                  ((kxxMyyFromfcNEQPPP - kxxMyyFromfcNEQMMM) + (kxxMyyFromfcNEQPPM - kxxMyyFromfcNEQMMP)) +
                                  ((kxxMyyFromfcNEQMPP - kxxMyyFromfcNEQPMM) + (kxxMyyFromfcNEQMPM - kxxMyyFromfcNEQPMP)) +
                                  ((kxyFromfcNEQMMM - kxyFromfcNEQPPP) + (kxyFromfcNEQMMP - kxyFromfcNEQPPM)) +
                                  ((kxyFromfcNEQMPP - kxyFromfcNEQPMM) + (kxyFromfcNEQMPM - kxyFromfcNEQPMP)) +
                                  ((kyzFromfcNEQMMM - kyzFromfcNEQPPP) + (kyzFromfcNEQPPM - kyzFromfcNEQMMP)) +
                                  ((kyzFromfcNEQPMM - kyzFromfcNEQMPP) + (kyzFromfcNEQMPM - kyzFromfcNEQPMP)) +
                                  ((vx1PPP + vx1MMM) + (vx1PPM + vx1MMP)) - ((vx1MPM + vx1MPP) + (vx1PMM + vx1PMP)) +
                                  ((vx3PPP + vx3MMM) - (vx3PPM + vx3MMP)) + ((vx3MPP + vx3PMM) - (vx3MPM + vx3PMP))) +
                        c8o1 * (((vx2PPP + vx2MMM) + (vx2PPM + vx2MMP)) + ((vx2MPP + vx2PMM) + (vx2MPM + vx2PMP))) +
                        ((kxxMzzFromfcNEQMMM - kxxMzzFromfcNEQPPP) + (kxxMzzFromfcNEQMMP - kxxMzzFromfcNEQPPM)) +
                        ((kxxMzzFromfcNEQPMM - kxxMzzFromfcNEQMPP) + (kxxMzzFromfcNEQPMP - kxxMzzFromfcNEQMPM)));
      c000 = c1o64 * (
                        c2o1 * (
                                  ((kxxMzzFromfcNEQPPP - kxxMzzFromfcNEQMMM) + (kxxMzzFromfcNEQMMP - kxxMzzFromfcNEQPPM)) +
                                  ((kxxMzzFromfcNEQMPP - kxxMzzFromfcNEQPMM) + (kxxMzzFromfcNEQPMP - kxxMzzFromfcNEQMPM)) +
                                  ((kxzFromfcNEQMMM - kxzFromfcNEQPPP) + (kxzFromfcNEQMMP - kxzFromfcNEQPPM)) +
                                  ((kxzFromfcNEQMPP - kxzFromfcNEQPMM) + (kxzFromfcNEQMPM - kxzFromfcNEQPMP)) +
                                  ((kyzFromfcNEQMMM - kyzFromfcNEQPPP) + (kyzFromfcNEQMMP - kyzFromfcNEQPPM)) +
                                  ((kyzFromfcNEQPMM - kyzFromfcNEQMPP) + (kyzFromfcNEQPMP - kyzFromfcNEQMPM)) +
                                  ((vx1PPP + vx1MMM) - (vx1MMP + vx1PPM)) + ((vx1MPM + vx1PMP) - (vx1MPP + vx1PMM)) +
                                  ((vx2PPP + vx2MMM) - (vx2MMP + vx2PPM)) + ((vx2MPP + vx2PMM) - (vx2MPM + vx2PMP))) +
                        c8o1 * (((vx3PPP + vx3MMM) + (vx3PPM + vx3MMP)) + ((vx3PMM + vx3MPP) + (vx3PMP + vx3MPM))) +
                        ((kxxMyyFromfcNEQMMM - kxxMyyFromfcNEQPPP) + (kxxMyyFromfcNEQPPM - kxxMyyFromfcNEQMMP)) +
                        ((kxxMyyFromfcNEQPMM - kxxMyyFromfcNEQMPP) + (kxxMyyFromfcNEQMPM - kxxMyyFromfcNEQPMP)));

      a100 = c1o4 * (((vx1PPP - vx1MMM) + (vx1PPM - vx1MMP)) + ((vx1PMM - vx1MPP) + (vx1PMP - vx1MPM)));
      b100 = c1o4 * (((vx2PPP - vx2MMM) + (vx2PPM - vx2MMP)) + ((vx2PMM - vx2MPP) + (vx2PMP - vx2MPM)));
      c100 = c1o4 * (((vx3PPP - vx3MMM) + (vx3PPM - vx3MMP)) + ((vx3PMM - vx3MPP) + (vx3PMP - vx3MPM)));

      a200 = c1o16 * (
                        c2o1 * (
                                  ((vx2PPP + vx2MMM) + (vx2PPM - vx2MPP)) + ((vx2MMP - vx2PMM) - (vx2MPM + vx2PMP)) +
                                  ((vx3PPP + vx3MMM) - (vx3PPM + vx3MPP)) + ((vx3MPM + vx3PMP) - (vx3MMP + vx3PMM))) +
                        ((kxxMyyFromfcNEQPPP - kxxMyyFromfcNEQMMM) + (kxxMyyFromfcNEQPPM - kxxMyyFromfcNEQMMP)) +
                        ((kxxMyyFromfcNEQPMM - kxxMyyFromfcNEQMPP) + (kxxMyyFromfcNEQPMP - kxxMyyFromfcNEQMPM)) +
                        ((kxxMzzFromfcNEQPPP - kxxMzzFromfcNEQMMM) + (kxxMzzFromfcNEQPPM - kxxMzzFromfcNEQMMP)) +
                        ((kxxMzzFromfcNEQPMM - kxxMzzFromfcNEQMPP) + (kxxMzzFromfcNEQPMP - kxxMzzFromfcNEQMPM)));
      b200 = c1o8 * (
                       c2o1 * (
                                 -((vx1PPP + vx1MMM) + (vx1PPM + vx1MMP)) + ((vx1MPP + vx1PMM) + (vx1MPM + vx1PMP))) +
                       ((kxyFromfcNEQPPP - kxyFromfcNEQMMM) + (kxyFromfcNEQPPM - kxyFromfcNEQMMP)) +
                       ((kxyFromfcNEQPMM - kxyFromfcNEQMPP) + (kxyFromfcNEQPMP - kxyFromfcNEQMPM)));
      c200 = c1o8 * (
                       c2o1 * (
                                 ((vx1PPM + vx1MMP) - (vx1PPP + vx1MMM)) + ((vx1MPP + vx1PMM) - (vx1MPM + vx1PMP))) +
                       ((kxzFromfcNEQPPP - kxzFromfcNEQMMM) + (kxzFromfcNEQPPM - kxzFromfcNEQMMP)) +
                       ((kxzFromfcNEQPMM - kxzFromfcNEQMPP) + (kxzFromfcNEQPMP - kxzFromfcNEQMPM)));

      a010 = c1o4 * (((vx1PPP - vx1MMM) + (vx1PPM - vx1MMP)) + ((vx1MPP - vx1PMM) + (vx1MPM - vx1PMP)));
      b010 = c1o4 * (((vx2PPP - vx2MMM) + (vx2PPM - vx2MMP)) + ((vx2MPP - vx2PMM) + (vx2MPM - vx2PMP)));
      c010 = c1o4 * (((vx3PPP - vx3MMM) + (vx3PPM - vx3MMP)) + ((vx3MPP - vx3PMM) + (vx3MPM - vx3PMP)));

      a020 = c1o8 * (
                       c2o1 * (-((vx2PPP + vx2MMM) + (vx2MMP + vx2PPM)) + ((vx2MPP + vx2PMM) + (vx2MPM + vx2PMP))) +
                       ((kxyFromfcNEQPPP - kxyFromfcNEQMMM) + (kxyFromfcNEQPPM - kxyFromfcNEQMMP)) +
                       ((kxyFromfcNEQMPP - kxyFromfcNEQPMM) + (kxyFromfcNEQMPM - kxyFromfcNEQPMP)));
      b020 = c1o16 * (
                        c2o1 * (
                                  ((kxxMyyFromfcNEQMMM - kxxMyyFromfcNEQPPP) + (kxxMyyFromfcNEQMMP - kxxMyyFromfcNEQPPM)) +
                                  ((kxxMyyFromfcNEQPMM - kxxMyyFromfcNEQMPP) + (kxxMyyFromfcNEQPMP - kxxMyyFromfcNEQMPM)) +
                                  ((vx1PPP + vx1MMM) + (vx1PPM + vx1MMP)) - ((vx1MPP + vx1PMM) + (vx1PMP + vx1MPM)) +
                                  ((vx3PPP + vx3MMM) - (vx3PPM + vx3MMP)) + ((vx3MPP + vx3PMM) - (vx3MPM + vx3PMP))) +
                        ((kxxMzzFromfcNEQPPP - kxxMzzFromfcNEQMMM) + (kxxMzzFromfcNEQPPM - kxxMzzFromfcNEQMMP)) +
                        ((kxxMzzFromfcNEQMPP - kxxMzzFromfcNEQPMM) + (kxxMzzFromfcNEQMPM - kxxMzzFromfcNEQPMP)));
      c020 = c1o8 * (
                       c2o1 * (((vx2MMP + vx2PPM) - (vx2PPP + vx2MMM)) + ((vx2PMP + vx2MPM) - (vx2MPP + vx2PMM))) +
                       ((kyzFromfcNEQPPP - kyzFromfcNEQMMM) + (kyzFromfcNEQPPM - kyzFromfcNEQMMP)) +
                       ((kyzFromfcNEQMPP - kyzFromfcNEQPMM) + (kyzFromfcNEQMPM - kyzFromfcNEQPMP)));

      a001 = c1o4 * (((vx1PPP - vx1MMM) + (vx1MMP - vx1PPM)) + ((vx1MPP - vx1PMM) + (vx1PMP - vx1MPM)));
      b001 = c1o4 * (((vx2PPP - vx2MMM) + (vx2MMP - vx2PPM)) + ((vx2MPP - vx2PMM) + (vx2PMP - vx2MPM)));
      c001 = c1o4 * (((vx3PPP - vx3MMM) + (vx3MMP - vx3PPM)) + ((vx3MPP - vx3PMM) + (vx3PMP - vx3MPM)));

      a002 = c1o8 * (
                       c2o1 * (((vx3PPM + vx3MMP) - (vx3PPP + vx3MMM)) + ((vx3MPP + vx3PMM) - (vx3PMP + vx3MPM))) +
                       ((kxzFromfcNEQPPP - kxzFromfcNEQMMM) + (kxzFromfcNEQMMP - kxzFromfcNEQPPM)) +
                       ((kxzFromfcNEQPMP - kxzFromfcNEQMPM) + (kxzFromfcNEQMPP - kxzFromfcNEQPMM)));
      b002 = c1o8 * (
                       c2o1 * (((vx3PPM + vx3MMP) - (vx3PPP + vx3MMM)) + ((vx3MPM + vx3PMP) - (vx3PMM + vx3MPP))) +
                       ((kyzFromfcNEQPPP - kyzFromfcNEQMMM) + (kyzFromfcNEQMMP - kyzFromfcNEQPPM)) +
                       ((kyzFromfcNEQPMP - kyzFromfcNEQMPM) + (kyzFromfcNEQMPP - kyzFromfcNEQPMM)));
      c002 = c1o16 * (
                        c2o1 * (
                                  ((kxxMzzFromfcNEQMMM - kxxMzzFromfcNEQPPP) + (kxxMzzFromfcNEQPPM - kxxMzzFromfcNEQMMP)) +
                                  ((kxxMzzFromfcNEQMPM - kxxMzzFromfcNEQPMP) + (kxxMzzFromfcNEQPMM - kxxMzzFromfcNEQMPP)) +
                                  ((vx1PPP + vx1MMM) - (vx1MMP + vx1PPM)) + ((vx1MPM + vx1PMP) - (vx1PMM + vx1MPP)) +
                                  ((vx2PPP + vx2MMM) - (vx2MMP + vx2PPM)) + ((vx2PMM + vx2MPP) - (vx2MPM + vx2PMP))) +
                        ((kxxMyyFromfcNEQPPP - kxxMyyFromfcNEQMMM) + (kxxMyyFromfcNEQMMP - kxxMyyFromfcNEQPPM)) +
                        ((kxxMyyFromfcNEQPMP - kxxMyyFromfcNEQMPM) + (kxxMyyFromfcNEQMPP - kxxMyyFromfcNEQPMM)));

      a110 = c1o2 * (((vx1PPP + vx1MMM) + (vx1MMP + vx1PPM)) - ((vx1MPM + vx1PMP) + (vx1PMM + vx1MPP)));
      b110 = c1o2 * (((vx2PPP + vx2MMM) + (vx2MMP + vx2PPM)) - ((vx2MPM + vx2PMP) + (vx2PMM + vx2MPP)));
      c110 = c1o2 * (((vx3PPP + vx3MMM) + (vx3MMP + vx3PPM)) - ((vx3MPM + vx3PMP) + (vx3PMM + vx3MPP)));

      a101 = c1o2 * (((vx1PPP + vx1MMM) - (vx1MMP + vx1PPM)) + ((vx1MPM + vx1PMP) - (vx1PMM + vx1MPP)));
      b101 = c1o2 * (((vx2PPP + vx2MMM) - (vx2MMP + vx2PPM)) + ((vx2MPM + vx2PMP) - (vx2PMM + vx2MPP)));
      c101 = c1o2 * (((vx3PPP + vx3MMM) - (vx3MMP + vx3PPM)) + ((vx3MPM + vx3PMP) - (vx3PMM + vx3MPP)));

      a011 = c1o2 * (((vx1PPP + vx1MMM) - (vx1MMP + vx1PPM)) + ((vx1PMM + vx1MPP) - (vx1MPM + vx1PMP)));
      b011 = c1o2 * (((vx2PPP + vx2MMM) - (vx2MMP + vx2PPM)) + ((vx2PMM + vx2MPP) - (vx2MPM + vx2PMP)));
      c011 = c1o2 * (((vx3PPP + vx3MMM) - (vx3MMP + vx3PPM)) + ((vx3PMM + vx3MPP) - (vx3MPM + vx3PMP)));

      a111 = ((vx1PPP - vx1MMM) + (vx1MMP - vx1PPM)) + ((vx1MPM - vx1PMP) + (vx1PMM - vx1MPP));
      b111 = ((vx2PPP - vx2MMM) + (vx2MMP - vx2PPM)) + ((vx2MPM - vx2PMP) + (vx2PMM - vx2MPP));
      c111 = ((vx3PPP - vx3MMM) + (vx3MMP - vx3PPM)) + ((vx3MPM - vx3PMP) + (vx3PMM - vx3MPP));

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //!- Calculate coefficients for the polynomial interpolation of the pressure
      //!
      const real_t xoff = 0.0;
      const real_t yoff = 0.0;
      const real_t zoff = 0.0;
      const real_t xoffsq = xoff * xoff;
      const real_t yoffsq = yoff * yoff;
      const real_t zoffsq = zoff * zoff;

      LaplaceRho =
         ((xoff != c0o1) || (yoff != c0o1) || (zoff != c0o1))
            ? c0o1 : -c3o1 * (a100 * a100 + b010 * b010 + c001 * c001) - c6o1 * (b100 * a010 + c100 * a001 + c010 * b001);
      d000 = c1o8 * (((drhoPPP + drhoMMM) + (drhoPPM + drhoMMP)) + ((drhoPMM + drhoMPP) + (drhoPMP + drhoMPM)));
      d100 = c1o4 * (((drhoPPP - drhoMMM) + (drhoPPM - drhoMMP)) + ((drhoPMM - drhoMPP) + (drhoPMP - drhoMPM)));
      d010 = c1o4 * (((drhoPPP - drhoMMM) + (drhoPPM - drhoMMP)) + ((drhoMPP - drhoPMM) + (drhoMPM - drhoPMP)));
      d001 = c1o4 * (((drhoPPP - drhoMMM) + (drhoMMP - drhoPPM)) + ((drhoMPP - drhoPMM) + (drhoPMP - drhoMPM)));
      d110 = c1o2 * (((drhoPPP + drhoMMM) + (drhoPPM + drhoMMP)) - ((drhoPMM + drhoMPP) + (drhoPMP + drhoMPM)));
      d101 = c1o2 * (((drhoPPP + drhoMMM) - (drhoPPM + drhoMMP)) + ((drhoPMP + drhoMPM) - (drhoPMM + drhoMPP)));
      d011 = c1o2 * (((drhoPPP + drhoMMM) - (drhoPPM + drhoMMP)) + ((drhoPMM + drhoMPP) - (drhoPMP + drhoMPM)));

      d111 = (((drhoPPP - drhoMMM) + (drhoMMP - drhoPPM)) + ((drhoPMM - drhoMPP) + (drhoMPM - drhoPMP)));

      //////////////////////////////////////////////////////////////////////////
      //! - Extrapolation for refinement in to the wall (polynomial coefficients)
      //!
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //
      // x------x
      // |      |
      // |   ---+--->X
      // |      |  |
      // x------x  |
      //          offset-vector
      //
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      a000 = a000 + xoff * a100 + yoff * a010 + zoff * a001 + xoffsq * a200 + yoffsq * a020 + zoffsq * a002 +
             xoff * yoff * a110 + xoff * zoff * a101 + yoff * zoff * a011;
      a100 = a100 + c2o1 * xoff * a200 + yoff * a110 + zoff * a101;
      a010 = a010 + c2o1 * yoff * a020 + xoff * a110 + zoff * a011;
      a001 = a001 + c2o1 * zoff * a002 + xoff * a101 + yoff * a011;
      b000 = b000 + xoff * b100 + yoff * b010 + zoff * b001 + xoffsq * b200 + yoffsq * b020 + zoffsq * b002 +
             xoff * yoff * b110 + xoff * zoff * b101 + yoff * zoff * b011;
      b100 = b100 + c2o1 * xoff * b200 + yoff * b110 + zoff * b101;
      b010 = b010 + c2o1 * yoff * b020 + xoff * b110 + zoff * b011;
      b001 = b001 + c2o1 * zoff * b002 + xoff * b101 + yoff * b011;
      c000 = c000 + xoff * c100 + yoff * c010 + zoff * c001 + xoffsq * c200 + yoffsq * c020 + zoffsq * c002 +
             xoff * yoff * c110 + xoff * zoff * c101 + yoff * zoff * c011;
      c100 = c100 + c2o1 * xoff * c200 + yoff * c110 + zoff * c101;
      c010 = c010 + c2o1 * yoff * c020 + xoff * c110 + zoff * c011;
      c001 = c001 + c2o1 * zoff * c002 + xoff * c101 + yoff * c011;
      d000 = d000 + xoff * d100 + yoff * d010 + zoff * d001 +
             xoff * yoff * d110 + xoff * zoff * d101 + yoff * zoff * d011;

      d100 = d100 + yoff * d110 + zoff * d101;
      d010 = d010 + xoff * d110 + zoff * d011;
      d001 = d001 + xoff * d101 + yoff * d011;
   }

};


class Interpolation
{
 public:
   Interpolation(const shared_ptr< StructuredBlockForest > & blocks, const IDs& ids, const real_t omega)
      :blocks_(blocks), ids_(ids)
   {
      for (uint_t level = 0; level < blocks->getNumberOfLevels(); level++)
      {
         const double level_scale_factor = double(uint_t(1) << level);
         const double one                = double(1.0);
         const double half               = double(0.5);

         omegaVector.push_back( double(omega / (level_scale_factor * (-omega * half + one) + omega * half)) );

         std::vector<Block *> tmp;
         blocks_->getBlocks(tmp, level);
         blocksPerLevel_.push_back(tmp);

         for(auto block: tmp)
         {
            fineCoefficients[block].resize(9);
         }
      }

      std::vector<AABB> coarseAABBs;
      coarseAABBs.resize(8);
      std::vector<AABB> fineAABBs;
      for(auto block: blocksPerLevel_[0])
      {
         const AABB coarseAABB = block->getAABB();
         const real_t xMid     = (coarseAABB.xMin() + coarseAABB.xMax()) / real_c(2.0);
         const real_t yMid     = (coarseAABB.yMin() + coarseAABB.yMax()) / real_c(2.0);
         const real_t zMid     = (coarseAABB.zMin() + coarseAABB.zMax()) / real_c(2.0);

         for (uint_t c = 0; c != 8; ++c)
         {
            coarseAABBs[c] = AABB(((c & 1) ? xMid : coarseAABB.xMin()),  // xmin (incl.)
                                  ((c & 2) ? yMid : coarseAABB.yMin()),  // ymin (incl.)
                                  ((c & 4) ? zMid : coarseAABB.zMin()),  // zmin (incl.)
                                  ((c & 1) ? coarseAABB.xMax() : xMid),  // xmax (excl.)
                                  ((c & 2) ? coarseAABB.yMax() : yMid),  // ymax (excl.)
                                  ((c & 4) ? coarseAABB.zMax() : zMid)); // zmax (excl.))
         }

         fineAABBs.clear();
         for (auto dir = Stencil_T::beginNoCenter(); dir != Stencil_T::end(); ++dir)
         {
            const auto neighborIdx = blockforest::getBlockNeighborhoodSectionIndex(*dir);

            if (block->getNeighborhoodSectionSize(neighborIdx) == uint_t(0)) continue;
            if (!(block->neighborhoodSectionHasSmallerBlocks(neighborIdx))) continue;
            for (uint_t n = 0; n != block->getNeighborhoodSectionSize(neighborIdx); ++n)
            {
               auto fineAABB = block->getNeighborAABB(neighborIdx, n);
               if (std::find(fineAABBs.begin(), fineAABBs.end(), fineAABB) != fineAABBs.end()) continue;
               fineAABBs.emplace_back(fineAABB);
            }
         }

         for (auto dir = Stencil_T::beginNoCenter(); dir != Stencil_T::end(); ++dir)
         {
            for (uint_t c = 0; c != 8; ++c)
            {
               CellInterval ci(0, 0, 0, 3, 3, 3);
               Vector3< real_t > p(((coarseAABBs[c].xMin() + coarseAABBs[c].xMax()) / real_c(2.0)) + real_c(dir.cx() * coarseAABBs[c].xSize()),
                                   ((coarseAABBs[c].yMin() + coarseAABBs[c].yMax()) / real_c(2.0)) + real_c(dir.cy() * coarseAABBs[c].ySize()),
                                   ((coarseAABBs[c].zMin() + coarseAABBs[c].zMax()) / real_c(2.0)) + real_c(dir.cz() * coarseAABBs[c].zSize()));
               blocks_->mapToPeriodicDomain(p);

               ci.min()[0] = (c & 1) ? cell_idx_c(3) : cell_idx_c(1);
               ci.min()[1] = (c & 2) ? cell_idx_c(3) : cell_idx_c(1);
               ci.min()[2] = (c & 4) ? cell_idx_c(3) : cell_idx_c(1);
               ci.max()[0] = (c & 1) ? cell_idx_c(3) : cell_idx_c(1);
               ci.max()[1] = (c & 2) ? cell_idx_c(3) : cell_idx_c(1);
               ci.max()[2] = (c & 4) ? cell_idx_c(3) : cell_idx_c(1);

               for (auto fineAABB : fineAABBs)
               {
                  if (fineAABB.contains(p))
                  {
                     auto fineBlock = dynamic_cast< Block* >(blocks->getBlock(p));
                     auto t = std::make_tuple(block, fineBlock, *dir, ci);
                     mapping_.emplace_back(t);
                  }
               }
            }
         }
      }
   }

//   void getCoarsePDFs(Vector3<real_t>& globalPoint, real_t* local)
//   {
//      blocks_->mapToPeriodicDomain(globalPoint);
//      auto b = blocks_->getBlock(globalPoint);
//      Cell localCell = blocks_->getBlockLocalCell(*b, globalPoint);
//      auto f = b->getData<PdfField_T>(ids_.pdfField);
//
//      for(cell_idx_t i = 0; i < Stencil_T::Q; ++i)
//      {
//         local[i] = f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[i],
//                           localCell.y() + StorageSpecification_T::AccessorEVEN::readY[i],
//                           localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[i],
//                           StorageSpecification_T::AccessorEVEN::readD[i]);
//      }
//   }

   void getCoarsePDFs(Block* block, Cell& cell, real_t* local)
   {
      auto f = block->getData<PdfField_T>(ids_.pdfField);
      for(cell_idx_t i = 0; i < Stencil_T::Q; ++i)
      {
         local[i] = f->get(cell.x() + StorageSpecification_T::AccessorEVEN::writeX[i],
                           cell.y() + StorageSpecification_T::AccessorEVEN::writeY[i],
                           cell.z() + StorageSpecification_T::AccessorEVEN::writeZ[i],
                           StorageSpecification_T::AccessorEVEN::writeD[i]);
      }
   }

   void getFinePDFs(Block* block, Cell& cell, real_t* local)
   {
      auto f = block->getData<PdfField_T>(ids_.pdfField);
      for(cell_idx_t i = 0; i < Stencil_T::Q; ++i)
      {
         local[i] = f->get(cell.x() + StorageSpecification_T::AccessorEVEN::writeX[i],
                           cell.y() + StorageSpecification_T::AccessorEVEN::writeY[i],
                           cell.z() + StorageSpecification_T::AccessorEVEN::writeZ[i],
                           StorageSpecification_T::AccessorEVEN::writeD[i]);
      }
   }

   inline void backwardInverseChimeraWithK(real_t &mfa, real_t &mfb, real_t &mfc, real_t vv,
                                           real_t v2, real_t Kinverse, real_t K)
   {
      const real_t m0 = (((mfc - mfb) * c1o2 + mfb * vv) * Kinverse + (mfa * Kinverse + c1o1) * (v2 - vv) * c1o2) * K;
      const real_t m1 = (((mfa - mfc) - c2o1 * mfb * vv) * Kinverse + (mfa * Kinverse + c1o1) * (-v2)) * K;

      mfc = (((mfc + mfb) * c1o2 + mfb * vv) * Kinverse + (mfa * Kinverse + c1o1) * (v2 + vv) * c1o2) * K;
      mfa = m0;
      mfb = m1;
   }

   inline void backwardChimera(real_t &mfa, real_t &mfb, real_t &mfc, real_t vv, real_t v2)
   {
      const real_t ma = (mfc + mfa * (v2 - vv)) * c1o2 + mfb * (vv - c1o2);
      const real_t mb = ((mfa - mfc) - mfa * v2) - c2o1 * mfb * vv;

      mfc = (mfc + mfa * (v2 + vv)) * c1o2 + mfb * (vv + c1o2);
      mfb = mb;
      mfa = ma;
   }


   void interpolateCF(Block* block, Cell& localCell, Vector3<real_t>& offset, const real_t omegaF, const InterpolationCoefficients &coefficients)
   {
      const real_t epsnew = c1o2;
      const real_t x = offset[0];
      const real_t y = offset[1];
      const real_t z = offset[2];

      const real_t useNEQ = c1o1;

      const real_t kxyAverage    = c0o1;
      const real_t kyzAverage    = c0o1;
      const real_t kxzAverage    = c0o1;
      const real_t kxxMyyAverage = c0o1;
      const real_t kxxMzzAverage = c0o1;

      const real_t& a000 = coefficients.a000;
      const real_t& b000 = coefficients.b000;
      const real_t& c000 = coefficients.c000;
      const real_t& d000 = coefficients.d000;

      const real_t& a100 = coefficients.a100;
      const real_t& b100 = coefficients.b100;
      const real_t& c100 = coefficients.c100;
      const real_t& d100 = coefficients.d100;

      const real_t& a010 = coefficients.a010;
      const real_t& b010 = coefficients.b010;
      const real_t& c010 = coefficients.c010;
      const real_t& d010 = coefficients.d010;

      const real_t& a001 = coefficients.a001;
      const real_t& b001 = coefficients.b001;
      const real_t& c001 = coefficients.c001;
      const real_t& d001 = coefficients.d001;

      const real_t& d110 = coefficients.d110, &d101 = coefficients.d101, &d011 = coefficients.d011;

      const real_t& a200 = coefficients.a200, &a020 = coefficients.a020, &a002 = coefficients.a002;
      const real_t& b200 = coefficients.b200, &b020 = coefficients.b020, &b002 = coefficients.b002;
      const real_t& c200 = coefficients.c200, &c020 = coefficients.c020, &c002 = coefficients.c002;

      const real_t& a110 = coefficients.a110, &a101 = coefficients.a101, &a011 = coefficients.a011;
      const real_t& b110 = coefficients.b110, &b101 = coefficients.b101, &b011 = coefficients.b011;
      const real_t& c110 = coefficients.c110, &c101 = coefficients.c101, &c011 = coefficients.c011;

      const real_t &a111 = coefficients.a111, &b111 = coefficients.b111, &c111 = coefficients.c111, &d111 = coefficients.d111;

      const real_t &LaplaceRho = coefficients.LaplaceRho;


      ////////////////////////////////////////////////////////////////////////////////////
      //! - Set all moments to zero
      //!
      real_t m111 = real_c(0.0);
      real_t m211 = real_c(0.0);
      real_t m011 = real_c(0.0);
      real_t m121 = real_c(0.0);
      real_t m101 = real_c(0.0);
      real_t m112 = real_c(0.0);
      real_t m110 = real_c(0.0);
      real_t m221 = real_c(0.0);
      real_t m001 = real_c(0.0);
      real_t m201 = real_c(0.0);
      real_t m021 = real_c(0.0);
      real_t m212 = real_c(0.0);
      real_t m010 = real_c(0.0);
      real_t m210 = real_c(0.0);
      real_t m012 = real_c(0.0);
      real_t m122 = real_c(0.0);
      real_t m100 = real_c(0.0);
      real_t m120 = real_c(0.0);
      real_t m102 = real_c(0.0);
      real_t m222 = real_c(0.0);
      real_t m022 = real_c(0.0);
      real_t m202 = real_c(0.0);
      real_t m002 = real_c(0.0);
      real_t m220 = real_c(0.0);
      real_t m020 = real_c(0.0);
      real_t m200 = real_c(0.0);
      real_t m000 = real_c(0.0);

      real_t& f000 = m111;
      real_t& fP00 = m211;
      real_t& fM00 = m011;
      real_t& f0P0 = m121;
      real_t& f0M0 = m101;
      real_t& f00P = m112;
      real_t& f00M = m110;
      real_t& fPP0 = m221;
      real_t& fMM0 = m001;
      real_t& fPM0 = m201;
      real_t& fMP0 = m021;
      real_t& fP0P = m212;
      real_t& fM0M = m010;
      real_t& fP0M = m210;
      real_t& fM0P = m012;
      real_t& f0PP = m122;
      real_t& f0MM = m100;
      real_t& f0PM = m120;
      real_t& f0MP = m102;
      real_t& fPPP = m222;
      real_t& fMPP = m022;
      real_t& fPMP = m202;
      real_t& fMMP = m002;
      real_t& fPPM = m220;
      real_t& fMPM = m020;
      real_t& fPMM = m200;
      real_t& fMMM = m000;



      ////////////////////////////////////////////////////////////////////////////////
      //! - Set macroscopic values on destination node (zeroth and first order moments)
      //!
      real_t press = d000 + x * d100 + y * d010 + z * d001 +
                   x * y * d110 + x * z * d101 + y * z * d011 + x * y * z * d111 + c3o1 * x * x * LaplaceRho;
      real_t vvx   = a000 + x * a100 + y * a010 + z * a001 +
                 x * x * a200 + y * y * a020 + z * z * a002 +
                 x * y * a110 + x * z * a101 + y * z * a011 + x * y * z * a111;
      real_t vvy   = b000 + x * b100 + y * b010 + z * b001 +
                 x * x * b200 + y * y * b020 + z * z * b002 +
                 x * y * b110 + x * z * b101 + y * z * b011 + x * y * z * b111;
      real_t vvz   = c000 + x * c100 + y * c010 + z * c001 +
                 x * x * c200 + y * y * c020 + z * z * c002 +
                 x * y * c110 + x * z * c101 + y * z * c011 + x * y * z * c111;

      m000 = press; // m000 is press, if drho is interpolated directly

      ////////////////////////////////////////////////////////////////////////////////
      //! - Set moments (second to sixth order) on destination node
      //!
      // linear combinations for second order moments
      real_t mxxPyyPzz = m000;

      real_t mxxMyy = -c2o3 * (a100 - b010 + kxxMyyAverage + c2o1 * a200 * x - b110 * x + a110 * y
                             -c2o1 * b020 * y + a101 * z - b011 * z - b111 * x * z + a111 * y * z) * epsnew/ omegaF * (c1o1 + press);
      real_t mxxMzz = -c2o3 * (a100 - c001 + kxxMzzAverage + c2o1 * a200 * x - c101 * x + a110 * y
                             -c011 * y - c111 * x * y + a101 * z - c2o1 * c002 * z + a111 * y * z) * epsnew/ omegaF * (c1o1 + press);

      m011 = -c1o3 * (b001 + c010 + kyzAverage + b101 * x + c110 * x + b011 * y + c2o1 * c020 * y
                      + b111 * x * y + c2o1 * b002 * z + c011 * z + c111 * x * z) * epsnew / omegaF * (c1o1 + press);
      m101 = -c1o3 * (a001 + c100 + kxzAverage + a101 * x + c2o1 * c200 * x + a011 * y + c110 * y
                      + a111 * x * y + c2o1 * a002 * z + c101 * z + c111 * y * z) * epsnew / omegaF * (c1o1 + press);
      m110 = -c1o3 * (a010 + b100 + kxyAverage + a110 * x + c2o1 * b200 * x + c2o1 * a020 * y
                      + b110 * y + a011 * z + b101 * z + a111 * x * z + b111 * y * z) * epsnew / omegaF * (c1o1 + press);

      m200 = c1o3 * (        mxxMyy +        mxxMzz + mxxPyyPzz) * useNEQ;
      m020 = c1o3 * (-c2o1 * mxxMyy +        mxxMzz + mxxPyyPzz) * useNEQ;
      m002 = c1o3 * (        mxxMyy - c2o1 * mxxMzz + mxxPyyPzz) * useNEQ;

      // linear combinations for third order moments
      m111 = c0o1;

      real_t mxxyPyzz = c0o1;
      real_t mxxyMyzz = c0o1;
      real_t mxxzPyyz = c0o1;
      real_t mxxzMyyz = c0o1;
      real_t mxyyPxzz = c0o1;
      real_t mxyyMxzz = c0o1;

      m210 = ( mxxyMyzz + mxxyPyzz) * c1o2;
      m012 = (-mxxyMyzz + mxxyPyzz) * c1o2;
      m201 = ( mxxzMyyz + mxxzPyyz) * c1o2;
      m021 = (-mxxzMyyz + mxxzPyyz) * c1o2;
      m120 = ( mxyyMxzz + mxyyPxzz) * c1o2;
      m102 = (-mxyyMxzz + mxyyPxzz) * c1o2;

      // fourth order moments
      m022 = m000 * c1o9;
      m202 = m022;
      m220 = m022;

      // fifth order moments

      // sixth order moment
      m222 = m000 * c1o27;

      real_t vxsq = vvx * vvx;
      real_t vysq = vvy * vvy;
      real_t vzsq = vvz * vvz;


      ////////////////////////////////////////////////////////////////////////////////////
      // X - Dir
      backwardInverseChimeraWithK(m000, m100, m200, vvx, vxsq, c1o1, c1o1);
      backwardChimera(            m010, m110, m210, vvx, vxsq);
      backwardInverseChimeraWithK(m020, m120, m220, vvx, vxsq, c3o1, c1o3);
      backwardChimera(            m001, m101, m201, vvx, vxsq);
      backwardChimera(            m011, m111, m211, vvx, vxsq);
      backwardChimera(            m021, m121, m221, vvx, vxsq);
      backwardInverseChimeraWithK(m002, m102, m202, vvx, vxsq, c3o1, c1o3);
      backwardChimera(            m012, m112, m212, vvx, vxsq);
      backwardInverseChimeraWithK(m022, m122, m222, vvx, vxsq, c9o1, c1o9);

      ////////////////////////////////////////////////////////////////////////////////////
      // Y - Dir
      backwardInverseChimeraWithK(m000, m010, m020, vvy, vysq, c6o1, c1o6);
      backwardChimera(            m001, m011, m021, vvy, vysq);
      backwardInverseChimeraWithK(m002, m012, m022, vvy, vysq, c18o1, c1o18);
      backwardInverseChimeraWithK(m100, m110, m120, vvy, vysq, c3o2, c2o3);
      backwardChimera(            m101, m111, m121, vvy, vysq);
      backwardInverseChimeraWithK(m102, m112, m122, vvy, vysq, c9o2, c2o9);
      backwardInverseChimeraWithK(m200, m210, m220, vvy, vysq, c6o1, c1o6);
      backwardChimera(            m201, m211, m221, vvy, vysq);
      backwardInverseChimeraWithK(m202, m212, m222, vvy, vysq, c18o1, c1o18);

      ////////////////////////////////////////////////////////////////////////////////////
      // Z - Dir
      backwardInverseChimeraWithK(m000, m001, m002, vvz, vzsq, c36o1, c1o36);
      backwardInverseChimeraWithK(m010, m011, m012, vvz, vzsq, c9o1,  c1o9);
      backwardInverseChimeraWithK(m020, m021, m022, vvz, vzsq, c36o1, c1o36);
      backwardInverseChimeraWithK(m100, m101, m102, vvz, vzsq, c9o1,  c1o9);
      backwardInverseChimeraWithK(m110, m111, m112, vvz, vzsq, c9o4,  c4o9);
      backwardInverseChimeraWithK(m120, m121, m122, vvz, vzsq, c9o1,  c1o9);
      backwardInverseChimeraWithK(m200, m201, m202, vvz, vzsq, c36o1, c1o36);
      backwardInverseChimeraWithK(m210, m211, m212, vvz, vzsq, c9o1,  c1o9);
      backwardInverseChimeraWithK(m220, m221, m222, vvz, vzsq, c36o1, c1o36);

      auto f = block->getData<PdfField_T>(ids_.pdfField);
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[0],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[0],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[0], 0) = f000;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[4],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[4],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[4], 4)= fP00;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[3],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[3],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[3], 3)= fM00;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[1],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[1],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[1], 1)= f0P0;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[2],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[2],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[2], 2)= f0M0;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[5],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[5],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[5], 5)= f00P;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[6],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[6],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[6], 6)= f00M;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[8],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[8],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[8], 8)= fPP0;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[9],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[9],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[9], 9)= fMM0;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[10],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[10],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[10], 10) = fPM0;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[7],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[7],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[7], 7)= fMP0;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[14],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[14],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[14], 14) = fP0P;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[17],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[17],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[17], 17) = fM0M;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[18],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[18],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[18], 18) = fP0M;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[13],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[13],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[13], 13) = fM0P;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[11],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[11],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[11], 11) = f0PP;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[16],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[16],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[16], 16) = f0MM;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[15],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[15],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[15], 15) = f0PM;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[12],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[12],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[12], 12) = f0MP;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[19],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[19],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[19], 19) = fPPP;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[20],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[20],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[20], 20) = fMPP;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[21],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[21],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[21], 21) = fPMP;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[22],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[22],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[22], 22) = fMMP;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[23],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[23],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[23], 23) = fPPM;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[24],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[24],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[24], 24) = fMPM;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[25],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[25],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[25], 25) = fPMM;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[26],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[26],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[26], 26) = fMMM;
   }

   void interpolateFC(Block* block, Cell& localCell, const real_t omegaC, const InterpolationCoefficients &coefficients)
   {
      const real_t epsnew = c2o1;
      const real_t kxyAverage    = c0o1;
      const real_t kyzAverage    = c0o1;
      const real_t kxzAverage    = c0o1;
      const real_t kxxMyyAverage = c0o1;
      const real_t kxxMzzAverage = c0o1;

      ////////////////////////////////////////////////////////////////////////////////////
      //! - Set all moments to zero
      //!
      real_t m111 = real_c(0.0);
      real_t m211 = real_c(0.0);
      real_t m011 = real_c(0.0);
      real_t m121 = real_c(0.0);
      real_t m101 = real_c(0.0);
      real_t m112 = real_c(0.0);
      real_t m110 = real_c(0.0);
      real_t m221 = real_c(0.0);
      real_t m001 = real_c(0.0);
      real_t m201 = real_c(0.0);
      real_t m021 = real_c(0.0);
      real_t m212 = real_c(0.0);
      real_t m010 = real_c(0.0);
      real_t m210 = real_c(0.0);
      real_t m012 = real_c(0.0);
      real_t m122 = real_c(0.0);
      real_t m100 = real_c(0.0);
      real_t m120 = real_c(0.0);
      real_t m102 = real_c(0.0);
      real_t m222 = real_c(0.0);
      real_t m022 = real_c(0.0);
      real_t m202 = real_c(0.0);
      real_t m002 = real_c(0.0);
      real_t m220 = real_c(0.0);
      real_t m020 = real_c(0.0);
      real_t m200 = real_c(0.0);
      real_t m000 = real_c(0.0);

      real_t& f000 = m111;
      real_t& fP00 = m211;
      real_t& fM00 = m011;
      real_t& f0P0 = m121;
      real_t& f0M0 = m101;
      real_t& f00P = m112;
      real_t& f00M = m110;
      real_t& fPP0 = m221;
      real_t& fMM0 = m001;
      real_t& fPM0 = m201;
      real_t& fMP0 = m021;
      real_t& fP0P = m212;
      real_t& fM0M = m010;
      real_t& fP0M = m210;
      real_t& fM0P = m012;
      real_t& f0PP = m122;
      real_t& f0MM = m100;
      real_t& f0PM = m120;
      real_t& f0MP = m102;
      real_t& fPPP = m222;
      real_t& fMPP = m022;
      real_t& fPMP = m202;
      real_t& fMMP = m002;
      real_t& fPPM = m220;
      real_t& fMPM = m020;
      real_t& fPMM = m200;
      real_t& fMMM = m000;

      ////////////////////////////////////////////////////////////////////////////////
      //! - Declare local variables for destination nodes
      //!
      real_t vvx, vvy, vvz, vxsq, vysq, vzsq;
      real_t mxxPyyPzz, mxxMyy, mxxMzz, mxxyPyzz, mxxyMyzz, mxxzPyyz, mxxzMyyz, mxyyPxzz, mxyyMxzz;
      real_t useNEQ = c1o1; // zero; //one;   //.... one = on ..... zero = off
      real_t press;

      ////////////////////////////////////////////////////////////////////////////////
      //! - Set macroscopic values on destination node (zeroth and first order moments)
      //!
      press = coefficients.d000 - c2o1 * coefficients.LaplaceRho * c1o8;
      vvx   = coefficients.a000;
      vvy   = coefficients.b000;
      vvz   = coefficients.c000;

      m000 = press; // m000 is press, if drho is interpolated directly

      vxsq = vvx * vvx;
      vysq = vvy * vvy;
      vzsq = vvz * vvz;

      ////////////////////////////////////////////////////////////////////////////////
      //! - Set moments (second to sixth order) on destination node
      //!
      // linear combinations for second order moments
      mxxPyyPzz = m000;

      mxxMyy = -c2o3 * ((coefficients.a100 - coefficients.b010) + kxxMyyAverage) * epsnew / omegaC * (c1o1 + press);
      mxxMzz = -c2o3 * ((coefficients.a100 - coefficients.c001) + kxxMzzAverage) * epsnew / omegaC * (c1o1 + press);

      m011 = -c1o3 * ((coefficients.b001 + coefficients.c010) + kyzAverage) * epsnew / omegaC * (c1o1 + press);
      m101 = -c1o3 * ((coefficients.a001 + coefficients.c100) + kxzAverage) * epsnew / omegaC * (c1o1 + press);
      m110 = -c1o3 * ((coefficients.a010 + coefficients.b100) + kxyAverage) * epsnew / omegaC * (c1o1 + press);

      m200 = c1o3 * (        mxxMyy +        mxxMzz + mxxPyyPzz) * useNEQ;
      m020 = c1o3 * (-c2o1 * mxxMyy +        mxxMzz + mxxPyyPzz) * useNEQ;
      m002 = c1o3 * (        mxxMyy - c2o1 * mxxMzz + mxxPyyPzz) * useNEQ;

      // linear combinations for third order moments
      m111 = c0o1;

      mxxyPyzz = c0o1;
      mxxyMyzz = c0o1;
      mxxzPyyz = c0o1;
      mxxzMyyz = c0o1;
      mxyyPxzz = c0o1;
      mxyyMxzz = c0o1;

      m210 = ( mxxyMyzz + mxxyPyzz) * c1o2;
      m012 = (-mxxyMyzz + mxxyPyzz) * c1o2;
      m201 = ( mxxzMyyz + mxxzPyyz) * c1o2;
      m021 = (-mxxzMyyz + mxxzPyyz) * c1o2;
      m120 = ( mxyyMxzz + mxyyPxzz) * c1o2;
      m102 = (-mxyyMxzz + mxyyPxzz) * c1o2;

      // fourth order moments
      m022 = m000 * c1o9;
      m202 = m022;
      m220 = m022;

      // fifth order moments

      // sixth order moments
      m222 = m000 * c1o27;

      ////////////////////////////////////////////////////////////////////////////////////
      //! - Chimera transform from central moments to well conditioned distributions as defined in Appendix J in
      //! <a href="https://doi.org/10.1016/j.camwa.2015.05.001"><b>[ M. Geier et al. (2015),
      //! DOI:10.1016/j.camwa.2015.05.001 ]</b></a> see also Eq. (88)-(96) in <a
      //! href="https://doi.org/10.1016/j.jcp.2017.05.040"><b>[ M. Geier et al. (2017), DOI:10.1016/j.jcp.2017.05.040
      //! ]</b></a>
      //!
      ////////////////////////////////////////////////////////////////////////////////////
      // X - Dir
      backwardInverseChimeraWithK(m000, m100, m200, vvx, vxsq, c1o1, c1o1);
      backwardChimera(            m010, m110, m210, vvx, vxsq);
      backwardInverseChimeraWithK(m020, m120, m220, vvx, vxsq, c3o1, c1o3);
      backwardChimera(            m001, m101, m201, vvx, vxsq);
      backwardChimera(            m011, m111, m211, vvx, vxsq);
      backwardChimera(            m021, m121, m221, vvx, vxsq);
      backwardInverseChimeraWithK(m002, m102, m202, vvx, vxsq, c3o1, c1o3);
      backwardChimera(            m012, m112, m212, vvx, vxsq);
      backwardInverseChimeraWithK(m022, m122, m222, vvx, vxsq, c9o1, c1o9);

      ////////////////////////////////////////////////////////////////////////////////////
      // Y - Dir
      backwardInverseChimeraWithK(m000, m010, m020, vvy, vysq, c6o1, c1o6);
      backwardChimera(            m001, m011, m021, vvy, vysq);
      backwardInverseChimeraWithK(m002, m012, m022, vvy, vysq, c18o1, c1o18);
      backwardInverseChimeraWithK(m100, m110, m120, vvy, vysq, c3o2, c2o3);
      backwardChimera(            m101, m111, m121, vvy, vysq);
      backwardInverseChimeraWithK(m102, m112, m122, vvy, vysq, c9o2, c2o9);
      backwardInverseChimeraWithK(m200, m210, m220, vvy, vysq, c6o1, c1o6);
      backwardChimera(            m201, m211, m221, vvy, vysq);
      backwardInverseChimeraWithK(m202, m212, m222, vvy, vysq, c18o1, c1o18);

      ////////////////////////////////////////////////////////////////////////////////////
      // Z - Dir
      backwardInverseChimeraWithK(m000, m001, m002, vvz, vzsq, c36o1, c1o36);
      backwardInverseChimeraWithK(m010, m011, m012, vvz, vzsq, c9o1,  c1o9);
      backwardInverseChimeraWithK(m020, m021, m022, vvz, vzsq, c36o1, c1o36);
      backwardInverseChimeraWithK(m100, m101, m102, vvz, vzsq, c9o1,  c1o9);
      backwardInverseChimeraWithK(m110, m111, m112, vvz, vzsq, c9o4,  c4o9);
      backwardInverseChimeraWithK(m120, m121, m122, vvz, vzsq, c9o1,  c1o9);
      backwardInverseChimeraWithK(m200, m201, m202, vvz, vzsq, c36o1, c1o36);
      backwardInverseChimeraWithK(m210, m211, m212, vvz, vzsq, c9o1,  c1o9);
      backwardInverseChimeraWithK(m220, m221, m222, vvz, vzsq, c36o1, c1o36);

      auto f = block->getData<PdfField_T>(ids_.pdfField);
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[0],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[0],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[0], 0) = f000;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[4],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[4],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[4], 4)= fP00;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[3],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[3],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[3], 3)= fM00;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[1],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[1],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[1], 1)= f0P0;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[2],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[2],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[2], 2)= f0M0;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[5],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[5],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[5], 5)= f00P;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[6],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[6],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[6], 6)= f00M;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[8],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[8],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[8], 8)= fPP0;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[9],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[9],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[9], 9)= fMM0;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[10],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[10],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[10], 10) = fPM0;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[7],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[7],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[7], 7)= fMP0;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[14],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[14],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[14], 14) = fP0P;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[17],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[17],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[17], 17) = fM0M;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[18],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[18],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[18], 18) = fP0M;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[13],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[13],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[13], 13) = fM0P;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[11],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[11],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[11], 11) = f0PP;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[16],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[16],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[16], 16) = f0MM;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[15],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[15],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[15], 15) = f0PM;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[12],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[12],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[12], 12) = f0MP;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[19],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[19],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[19], 19) = fPPP;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[20],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[20],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[20], 20) = fMPP;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[21],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[21],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[21], 21) = fPMP;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[22],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[22],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[22], 22) = fMMP;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[23],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[23],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[23], 23) = fPPM;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[24],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[24],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[24], 24) = fMPM;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[25],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[25],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[25], 25) = fPMM;
      f->get(localCell.x() + StorageSpecification_T::AccessorEVEN::readX[26],
             localCell.y() + StorageSpecification_T::AccessorEVEN::readY[26],
             localCell.z() + StorageSpecification_T::AccessorEVEN::readZ[26], 26) = fMMM;
   }


   void setCoefficients(Block * sender, Block * receiver, CellInterval& ci, real_t omega, stencil::Direction dir)
   {
      real_t f_coarse[27];
      switch (dir) {
      case stencil::N : {
         for (auto it: ci)
         {
            Cell c0 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c0, f_coarse);
            fineCoefficients[receiver][0].calculatePMP(f_coarse, omega);
            fineCoefficients[receiver][1].calculateMMP(f_coarse, omega);
            fineCoefficients[receiver][3].calculatePMM(f_coarse, omega);
            fineCoefficients[receiver][4].calculateMMM(f_coarse, omega);


            Cell c1 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c1, f_coarse);
            fineCoefficients[receiver][3].calculatePMP(f_coarse, omega);
            fineCoefficients[receiver][4].calculateMMP(f_coarse, omega);
            fineCoefficients[receiver][6].calculatePMM(f_coarse, omega);
            fineCoefficients[receiver][7].calculateMMM(f_coarse, omega);


            Cell c2 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c2, f_coarse);
            fineCoefficients[receiver][4].calculatePMP(f_coarse, omega);
            fineCoefficients[receiver][5].calculateMMP(f_coarse, omega);
            fineCoefficients[receiver][7].calculatePMM(f_coarse, omega);
            fineCoefficients[receiver][8].calculateMMM(f_coarse, omega);


            Cell c3 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c3, f_coarse);
            fineCoefficients[receiver][1].calculatePMP(f_coarse, omega);
            fineCoefficients[receiver][2].calculateMMP(f_coarse, omega);
            fineCoefficients[receiver][4].calculatePMM(f_coarse, omega);
            fineCoefficients[receiver][5].calculateMMM(f_coarse, omega);


            Cell c4 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(0), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c4, f_coarse);
            fineCoefficients[receiver][0].calculatePPP(f_coarse, omega);
            fineCoefficients[receiver][1].calculateMPP(f_coarse, omega);
            fineCoefficients[receiver][3].calculatePPM(f_coarse, omega);
            fineCoefficients[receiver][4].calculateMPM(f_coarse, omega);


            Cell c5 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(0), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c5, f_coarse);
            fineCoefficients[receiver][3].calculatePPP(f_coarse, omega);
            fineCoefficients[receiver][4].calculateMPP(f_coarse, omega);
            fineCoefficients[receiver][6].calculatePPM(f_coarse, omega);
            fineCoefficients[receiver][7].calculateMPM(f_coarse, omega);


            Cell c6 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(0), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c6, f_coarse);
            fineCoefficients[receiver][4].calculatePPP(f_coarse, omega);
            fineCoefficients[receiver][5].calculateMPP(f_coarse, omega);
            fineCoefficients[receiver][7].calculatePPM(f_coarse, omega);
            fineCoefficients[receiver][8].calculateMPM(f_coarse, omega);


            Cell c7 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(0), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c7, f_coarse);
            fineCoefficients[receiver][1].calculatePPP(f_coarse, omega);
            fineCoefficients[receiver][2].calculateMPP(f_coarse, omega);
            fineCoefficients[receiver][4].calculatePPM(f_coarse, omega);
            fineCoefficients[receiver][5].calculateMPM(f_coarse, omega);
         }

         break;
      }
      case stencil::NW : {
         for (auto it: ci)
         {
            Cell c0 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c0, f_coarse);
            fineCoefficients[receiver][2].calculatePMP(f_coarse, omega);
            fineCoefficients[receiver][5].calculatePMM(f_coarse, omega);


            Cell c1 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c1, f_coarse);
            fineCoefficients[receiver][5].calculatePMP(f_coarse, omega);
            fineCoefficients[receiver][8].calculatePMM(f_coarse, omega);


            Cell c4 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(0), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c4, f_coarse);
            fineCoefficients[receiver][2].calculatePPP(f_coarse, omega);
            fineCoefficients[receiver][5].calculatePPM(f_coarse, omega);


            Cell c5 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(0), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c5, f_coarse);
            fineCoefficients[receiver][5].calculatePPP(f_coarse, omega);
            fineCoefficients[receiver][8].calculatePPM(f_coarse, omega);

         }
         break;
      }
      case stencil::NE : {
         for (auto it: ci)
         {
            Cell c2 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c2, f_coarse);
            fineCoefficients[receiver][3].calculateMMP(f_coarse, omega);
            fineCoefficients[receiver][6].calculateMMM(f_coarse, omega);


            Cell c3 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c3, f_coarse);
            fineCoefficients[receiver][0].calculateMMP(f_coarse, omega);
            fineCoefficients[receiver][3].calculateMMM(f_coarse, omega);


            Cell c6 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(0), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c6, f_coarse);
            fineCoefficients[receiver][3].calculateMPP(f_coarse, omega);
            fineCoefficients[receiver][6].calculateMPM(f_coarse, omega);


            Cell c7 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(0), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c7, f_coarse);
            fineCoefficients[receiver][0].calculateMPP(f_coarse, omega);
            fineCoefficients[receiver][3].calculateMPM(f_coarse, omega);
         }
         break;
      }
      case stencil::TN : {
         for (auto it: ci)
         {
            Cell c1 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c1, f_coarse);
            fineCoefficients[receiver][0].calculatePMM(f_coarse, omega);
            fineCoefficients[receiver][1].calculateMMM(f_coarse, omega);


            Cell c2 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c2, f_coarse);
            fineCoefficients[receiver][1].calculatePMM(f_coarse, omega);
            fineCoefficients[receiver][2].calculateMMM(f_coarse, omega);


            Cell c5 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(0), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c5, f_coarse);
            fineCoefficients[receiver][0].calculatePPM(f_coarse, omega);
            fineCoefficients[receiver][1].calculateMPM(f_coarse, omega);


            Cell c6 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(0), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c6, f_coarse);
            fineCoefficients[receiver][1].calculatePPM(f_coarse, omega);
            fineCoefficients[receiver][2].calculateMPM(f_coarse, omega);

         }
         break;
      }
      case stencil::BN : {
         for (auto it: ci)
         {
            Cell c0 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c0, f_coarse);
            fineCoefficients[receiver][6].calculatePMP(f_coarse, omega);
            fineCoefficients[receiver][7].calculateMMP(f_coarse, omega);


            Cell c3 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c3, f_coarse);
            fineCoefficients[receiver][7].calculatePMP(f_coarse, omega);
            fineCoefficients[receiver][8].calculateMMP(f_coarse, omega);


            Cell c4 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(0), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c4, f_coarse);
            fineCoefficients[receiver][6].calculatePPP(f_coarse, omega);
            fineCoefficients[receiver][7].calculateMPP(f_coarse, omega);


            Cell c7 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(0), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c7, f_coarse);
            fineCoefficients[receiver][7].calculatePPP(f_coarse, omega);
            fineCoefficients[receiver][8].calculateMPP(f_coarse, omega);
         }
         break;
      }
      case stencil::TNE : {
         for (auto it: ci)
         {
            Cell c2 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c2, f_coarse);
            fineCoefficients[receiver][0].calculateMMM(f_coarse, omega);


            Cell c6 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(0), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c6, f_coarse);
            fineCoefficients[receiver][0].calculateMPM(f_coarse, omega);
         }
         break;
      }
      case stencil::TNW : {
         for (auto it: ci)
         {
            Cell c1 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c1, f_coarse);
            fineCoefficients[receiver][2].calculatePMM(f_coarse, omega);


            Cell c5 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(0), it.z() + cell_idx_c(0));
            getCoarsePDFs(sender, c5, f_coarse);
            fineCoefficients[receiver][2].calculatePPM(f_coarse, omega);
         }
         break;
      }
      case stencil::BNE : {
         for (auto it: ci)
         {
            Cell c3 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c3, f_coarse);
            fineCoefficients[receiver][6].calculateMMP(f_coarse, omega);


            Cell c7 = Cell(it.x() + cell_idx_c(0), it.y() + cell_idx_c(0), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c7, f_coarse);
            fineCoefficients[receiver][6].calculateMPP(f_coarse, omega);
         }
         break;
      }
      case stencil::BNW : {
         for (auto it: ci)
         {
            Cell c0 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(-1), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c0, f_coarse);
            fineCoefficients[receiver][8].calculatePMP(f_coarse, omega);


            Cell c4 = Cell(it.x() + cell_idx_c(-1), it.y() + cell_idx_c(0), it.z() + cell_idx_c(-1));
            getCoarsePDFs(sender, c4, f_coarse);
            fineCoefficients[receiver][8].calculatePPP(f_coarse, omega);
         }
         break;
      }default:
         WALBERLA_ABORT("This should not happen");
      }
   }



   void coarseToFine()
   {
      const real_t omegaC = omegaVector[0];
      const real_t omegaF = omegaVector[1];
      for(auto t: mapping_)
      {
         auto [coarseBlock, fineBlock, dir, ci] = t;
         setCoefficients(coarseBlock, fineBlock, ci, omegaC, dir);
      }

      for(auto block: blocksPerLevel_[1])
      {
         const auto neighborIdx = blockforest::getBlockNeighborhoodSectionIndex(stencil::Direction::S);
         if (!(block->neighborhoodSectionHasLargerBlock(neighborIdx))) continue;
         auto momentSets = fineCoefficients[block];
         for(size_t i = 0; i < 3; ++i)
         {
            for(size_t j = 0; j < 3; ++j)
            {
               size_t index = 3 * i + j;
               InterpolationCoefficients coefficients;
               momentSets[index].calculateCoefficients(coefficients);

               Vector3<real_t> mmm(-real_c(0.25), -real_c(0.25), -real_c(0.25));
               Cell c0(-1 + 2*cell_idx_c(j), -1, -1 + 2*cell_idx_c(i));
               interpolateCF(block, c0, mmm, omegaF, coefficients);

               Vector3<real_t> mmp(-real_c(0.25), -real_c(0.25), real_c(0.25));
               Cell c1(-1 + 2*cell_idx_c(j), -1, 0 + 2*cell_idx_c(i));
               interpolateCF(block, c1, mmp, omegaF, coefficients);

               Vector3<real_t> pmp(real_c(0.25), -real_c(0.25), real_c(0.25));
               Cell c2(0 + 2*cell_idx_c(j), -1, 0 + 2*cell_idx_c(i));
               interpolateCF(block, c2, pmp, omegaF, coefficients);

               Vector3<real_t> pmm(real_c(0.25), -real_c(0.25), -real_c(0.25));
               Cell c3(0 + 2*cell_idx_c(j), -1, -1 + 2*cell_idx_c(i));
               interpolateCF(block, c3, pmm, omegaF, coefficients);

               Vector3<real_t> mpm(-real_c(0.25), real_c(0.25), -real_c(0.25));
               Cell c4(-1 + 2*cell_idx_c(j), -1, -1 + 2*cell_idx_c(i));
               interpolateCF(block, c4, mpm, omegaF, coefficients);

               Vector3<real_t> mpp(-real_c(0.25), real_c(0.25), real_c(0.25));
               Cell c5(-1 + 2*cell_idx_c(j), -1, 0 + 2*cell_idx_c(i));
               interpolateCF(block, c5, mpp, omegaF, coefficients);

               Vector3<real_t> ppp(real_c(0.25), real_c(0.25), real_c(0.25));
               Cell c6(0 + 2*cell_idx_c(j), -1, 0 + 2*cell_idx_c(i));
               interpolateCF(block, c6, ppp, omegaF, coefficients);

               Vector3<real_t> ppm(real_c(0.25), real_c(0.25), -real_c(0.25));
               Cell c7(0 + 2*cell_idx_c(j), -1, -1 + 2*cell_idx_c(i));
               interpolateCF(block, c7, ppm, omegaF, coefficients);

            }
         }
      }
   }

   void fineToCoarse()
   {
      const real_t omegaC = omegaVector[0];
      const real_t omegaF = omegaVector[1];

      for( auto it = blocks_->begin(); it != blocks_->end(); ++it )
      {
         if(blocks_->getLevel(*it.get()) != 0)
            continue;

         for(cell_idx_t z = 0; z < 4; z++)
         {
            for(cell_idx_t x = 0; x < 4; x++)
            {
               real_t f_fine[27];
               MomentsOnSourceNodeSet moments;

               Vector3<real_t> p(real_c(x), real_c(3.0), real_c(z));
               blocks_->transformBlockLocalToGlobal(p, *it);
               Vector3<real_t> cellCenter(p[0] + real_c(0.5), p[1] + real_c(1.5), p[2] + real_c(0.5));
               Block* fineBlock = dynamic_cast< Block* >(blocks_->getBlock(p[0], p[1] + real_c(2.0), p[2]));

               Vector3<real_t> mmm(-real_c(0.25), -real_c(0.25), -real_c(0.25));
               Cell c0 = blocks_->getBlockLocalCell(*fineBlock, cellCenter + mmm);
               getFinePDFs(fineBlock, c0, f_fine);
               moments.calculateMMM(f_fine, omegaF);

               Vector3<real_t> mmp(-real_c(0.25), -real_c(0.25), real_c(0.25));
               Cell c1 = blocks_->getBlockLocalCell(*fineBlock, cellCenter + mmp);
               getFinePDFs(fineBlock, c1, f_fine);
               moments.calculateMMP(f_fine, omegaF);

               Vector3<real_t> pmp(real_c(0.25), -real_c(0.25), real_c(0.25));
               Cell c2 = blocks_->getBlockLocalCell(*fineBlock, cellCenter + pmp);
               getFinePDFs(fineBlock, c2, f_fine);
               moments.calculatePMP(f_fine, omegaF);

               Vector3<real_t> pmm(real_c(0.25), -real_c(0.25), -real_c(0.25));
               Cell c3 = blocks_->getBlockLocalCell(*fineBlock, cellCenter + pmm);
               getFinePDFs(fineBlock, c3, f_fine);
               moments.calculatePMM(f_fine, omegaF);

               Vector3<real_t> mpm(-real_c(0.25), real_c(0.25), -real_c(0.25));
               Cell c4 = blocks_->getBlockLocalCell(*fineBlock, cellCenter + mpm);
               getFinePDFs(fineBlock, c4, f_fine);
               moments.calculateMPM(f_fine, omegaF);

               Vector3<real_t> mpp(-real_c(0.25), real_c(0.25), real_c(0.25));
               Cell c5 = blocks_->getBlockLocalCell(*fineBlock, cellCenter + mpp);
               getFinePDFs(fineBlock, c5, f_fine);
               moments.calculateMPP(f_fine, omegaF);

               Vector3<real_t> ppp(real_c(0.25), real_c(0.25), real_c(0.25));
               Cell c6 = blocks_->getBlockLocalCell(*fineBlock, cellCenter + ppp);
               getFinePDFs(fineBlock, c6, f_fine);
               moments.calculatePPP(f_fine, omegaF);

               Vector3<real_t> ppm(real_c(0.25), real_c(0.25), -real_c(0.25));
               Cell c7 = blocks_->getBlockLocalCell(*fineBlock, cellCenter + ppm);
               getFinePDFs(fineBlock, c7, f_fine);
               moments.calculatePPM(f_fine, omegaF);

               InterpolationCoefficients coefficients;
               moments.calculateCoefficients(coefficients);

               Block* coarseBlock = dynamic_cast< Block* >(it.get());
               Cell coarseLocalCell(x, cell_idx_c(3), z);
               interpolateFC(coarseBlock, coarseLocalCell, omegaF, coefficients);
            }
         }
      }
   }

 private:
   std::shared_ptr< StructuredBlockForest > blocks_;
   IDs ids_;
   std::vector<real_t> omegaVector;
   std::vector<std::vector<Block *>> blocksPerLevel_;
   std::map< Block*, std::vector< MomentsOnSourceNodeSet > > fineCoefficients;
   std::vector<std::tuple<Block*, Block*, stencil::Direction, CellInterval>> mapping_;
};


class Timestep
{
 public:
   Timestep(std::shared_ptr< StructuredBlockForest >& blocks, const IDs& ids,
            StreamCollide & streamCollide, BoundaryCollection_T& boundaryCollection, Interpolation& interpolation,
            std::shared_ptr< PackInfo_T >& packInfo, std::shared_ptr< NonUniformBufferedScheme<CommunicationStencil_T> > communication)
      : blocks_(blocks), ids_(ids), streamCollide_(streamCollide), boundaryCollection_(boundaryCollection), interpolation_(interpolation),
        packInfo_(packInfo), communication_(communication)
   {
      maxLevel_ = blocks_->getDepth();

      for (uint_t level = 0; level <= maxLevel_; level++)
      {
         std::vector<Block *> tmp;
         blocks_->getBlocks(tmp, level);
         blocksPerLevel_.push_back(tmp);
      }
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
            cell_idx_t gls = 1; //skipsThroughCoarseBlock(sender, *dir) ? 2 : 1;
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

   inline bool skipsThroughCoarseBlock(const Block* block, const stencil::Direction dir) const
   {
      Vector3< cell_idx_t > dirVec(stencil::cx[dir], stencil::cy[dir], stencil::cz[dir]);
      bool coarseBlockFound = false;
      forEachSubdirectionCancel(dirVec, [&](Vector3< cell_idx_t > subdir) {
         coarseBlockFound =
            coarseBlockFound || block->neighborhoodSectionHasLargerBlock(
                                   blockforest::getBlockNeighborhoodSectionIndex(subdir[0], subdir[1], subdir[2]));
         return !coarseBlockFound;
      });

      return coarseBlockFound;
   }

   void ghostLayerPropagation(Block * block)
   {
      auto pdfField = block->getData<PdfField_T>(ids_.pdfField);

      for(auto it = CommunicationStencil_T ::beginNoCenter(); it != CommunicationStencil_T ::end(); ++it){
         uint_t nSecIdx = blockforest::getBlockNeighborhoodSectionIndex(*it);
         // Propagate on ghost layers shadowing coarse or no blocks
         if(block->neighborhoodSectionHasLargerBlock(nSecIdx)){
            CellInterval ci;
            pdfField->getGhostRegion(*it, ci, 1);
            streamCollide_.streamCollide(block, ci);
         }
      }
   }

   void timestep()
   {
      for(auto b: blocksPerLevel_[1])
      {
         ghostLayerPropagation(b);
         streamCollide_(b);
         boundaryCollection_(b);
      }
      commLocal(1);


      for(auto b: blocksPerLevel_[1])
      {
         streamCollide_(b);
         boundaryCollection_(b);
      }
      commLocal(1);

//      for(auto b: blocksPerLevel_[0])
//      {
//         streamCollide_(b);
//         boundaryCollection_(b);
//      }
//      commLocal(0);

      interpolation_.coarseToFine();
      //interpolation_.fineToCoarse();
   }

   void operator()(){ timestep(); };

 private:
   std::shared_ptr< StructuredBlockForest > blocks_;
   IDs ids_;
   StreamCollide & streamCollide_;
   BoundaryCollection_T & boundaryCollection_;
   Interpolation & interpolation_;
   std::shared_ptr< PackInfo_T > packInfo_;
   std::shared_ptr< NonUniformBufferedScheme<CommunicationStencil_T> > communication_;

   uint_t maxLevel_;
   std::vector<std::vector<Block *>> blocksPerLevel_;
   uint_t t{0};
}; // class RefinementSelection

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
   auto EvaluationParameters = config->getOneBlock("Evaluation");

   const real_t omega              = parameters.getParameter< real_t >("omega");
   const real_t maxLatticeVelocity = parameters.getParameter< real_t >("maxLatticeVelocity");
   const uint_t timesteps          = parameters.getParameter< uint_t >("timesteps") + uint_c(1);

   const uint_t refinementLevels   = domainParameters.getParameter< uint_t >("refinementLevels");

   Setup setup;
   setup.blocks        = domainParameters.getParameter< Vector3< uint_t > >("blocks");
   setup.cellsPerBlock = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
   setup.cells = Vector3< uint_t >(setup.blocks[0] * setup.cellsPerBlock[0], setup.blocks[1] * setup.cellsPerBlock[1],
                                   setup.blocks[2] * setup.cellsPerBlock[2]);
   setup.periodic       = domainParameters.getParameter< Vector3< bool > >("periodic");
   setup.numGhostLayers = refinementLevels > 0 ? uint_c(2) : uint_c(1);


   const uint_t valuesPerCell = (uint_c(2) * Stencil_T::Q + VectorField_T ::F_SIZE + uint_c(2) * ScalarField_T::F_SIZE);
   const uint_t sizePerValue  = sizeof(PdfField_T::value_type);
   const memory_t memoryPerCell = memory_t(valuesPerCell * sizePerValue + uint_c(1));

   blockforest::RefinementSelectionFunctions refinementSelectionFunctions;

   RefinementSelection refinementSelection(refinementLevels);
   refinementSelectionFunctions.add(refinementSelection);

   bool writeSetupForestAndReturn = loggingParameters.getParameter< bool >("writeSetupForestAndReturn", false);
   if (uint_c(MPIManager::instance()->numProcesses()) > 1) writeSetupForestAndReturn = false;

   if (writeSetupForestAndReturn)
   {
      std::string sbffile = "Channel.bfs";

      std::ostringstream infoString;
      infoString << "You have selected the option of just creating the block structure (= domain decomposition) and "
                    "saving the result to file\n"
                    "by specifying the output file name \'"
                 << sbffile << "\' AND also specifying \'saveToFile\'.\n";

      if (MPIManager::instance()->numProcesses() > 1)
         WALBERLA_ABORT(infoString.str() << "In this mode you need to start " << argv[0] << " with just one process!")

      WALBERLA_LOG_INFO_ON_ROOT(infoString.str() << "Creating the block structure ...")

      const uint_t numberProcesses = domainParameters.getParameter< uint_t >("numberProcesses");

      shared_ptr< SetupBlockForest > sforest =
         createSetupBlockForest(refinementSelectionFunctions, setup, numberProcesses, memoryPerCell);
      sforest->writeVTKOutput("domain_decomposition");
      sforest->saveToFile(sbffile.c_str());

      logging::Logging::printFooterOnStream();
      return EXIT_SUCCESS;
   }

   auto blocks = createStructuredBlockForest(refinementSelectionFunctions, setup, memoryPerCell);

   IDs ids;
   const StorageSpecification_T StorageSpec = StorageSpecification_T();
   ids.pdfField      = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, setup.numGhostLayers, field::fzyx);
   ids.pdfFieldTmp   = lbm_generated::addPdfFieldToStorage(blocks, "pdfs tmp", StorageSpec, setup.numGhostLayers, field::fzyx);
   ids.velocityField = field::addToStorage< VectorField_T >(blocks, "vel", real_c(0.0), field::fzyx, setup.numGhostLayers);
   ids.densityField = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1.0), field::fzyx, setup.numGhostLayers);
   ids.flagField    = field::addFlagFieldToStorage< FlagField_T >(blocks, "Boundary Flag Field", uint_c(3));

//   for (auto& block : *blocks)
//   {
//      const uint_t level = blocks->getLevel(block);
//      if(level == 0)
//      {
//         auto velocityField = block.getData<VectorField_T>(ids.velocityField);
//
//         for (cell_idx_t ctr_2 = 0; ctr_2 < velocityField->zSize(); ++ctr_2)
//         {
//            for(cell_idx_t ctr_1 = 0; ctr_1 < velocityField->ySize(); ++ctr_1)
//            {
//               for (cell_idx_t ctr_0 = 0; ctr_0 < velocityField->xSize(); ++ctr_0)
//               {
//                  Cell cell(ctr_0, ctr_1, ctr_2);
//                  blocks->transformBlockLocalToGlobalCell(cell, block);
//                  blocks->mapToPeriodicDomain(cell);
//                  velocityField->get(ctr_0, ctr_1, ctr_2, 0) = real_c(cell.x());
//                  velocityField->get(ctr_0, ctr_1, ctr_2, 1) = real_c(cell.y());
//                  velocityField->get(ctr_0, ctr_1, ctr_2, 2) = real_c(cell.z());
//               }
//            }
//         }
//      }
//   }


//      for (auto& block : *blocks)
//      {
//         const uint_t level = blocks->getLevel(block);
//         if(level == 0)
//         {
//            auto velocityField = block.getData<VectorField_T>(ids.velocityField);
//
//            for (cell_idx_t ctr_2 = 0; ctr_2 < velocityField->zSize(); ++ctr_2)
//            {
//               for(cell_idx_t ctr_1 = 0; ctr_1 < velocityField->ySize(); ++ctr_1)
//               {
//                  for (cell_idx_t ctr_0 = 0; ctr_0 < velocityField->xSize(); ++ctr_0)
//                  {
//                     velocityField->get(ctr_0, ctr_1, ctr_2, 0) = real_c(0.01);
//                  }
//               }
//            }
//         }
//      }


   SweepCollection_T sweepCollection(blocks, ids.pdfField, ids.densityField, ids.velocityField, omega);
   for (auto& block : *blocks)
   {
      sweepCollection.initialise(&block, cell_idx_c(setup.numGhostLayers - uint_c(1)));
   }

   const FlagUID fluidFlagUID("Fluid");
   auto boundariesConfig = config->getBlock("Boundaries");
   geometry::initBoundaryHandling< FlagField_T >(*blocks, ids.flagField, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, ids.flagField, fluidFlagUID, cell_idx_c(0));

   BoundaryCollection_T boundaryCollection(blocks, ids.flagField, ids.pdfField, fluidFlagUID, maxLatticeVelocity);

   WALBERLA_LOG_INFO_ON_ROOT("Setting up communication...")
   auto communication      = std::make_shared< NonUniformBufferedScheme< CommunicationStencil_T > >(blocks);
   auto nonUniformPackInfo = lbm_generated::setupNonuniformPdfCommunication< PdfField_T >(blocks, ids.pdfField);
   communication->addPackInfo(nonUniformPackInfo);

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
         vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_Channel", "simulation_step",
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
   Interpolation interpolation(blocks, ids, omega);
   Timestep timeloopFunction(blocks, ids, streamCollide, boundaryCollection, interpolation, nonUniformPackInfo, communication);
   timeloop.addFuncBeforeTimeStep(timeloopFunction, "Refinement Cycle");

   const real_t remainingTimeLoggerFrequency =
      loggingParameters.getParameter< real_t >("remainingTimeLoggerFrequency", 3.0); // in seconds
   if (uint_c(remainingTimeLoggerFrequency) > 0)
   {
      timeloop.addFuncAfterTimeStep(
         timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
         "remaining time logger");
   }

   const uint_t evaluationCheckFrequency = EvaluationParameters.getParameter< uint_t >("evaluationCheckFrequency");
   const bool evaluationLogToStream      = EvaluationParameters.getParameter< bool >("logToStream");
   const bool evaluationLogToFile        = EvaluationParameters.getParameter< bool >("logToFile");
   const std::string evaluationFilename  = EvaluationParameters.getParameter< std::string >("filename");

   std::shared_ptr< AccuracyEvaluation > evaluation =
      std::make_shared< AccuracyEvaluation >(blocks, ids, sweepCollection, maxLatticeVelocity, evaluationCheckFrequency,
                                             evaluationLogToStream, evaluationLogToFile);
   timeloop.addFuncBeforeTimeStep(SharedFunctor< AccuracyEvaluation >(evaluation), "evaluation");

   //////////////////////
   /// RUN SIMULATION ///
   //////////////////////
   const lbm_generated::PerformanceEvaluation< FlagField_T > performance(blocks, ids.flagField, fluidFlagUID);
   field::CellCounter< FlagField_T > fluidCells(blocks, ids.flagField, fluidFlagUID);
   fluidCells();

   WALBERLA_LOG_INFO_ON_ROOT("Blocks created: " << blocks->getNumberOfBlocks())
   for (uint_t level = 0; level <= refinementLevels; level++)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Level " << level << " Blocks: " << blocks->getNumberOfBlocks(level))
   }

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
