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
//! \file   PipingEvaluators.h
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/mpi/Gatherv.h"
#include "core/mpi/Reduce.h"

#include "field/FlagField.h"

#include "lbm_mesapd_coupling/DataTypesGPU.h"

#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/shape/Sphere.h"

#include "ParticleSelectors.h"

namespace walberla
{
namespace piping
{

template< typename ParticleAccessor_T >
real_t computeVoidRatio(const shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& BFieldID,
                        const BlockDataID& BFieldGPUID, const BlockDataID& flagFieldID, field::FlagUID fluidFlagID,
                        const shared_ptr< ParticleAccessor_T >& accessor,
                        const shared_ptr< mesa_pd::data::ParticleStorage >& ps, const real_t tau)
{
   using namespace lbm_mesapd_coupling::psm::gpu;

   // Compute max particle height (only considers particle centers, not particle radii)
   real_t maxParticleHeight = real_t(0);
   ps->forEachParticle(
      false, SphereSelector(), *accessor,
      [&maxParticleHeight](const size_t idx, auto& ac) {
         maxParticleHeight = std::max(ac.getPosition(idx)[2], maxParticleHeight);
      },
      *accessor);

   WALBERLA_MPI_SECTION() { mpi::allReduceInplace(maxParticleHeight, mpi::MAX); }

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(maxParticleHeight)
   gpu::fieldCpy< GhostLayerField< real_t, 1 >, BFieldGPU_T >(blocks, BFieldID, BFieldGPUID);

   // Compute cell-averaged overlap fraction for cells in the sediment bed (= below the maximum particle height)
   real_t sumOverlapFractions = real_t(0);
   uint_t numFluidCells       = uint_t(0);
   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      BField_T* BField                          = blockIt->getData< BField_T >(BFieldID);
      FlagField< walberla::uint8_t >* flagField = blockIt->getData< FlagField< walberla::uint8_t > >(flagFieldID);
      auto fluidFlag                            = flagField->getOrRegisterFlag(fluidFlagID);
      WALBERLA_FOR_ALL_CELLS_XYZ(
         BField, Cell cell(x, y, z); blocks->transformBlockLocalToGlobalCell(cell, *blockIt);
         const Vector3< real_t > globalCellCenter = blocks->getCellCenter(cell);
         // Only consider cells inside the soil (< maxParticleHeight) and outside the bucket (= fluidFlag)
         // TODO: estimated void ratio is too small if movingBucket is true because the bucket is also counted
         if (globalCellCenter[2] < maxParticleHeight && flagField->get(x, y, z) == fluidFlag) {
            // TODO: fix this if BField does not contain overlap fraction if weighting=2
            auto overlapFraction = BField->get(x, y, z);
            // If Weighting == 2, the BField is not equal to the solid volume fraction
            // TODO: check why e_init is not exactly the same for both weightings
            if (Weighting == 2)
            {
               overlapFraction = ((real_t(2) * tau + real_t(1)) * overlapFraction) /
                                 (real_t(2) * tau + real_t(2) * overlapFraction - real_t(1));
            }
            sumOverlapFractions += overlapFraction;
            ++numFluidCells;
         })
   }

   WALBERLA_MPI_SECTION()
   {
      mpi::allReduceInplace(sumOverlapFractions, mpi::SUM);
      mpi::allReduceInplace(numFluidCells, mpi::SUM);
   }

   // Compute void fraction
   return real_t(1) - sumOverlapFractions / real_t(numFluidCells);
}

template< typename ParticleAccessor_T >
real_t computeMaxParticleSeepageHeight(const shared_ptr< ParticleAccessor_T >& accessor,
                                       const shared_ptr< mesa_pd::data::ParticleStorage >& ps,
                                       const mesa_pd::Vec3& boxPosition, const mesa_pd::Vec3& boxEdgeLength)
{
   // Compute max particle height (only considers particle close to the bucket)
   // Assuming uniform height on both sides of the bucket (no uplift or subsidence)
   real_t maxParticleSeepageHeight = real_t(0);
   ps->forEachParticle(
      false, SphereSelector(), *accessor,
      [&maxParticleSeepageHeight, &boxPosition, &boxEdgeLength](const size_t idx, auto& ac) {
         if (ac.getPosition(idx)[0] > boxPosition[0] - boxEdgeLength[0] &&
             ac.getPosition(idx)[0] < boxPosition[0] + boxEdgeLength[0])
         {
            maxParticleSeepageHeight = std::max(ac.getPosition(idx)[2], maxParticleSeepageHeight);
         }
      },
      *accessor);

   WALBERLA_MPI_SECTION() { mpi::allReduceInplace(maxParticleSeepageHeight, mpi::MAX); }

   return maxParticleSeepageHeight;
}

template< typename ParticleAccessor_T >
real_t computeSeepageLength(const shared_ptr< ParticleAccessor_T >& accessor,
                            const shared_ptr< mesa_pd::data::ParticleStorage >& ps, const mesa_pd::Vec3& boxPosition,
                            const mesa_pd::Vec3& boxEdgeLength)
{
   using namespace lbm_mesapd_coupling::psm::gpu;

   const real_t maxParticleSeepageHeight = computeMaxParticleSeepageHeight(accessor, ps, boxPosition, boxEdgeLength);
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(maxParticleSeepageHeight)
   const real_t penetrationDepth = maxParticleSeepageHeight - real_t(boxPosition[2] - boxEdgeLength[2] / 2);
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(penetrationDepth)
   WALBERLA_CHECK(penetrationDepth > 0, "Bucket does not penetrate the soil.")

   // Two times the penetration depth plus the wall thickness
   return real_t(2 * penetrationDepth + boxEdgeLength[0]);
}

// TODO: further test this functionality (particle UIDs already tested)
// Computes the average particle displacement (uplift/subsidence) in z direction. The corresponding observation domains
// are aligned with the bucket and the soil (see Fukumoto et al., 2021) height and the size is specified in the
// parameter file. All particles overlapping with the observation domain are tracked.
template< typename ParticleAccessor_T >
class UpliftSubsidenceEvaluator
{
 public:
   UpliftSubsidenceEvaluator(const shared_ptr< ParticleAccessor_T >& accessor,
                             const shared_ptr< mesa_pd::data::ParticleStorage >& ps, const mesa_pd::Vec3& boxPosition,
                             const mesa_pd::Vec3& boxEdgeLength, const Vector3< real_t >& observationDomainSize,
                             const bool pressureDrivenFlow)
   {
      const real_t maxParticleSeepageHeight = computeMaxParticleSeepageHeight(accessor, ps, boxPosition, boxEdgeLength);
      const Vector3< real_t > upliftDomainCenter(boxPosition[0] + boxEdgeLength[0] / 2 + observationDomainSize[0] / 2,
                                                 boxPosition[1],
                                                 maxParticleSeepageHeight - observationDomainSize[2] / 2);
      const Vector3< real_t > subsidenceDomainCenter(
         boxPosition[0] - boxEdgeLength[0] / 2 - observationDomainSize[0] / 2, boxPosition[1],
         maxParticleSeepageHeight - observationDomainSize[2] / 2);

      // WALBERLA_LOG_DEVEL_VAR_ON_ROOT(upliftDomainCenter)
      // WALBERLA_LOG_DEVEL_VAR_ON_ROOT(subsidenceDomainCenter)
      // WALBERLA_LOG_DEVEL_VAR_ON_ROOT(observationDomainSize)

      std::vector< walberla::id_t > upliftParticlesUIDsLocal;
      std::vector< walberla::id_t > subsidenceParticlesUIDsLocal;
      ps->forEachParticle(
         false, SphereSelectorExcludeGhost(), *accessor,
         [this, &upliftDomainCenter, &subsidenceDomainCenter, &observationDomainSize, &upliftParticlesUIDsLocal,
          &subsidenceParticlesUIDsLocal](const size_t idx, auto& ac) {
            const real_t particleRadius =
               static_cast< mesa_pd::data::Sphere* >(ac.getBaseShape(idx).get())->getRadius();
            if (sphereBoxOverlap(ac.getPosition(idx), particleRadius, upliftDomainCenter, observationDomainSize))
            {
               upliftParticlesUIDsLocal.push_back(ac.getUid(idx));
               initialUpliftPosition_ += ac.getPosition(idx)[2];
            }
            if (sphereBoxOverlap(ac.getPosition(idx), particleRadius, subsidenceDomainCenter, observationDomainSize))
            {
               subsidenceParticlesUIDsLocal.push_back(ac.getUid(idx));
               initialSubsidencePosition_ += ac.getPosition(idx)[2];
            }
         },
         *accessor);

      WALBERLA_MPI_SECTION()
      {
         upliftParticlesUIDs_ = walberla::mpi::allGatherv(upliftParticlesUIDsLocal);
         walberla::mpi::allReduceInplace(initialUpliftPosition_, walberla::mpi::SUM);
         subsidenceParticlesUIDs_ = walberla::mpi::allGatherv(subsidenceParticlesUIDsLocal);
         walberla::mpi::allReduceInplace(initialSubsidencePosition_, walberla::mpi::SUM);
      }

      initialUpliftPosition_ /= real_t(upliftParticlesUIDs_.size());
      initialSubsidencePosition_ /= real_t(subsidenceParticlesUIDs_.size());

      // WALBERLA_LOG_DEVEL_VAR_ON_ROOT(upliftParticlesUIDs_.size())
      // WALBERLA_LOG_DEVEL_VAR_ON_ROOT(initialUpliftPosition_)
      // WALBERLA_LOG_DEVEL_VAR_ON_ROOT(subsidenceParticlesUIDs_.size())
      // WALBERLA_LOG_DEVEL_VAR_ON_ROOT(initialSubsidencePosition_)

      outFile_.open("upliftSubsidenceEvaluation.dat", std::ios::out);
      if (pressureDrivenFlow) { outFile_ << "# HydraulicGradient Uplift Subsidence" << std::endl; }
      else { outFile_ << "# OutflowVelocity Uplift Subsidence" << std::endl; }
   }

   ~UpliftSubsidenceEvaluator() { outFile_.close(); };

   // Write either the hydraulic gradient or the outflow velocity into the file
   void operator()(const real_t outflowQuantity, const shared_ptr< ParticleAccessor_T >& accessor,
                   const shared_ptr< mesa_pd::data::ParticleStorage >& ps)
   {
      outFile_ << outflowQuantity << " ";

      real_t upliftPosition     = real_t(0);
      real_t subsidencePosition = real_t(0);

      ps->forEachParticle(
         false, SphereSelectorExcludeGhost(), *accessor,
         [this, &upliftPosition, &subsidencePosition](const size_t idx, auto& ac) {
            if (std::find(upliftParticlesUIDs_.begin(), upliftParticlesUIDs_.end(), ac.getUid(idx)) !=
                upliftParticlesUIDs_.end())
            {
               upliftPosition += ac.getPosition(idx)[2];
            }
            if (std::find(subsidenceParticlesUIDs_.begin(), subsidenceParticlesUIDs_.end(), ac.getUid(idx)) !=
                subsidenceParticlesUIDs_.end())
            {
               subsidencePosition += ac.getPosition(idx)[2];
            }
         },
         *accessor);

      WALBERLA_MPI_SECTION()
      {
         walberla::mpi::allReduceInplace(upliftPosition, walberla::mpi::SUM);
         walberla::mpi::allReduceInplace(subsidencePosition, walberla::mpi::SUM);
      }

      upliftPosition /= real_t(upliftParticlesUIDs_.size());
      subsidencePosition /= real_t(subsidenceParticlesUIDs_.size());

      outFile_ << upliftPosition - initialUpliftPosition_ << " " << subsidencePosition - initialSubsidencePosition_
               << std::endl;
   }

 private:
   std::vector< walberla::id_t > upliftParticlesUIDs_;
   real_t initialUpliftPosition_ = real_t(0);
   std::vector< walberla::id_t > subsidenceParticlesUIDs_;
   real_t initialSubsidencePosition_ = real_t(0);
   std::ofstream outFile_;
};

} // namespace piping
} // namespace walberla
