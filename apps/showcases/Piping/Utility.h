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
//! \file   Utility.h
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/mpi/Broadcast.h"
#include "core/mpi/Gatherv.h"
#include "core/mpi/MPITextFile.h"
#include "core/mpi/Reduce.h"

#include "field/FlagField.h"

#include "lbm_mesapd_coupling/DataTypesGPU.h"

#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/shape/Sphere.h"

#include <functional>
#include <iterator>

namespace walberla
{
namespace piping
{

// Some functions in this file (as the one below) are based on showcases/Antidunes/Utility.cpp

void writeSphereInformationToFile(const std::string& filename, walberla::mesa_pd::data::ParticleStorage& ps,
                                  Vector3< real_t >& domainSize, int precision = 12)
{
   std::ostringstream ossData;
   ossData << std::setprecision(precision);

   WALBERLA_ROOT_SECTION() { ossData << domainSize[0] << " " << domainSize[1] << " " << domainSize[2] << "\n"; }

   for (auto pIt : ps)
   {
      using namespace walberla::mesa_pd::data;
      if (pIt->getBaseShape()->getShapeType() != Sphere::SHAPE_TYPE) continue;
      using namespace walberla::mesa_pd::data::particle_flags;
      if (isSet(pIt->getFlags(), GHOST)) continue;
      auto sp = static_cast< Sphere* >(pIt->getBaseShape().get());

      auto position = pIt->getPosition();

      ossData << position[0] << " " << position[1] << " " << position[2] << " " << sp->getRadius() << '\n';
   }

   walberla::mpi::writeMPITextFile(filename, ossData.str());
}

bool sphereBoxOverlap(const mesa_pd::Vec3& spherePosition, const real_t sphereRadius, const mesa_pd::Vec3& boxPosition,
                      const mesa_pd::Vec3& boxEdgeLength)
{
   if ((spherePosition[0] + sphereRadius < boxPosition[0] - boxEdgeLength[0] / real_t(2)) ||
       (spherePosition[1] + sphereRadius < boxPosition[1] - boxEdgeLength[1] / real_t(2)) ||
       (spherePosition[2] + sphereRadius < boxPosition[2] - boxEdgeLength[2] / real_t(2)) ||
       (spherePosition[0] - sphereRadius > boxPosition[0] + boxEdgeLength[0] / real_t(2)) ||
       (spherePosition[1] - sphereRadius > boxPosition[1] + boxEdgeLength[1] / real_t(2)) ||
       (spherePosition[2] - sphereRadius > boxPosition[2] + boxEdgeLength[2] / real_t(2)))
   {
      return false;
   }
   return true;
}

void initSpheresFromFile(const std::string& fileName, walberla::mesa_pd::data::ParticleStorage& ps,
                         const walberla::mesa_pd::domain::IDomain& domain, walberla::real_t particleDensity,
                         math::AABB& simulationDomain, const Vector3< uint_t >& domainSize,
                         const mesa_pd::Vec3& boxPosition, const mesa_pd::Vec3& boxEdgeLength)
{
   using namespace walberla::mesa_pd::data;

   auto rank = walberla::mpi::MPIManager::instance()->rank();

   std::string textFile;

   WALBERLA_ROOT_SECTION()
   {
      std::ifstream t(fileName.c_str());
      if (!t) { WALBERLA_ABORT("Invalid input file " << fileName << "\n"); }
      std::stringstream buffer;
      buffer << t.rdbuf();
      textFile = buffer.str();
   }

   walberla::mpi::broadcastObject(textFile);

   std::istringstream fileIss(textFile);
   std::string line;

   // first line contains generation domain sizes
   std::getline(fileIss, line);
   Vector3< real_t > generationDomainSize_SI(0_r);
   std::istringstream firstLine(line);
   firstLine >> generationDomainSize_SI[0] >> generationDomainSize_SI[1] >> generationDomainSize_SI[2];
   real_t scalingFactor = real_t(domainSize[0]) / generationDomainSize_SI[0];
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(generationDomainSize_SI)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(scalingFactor)

   real_t minParticleDiameter = std::numeric_limits< real_t >::max();
   real_t maxParticleDiameter = real_t(0);

   while (std::getline(fileIss, line))
   {
      std::istringstream iss(line);

      ParticleStorage::position_type position;
      real_t radius;
      iss >> position[0] >> position[1] >> position[2] >> radius;
      position *= scalingFactor;
      radius *= scalingFactor;

      WALBERLA_CHECK(simulationDomain.contains(position),
                     "Particle read from file is not contained in simulation domain");

      if (!domain.isContainedInProcessSubdomain(uint_c(rank), position)) continue;
      if (sphereBoxOverlap(position, radius, boxPosition, boxEdgeLength)) continue;

      auto pIt = ps.create();
      pIt->setPosition(position);
      pIt->getBaseShapeRef() = std::make_shared< data::Sphere >(radius);
      pIt->getBaseShapeRef()->updateMassAndInertia(particleDensity);
      pIt->setInteractionRadius(radius);
      pIt->setOwner(rank);
      pIt->setType(0);

      minParticleDiameter = std::min(real_t(2) * radius, minParticleDiameter);
      maxParticleDiameter = std::max(real_t(2) * radius, maxParticleDiameter);

      WALBERLA_CHECK_EQUAL(iss.tellg(), -1);
   }

   WALBERLA_MPI_SECTION() { walberla::mpi::allReduceInplace(minParticleDiameter, walberla::mpi::MIN); }
   WALBERLA_MPI_SECTION() { walberla::mpi::allReduceInplace(maxParticleDiameter, walberla::mpi::MAX); }
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(minParticleDiameter)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(maxParticleDiameter)
}

template< typename ParticleAccessor_T >
void getParticleVelocities(const ParticleAccessor_T& ac, real_t& maxVelocity, real_t& averageVelocity)
{
   maxVelocity         = real_t(0);
   averageVelocity     = real_t(0);
   uint_t numParticles = uint_t(0);

   for (uint_t i = 0; i < ac.size(); ++i)
   {
      if (isSet(ac.getFlags(i), walberla::mesa_pd::data::particle_flags::GHOST)) continue;
      if (isSet(ac.getFlags(i), walberla::mesa_pd::data::particle_flags::GLOBAL)) continue;

      ++numParticles;
      real_t velMagnitude = ac.getLinearVelocity(i).length();
      maxVelocity         = std::max(maxVelocity, velMagnitude);
      averageVelocity += velMagnitude;
   }

   WALBERLA_MPI_SECTION()
   {
      walberla::mpi::allReduceInplace(maxVelocity, walberla::mpi::MAX);
      walberla::mpi::allReduceInplace(averageVelocity, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(numParticles, walberla::mpi::SUM);
   }

   averageVelocity /= real_t(numParticles);
}

// TODO: maybe set different types and density for plane and sphere
auto createPlane(mesa_pd::data::ParticleStorage& ps, const mesa_pd::Vec3& pos, const mesa_pd::Vec3& normal)
{
   auto p0 = ps.create(true);
   p0->setPosition(pos);
   p0->setBaseShape(std::make_shared< mesa_pd::data::HalfSpace >(normal));
   p0->getBaseShapeRef()->updateMassAndInertia(real_t(1));
   p0->setOwner(walberla::mpi::MPIManager::instance()->rank());
   p0->setType(0);
   p0->setInteractionRadius(std::numeric_limits< real_t >::infinity());
   mesa_pd::data::particle_flags::set(p0->getFlagsRef(), mesa_pd::data::particle_flags::GLOBAL);
   mesa_pd::data::particle_flags::set(p0->getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
   mesa_pd::data::particle_flags::set(p0->getFlagsRef(), mesa_pd::data::particle_flags::FIXED);
   mesa_pd::data::particle_flags::set(p0->getFlagsRef(), mesa_pd::data::particle_flags::NON_COMMUNICATING);
   return p0;
}

// TODO: these box flags can only be used for non-moving box
auto createBox(mesa_pd::data::ParticleStorage& ps, const mesa_pd::Vec3& pos, const mesa_pd::Vec3& edgeLength)
{
   auto p0 = ps.create(true);
   p0->setPosition(pos);
   p0->setBaseShape(std::make_shared< mesa_pd::data::Box >(edgeLength));
   p0->getBaseShapeRef()->updateMassAndInertia(real_t(1));
   p0->setOwner(walberla::mpi::MPIManager::instance()->rank());
   p0->setType(0);
   p0->setInteractionRadius(std::numeric_limits< real_t >::infinity());
   mesa_pd::data::particle_flags::set(p0->getFlagsRef(), mesa_pd::data::particle_flags::GLOBAL);
   mesa_pd::data::particle_flags::set(p0->getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
   mesa_pd::data::particle_flags::set(p0->getFlagsRef(), mesa_pd::data::particle_flags::FIXED);
   mesa_pd::data::particle_flags::set(p0->getFlagsRef(), mesa_pd::data::particle_flags::NON_COMMUNICATING);
   return p0;
}

void assembleBoundaryBlock(const Vector3< uint_t >& domainSize, const mesa_pd::Vec3& boxPosition,
                           const mesa_pd::Vec3& boxEdgeLength, const bool periodicInY)
{
   // TODO: improve readability of boundary conditions
   std::string boundariesBlockString =
      " Boundaries"
      "{"
      "Border { direction W;    walldistance -1;  flag NoSlip; }"
      "Border { direction E;    walldistance -1;  flag NoSlip; }"
      "Border { direction B;    walldistance -1;  flag NoSlip; }"
      "CellInterval { min < 0,-1," +
      std::to_string(domainSize[2]) + ">; max < " + std::to_string(uint_t(boxPosition[0] - boxEdgeLength[0] / 2 - 1)) +
      "," + std::to_string(domainSize[1] + 1) + "," + std::to_string(domainSize[2] + 1) +
      ">; flag Density0; }"
      "CellInterval { min < " +
      std::to_string(uint_t(boxPosition[0] - boxEdgeLength[0] / 2)) + ",-1," + std::to_string(domainSize[2]) +
      ">; max < " + std::to_string(uint_t(boxPosition[0] + boxEdgeLength[0] / 2 - 1)) + "," +
      std::to_string(domainSize[1] + 1) + "," + std::to_string(domainSize[2] + 1) +
      ">; flag NoSlip; }"
      "CellInterval { min < " +
      std::to_string(uint_t(boxPosition[0] + boxEdgeLength[0] / 2)) + ",-1," + std::to_string(domainSize[2]) +
      ">; max < " + std::to_string(domainSize[0]) + "," + std::to_string(domainSize[1] + 1) + "," +
      std::to_string(domainSize[2] + 1) +
      ">; flag Density1; }"
      "Body { shape box; min <" +
      std::to_string(boxPosition[0] - boxEdgeLength[0] / 2) + "," +
      std::to_string(boxPosition[1] - boxEdgeLength[1] / 2) + "," +
      std::to_string(boxPosition[2] - boxEdgeLength[2] / 2) + ">; max <" +
      std::to_string(boxPosition[0] + boxEdgeLength[0] / 2) + "," +
      std::to_string(boxPosition[1] + boxEdgeLength[1] / 2) + "," +
      std::to_string(boxPosition[2] + boxEdgeLength[2] / 2) + ">; flag NoSlip; }";

   if (!periodicInY)
   {
      boundariesBlockString += "Border { direction S;    walldistance -1;  flag NoSlip; }"
                               "Border { direction N;    walldistance -1;  flag NoSlip; }";
   }

   boundariesBlockString += "}";

   WALBERLA_ROOT_SECTION()
   {
      std::ofstream boundariesFile("boundaries.prm");
      boundariesFile << boundariesBlockString;
      boundariesFile.close();
   }
   WALBERLA_MPI_BARRIER()
}

struct SphereSelector
{
   template< typename ParticleAccessor_T >
   bool inline operator()(const size_t particleIdx, const ParticleAccessor_T& ac) const
   {
      return ac.getBaseShape(particleIdx)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE;
   }
};

struct SphereSelectorExcludeGhost
{
   template< typename ParticleAccessor_T >
   bool inline operator()(const size_t particleIdx, const ParticleAccessor_T& ac) const
   {
      using namespace walberla::mesa_pd::data::particle_flags;
      return (ac.getBaseShape(particleIdx)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE) &&
             !isSet(ac.getFlags(particleIdx), GHOST);
   }
};

class SelectBoxEdgeLength
{
 public:
   using return_type = walberla::mesa_pd::Vec3;
   walberla::mesa_pd::Vec3 operator()(mesa_pd::data::Particle& p) const
   {
      return static_cast< mesa_pd::data::Box* >(p->getBaseShape().get())->getEdgeLength();
   }
   walberla::mesa_pd::Vec3 operator()(mesa_pd::data::Particle&& p) const
   {
      return static_cast< mesa_pd::data::Box* >(p->getBaseShape().get())->getEdgeLength();
   }
   walberla::mesa_pd::Vec3 const operator()(const mesa_pd::data::Particle& p) const
   {
      auto p_tmp = p;
      return static_cast< mesa_pd::data::Box* >(p_tmp->getBaseShape().get())->getEdgeLength();
   }
};

template< typename ParticleAccessor_T >
real_t computeVoidRatio(const shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& BFieldID,
                        const BlockDataID& BFieldGPUID, const BlockDataID& flagFieldID, field::FlagUID fluidFlagID,
                        const shared_ptr< ParticleAccessor_T >& accessor,
                        const shared_ptr< mesa_pd::data::ParticleStorage >& ps)
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
         if (globalCellCenter[2] < maxParticleHeight && flagField->get(x, y, z) == fluidFlag) {
            sumOverlapFractions += BField->get(x, y, z);
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
                             const mesa_pd::Vec3& boxEdgeLength, const Vector3< real_t >& observationDomainSize)
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
      outFile_ << "# HydraulicGradient Uplift Subsidence" << std::endl;
   }

   ~UpliftSubsidenceEvaluator() { outFile_.close(); };

   void operator()(const real_t hydraulicGradient, const shared_ptr< ParticleAccessor_T >& accessor,
                   const shared_ptr< mesa_pd::data::ParticleStorage >& ps)
   {
      outFile_ << hydraulicGradient << " ";

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
