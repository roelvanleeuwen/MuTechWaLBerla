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
//! \brief Based on showcases/Antidunes/Utility.cpp
//
//======================================================================================================================

#pragma once

#include "core/mpi/Broadcast.h"
#include "core/mpi/MPITextFile.h"
#include "core/mpi/Reduce.h"

#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/shape/Sphere.h"

#include <functional>
#include <iterator>

namespace walberla
{
namespace piping
{

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

bool sphereBucketOverlap(const mesa_pd::Vec3& spherePosition, const real_t sphereRadius,
                         const mesa_pd::Vec3& boxPosition, const mesa_pd::Vec3& boxEdgeLength)
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
      if (sphereBucketOverlap(position, radius, boxPosition, boxEdgeLength)) continue;

      auto pIt = ps.create();
      pIt->setPosition(position);
      pIt->getBaseShapeRef() = std::make_shared< data::Sphere >(radius);
      pIt->getBaseShapeRef()->updateMassAndInertia(particleDensity);
      pIt->setInteractionRadius(radius);
      pIt->setOwner(rank);
      pIt->setType(0);

      WALBERLA_CHECK_EQUAL(iss.tellg(), -1);
   }
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

} // namespace piping
} // namespace walberla
