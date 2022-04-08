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
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/mpi/MPIManager.h"
#include "core/mpi/Broadcast.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "core/mpi/MPITextFile.h"

namespace walberla {
namespace mesa_pd {


template< typename ParticleAccessor_T>
real_t getMaximumParticleVelocityInSystem(ParticleAccessor_T & accessor, size_t sphereShape)
{
   real_t maximumVelocityMagnitude{0};
   for(uint_t idx = 0; idx < accessor.size(); ++idx)
   {
      if(accessor.getShapeID(idx) == sphereShape)
      {
         real_t particleVelocityMagnitude = accessor.getLinearVelocity(idx).length();
         maximumVelocityMagnitude = std::max(maximumVelocityMagnitude,particleVelocityMagnitude);
      }
   }

   walberla::mpi::allReduceInplace(maximumVelocityMagnitude, walberla::mpi::MAX);
   return maximumVelocityMagnitude;
}

template< typename ParticleAccessor_T>
real_t getMaximumSphereHeightInSystem(ParticleAccessor_T & accessor, size_t sphereShape)
{
   real_t maximumHeight{0};
   for(uint_t idx = 0; idx < accessor.size(); ++idx)
   {
      if(accessor.getShapeID(idx) == sphereShape)
      {
         real_t height = accessor.getPosition(idx)[2];
         maximumHeight = std::max(maximumHeight,height);
      }
   }

   walberla::mpi::allReduceInplace(maximumHeight, walberla::mpi::MAX);

   return maximumHeight;
}

mesa_pd::data::ParticleStorage::iterator createPlane( mesa_pd::data::ParticleStorage& ps,
                                                      mesa_pd::data::ShapeStorage& ss,
                                                      const Vector3<real_t>& pos,
                                                      const Vector3<real_t>& normal ) {
   auto p0              = ps.create(true);
   p0->getPositionRef() = pos;
   p0->getShapeIDRef()  = ss.create<mesa_pd::data::HalfSpace>( normal );
   p0->getOwnerRef()    = walberla::mpi::MPIManager::instance()->rank();
   p0->getTypeRef()     = 0;
   mesa_pd::data::particle_flags::set(p0->getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
   mesa_pd::data::particle_flags::set(p0->getFlagsRef(), mesa_pd::data::particle_flags::FIXED);
   mesa_pd::data::particle_flags::set(p0->getFlagsRef(), mesa_pd::data::particle_flags::NON_COMMUNICATING);
   return p0;
}

class ExcludeGlobalGlobal
{
 public:
   template <typename Accessor>
   bool operator()(const size_t idx, const size_t jdx, Accessor& ac) const
   {
      using namespace data::particle_flags;
      return !(isSet(ac.getFlags(idx), GLOBAL) && isSet(ac.getFlags(jdx), GLOBAL));
   }
};

class SelectSphere
{
 public:
   template <typename Accessor>
   bool operator()(const size_t idx, Accessor& ac) const {
      return ac.getShape(idx)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE;
   }

   template <typename Accessor>
   bool operator()(const size_t idx1, const size_t idx2, Accessor& ac) const {
      return ac.getShape(idx1)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE &&
             ac.getShape(idx2)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE;
   }
};

class SelectBoxEdgeLengths
{
 public:
   using return_type = walberla::mesa_pd::Vec3;

   SelectBoxEdgeLengths(const mesa_pd::data::ShapeStorage& ss) : ss_(ss) {}

   walberla::mesa_pd::Vec3 operator()(mesa_pd::data::Particle&& p) const {
      const auto& shape = ss_.shapes[p.getShapeID()];
      if (shape->getShapeType() == mesa_pd::data::Box::SHAPE_TYPE) {
         return static_cast<mesa_pd::data::Box*>(ss_.shapes[p->getShapeID()].get())->getEdgeLength();
      } else {
         return mesa_pd::Vec3(0_r);
      }
   }
   walberla::mesa_pd::Vec3 operator()(const mesa_pd::data::Particle& p) const {
      const auto& shape = ss_.shapes[p.getShapeID()];
      if (shape->getShapeType() == mesa_pd::data::Box::SHAPE_TYPE) {
         return static_cast<mesa_pd::data::Box*>(ss_.shapes[p.getShapeID()].get())->getEdgeLength();
      } else {
         return mesa_pd::Vec3(0_r);
      }
   }

 private:
   const mesa_pd::data::ShapeStorage& ss_; //TODO change this!
};

class SelectRotation
{
 public:
   using return_type = walberla::mesa_pd::Vec3;

   walberla::mesa_pd::Vec3 operator()(mesa_pd::data::Particle&& p) const {
      return p.getRotation().getMatrix() * Vector3<real_t>(1_r, 0_r, 0_r);
   }
   walberla::mesa_pd::Vec3 operator()(const mesa_pd::data::Particle& p) const {
      return p.getRotation().getMatrix() * Vector3<real_t>(1_r, 0_r, 0_r);
   }
};


//saving spheres position to txt file
template< typename ParticleAccessor_T>
void writeSpherePropertiesToFile(ParticleAccessor_T & accessor, std::string fileName,size_t sphereShape)
{
   std::ostringstream ossData;

   for (uint_t idx = 0; idx < accessor.size(); ++idx)
   {
      if(accessor.getShapeID(idx) == sphereShape)
      {
         if(!isSet(accessor.getFlags(idx), mesa_pd::data::particle_flags::GHOST))
         {
            auto uid = accessor.getUid(idx);
            auto position = accessor.getPosition(idx);
            auto radii    = accessor.getInteractionRadius(idx);
            ossData << uid << " " << position[0] << " " << position[1] << " " << position[2] << " " << radii << "\n";
         }
      }
   }
   walberla::mpi::writeMPITextFile( fileName, ossData.str() );
}

void initSpheresFromFile(const std::string& filename,
                         walberla::mesa_pd::data::ParticleStorage& ps,
                         const walberla::mesa_pd::domain::IDomain& domain,
                         size_t sphereShape)
{
   using namespace walberla;
   using namespace walberla::mesa_pd;
   using namespace walberla::mesa_pd::data;

   auto rank = walberla::mpi::MPIManager::instance()->rank();

   std::string textFile;

   WALBERLA_ROOT_SECTION()
   {
      std::ifstream t( filename.c_str() );
      if( !t )
      {
         WALBERLA_ABORT("Invalid input file " << filename << "\n");
      }
      std::stringstream buffer;
      buffer << t.rdbuf();
      textFile = buffer.str();
   }

   walberla::mpi::broadcastObject( textFile );

   std::istringstream fileIss( textFile );
   std::string line;

   while( std::getline( fileIss, line ) )
   {
      std::istringstream iss( line );

      data::ParticleStorage::uid_type      uID;
      data::ParticleStorage::position_type pos;
      walberla::real_t radius;
      iss >> uID >> pos[0] >> pos[1] >> pos[2] >> radius;

      if (!domain.isContainedInProcessSubdomain(uint_c(rank), pos)) continue;

      auto pIt = ps.create();
      pIt->setPosition(pos);
      //pIt->getBaseShapeRef() = std::make_shared<data::Sphere>(radius);
      //pIt->getBaseShapeRef()->updateMassAndInertia(density);
      pIt->setShapeID(sphereShape);
      pIt->setInteractionRadius( radius );
      pIt->setOwner(rank);
      pIt->setType(1);

      WALBERLA_CHECK_EQUAL( iss.tellg(), -1);
   }
}


} // namespace mesa_pd
} // namespace walberla
