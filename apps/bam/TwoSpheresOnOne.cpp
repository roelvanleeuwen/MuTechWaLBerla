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
//! \file
//! \author
//
//======================================================================================================================


#include <blockforest/Initialization.h>
#include <blockforest/StructuredBlockForest.h>

#include <mesa_pd/collision_detection/AnalyticContactDetection.h>

#include <mesa_pd/data/ParticleAccessorWithShape.h>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/ShapeStorage.h>
#include <mesa_pd/domain/BlockForestDomain.h>

#include <mesa_pd/kernel/DoubleCast.h>
#include <mesa_pd/kernel/CohesionInitialization.h>
#include <mesa_pd/kernel/Cohesion.h>
#include <mesa_pd/kernel/SpringDashpot.h>
#include <mesa_pd/kernel/SemiImplicitEuler.h>
#include <mesa_pd/mpi/ReduceContactHistory.h>
#include <mesa_pd/mpi/ReduceProperty.h>
#include <mesa_pd/mpi/SyncNextNeighbors.h>
#include <mesa_pd/mpi/notifications/ForceTorqueNotification.h>
#include <mesa_pd/mpi/ContactFilter.h>

#include <mesa_pd/vtk/OutputSelector.h>
#include <mesa_pd/vtk/ParticleVtkOutput.h>

#include "vtk/all.h"

#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <iostream>

#include "Utility.h"

namespace walberla {

using namespace walberla::mesa_pd;



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

int main( int argc, char ** argv )
{
   Environment env(argc, argv);
   WALBERLA_UNUSED(env);
   walberla::mpi::MPIManager::instance()->useWorldComm();

   real_t sphereRadius = 0.05_r; // m
   real_t sphereDensity = 2600_r; // kg / m^3
   real_t y_n = 200_r; // N = tensile force, where bond breaks

   uint_t numberOfTimeSteps = 100000;
   uint_t vtkSpacing = 1000;

   real_t dt = 1e-5_r; // s

   // simulation domain
   Vector3<uint_t> blocks(2,2,2);
   Vector3<real_t> minCorner(0_r);
   Vector3<real_t> maxCorner(1_r);
   auto forest = blockforest::createBlockForest(math::AABB(minCorner, maxCorner),
                                                blocks, Vector3<bool>(false, false, false));
   auto domainAABB = forest->getDomain();

   /// MESAPD Domain
   auto domain = std::make_shared<mesa_pd::domain::BlockForestDomain>(forest);
   auto localDomain = forest->begin()->getAABB();
   for (auto& blk : *forest) {
      localDomain.merge(blk.getAABB());
   }

   //init data structures
   auto ps = std::make_shared<data::ParticleStorage>(2);
   auto ss = std::make_shared<data::ShapeStorage>();
   mesa_pd::data::ParticleAccessorWithShape ac(ps, ss);

   real_t sphereMass = sphereDensity * math::pi * 4_r / 3_r * sphereRadius * sphereRadius * sphereRadius;
   auto sphereShape = ss->create<data::Sphere>( sphereRadius);
   ss->shapes[sphereShape]->updateMassAndInertia(sphereDensity);

   Vector3<real_t> centerPoint = (domainAABB.maxCorner() - domainAABB.minCorner()) / real_t(2);

   std::vector<Vec3> positions;
   positions.push_back(centerPoint);
   positions.push_back(centerPoint + Vec3(sphereRadius*1.9999, real_t(0), real_t(0)));
   positions.push_back(centerPoint + Vector3<real_t>(sphereRadius*1.9999_r*2_r, real_t(0), real_t(0)));
   positions.push_back(Vec3{centerPoint[0], centerPoint[1], sphereRadius});

   std::vector<walberla::id_t> sphereUids(positions.size());

   for (uint_t i = 0; i < positions.size(); ++i) {
      Vec3 pos = positions[i];
      if (domain->isContainedInProcessSubdomain(uint_c(walberla::mpi::MPIManager::instance()->rank()), pos)) {
         auto sphereParticle = ps->create();

         sphereParticle->setShapeID(sphereShape);
         sphereParticle->setType(0);
         sphereParticle->setPosition(pos);
         sphereParticle->setOwner(walberla::MPIManager::instance()->rank());
         sphereParticle->setInteractionRadius(sphereRadius);
         sphereUids[i] = sphereParticle->getUid();

         //if(i == positions.size()-1)  mesa_pd::data::particle_flags::set(sphereParticle->getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

         WALBERLA_LOG_INFO("particle created");
      }
   }
   walberla::mpi::reduceInplace(sphereUids, walberla::mpi::SUM);
   createPlane(*ps, *ss, domainAABB.minCorner(), Vector3<real_t>(0,0,1_r));

   // Init kernels
   kernel::CohesionInitialization cohesionInitKernel;
   kernel::Cohesion cohesionKernel(1);

   mesa_pd::mpi::ReduceContactHistory reduceAndSwapContactHistory;
   mesa_pd::mpi::ReduceProperty reduceProperty;
   mesa_pd::mpi::SyncNextNeighbors syncNextNeighbors;
   mesa_pd::mpi::ContactFilter contactFilter;

   kernel::SemiImplicitEuler particleIntegration(dt);
   SelectSphere sphereSelector;

   syncNextNeighbors(*ps, *domain);

   real_t E = 1e6_r; // kg / (m * s^2)
   real_t en = 0.2_r; // coefficient of restitution
   real_t kn = 2_r * E * (sphereRadius * sphereRadius / (sphereRadius + sphereRadius));
   real_t meff = sphereMass * sphereMass / (sphereMass + sphereMass);
   real_t damping = -std::log(en) / std::sqrt((std::log(en) * std::log(en) + math::pi * math::pi));
   real_t nun = 2_r * std::sqrt(kn * meff) * damping;

   WALBERLA_LOG_INFO("kn = " << kn << ", nun = " << nun);

   cohesionKernel.setKn(0,0,kn);
   cohesionKernel.setKsFactor(0,0,0_r);
   cohesionKernel.setKrFactor(0,0,0_r);
   cohesionKernel.setKoFactor(0,0,0_r);

   cohesionKernel.setNun(0,0,nun);
   cohesionKernel.setNusFactor(0,0,0_r);
   cohesionKernel.setNurFactor(0,0,0_r);
   cohesionKernel.setNuoFactor(0,0,0_r);

   cohesionKernel.setFrictionCoefficient(0,0,0.5_r);

   cohesionKernel.setYn(0,0,y_n);
   cohesionKernel.setYs(0,0,y_n);
   cohesionKernel.setYr(0,0,y_n);
   cohesionKernel.setYo(0,0,y_n);

   // vtk
   std::string vtkOutputFolder = "vtk_out_particles_cen";
   auto vtkOutput = make_shared<mesa_pd::vtk::ParticleVtkOutput>(ps);
   vtkOutput->addOutput<mesa_pd::data::SelectParticleInteractionRadius>("interactionRadius");
   vtkOutput->addOutput<data::SelectParticleLinearVelocity>("velocity");
   vtkOutput->setParticleSelector([sphereShape](const mesa_pd::data::ParticleStorage::iterator& pIt){
     return pIt->getShapeID() == sphereShape;
   });
   auto vtkWriter = walberla::vtk::createVTKOutput_PointData(vtkOutput, "particles_cen", 1, vtkOutputFolder, "simulation_step");

   // domain decomposition
   auto vtkDomainOutput = walberla::vtk::createVTKOutput_DomainDecomposition(forest, "domain_decomposition",
                                                                   uint_t(1), vtkOutputFolder, "simulation_step");
   vtkDomainOutput->write();

   // gravity
   Vector3<real_t> globalAcceleration(real_t(0), real_t(0), real_t(-9.81));
   auto addGravitationalForce = [&globalAcceleration, sphereMass](const size_t idx, mesa_pd::data::ParticleAccessorWithShape& ac_) {
     auto force = sphereMass * globalAcceleration;
     mesa_pd::addForceAtomic(idx, ac_, force);
   };

   //cohesion init
   bool openmp = false;
   ps->forEachParticlePairHalf(openmp, sphereSelector, ac,
                               [&](const size_t idx1, const size_t idx2){
                                 mesa_pd::collision_detection::AnalyticContactDetection acd;
                                 mesa_pd::kernel::DoubleCast double_cast;
                                 if (double_cast(idx1, idx2, ac, acd, ac)) {
                                    // particles overlap
                                    if (contactFilter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(),  *domain))
                                    {
                                       cohesionInitKernel(acd.getIdx1(), acd.getIdx2(), ac, acd.getPenetrationDepth());
                                    }
                                 }
                               });
   reduceAndSwapContactHistory(*ps);

   for(uint_t t = 0; t < numberOfTimeSteps; ++t) {

      if(t % vtkSpacing == 0){
         WALBERLA_LOG_INFO_ON_ROOT( "Timestep " << t << " / " << numberOfTimeSteps );
      }

      if(t % vtkSpacing == 0) {
         vtkWriter->write();
      }

      ps->forEachParticlePairHalf(openmp, kernel::SelectAll(), ac,
                                  [&](size_t idx1, size_t idx2){
                                    mesa_pd::collision_detection::AnalyticContactDetection acd;
                                    mesa_pd::kernel::DoubleCast double_cast;
                                    bool contactExists = double_cast(idx1, idx2, ac, acd, ac);

                                    Vector3<real_t> filteringPoint;
                                    if (contactExists)  {
                                       filteringPoint = acd.getContactPoint();
                                    } else {
                                       filteringPoint = (ac.getPosition(idx1) + ac.getPosition(idx2)) / real_t(2);
                                    }

                                    if (contactFilter(idx1, idx2, ac, filteringPoint, *domain))
                                    {
                                       bool contactTreatedByCohesionKernel = false;
                                       if (sphereSelector(idx1, idx2, ac))
                                       {
                                          if (cohesionKernel.isCohesiveBondActive(idx1, idx2, ac))
                                          { contactTreatedByCohesionKernel = cohesionKernel(idx1, idx2, ac, dt); }
                                       }
                                       if (contactExists && !contactTreatedByCohesionKernel)
                                       {
                                          cohesionKernel.nonCohesiveInteraction(
                                             acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(),
                                             acd.getContactNormal(), acd.getPenetrationDepth(), dt);
                                       }
                                    }

                                  });


      reduceAndSwapContactHistory(*ps);

      reduceProperty.operator()<mesa_pd::ForceTorqueNotification>(*ps);

      ps->forEachParticle(true, mesa_pd::kernel::SelectLocal(), ac, addGravitationalForce, ac);

      ps->forEachParticle(openmp, kernel::SelectLocal(), ac, particleIntegration, ac);

      syncNextNeighbors(*ps, *domain);

   }

   return EXIT_SUCCESS;
}

} //namespace walberla

int main( int argc, char ** argv )
{
   return walberla::main(argc, argv);
}
