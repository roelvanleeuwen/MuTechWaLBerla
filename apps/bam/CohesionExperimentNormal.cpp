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

#include <mesa_pd/collision_detection/AnalyticContactDetection.h>

#include <mesa_pd/data/ParticleAccessorWithShape.h>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/ShapeStorage.h>

#include <mesa_pd/kernel/DoubleCast.h>
#include <mesa_pd/kernel/CohesionInitialization.h>
#include <mesa_pd/kernel/Cohesion.h>
#include <mesa_pd/kernel/SpringDashpot.h>
#include <mesa_pd/kernel/SemiImplicitEuler.h>
#include <mesa_pd/mpi/ReduceContactHistory.h>
#include <mesa_pd/mpi/ReduceProperty.h>
#include <mesa_pd/mpi/SyncNextNeighbors.h>
#include <mesa_pd/mpi/notifications/ForceTorqueNotification.h>

#include <mesa_pd/vtk/OutputSelector.h>
#include <mesa_pd/vtk/ParticleVtkOutput.h>

#include "vtk/all.h"

#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <iostream>

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

   real_t sphereRadius = 2e-3_r; // m
   real_t y_n = 0.4_r; // N = tensile force, where bond breaks

   uint_t numberOfTimeSteps = 100;

   real_t surfaceOverlap = 1e-6_r; // m
   real_t dt = 1e-5_r; // s

   //init data structures
   auto ps = std::make_shared<data::ParticleStorage>(2);
   auto ss = std::make_shared<data::ShapeStorage>();

   real_t sphereDensity = 2600_r; // kg / m^3
   real_t sphereMass = sphereDensity * math::pi * 4_r / 3_r * sphereRadius * sphereRadius * sphereRadius;
   auto sphereShape = ss->create<data::Sphere>( sphereRadius);
   ss->shapes[sphereShape]->updateMassAndInertia(sphereDensity);

   mesa_pd::data::ParticleAccessorWithShape ac(ps, ss);

   data::Particle&& p1 = *ps->create();
   p1.getPositionRef() = Vec3(0,0,0);
   p1.getShapeIDRef()  = sphereShape;
   p1.getTypeRef()     = 0;
   p1.getInteractionRadiusRef() = sphereRadius;
   mesa_pd::data::particle_flags::set(p1.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

   data::Particle&& p2 = *ps->create();
   p2.getPositionRef() = Vec3(0,0,2_r * sphereRadius-surfaceOverlap);
   p2.getShapeIDRef()  = sphereShape;
   p2.getTypeRef()     = 0;
   p2.getInteractionRadiusRef() = sphereRadius;

   // Init kernels
   kernel::CohesionInitialization cohesionInitKernel;
   kernel::Cohesion cohesionKernel(1);

   mesa_pd::mpi::ReduceContactHistory reduceAndSwapContactHistory;
   mesa_pd::mpi::ReduceProperty reduceProperty;
   mesa_pd::mpi::SyncNextNeighbors syncNextNeighbors;

   kernel::SemiImplicitEuler particleIntegration(dt);
   SelectSphere sphereSelector;

   real_t E = 1e6_r; // kg / (m * s^2)
   real_t en = 0.2_r; // coefficient of restitution
   real_t kn = 2_r * E * (sphereRadius * sphereRadius / (sphereRadius + sphereRadius));
   real_t meff = sphereMass * sphereMass / (sphereMass + sphereMass);
   real_t damping = -std::log(en) / std::sqrt((std::log(en) * std::log(en) + math::pi * math::pi));
   real_t nun = 2_r * std::sqrt(kn * meff) * damping;

   WALBERLA_LOG_INFO("kn = " << kn << ", nun = " << nun);

   cohesionKernel.setKn(0,0,kn);
   cohesionKernel.setNun(0,0,nun);
   cohesionKernel.setYn(0,0,y_n);
   cohesionKernel.setYs(0,0,y_n);
   cohesionKernel.setYr(0,0,y_n);
   cohesionKernel.setYo(0,0,y_n);


   // vtk
   uint_t vtkSpacing = 10;
   std::string vtkOutputFolder = "vtk_out_particles_cen";
   auto vtkOutput = make_shared<mesa_pd::vtk::ParticleVtkOutput>(ps);
   vtkOutput->addOutput<mesa_pd::data::SelectParticleInteractionRadius>("interactionRadius");
   vtkOutput->addOutput<data::SelectParticleLinearVelocity>("velocity");
   vtkOutput->setParticleSelector([sphereShape](const mesa_pd::data::ParticleStorage::iterator& pIt){
     return pIt->getShapeID() == sphereShape;
   });
   auto vtkWriter = walberla::vtk::createVTKOutput_PointData(vtkOutput, "particles_cen", 1, vtkOutputFolder, "simulation_step");


   //cohesion init
   bool openmp = false;
   ps->forEachParticlePairHalf(openmp, sphereSelector, ac,
                               [&](const size_t idx1, const size_t idx2){
                                 mesa_pd::collision_detection::AnalyticContactDetection acd;
                                 mesa_pd::kernel::DoubleCast double_cast;
                                 if (double_cast(idx1, idx2, ac, acd, ac)) {
                                    // particles overlap
                                    cohesionInitKernel(idx1, idx2, ac, acd.getPenetrationDepth());
                                 }
                               });
   reduceAndSwapContactHistory(*ps);


   real_t currentExternalForce = 0_r;
   real_t externalForceIncrease = 2_r * y_n / real_c(numberOfTimeSteps);
   for(uint_t t = 0; t < numberOfTimeSteps; ++t) {


      WALBERLA_LOG_INFO_ON_ROOT("Time step: " << t);

      mesa_pd::addForceAtomic(1, ac, Vec3(0_r, 0_r, currentExternalForce));

      ps->forEachParticlePairHalf(openmp, sphereSelector, ac,
                                  [&](size_t idx1, size_t idx2){
                                    mesa_pd::collision_detection::AnalyticContactDetection acd;
                                    mesa_pd::kernel::DoubleCast double_cast;
                                    bool contactExists = double_cast(idx1, idx2, ac, acd, ac);
                                    bool contactTreatedByCohesionKernel = false;
                                    if( cohesionKernel.isCohesiveBondActive(idx1, idx2, ac) )
                                    {
                                       contactTreatedByCohesionKernel = cohesionKernel(idx1, idx2,ac, dt);
                                    }
                                    if(contactExists && !contactTreatedByCohesionKernel)
                                    {
                                       cohesionKernel.nonCohesiveInteraction(acd.getIdx1(), acd.getIdx2(), ac,
                                                                             acd.getContactPoint(),
                                                                             acd.getContactNormal(),
                                                                             acd.getPenetrationDepth(),
                                                                             dt);
                                    }
                                  });


      reduceAndSwapContactHistory(*ps);

      reduceProperty.operator()<mesa_pd::ForceTorqueNotification>(*ps);


      WALBERLA_LOG_INFO_ON_ROOT("Sphere 2: " << p2.getOldContactHistoryRef()[p1.getUid()].getCohesionBound()
                                             << " " << p2.getPosition() << " " << p2.getLinearVelocity()
                                             << " " << p2.getForce());


      ps->forEachParticle(openmp, kernel::SelectLocal(), ac, particleIntegration, ac);

      //TODO sync position and velocity

      if(t % vtkSpacing == 0) {
         vtkWriter->write();
      }

      currentExternalForce += externalForceIncrease;
   }

   return EXIT_SUCCESS;
}

} //namespace walberla

int main( int argc, char ** argv )
{
   return walberla::main(argc, argv);
}
