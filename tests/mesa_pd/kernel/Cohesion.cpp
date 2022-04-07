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
#include <mesa_pd/mpi/ReduceContactHistory.h>

#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <iostream>

namespace walberla {

using namespace walberla::mesa_pd;

int main( int argc, char ** argv )
{

   Environment env(argc, argv);
   WALBERLA_UNUSED(env);
   walberla::mpi::MPIManager::instance()->useWorldComm();

   real_t sphereRadius = 2_r;
   real_t surfaceOverlap = 0.1_r* sphereRadius;
   real_t dt = 1_r;

   //init data structures
   auto ps = std::make_shared<data::ParticleStorage>(2);
   auto ss = std::make_shared<data::ShapeStorage>();

   real_t sphereDensity = 1000_r;
   real_t sphereMass = sphereDensity * math::pi * 4_r / 3_r * sphereRadius * sphereRadius * sphereRadius;
   auto sphere = ss->create<data::Sphere>( sphereRadius);
   ss->shapes[sphere]->updateMassAndInertia(sphereDensity);

   mesa_pd::data::ParticleAccessorWithShape ac(ps, ss);

   data::Particle&& p1 = *ps->create();
   p1.getPositionRef() = Vec3(0,0,0);
   p1.getShapeIDRef()  = sphere;
   p1.getTypeRef()     = 0;
   p1.getInteractionRadiusRef() = sphereRadius;

   data::Particle&& p2 = *ps->create();
   p2.getPositionRef() = Vec3(0,0,2_r * sphereRadius-surfaceOverlap);
   p2.getShapeIDRef()  = sphere;
   p2.getTypeRef()     = 0;
   p2.getInteractionRadiusRef() = sphereRadius;

   // Init kernels
   kernel::CohesionInitialization cohesionInitKernel;
   kernel::Cohesion cohesionKernel(1);
   mesa_pd::mpi::ReduceContactHistory reduceAndSwapContactHistory;

   real_t E = 1e6_r; // kg / (m * s^2)
   real_t en = 0.2_r; // coefficient of restitution
   real_t kn = 2_r * E * (sphereRadius * sphereRadius / (sphereRadius + sphereRadius));
   real_t meff = sphereMass * sphereMass / (sphereMass + sphereMass);
   real_t damping = 0_r; //-1_r * std::log(en) / std::sqrt((std::log(en) * std::log(en) + math::pi * math::pi));
   real_t nun = 2_r * std::sqrt(kn * meff) * damping;

   WALBERLA_LOG_INFO("kn = " << kn << ", nun = " << nun);

   real_t y_n = 100_r; // N
   real_t y_s = 100_r; // N

   cohesionKernel.setKn(0,0,kn);
   cohesionKernel.setNun(0,0,nun);
   cohesionKernel.setYn(0,0,y_n);

   cohesionKernel.setKsFactor(0,0,1_r);
   cohesionKernel.setNusFactor(0,0,1_r);
   cohesionKernel.setYs(0,0,y_s);

   cohesionKernel.setFrictionCoefficient(0,0,0.5_r);

   collision_detection::AnalyticContactDetection contact;
   kernel::DoubleCast       double_cast;
   bool contactExists = double_cast(0, 1, ac, contact, ac );
   WALBERLA_CHECK(contactExists);
   WALBERLA_LOG_INFO(contact);
   //WALBERLA_CHECK_FLOAT_EQUAL(contact.getPenetrationDepth(), -surfaceOverlap);

   //cohesion init
   cohesionInitKernel(0,1,ac,contact.getPenetrationDepth());
   reduceAndSwapContactHistory(*ps);

   WALBERLA_LOG_INFO("First cohesion kernel call");
   // call cohesion kernel -> no force
   cohesionKernel(contact.getIdx1(), contact.getIdx2(), ac,
                  contact.getContactPoint(),
                  contact.getContactNormal(),
                  contact.getPenetrationDepth(),
                  dt);
   reduceAndSwapContactHistory(*ps);

   WALBERLA_CHECK_FLOAT_EQUAL(ac.getForce(0).length(), 0_r);
   WALBERLA_CHECK_FLOAT_EQUAL(ac.getForce(1).length(), 0_r);

   // change sphere 2
   p2.setPosition(Vec3(0,0,2_r * sphereRadius-0.5_r*surfaceOverlap));
   p2.setLinearVelocity( Vec3(1_r, 0_r, 0_r));

   contactExists = double_cast(0, 1, ac, contact, ac );
   WALBERLA_LOG_INFO(contactExists << " " << contact);

   WALBERLA_LOG_INFO(p2);

   WALBERLA_LOG_INFO("Second kernel call");
   if( cohesionKernel.isCohesiveBondActive(0, 1, ac) )
   {
      WALBERLA_LOG_INFO(" -> cohesion kernel call");
      cohesionKernel(contact.getIdx1(), contact.getIdx2(),ac,
                     contact.getContactPoint(),
                     contact.getContactNormal(),
                     contact.getPenetrationDepth(),
                     dt);
   } else
   {
      WALBERLA_LOG_INFO(" -> non-cohesion kernel call");
      cohesionKernel.nonCohesiveInteraction(contact.getIdx1(), contact.getIdx2(), ac,
                                            contact.getContactPoint(),
                                            contact.getContactNormal(),
                                            contact.getPenetrationDepth(),
                                            dt);
   }
   reduceAndSwapContactHistory(*ps);


   //WALBERLA_LOG_INFO(p1);
   WALBERLA_LOG_INFO(p2);
   WALBERLA_CHECK_FLOAT_UNEQUAL(ac.getForce(0).length(), 0_r);
   WALBERLA_CHECK_FLOAT_UNEQUAL(ac.getForce(1).length(), 0_r);


   // call cohesion kernel -> break if force > threshold

   // single contact test


   return EXIT_SUCCESS;
}

} //namespace walberla

int main( int argc, char ** argv )
{
   return walberla::main(argc, argv);
}
