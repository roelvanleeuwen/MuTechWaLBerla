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
//! \file   BucketCohesion.cpp
//! \author Lukas Werner
//
//======================================================================================================================

#include "blockforest/Initialization.h"
#include "blockforest/StructuredBlockForest.h"

#include "vtk/all.h"

#include <core/Environment.h>
#include <core/grid_generator/SCIterator.h>
#include <core/logging/Logging.h>
#include <core/math/Random.h>
#include <mesa_pd/collision_detection/GeneralContactDetection.h>
#include <mesa_pd/common/ParticleFunctions.h>
#include <mesa_pd/data/DataTypes.h>
#include <mesa_pd/data/Flags.h>
#include <mesa_pd/data/LinkedCells.h>
#include <mesa_pd/data/ParticleAccessorWithShape.h>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/ShapeStorage.h>
#include <mesa_pd/data/shape/Box.h>
#include <mesa_pd/data/shape/HalfSpace.h>
#include <mesa_pd/domain/BlockForestDomain.h>
#include <mesa_pd/kernel/DoubleCast.h>
#include <mesa_pd/kernel/SemiImplicitEuler.h>
#include <mesa_pd/kernel/InsertParticleIntoLinkedCells.h>
#include <mesa_pd/kernel/LinearSpringDashpot.h>
#include <mesa_pd/kernel/SpringDashpot.h>
#include <mesa_pd/kernel/SpringDashpotSpring.h>
#include <mesa_pd/kernel/CohesionInitialization.h>
#include <mesa_pd/kernel/Cohesion.h>
#include <mesa_pd/mpi/ContactFilter.h>
#include <mesa_pd/mpi/ReduceContactHistory.h>
#include <mesa_pd/mpi/ReduceProperty.h>
#include <mesa_pd/mpi/SyncNextNeighbors.h>
#include <mesa_pd/mpi/notifications/ForceTorqueNotification.h>
#include <mesa_pd/vtk/OutputSelector.h>
#include <mesa_pd/vtk/ParticleVtkOutput.h>
#include "mesa_pd/data/Flags.h"
#include "mesa_pd/data/shape/Sphere.h"
#include "core/mpi/Broadcast.h"
#include "core/mpi/MPITextFile.h"
#include "core/mpi/Reduce.h"

#include "Utility.h"

namespace walberla::mesa_pd {

int main(int argc, char** argv) {
   /// Setup
   Environment env(argc, argv);

   /// Configuration
   real_t diameter_SI {3e-3_r};
   real_t gravity_SI {5_r};
   real_t density_SI {2610_r};
   real_t frictionCoefficient {0.3_r};
   real_t restitutionCoefficient {0.9_r};

   const Vector3<real_t> domainSize_SI {0.02_r, 0.02_r, 0.02_r}; //box 10 cm X 10cm X 10cm
   const Vector3<real_t> numberOfBlocksPerDirection {2_r,2_r,2_r}; // 8 blocks

   const uint_t numberOfTimeSteps {20000};
   const real_t timeStepSize_SI {1e-5_r}; // dt critical
   const uint_t collisionTime {10};
   const uint_t vtkSpacing {100};

   const real_t E {1e5_r};
   const real_t b_c {2_r};
   const real_t en {.2_r};
   const real_t damp {-log(en) / sqrt(log(en)*log(en) + math::pi*math::pi)};



   const std::string vtkOutputFolder = "vtk_coh";
   const std::string collisionForceLoggingFile = "collisionForce.txt";

   /// Blockforest
   auto domainAABB = math::AABB{Vector3<real_t>{0_r}, domainSize_SI};

   auto forest = blockforest::createBlockForest(domainAABB, numberOfBlocksPerDirection, Vector3<bool>(false),
                                                uint_c(walberla::mpi::MPIManager::instance()->numProcesses()),
                                                uint_t(0), false);
   /// MESAPD Domain
   auto mesapdDomain = std::make_shared<mesa_pd::domain::BlockForestDomain>(forest);
   auto localDomain = forest->begin()->getAABB();
   for (auto& blk : *forest) {
      localDomain.merge(blk.getAABB());
   }

   /// MESAPD Data
   auto shapeStorage = std::make_shared<data::ShapeStorage>();
   auto particleStorage = std::make_shared<data::ParticleStorage>(1);
   auto accessor = std::make_shared<data::ParticleAccessorWithShape>(particleStorage, shapeStorage);
   // Initialize the size of a linked cell such that
   // (1) a possible contact between two spheres is assuredly detected and
   // (2) a possible cohesive bound between two non-contacting spheres is still detected.
   real_t lcSpacing {2.1_r * diameter_SI * 0.5_r};
   mesa_pd::data::LinkedCells lc{localDomain.getExtended(lcSpacing), lcSpacing};
   auto sphereShape = shapeStorage->create<data::Sphere>( diameter_SI * 0.5_r );
   shapeStorage->shapes[sphereShape]->updateMassAndInertia(density_SI);

   /// MESAPD Particles

   initSpheresFromFile("logging_sphere1.txt", *particleStorage, *mesapdDomain, sphereShape);

   int64_t numParticles = int64_c(particleStorage->size());
   walberla::mpi::reduceInplace(numParticles, walberla::mpi::SUM);
   WALBERLA_LOG_INFO_ON_ROOT("Created " << numParticles << " particles.");

   /// MESAPD Planes
   createPlane(*particleStorage, *shapeStorage, domainAABB.minCorner(), Vec3(1_r, 0_r, 0_r));
   createPlane(*particleStorage, *shapeStorage, domainAABB.minCorner(), Vec3(0_r, 1_r, 0_r));
   createPlane(*particleStorage, *shapeStorage, domainAABB.minCorner(), Vec3(0_r, 0_r, 1_r));
   createPlane(*particleStorage, *shapeStorage, domainAABB.maxCorner(), Vec3(-1_r, 0_r, 0_r));
   createPlane(*particleStorage, *shapeStorage, domainAABB.maxCorner(), Vec3(0_r, -1_r, 0_r));
   createPlane(*particleStorage, *shapeStorage, domainAABB.maxCorner(), Vec3(0_r, 0_r, -1_r));

   /// VTK Output
   // domain output
   auto vtkDomainOutput = walberla::vtk::createVTKOutput_DomainDecomposition(forest, "domain_decomposition",
                                                                             uint_t(1), vtkOutputFolder, "simulation_step");
   vtkDomainOutput->write();

   // mesapd particle output
   auto preParticleOutput = make_shared<mesa_pd::vtk::ParticleVtkOutput>(particleStorage);
   preParticleOutput->addOutput<mesa_pd::data::SelectParticleInteractionRadius>("interactionRadius");
   preParticleOutput->addOutput<data::SelectParticleLinearVelocity>("velocity");
   preParticleOutput->setParticleSelector([sphereShape](const mesa_pd::data::ParticleStorage::iterator& pIt){
     return pIt->getShapeID() == sphereShape;
   });
   auto postParticleVtkWriter = walberla::vtk::createVTKOutput_PointData(preParticleOutput, "particles_coh", 1,
                                                                        vtkOutputFolder, "simulation_step");

   /// MESAPD Kernels
   kernel::LinearSpringDashpot dem(2);
   const real_t collisionTime_SI = collisionTime * timeStepSize_SI;
   const real_t poissonsRatio = 0.22_r;
   const real_t kappa = 2_r * ( 1_r - poissonsRatio ) / ( 2_r - poissonsRatio ) ;

   real_t volumeSphere_SI = math::pi / 6_r * diameter_SI * diameter_SI * diameter_SI;
   real_t massSphere_SI = density_SI * volumeSphere_SI;
   real_t effectiveMass_SpherePlane = massSphere_SI;

   // sphere-wall
   dem.setStiffnessAndDamping(0, 1,restitutionCoefficient, collisionTime_SI, kappa, effectiveMass_SpherePlane);
   dem.setFrictionCoefficientDynamic(0, 1, frictionCoefficient);

   real_t gravitationalForce = gravity_SI * massSphere_SI;

   mpi::ReduceProperty reduceProperty;
   mpi::ReduceContactHistory reduceAndSwapContactHistory;
   mpi::SyncNextNeighbors syncNextNeighbors;
   kernel::InsertParticleIntoLinkedCells insertIntoLinkedCells;

   kernel::SemiImplicitEuler particleIntegration(timeStepSize_SI); //verlet to update particles position and velocity

   /// Cohesion
   mesa_pd::kernel::Cohesion cohesion(timeStepSize_SI, E, damp, b_c);
   mesa_pd::kernel::CohesionInitialization cohesionInitialization;

   // Initialize Cohesion
   WALBERLA_LOG_INFO_ON_ROOT("Initializing cohesion bounds");
   lc.clear();
   particleStorage->forEachParticle(false, mesa_pd::kernel::SelectAll(), *accessor, insertIntoLinkedCells, *accessor, lc);

   lc.forEachParticlePairHalf(false, mesa_pd::kernel::SelectAll(), *accessor,
                              [&](const size_t idx1, const size_t idx2){
                                // call the general contact detection kernel (gcd) for particles with idx1 and idx2
                                mesa_pd::collision_detection::AnalyticContactDetection acd;
                                mesa_pd::kernel::DoubleCast double_cast;
                                mesa_pd::mpi::ContactFilter contact_filter;
                                if(accessor->getShape(idx1)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE &&
                                   accessor->getShape(idx2)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE) {
                                   if (double_cast(idx1, idx2, *accessor, acd, *accessor)) {
                                      // particles overlap
                                      // check if the overlap should be treated on this process to avoid duplicate calculations
                                      if (contact_filter(acd.getIdx1(), acd.getIdx2(), *accessor, acd.getContactPoint(),
                                                         *mesapdDomain)) {
                                         WALBERLA_LOG_INFO("Found cohesive particles");
                                         // initialize cohesion
                                         cohesionInitialization(idx1, idx2, *accessor, acd.getPenetrationDepth());
                                      }
                                   }
                                }
                              });

   reduceAndSwapContactHistory(*particleStorage);
   syncNextNeighbors(*particleStorage, *mesapdDomain);


   /// Simulation Loop 2
   for(uint_t t = 0; t < numberOfTimeSteps; ++t) {
      // Prepare Data Structures
      lc.clear();
      particleStorage->forEachParticle(true, mesa_pd::kernel::SelectAll(), *accessor, insertIntoLinkedCells, *accessor, lc);

      //collision (particle-particle & particle-movingBox)
      // compute collision forces -> DEM
      // Collision Resolution
      lc.forEachParticlePairHalf(true, mesa_pd::kernel::SelectAll(), *accessor,
                                 [&](const size_t idx1, const size_t idx2){
                                   mesa_pd::collision_detection::AnalyticContactDetection acd;
                                   mesa_pd::kernel::DoubleCast double_cast;
                                   mesa_pd::mpi::ContactFilter contact_filter;

                                   // call the general contact detection kernel (gcd) for particles with idx1 and idx2
                                   bool contactExists = double_cast(idx1, idx2, *accessor, acd, *accessor);

                                   if(accessor->getShape(idx1)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE &&
                                      accessor->getShape(idx2)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE) {
                                      // two spheres -> might be a cohesive contact, requires special treatment
                                      Vector3<real_t> filteringPoint; // the point which is used to decide which process should handle those two particles
                                      if (contactExists)  {
                                         // use the contact point, if both particles overlap
                                         filteringPoint = acd.getContactPoint();
                                      } else {
                                         // use the center point between the non-overlapping particles
                                         filteringPoint = (accessor->getPosition(idx1) + accessor->getPosition(idx2)) / real_t(2);
                                      }
                                      // based on filteringPoint, check if the overlap should be treated on this process to avoid duplicate calculations
                                      if (contact_filter(acd.getIdx1(), acd.getIdx2(), *accessor, filteringPoint, *mesapdDomain)) {
                                         // let the cohesive kernel check for cohesion and handle it (returns true if it did)
                                         bool cohesiveContactTreated = cohesion(idx1, idx2, *accessor,
                                                                                contactExists, acd.getContactNormal(), acd.getPenetrationDepth());
                                         if (cohesiveContactTreated) {
                                            // cohesive contact has been treated
                                            // WALBERLA_LOG_INFO("Cohesive contact between " << idx1 << " and " << idx2 << " treated.");
                                         } else if (contactExists) {
                                            // non-cohesive, but contact exists: normal dem has to run
                                            // this could also be handled inside the cohesion kernel, where we can implement our own non-cohesive dem
                                            // WALBERLA_LOG_INFO("Non-Cohesive contact between " << idx1 << " and " << idx2 << " treated.");


                                            cohesion(idx1, idx2, *accessor,
                                                     contactExists, acd.getContactNormal(), acd.getPenetrationDepth());
//                                            auto force = accessor->getForce(idx1);
//                                            dem(acd.getIdx1(), acd.getIdx2(), *accessor, acd.getContactPoint(),
//                                                acd.getContactNormal(), acd.getPenetrationDepth(), timeStepSize_SI);
                                         }
                                      }
                                   } else {
                                      // sphere and other geometry -> always non-cohesive
                                      if (contact_filter(acd.getIdx1(), acd.getIdx2(), *accessor, acd.getContactPoint(), *mesapdDomain)) {
                                         dem(acd.getIdx1(), acd.getIdx2(), *accessor, acd.getContactPoint(),
                                             acd.getContactNormal(), acd.getPenetrationDepth(), timeStepSize_SI);
                                      }
                                   }
                                 });



      // synchronize collision information
      reduceAndSwapContactHistory(*particleStorage);

      // add gravitational force
      particleStorage->forEachParticle(false, kernel::SelectLocal(), *accessor, [gravitationalForce](const size_t idx, data::ParticleAccessorWithShape& ac){mesa_pd::addForceAtomic(idx, ac, Vec3(real_t(0), real_t(0),-gravitationalForce));},*accessor);



      // synchronize forces
      reduceProperty.operator()<mesa_pd::ForceTorqueNotification>(*particleStorage);

      // update position and velocity
      // .. of spheres
      particleStorage->forEachParticle(false, kernel::SelectLocal(), *accessor, particleIntegration, *accessor);
      // .. of bucket
//      updateBucketPosition(*accessor, bucketElementUids, timeStepSize_SI);

      // synchronize position and velocity
      syncNextNeighbors(*particleStorage, *mesapdDomain);

      //saving vtk of the box and particles
      if(t % vtkSpacing == 0) {
         postParticleVtkWriter->write();
      }

      if(t % 1000 == 0){
         WALBERLA_LOG_INFO_ON_ROOT("Time step: " << t);
      }
   }

   return EXIT_SUCCESS;
}

}

int main( int argc, char* argv[] ) {
   return walberla::mesa_pd::main( argc, argv );
}
