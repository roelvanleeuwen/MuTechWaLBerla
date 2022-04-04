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

#include "core/mpi/Broadcast.h"
#include "core/mpi/MPITextFile.h"
#include "core/mpi/Reduce.h"

#include "vtk/all.h"

#include <core/Environment.h>
#include <core/grid_generator/SCIterator.h>
#include <core/logging/Logging.h>
#include <core/math/Random.h>
#include <core/timing/TimingTree.h>
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
#include <mesa_pd/kernel/Cohesion.h>
#include <mesa_pd/kernel/CohesionInitialization.h>
#include <mesa_pd/kernel/DoubleCast.h>
#include <mesa_pd/kernel/ExplicitEuler.h>
#include <mesa_pd/kernel/InsertParticleIntoLinkedCells.h>
#include <mesa_pd/kernel/LinearSpringDashpot.h>
#include <mesa_pd/kernel/SemiImplicitEuler.h>
#include <mesa_pd/kernel/SpringDashpot.h>
#include <mesa_pd/kernel/SpringDashpotSpring.h>
#include <mesa_pd/mpi/ContactFilter.h>
#include <mesa_pd/mpi/ReduceContactHistory.h>
#include <mesa_pd/mpi/ReduceProperty.h>
#include <mesa_pd/mpi/SyncNextNeighbors.h>
#include <mesa_pd/mpi/notifications/ForceTorqueNotification.h>
#include <mesa_pd/vtk/OutputSelector.h>
#include <mesa_pd/vtk/ParticleVtkOutput.h>

#include "Utility.h"
#include "mesa_pd/data/Flags.h"
#include "mesa_pd/data/shape/Sphere.h"

namespace walberla::mesa_pd {


int main(int argc, char** argv) {
   /// Setup
   Environment env(argc, argv);

   /// Configuration
   real_t diameter_SI {3e-3_r};
   real_t gravity_SI {9.81_r};
   real_t density_SI {2610_r};
   real_t frictionCoefficient {0.3_r};
   real_t restitutionCoefficient {0.9_r};
   real_t sphereGenerationSpacing_SI = 4e-3; // for particle generator, the distance between two particles. It should be more than particle diameter

   const Vector3<real_t> domainSize_SI {0.02_r, 0.02_r, 0.02_r}; //box 10 cm X 10cm X 10cm
   const Vector3<real_t> numberOfBlocksPerDirection {2_r,2_r,2_r}; // 8 blocks

   const uint_t numberOfTimeSteps {50000};
   const real_t timeStepSize_SI {1e-5_r}; // dt critical
   const uint_t collisionTime {10};
   const uint_t vtkSpacing {100};
   const real_t terminalVelocityMagnitude {0.01_r};

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
   uint_t randomSeed = 1;
   std::mt19937 randomNumberGenerator{static_cast<unsigned int>(randomSeed)}; // rand()

   WALBERLA_CHECK(sphereGenerationSpacing_SI > diameter_SI, "Spacing should be larger than diameter!");

   for (auto& iBlk : *forest)
   {
      for (auto position : grid_generator::SCGrid{domainAABB,
                                                  Vector3<real_t>{sphereGenerationSpacing_SI} * real_c(0.5),
                                                  sphereGenerationSpacing_SI})
      {
         Vec3 positionOffset{math::realRandom<real_t>(-0.1_r, 0.1_r, randomNumberGenerator),
                             math::realRandom<real_t>(-0.1_r, 0.1_r, randomNumberGenerator),
                             math::realRandom<real_t>(-0.1_r, 0.1_r, randomNumberGenerator)};

         if(iBlk.getAABB().contains(position))
         {
            auto particle = particleStorage->create();
            particle->setPosition(position + positionOffset * sphereGenerationSpacing_SI);
            shapeStorage->shapes[sphereShape]->updateMassAndInertia(density_SI);
            particle->setInteractionRadius( diameter_SI * 0.5_r );
            particle->setShapeID(sphereShape);
            particle->setOwner(walberla::mpi::MPIManager::instance()->rank());
            particle->setType(1);
         }
      }
   }
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
   auto preParticleVtkWriter = walberla::vtk::createVTKOutput_PointData(preParticleOutput, "particles_pre", 1,
                                                                        vtkOutputFolder, "simulation_step");

   // MESAPD Kernels
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

   kernel::SemiImplicitEuler particleIntegration(timeStepSize_SI);

   /// Cohesion
   mesa_pd::kernel::Cohesion cohesion(timeStepSize_SI, E, damp, b_c);

   WALBERLA_LOG_INFO_ON_ROOT("Initializing cohesion bounds");
   lc.clear();
   particleStorage->forEachParticle(true, mesa_pd::kernel::SelectAll(), *accessor, insertIntoLinkedCells, *accessor, lc);


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
                                         // initialize cohesion to false
                                         // contact history of particle 1 -> particle 2
                                         auto& nch1 = accessor->getNewContactHistoryRef(idx1)[accessor->getUid(idx2)];
                                         // contact history of particle 2 -> particle 1
                                         auto& nch2 = accessor->getNewContactHistoryRef(idx2)[accessor->getUid(idx1)];
                                         // save for each of the particles that they are bound to the other by cohesion
                                         nch1.setCohesionBound(false);
                                         nch2.setCohesionBound(false);
                                      }
                                   }
                                }
                              });


   WcTimingTree timing;

   timing.start("Simulation");

   /// Simulation Loop 1
   for(uint_t t = 0; t < numberOfTimeSteps; ++t) {
      //Timestep log
      if(t % 100 == 0) {
         WALBERLA_LOG_INFO_ON_ROOT("Time step: " << t);
      }


      timing.start("Linked cells");
      // Prepare Data Structures
      lc.clear();
      particleStorage->forEachParticle(true, mesa_pd::kernel::SelectAll(), *accessor, insertIntoLinkedCells, *accessor, lc);
      timing.stop("Linked cells");

      timing.start("Collision treatment");
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
                                         } else if (!cohesiveContactTreated && contactExists) {
                                            // non-cohesive, but contact exists: normal dem has to run
                                            // this could also be handled inside the cohesion kernel, where we can implement our own non-cohesive dem
                                            // WALBERLA_LOG_INFO("Non-Cohesive contact between " << idx1 << " and " << idx2 << " treated.");

                                            //TODO: isn't this case already handled inside the cohesion kernel? So it might be called twice here?
                                            cohesion(idx1, idx2, *accessor,
                                                     contactExists, acd.getContactNormal(), acd.getPenetrationDepth());
//
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
      timing.stop("Collision treatment");

      timing.start("Contact history");
      // synchronize collision information
      reduceAndSwapContactHistory(*particleStorage);
      timing.stop("Contact history");

      timing.start("Gravity");
      //add gravity to particles
      particleStorage->forEachParticle(false, kernel::SelectLocal(), *accessor,
                                       [gravitationalForce](const size_t idx, data::ParticleAccessorWithShape& ac){
                                         mesa_pd::addForceAtomic(idx, ac, Vec3(real_t(0), real_t(0),-gravitationalForce));
                                       }, *accessor);
      timing.stop("Gravity");

      timing.start("Force Sync");
      // synchronize forces for mpi
      reduceProperty.operator()<mesa_pd::ForceTorqueNotification>(*particleStorage);
      timing.stop("Force Sync");

      timing.start("Integration");
      // update position and velocity
      particleStorage->forEachParticle(false, kernel::SelectLocal(), *accessor, particleIntegration, *accessor);
      timing.stop("Integration");

      timing.start("Particle Sync");
      // synchronize position and velocity
      syncNextNeighbors(*particleStorage, *mesapdDomain);
      timing.stop("Particle Sync");

      timing.start("VTK");
      if(t % vtkSpacing == 0) { preParticleVtkWriter->write();
      }
      timing.stop("VTK");

      timing.start("Termination check");
      // break out of simulation when particles stop moving
      if(getMaximumParticleVelocityInSystem(*accessor,sphereShape) < terminalVelocityMagnitude && t > 100) {
         WALBERLA_LOG_INFO_ON_ROOT("Terminal velocity reached, simulation ended!");
         timing.stop("Termination check");
         break;
      }
      timing.stop("Termination check");

      timing.start("Evaluation");
      //log particles maximum velocity
      if(t % 1000 == 0) {
         auto velocityMagnitude = getMaximumParticleVelocityInSystem(*accessor,sphereShape);
         WALBERLA_LOG_INFO_ON_ROOT("Current maximum velocity magnitude: " << velocityMagnitude);
      }
      timing.stop("Evaluation");
   }

   timing.stop("Simulation");

   auto reducedTT = timing.getReduced();
   WALBERLA_LOG_INFO_ON_ROOT(reducedTT);

   writeSpherePropertiesToFile(*accessor, "logging_sphere1.txt", sphereShape);

   return EXIT_SUCCESS;
}

}

int main( int argc, char* argv[] ) {
   return walberla::mesa_pd::main( argc, argv );
}
