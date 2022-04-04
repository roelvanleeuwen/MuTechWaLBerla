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

// it work slowly when i decrease the size of particles
//! \file   Cohesion.cpp
//! \author Lukas Werner
//
//======================================================================================================================

#include "blockforest/Initialization.h"
#include "blockforest/StructuredBlockForest.h"

#include "vtk/all.h"

#include <core/Environment.h>
#include <core/grid_generator/SCIterator.h>
#include <core/logging/Logging.h>

#include <mesa_pd/data/DataTypes.h>
#include <mesa_pd/data/Flags.h>
#include <mesa_pd/data/LinkedCells.h>
#include <mesa_pd/data/shape/HalfSpace.h>
#include <mesa_pd/data/ParticleAccessorWithShape.h>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/ShapeStorage.h>

#include <mesa_pd/kernel/DoubleCast.h>
#include <mesa_pd/kernel/InsertParticleIntoLinkedCells.h>
#include <mesa_pd/kernel/SpringDashpot.h>
#include <mesa_pd/kernel/SpringDashpotSpring.h>
#include <mesa_pd/kernel/LinearSpringDashpot.h>
#include <mesa_pd/kernel/ExplicitEuler.h>
#include <mesa_pd/kernel/ParticleSelector.h>
#include <mesa_pd/kernel/CohesionInitialization.h>
#include <mesa_pd/kernel/Cohesion.h>

#include <mesa_pd/mpi/ContactFilter.h>
#include <mesa_pd/mpi/ReduceContactHistory.h>
#include <mesa_pd/mpi/ReduceProperty.h>
#include <mesa_pd/mpi/notifications/ForceTorqueNotification.h>

#include <mesa_pd/vtk/ParticleVtkOutput.h>
#include <mesa_pd/domain/BlockForestDomain.h>
#include <mesa_pd/mpi/SyncNextNeighbors.h>
#include <mesa_pd/collision_detection/GeneralContactDetection.h>
#include <mesa_pd/common/ParticleFunctions.h>

namespace walberla {

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

    int main( int argc, char ** argv ) {
        /// Setup
        Environment env(argc, argv);


        auto sphereRadius = real_t(0.05);
        auto sphereDensity = real_t(2600);
        auto maxCohesionDistance = sphereRadius;
        uint_t visSpacing = 1000;
        std::string vtk_out = "vtk_benchmark";
        uint_t simulationSteps = 100000;

        const auto dt = real_t(1e-5); //real_t(1e-5);
        const auto E = real_t(1e6);
        const auto b_c = real_t(2);
        const auto en = real_t(0.2);
        const auto damp = -log(en) / sqrt(log(en)*log(en) + math::pi*math::pi);

        /// BlockForest
        Vector3<uint_t> blocks(2, 2, 2);
        Vector3<real_t> minCorner(real_t(0));
        Vector3<real_t> maxCorner(real_t(4), real_t(1), real_t(4));
        auto forest = blockforest::createBlockForest(math::AABB(minCorner, maxCorner),
                                                     blocks, Vector3<bool>(false, false, false));
        auto domainAABB = forest->getDomain();

        /// MESAPD Domain
        auto domain = std::make_shared<mesa_pd::domain::BlockForestDomain>(forest);

        auto localDomain = forest->begin()->getAABB();
        for (auto& blk : *forest) {
            localDomain.merge(blk.getAABB());
        }

        /// MESAPD Data
        auto ps = std::make_shared<mesa_pd::data::ParticleStorage>(4);
        auto ss = std::make_shared<mesa_pd::data::ShapeStorage>();
        mesa_pd::data::ParticleAccessorWithShape ac(ps, ss);
        // Initialize the size of a linked cell such that
        // (1) a possible contact between two spheres is assuredly detected and
        // (2) a possible cohesive bound between two non-contacting spheres is still detected.
        mesa_pd::data::LinkedCells lc(localDomain.getExtended(real_t(1)),
                                      real_t(2.1) * sphereRadius + maxCohesionDistance);
        mesa_pd::mpi::SyncNextNeighbors SNN;

        /// MESAPD Particles

        // Spheres
        auto sphereShapeId = ss->create<mesa_pd::data::Sphere>(sphereRadius);
        ss->shapes[sphereShapeId]->updateMassAndInertia(real_t(sphereDensity));

        Vector3<real_t> centerPoint = (domainAABB.maxCorner() - domainAABB.minCorner()) / real_t(2);
//   Vector3<real_t> centerPoint (real_t(0.2),real_t(0.2),real_t(0.2));
        auto sphereOffset = sphereRadius - real_t(0.05);
        std::vector<Vector3<real_t>> positions;
        positions.push_back(centerPoint);
        positions.push_back(centerPoint + Vector3<real_t>(sphereRadius*1.9997, real_t(0), real_t(0)));
//   positions.push_back(centerPoint + Vector3<real_t>(sphereRadius + sphereOffset, real_t(0), real_t(0)));
//   positions.push_back(centerPoint - Vector3<real_t>(real_t(0), real_t(0), sphereRadius*real_t(2.5)));



        for (auto & pos : positions) {
            if (domain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), pos)) {
                auto sphereParticle = ps->create();

                sphereParticle->setShapeID(sphereShapeId);
                sphereParticle->setPosition(pos);
                sphereParticle->setOwner(walberla::MPIManager::instance()->rank());
                sphereParticle->setInteractionRadius(sphereRadius);
                auto particleID = sphereParticle->getUid();


                WALBERLA_LOG_INFO("particle created");
            }
        }

//   Vector3<real_t> particle3 = centerPoint - Vector3<real_t>(real_t(0), real_t(0), sphereRadius*real_t(2.5));
        Vector3<real_t> position3 = centerPoint - Vector3<real_t>(real_t(0), real_t(0), sphereRadius*real_t(2.5));

        walberla::id_t specialUid = 0;
        if (domain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), position3)) {
            auto specialParticle = ps->create();
            specialParticle->setShapeID(sphereShapeId);
            specialParticle->setPosition(position3);
            specialParticle->setOwner(walberla::MPIManager::instance()->rank());
            specialParticle->setInteractionRadius(sphereRadius);
            specialUid = specialParticle->getUid();
        }
        mpi::reduceInplace(specialUid, mpi::SUM);


        int64_t numParticles = int64_c(ps->size());
        mpi::reduceInplace(numParticles, mpi::SUM);
        WALBERLA_LOG_INFO_ON_ROOT("#particles created: " << numParticles);

        // Confining Planes
        auto planeShift = (real_t(100) - sphereRadius - sphereRadius) * real_t(0.5);
        auto shift = Vector3<real_t>(real_t(0.01)); //??
        auto confiningDomain = domainAABB.getExtended(planeShift);
        if (!forest->isPeriodic(0)) {
            createPlane(*ps, *ss, confiningDomain.minCorner() + shift, Vector3<real_t>(+1,0,0));
            createPlane(*ps, *ss, confiningDomain.maxCorner() + shift, Vector3<real_t>(-1,0,0));
        }
        if (!forest->isPeriodic(1)) {
            createPlane(*ps, *ss, confiningDomain.minCorner() + shift, Vector3<real_t>(0,+1,0));
            createPlane(*ps, *ss, confiningDomain.maxCorner() + shift, Vector3<real_t>(0,-1,0));
        }
        if (!forest->isPeriodic(2)) {
            createPlane(*ps, *ss, confiningDomain.minCorner() + shift, Vector3<real_t>(0,0,+1));
            createPlane(*ps, *ss, confiningDomain.maxCorner() + shift, Vector3<real_t>(0,0,-1));
        }

        SNN(*ps, *domain);

        /// VTK Output
        // domain output
        auto vtkDomainOutput = vtk::createVTKOutput_DomainDecomposition(forest, "domain_decomposition",
                                                                        uint_t(1), vtk_out, "simulation_step");
        vtkDomainOutput->write();
        // mesapd particle output
        auto particleVtkOutput = make_shared<mesa_pd::vtk::ParticleVtkOutput>(ps);
        particleVtkOutput->addOutput<mesa_pd::data::SelectParticleInteractionRadius>("interactionRadius");
        particleVtkOutput->setParticleSelector([sphereShapeId](const mesa_pd::data::ParticleStorage::iterator& pIt){
            return pIt->getShapeID() == sphereShapeId;
        });
        auto particleVtkWriter = walberla::vtk::createVTKOutput_PointData(particleVtkOutput, "particles", visSpacing,
                                                                          vtk_out, "simulation_step");


        /// MESAPD kernels
        mesa_pd::collision_detection::GeneralContactDetection gcd;
        mesa_pd::kernel::ExplicitEuler explicitEuler(dt);
        mesa_pd::kernel::InsertParticleIntoLinkedCells ipilc;
        mesa_pd::kernel::DoubleCast double_cast;
        mesa_pd::mpi::ReduceProperty RP;
        mesa_pd::mpi::ReduceContactHistory RCH;
        mesa_pd::mpi::ContactFilter contact_filter;
//   mesa_pd::kernel::SpringDashpotSpring sds(1); // this DEM kernel does not take into account rolling
        mesa_pd::kernel::LinearSpringDashpot sds(1);

        real_t collisionTime_SI = 20 * real_t(1e-5) ;
        real_t restitutionCoefficient =  0.9;
        real_t frictionCoefficient = 0.1;
        const real_t poissonsRatio = real_t(0.22);
        const real_t kappa = real_t(2) * ( real_t(1) - poissonsRatio ) / ( real_t(2) - poissonsRatio ) ;

        real_t volumeSphere_SI = math::pi / real_t(3) * real_t(4) * sphereRadius * sphereRadius * sphereRadius;
        real_t massSphere_SI = sphereDensity * volumeSphere_SI;
        real_t effectiveMass_SphereSphere = massSphere_SI * massSphere_SI / (real_t(2) * massSphere_SI);

        sds.setStiffnessAndDamping(0,0,restitutionCoefficient, collisionTime_SI, kappa, effectiveMass_SphereSphere);
        sds.setFrictionCoefficientDynamic(0,0,frictionCoefficient);

        // those parameters might not be correct / maybe something else
//   real_t mass = sphereDensity * real_t(4)/real_t(3) * sphereRadius * sphereRadius * sphereRadius * math::pi;
//   sds.setParametersFromCOR(0, 0, real_t(0.9), dt*real_t(20), mass * real_t(0.5));
//   sds.setCoefficientOfFriction(0,0,real_t(0.3));
//   sds.setStiffnessT(0,0,real_t(0.5) * sds.getStiffnessN(0,0));

        mesa_pd::kernel::CohesionInitialization cohesionInitialization;
        mesa_pd::kernel::Cohesion cohesion(dt, E, damp, b_c);

        Vector3<real_t> globalAcceleration(real_t(0), real_t(0), real_t(-9.81));
        auto addGravitationalForce = [&globalAcceleration, massSphere_SI, specialUid](const size_t idx, mesa_pd::data::ParticleAccessorWithShape& ac_) {
            if (ac_.getUid(idx) == specialUid) return; // don't add gravitational acceleration to particle 3 to have the other ones drop onto it
            auto force = massSphere_SI * globalAcceleration;
            mesa_pd::addForceAtomic(idx, ac_, force);
        };





//        // Initialize cohesion bounds
//        WALBERLA_LOG_INFO_ON_ROOT("Initializing cohesion bounds");
//        lc.clear();
//        ps->forEachParticle(true, mesa_pd::kernel::SelectAll(), ac, ipilc, ac, lc);
//
//        lc.forEachParticlePairHalf(false, mesa_pd::kernel::SelectAll(), ac, [&](const size_t idx1, const size_t idx2){
//            // call the general contact detection kernel (gcd) for particles with idx1 and idx2
//            if(double_cast(idx1, idx2, ac, gcd, ac)) {
//                // particles overlap
//                // check if the overlap should be treated on this process to avoid duplicate calculations
//                if (contact_filter(gcd.getIdx1(), gcd.getIdx2(), ac, gcd.getContactPoint(), *domain)) {
//                    WALBERLA_LOG_INFO("Found cohesive particles");
//                    // initialize cohesion
//                    cohesionInitialization(idx1, idx2, ac, gcd.getPenetrationDepth());
//                }
//            }
//        });
//
//        RCH(*ps);
//        SNN(*ps, *domain);
// Initialize Cohesion
        WALBERLA_LOG_INFO_ON_ROOT("Initializing cohesion bounds");
        lc.clear();
        ps->forEachParticle(true, mesa_pd::kernel::SelectAll(), ac, ipilc, ac, lc);

        lc.forEachParticlePairHalf(false, mesa_pd::kernel::SelectAll(), ac,
                                   [&](const size_t idx1, const size_t idx2){
                                       // call the general contact detection kernel (gcd) for particles with idx1 and idx2
                                       mesa_pd::collision_detection::AnalyticContactDetection gcd;
                                       mesa_pd::kernel::DoubleCast double_cast;
                                       mesa_pd::mpi::ContactFilter contact_filter;
                                       if(ac.getShape(idx1)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE &&
                                          ac.getShape(idx2)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE) {
                                           if (double_cast(idx1, idx2, ac, gcd, ac)) {
                                               // particles overlap
                                               // check if the overlap should be treated on this process to avoid duplicate calculations
                                               if (contact_filter(gcd.getIdx1(), gcd.getIdx2(), ac, gcd.getContactPoint(),
                                                                  *domain)) {
                                                   WALBERLA_LOG_INFO("Found cohesive particles");
                                                   // initialize cohesion
                                                   cohesionInitialization(idx1, idx2, ac, gcd.getPenetrationDepth());
                                               }
                                           }
                                       }
                                   });

        RCH(*ps);
        SNN(*ps, *domain);

        // Time loop

        for (uint_t i = 0; i < simulationSteps; ++i) {
            if(i % visSpacing == 0){
                WALBERLA_LOG_INFO_ON_ROOT( "Timestep " << i << " / " << simulationSteps );
            }

            // VTK
            particleVtkWriter->write();

            // Prepare Data Structures
            lc.clear();
            ps->forEachParticle(true, mesa_pd::kernel::SelectAll(), ac, ipilc, ac, lc);

            // Collision Resolution
//            lc.forEachParticlePairHalf(true, mesa_pd::kernel::SelectAll(), ac,
//                                       [&](const size_t idx1, const size_t idx2){
//                                           // call the general contact detection kernel (gcd) for particles with idx1 and idx2
//                                           bool contactExists = double_cast(idx1, idx2, ac, gcd, ac);
//                                           Vector3<real_t> filteringPoint; // the point which is used to decide which process should handle those two particles
//                                           if (contactExists)  {
//                                               // use the contact point, if both particles overlap
//                                               filteringPoint = gcd.getContactPoint();
//                                           } else {
//                                               // use the center point between the non-overlapping particles
//                                               filteringPoint = (ac.getPosition(idx1) + ac.getPosition(idx2)) / real_t(2);
//                                           }
//                                           // based on filteringPoint, check if the overlap should be treated on this process to avoid duplicate calculations
//                                           if (contact_filter(gcd.getIdx1(), gcd.getIdx2(), ac, filteringPoint, *domain)) {
//                                               // let the cohesive kernel check for cohesion and handle it (returns true if it did)
//                                               if (cohesion(idx1, idx2, ac,
//                                                            contactExists, gcd.getContactNormal(), gcd.getPenetrationDepth())) {
//                                                   // cohesive contact has been treated
////               WALBERLA_LOG_INFO("Cohesive contact between " << idx1 << " and " << idx2 << " treated.");
//                                               } else if (contactExists) {
//                                                   // non-cohesive, but contact exists: normal dem has to run
//                                                   // this could also be handled inside the cohesion kernel, where we can implement our own non-cohesive dem
////               WALBERLA_LOG_INFO("Non-Cohesive contact between " << idx1 << " and " << idx2 << " treated.");
//                                                   sds(gcd.getIdx1(), gcd.getIdx2(), ac, gcd.getContactPoint(),
//                                                       gcd.getContactNormal(), gcd.getPenetrationDepth(), dt);
//                                               }
//                                           }
//                                       });
//            RP.operator()<mesa_pd::ForceTorqueNotification>(*ps);
//            RCH(*ps);


            lc.forEachParticlePairHalf(true, mesa_pd::kernel::SelectAll(), ac,
                                       [&](const size_t idx1, const size_t idx2){
                                           mesa_pd::collision_detection::AnalyticContactDetection acd;
                                           mesa_pd::kernel::DoubleCast double_cast;
                                           mesa_pd::mpi::ContactFilter contact_filter;

                                           // call the general contact detection kernel (gcd) for particles with idx1 and idx2
                                           bool contactExists = double_cast(idx1, idx2, ac, acd, ac);

                                           if(ac.getShape(idx1)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE &&
                                              ac.getShape(idx2)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE) {
                                               // two spheres -> might be a cohesive contact, requires special treatment
                                               Vector3<real_t> filteringPoint; // the point which is used to decide which process should handle those two particles
                                               if (contactExists)  {
                                                   // use the contact point, if both particles overlap
                                                   filteringPoint = acd.getContactPoint();
                                               } else {
                                                   // use the center point between the non-overlapping particles
                                                   filteringPoint = (ac.getPosition(idx1) + ac.getPosition(idx2)) / real_t(2);
                                               }
                                               // based on filteringPoint, check if the overlap should be treated on this process to avoid duplicate calculations
                                               if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, filteringPoint, *domain)) {
                                                   // let the cohesive kernel check for cohesion and handle it (returns true if it did)
                                                   bool cohesiveContactTreated = cohesion(idx1, idx2, ac,
                                                                                          contactExists, acd.getContactNormal(), acd.getPenetrationDepth());
                                                   if (cohesiveContactTreated) {
                                                       // cohesive contact has been treated
                                                       // WALBERLA_LOG_INFO("Cohesive contact between " << idx1 << " and " << idx2 << " treated.");
                                                   } else if (contactExists) {
                                                       // non-cohesive, but contact exists: normal dem has to run
                                                       // this could also be handled inside the cohesion kernel, where we can implement our own non-cohesive dem
                                                       // WALBERLA_LOG_INFO("Non-Cohesive contact between " << idx1 << " and " << idx2 << " treated.");


                                                       cohesion(idx1, idx2, ac,
                                                                contactExists, acd.getContactNormal(), acd.getPenetrationDepth());
//                                            auto force = accessor->getForce(idx1);
//                                            dem(acd.getIdx1(), acd.getIdx2(), *accessor, acd.getContactPoint(),
//                                                acd.getContactNormal(), acd.getPenetrationDepth(), timeStepSize_SI);
                                                   }
                                               }
                                           } else {
                                               // sphere and other geometry -> always non-cohesive
                                               if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *domain)) {
                                                   sds(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(),
                                                       acd.getContactNormal(), acd.getPenetrationDepth(), dt);
                                               }
                                           }
                                       });



            // synchronize collision information
            RCH(*ps);

            // Force Application
            ps->forEachParticle(true, mesa_pd::kernel::SelectLocal(), ac, addGravitationalForce, ac);

            ps->forEachParticle(true, mesa_pd::kernel::SelectLocal(), ac, [](const size_t idx, mesa_pd::data::ParticleAccessorWithShape& ac_){
//        WALBERLA_LOG_INFO(ac_.getUid(idx) << ": pos = " << ac_.getPosition(idx) << ", f = " << ac_.getForce(idx));
            }, ac);

            // Integration
            ps->forEachParticle(true, mesa_pd::kernel::SelectLocal(), ac,
                                [specialUid](const size_t idx, mesa_pd::data::ParticleAccessorWithShape& ac_) {
                                    if (ac_.getUid(idx) == specialUid) {
                                        ac_.setLinearVelocity(idx, Vector3<real_t>(0_r, 0_r, 0_r));
                                        ac_.setAngularVelocity(idx, Vector3<real_t>(0_r, 0_r, 0_r));
                                    }

                                }, ac);
            ps->forEachParticle(true, mesa_pd::kernel::SelectLocal(), ac, explicitEuler, ac);

            SNN(*ps, *domain);
        }

        return EXIT_SUCCESS;
    }
}

int main( int argc, char* argv[] ) {
    return walberla::main( argc, argv );
}
