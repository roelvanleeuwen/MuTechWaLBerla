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
//! \author Mohammad Sanayei
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

#include "lbm/boundary/all.h"
#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/field/PdfField.h"
#include "lbm/lattice_model/D3Q19.h"
#include "lbm/sweeps/CellwiseSweep.h"
#include "lbm/sweeps/SweepWrappers.h"

namespace walberla::mesa_pd {


    int main(int argc, char** argv) {
        /// Setup
        Environment env(argc, argv);

        /// Configuration
        real_t diameter_SI {75e-5_r};
        real_t gravity_SI {5_r};
        real_t density_SI {1010_r};
        real_t frictionCoefficient {0.3_r};
        real_t restitutionCoefficient {0.9_r};
        real_t sphereGenerationSpacing_SI = 1e-3; // for particle generator, the distance between two particles. It should be more than particle diameter

        const Vector3<real_t> domainSize_SI {0.02_r, 0.02_r, 0.02_r}; //box 10 cm X 10cm X 10cm
        const Vector3<real_t> numberOfBlocksPerDirection {2_r,2_r,2_r}; // 8 blocks

        const uint_t numberOfTimeSteps {100000};
        const real_t timeStepSize_SI {1e-5_r}; // dt critical
        const uint_t vtkSpacing {50};
        const real_t terminalVelocityMagnitude_SI {0.01_r};
        const std::string vtkOutputFolder = "vtk_One_Pierre_presentation";
        const std::string collisionForceLoggingFile = "collisionForce.txt";


        ///LBM properties
        real_t cellsPerDiameter = real_t(10);
        real_t relaxationTime = real_t(0.65); // (0.5, \infty)
        real_t densityFluid_SI = real_t(1000);
        real_t kinematicViscosity_SI = real_t(1e-6);




        /// Simulation properties in lattice units
        real_t dx_SI = diameter_SI / cellsPerDiameter; // m
        real_t omega = real_t(1) / relaxationTime;
        real_t kinematicViscosity = lbm::collision_model::viscosityFromOmega(omega);//remove
        real_t dt_SI = kinematicViscosity / kinematicViscosity_SI * dx_SI * dx_SI; // s //?
        real_t gravitationalAcceleration = gravity_SI * dt_SI * dt_SI / dx_SI;
        real_t densityRatio = density_SI / densityFluid_SI;
        real_t densityParticle = densityRatio;
        real_t densityFluid = real_t(1);
        real_t diameter = diameter_SI / dx_SI;
        real_t sphereGenerationSpacing = sphereGenerationSpacing_SI/dx_SI;
//        real_t timeStepSize = real_t(1);
        Vector3<uint_t> domainSize( uint_c(domainSize_SI[0] / dx_SI), uint_c(domainSize_SI[1] / dx_SI), uint_c(domainSize_SI[2] / dx_SI));
        real_t volumeSphere = math::pi / real_t(6) * diameter * diameter * diameter;
        real_t buoyancyForce = gravitationalAcceleration * densityFluid * volumeSphere;
        real_t terminalVelocityMagnitude = terminalVelocityMagnitude_SI * dt_SI/dx_SI;
        real_t timeStepSize = 1;












        /// Blockforest
        auto domainAABB = math::AABB{Vector3<real_t>{0_r}, domainSize};
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
        real_t lcSpacing {2.1_r * diameter * 0.5_r};
        mesa_pd::data::LinkedCells lc{localDomain.getExtended(lcSpacing), lcSpacing};
        auto sphereShape = shapeStorage->create<data::Sphere>( diameter * 0.5_r );
        shapeStorage->shapes[sphereShape]->updateMassAndInertia(densityParticle);

        /// MESAPD Particles
        uint_t randomSeed = 1;
        std::mt19937 randomNumberGenerator{static_cast<unsigned int>(randomSeed)}; // rand()

        WALBERLA_CHECK(sphereGenerationSpacing > diameter, "Spacing should be larger than diameter!");

        for (auto& iBlk : *forest)
        {
            for (auto position : grid_generator::SCGrid{domainAABB,
                                                        Vector3<real_t>{sphereGenerationSpacing} * real_c(0.5),
                                                        sphereGenerationSpacing})
            {
                Vec3 positionOffset{math::realRandom<real_t>(-0.1_r, 0.1_r, randomNumberGenerator),
                                    math::realRandom<real_t>(-0.1_r, 0.1_r, randomNumberGenerator),
                                    math::realRandom<real_t>(-0.1_r, 0.1_r, randomNumberGenerator)};

                if(iBlk.getAABB().contains(position))
                {
                    auto particle = particleStorage->create();
                    particle->setPosition(position + positionOffset * sphereGenerationSpacing);
                    shapeStorage->shapes[sphereShape]->updateMassAndInertia(densityParticle);
                    particle->setInteractionRadius( diameter * 0.5_r );
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
        auto preParticleVtkWriter = walberla::vtk::createVTKOutput_PointData(preParticleOutput, "particles_settling", 1,
                                                                             vtkOutputFolder, "simulation_step");

        // MESAPD Kernels
        kernel::LinearSpringDashpot dem(2);
        const real_t collisionTime = real_t(10);
        const real_t poissonsRatio = 0.22_r;
        const real_t kappa = real_t(2) * ( real_t(1) - poissonsRatio ) / ( real_t(2) - poissonsRatio ) ;

        real_t massSphere = densityParticle * volumeSphere;

        real_t effectiveMass_SpherePlane = massSphere;
        real_t effectiveMass_SphereSphere = massSphere * massSphere / (real_t(2) * massSphere);

        //sphere-sphere
        dem.setStiffnessAndDamping(1, 1,restitutionCoefficient, collisionTime, kappa, effectiveMass_SphereSphere);
        dem.setFrictionCoefficientDynamic(1, 1, frictionCoefficient);

        // sphere-wall
        dem.setStiffnessAndDamping(0, 1,restitutionCoefficient, collisionTime, kappa, effectiveMass_SpherePlane);
        dem.setFrictionCoefficientDynamic(0, 1, frictionCoefficient);

        real_t gravitationalForce = gravitationalAcceleration * massSphere;
        mpi::ReduceProperty reduceProperty;
        mpi::ReduceContactHistory reduceAndSwapContactHistory;
        mpi::SyncNextNeighbors syncNextNeighbors;

//        kernel::SemiImplicitEuler particleIntegration(timeStepSize);
        mesa_pd::kernel::ExplicitEuler particleIntegration(timeStepSize);


        WcTimingTree timing;

        timing.start("Simulation");

        /// Simulation Loop 1
        for(uint_t t = 0; t < numberOfTimeSteps; ++t) {
            //Timestep log
            if(t % 100 == 0) {
                WALBERLA_LOG_INFO_ON_ROOT("Time step: " << t);
                WALBERLA_LOG_INFO_ON_ROOT("timeStepSize: " << timeStepSize);
                WALBERLA_LOG_INFO_ON_ROOT("terminalVelocityMagnitude: " << terminalVelocityMagnitude);
            }



            timing.start("Collision treatment");
            particleStorage->forEachParticlePairHalf(false, mesa_pd::kernel::ExcludeInfiniteInfinite(), *accessor,
                                                     [&dem, &mesapdDomain, timeStepSize]
                                                             (const size_t idx1, const size_t idx2, auto& ac)
                                                     {
                                                         mesa_pd::collision_detection::AnalyticContactDetection acd;
                                                         mesa_pd::kernel::DoubleCast double_cast;
                                                         mesa_pd::mpi::ContactFilter contact_filter;
                                                         if (double_cast(idx1, idx2, ac, acd, ac ))
                                                         {
                                                             if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *mesapdDomain))
                                                             {
                                                                 dem(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), acd.getContactNormal(), acd.getPenetrationDepth(), timeStepSize);
                                                             }
                                                         }
                                                     },
                                                     *accessor );
            timing.stop("Collision treatment");
            timing.start("Contact history");
            // synchronize collision information
            reduceAndSwapContactHistory(*particleStorage);
            timing.stop("Contact history");

            timing.start("Gravity");
            //add gravity to particles
            particleStorage->forEachParticle(false, kernel::SelectLocal(), *accessor,
                                             [gravitationalForce, buoyancyForce](const size_t idx, data::ParticleAccessorWithShape& ac){
                                                 mesa_pd::addForceAtomic(idx, ac, Vec3(real_t(0), real_t(0),-gravitationalForce + buoyancyForce));
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
//             break out of simulation when particles stop moving
            if(getMaximumParticleVelocityInSystem(*accessor,sphereShape) < terminalVelocityMagnitude && t > 200) {
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

        writeSpherePropertiesToFile(*accessor, "logging_spheres.txt", sphereShape);

        return EXIT_SUCCESS;
    }

}

int main( int argc, char* argv[] ) {
    return walberla::mesa_pd::main( argc, argv );
}
