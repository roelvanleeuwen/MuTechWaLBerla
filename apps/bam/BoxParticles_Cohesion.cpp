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

//real_t x = ((real_t)rand()) / ((real_t)RAND_MAX) / 1.5 + 0.6; // between 0.6 and 1.2
//
//! \file   PlatePenetration.cpp
//! \author Mohammad
//
//======================================================================================================================


#include <mesa_pd/vtk/ParticleVtkOutput.h>
#include <mesa_pd/collision_detection/AnalyticContactDetection.h>
#include <mesa_pd/common/ParticleFunctions.h>
#include <mesa_pd/data/LinkedCells.h>
#include <mesa_pd/data/ParticleAccessorWithShape.h>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/ShapeStorage.h>
#include <mesa_pd/domain/BlockForestDomain.h>
#include <mesa_pd/kernel/AssocToBlock.h>
#include <mesa_pd/kernel/DoubleCast.h>
#include <mesa_pd/kernel/ExplicitEulerWithShape.h>
#include <mesa_pd/kernel/InsertParticleIntoLinkedCells.h>
#include <mesa_pd/kernel/ParticleSelector.h>
#include <mesa_pd/kernel/LinearSpringDashpot.h>


#include <blockforest/BlockForest.h>
#include <blockforest/Initialization.h>
#include <core/Abort.h>
#include <core/Environment.h>
#include <core/math/Random.h>

#include <core/grid_generator/SCIterator.h>
#include <core/logging/Logging.h>
#include <core/OpenMP.h>
#include <core/timing/Timer.h>
#include <core/waLBerlaBuildInfo.h>
#include <vtk/VTKOutput.h>

#include "mesa_pd/mpi/SyncNextNeighbors.h"
#include "mesa_pd/mpi/ReduceProperty.h"
#include "mesa_pd/mpi/ReduceContactHistory.h"
#include "mesa_pd/mpi/ContactFilter.h"
#include "mesa_pd/mpi/notifications/ForceTorqueNotification.h"

#include "core/mpi/MPIManager.h"
#include "core/mpi/Broadcast.h"
#include "mesa_pd/data/ParticleStorage.h"
#include <mesa_pd/kernel/CohesionInitialization.h>
#include <mesa_pd/kernel/Cohesion.h>




namespace walberla {
    namespace mesa_pd {

        class ExcludeGlobalGlobal
        {
        public:
            template <typename Accessor>
            bool operator()(const size_t idx, const size_t jdx, Accessor& ac) const
            {
                using namespace walberla::mesa_pd::data::particle_flags;
                if (isSet(ac.getFlags(idx), GLOBAL) && isSet(ac.getFlags(jdx), GLOBAL)) return false;
                return true;
            }
        };


        data::ParticleStorage::iterator createPlane( data::ParticleStorage& ps,
                                                     data::ShapeStorage& ss,
                                                     const Vec3& pos,
                                                     const Vec3& normal )
        {
            auto p0              = ps.create(true);
            p0->getPositionRef() = pos;
            p0->getShapeIDRef()  = ss.create<data::HalfSpace>( normal );
            p0->getOwnerRef()    = walberla::mpi::MPIManager::instance()->rank();
            p0->getTypeRef()     = 0;
            data::particle_flags::set(p0->getFlagsRef(), data::particle_flags::INFINITE);
            data::particle_flags::set(p0->getFlagsRef(), data::particle_flags::FIXED);
            data::particle_flags::set(p0->getFlagsRef(), data::particle_flags::NON_COMMUNICATING);
            return p0;
        }

        real_t getMaximumParticleVelocityInSystem(data::ParticleAccessorWithShape & accessor, size_t sphereShape)
        {
            real_t maximumVelocityMagnitude = real_t(0);
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

        real_t getMaximumSphereHeightInSystem(data::ParticleAccessorWithShape & accessor, size_t sphereShape)
        {
            real_t maximumHeight = real_t(0);
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

//new
        data::ParticleStorage::iterator createBox( data::ParticleStorage& ps,
                                                   const Vec3& position,
                                                   size_t boxShape)
        {
            auto p0              = ps.create(true);
            p0->getPositionRef() = position;
            p0->getShapeIDRef()  = boxShape;
            p0->getOwnerRef()    = walberla::mpi::MPIManager::instance()->rank();
            p0->getTypeRef()     = 0;
            data::particle_flags::set(p0->getFlagsRef(), data::particle_flags::NON_COMMUNICATING);
            data::particle_flags::set(p0->getFlagsRef(), data::particle_flags::FIXED);
            return p0;
        }

        Vec3 getCollisionForceOnBox(data::ParticleAccessorWithShape & accessor, walberla::id_t boxUID)
        {
            auto boxIdx = accessor.uidToIdx(boxUID);
            WALBERLA_CHECK(boxIdx != accessor.getInvalidIdx());
            auto force = accessor.getForce(boxIdx);

            walberla::mpi::allReduceInplace(force[0], walberla::mpi::SUM);
            walberla::mpi::allReduceInplace(force[1], walberla::mpi::SUM);
            walberla::mpi::allReduceInplace(force[2], walberla::mpi::SUM);

            return force;
        }

        void writeCollisionForceToFile(uint_t timestep, Vec3 force, std::string fileName)
        {
            WALBERLA_ROOT_SECTION()
            {
                std::ofstream file;
                if(timestep == 0) file.open( fileName.c_str() );
                else file.open( fileName.c_str(), std::ofstream::app );

                file << timestep << " " << force[0] << " " << force[1] << " " << force[2] << "\n";
                file.close();
            }
        }

        void updateBoxPosition(data::ParticleAccessorWithShape & accessor, walberla::id_t boxUID, real_t dt)
        {
            auto boxIdx = accessor.uidToIdx(boxUID);
            WALBERLA_CHECK(boxIdx != accessor.getInvalidIdx());
            auto newBoxPosition = accessor.getPosition(boxIdx) + dt * accessor.getLinearVelocity(boxIdx);
            accessor.setPosition(boxIdx, newBoxPosition);
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



        int main( int argc, char ** argv )
        {
            //environment configuration
            Environment env(argc, argv);

            //input data
            real_t diameter_SI {3e-3};
            real_t gravity_SI {9.81};
            real_t density_SI {2610};
            real_t frictionCoefficient = 0.3;
            real_t restitutionCoefficient =  0.9;
            real_t sphereGenerationSpacing_SI = 4e-3; // for particle generator, the distance between two particles distance between particles, it should be more than particle daimeter
            const Vector3<real_t> domainSize_SI {0.02_r, 0.02_r, 0.02_r};
            const Vector3<real_t> numberOfBlocksPerDirection (real_t(2),real_t(2),real_t(2));
            const std::string vtkOutputFolder = "vtk";
            const uint_t numberOfTimeSteps {10000};
            const real_t timeStepSize_SI {5e-5}; // dt critical
            const uint_t collisionTime {10};
            const uint_t vtkSpacing {10};
            const real_t  terminalVelocityMagnitude {0.01};
            const Vector3<real_t> boxEdgeLengths (real_t(0.005),real_t(0.005),real_t(0.1));
            Vector3<real_t> boxPosition(real_t(domainSize_SI[0]/2),real_t(domainSize_SI[1]/2),real_t(0.15));
            Vector3<real_t> boxVelocity(real_t(0),real_t(0),real_t(-0.05));
            const std::string collisionForceLoggingFile = "collisionForce.txt";

            const real_t E {1e5_r};
            const real_t b_c {0.02_r};
            const real_t en {.2_r};
            const real_t damp {-log(en) / sqrt(log(en)*log(en) + math::pi*math::pi)};


            //creating the domain:

            auto domainAABB = math::AABB(Vector3<real_t>(real_t(0)), domainSize_SI); // uses two points

            auto forest = blockforest::createBlockForest(domainAABB, numberOfBlocksPerDirection, Vector3<bool>(false),
                                                         uint_c(walberla::mpi::MPIManager::instance()->numProcesses()), uint_t(0), false);
            auto mesapdDomain = std::make_shared<domain::BlockForestDomain>(forest); //pointer

//...................





            auto shapeStorage = std::make_shared<data::ShapeStorage>();
            auto particleStorage = std::make_shared<data::ParticleStorage>(1);
            auto accessor = std::make_shared<data::ParticleAccessorWithShape>(particleStorage, shapeStorage);
            auto sphereShape = shapeStorage->create<data::Sphere>( diameter_SI * real_t(0.5) );
            shapeStorage->shapes[sphereShape]->updateMassAndInertia(density_SI);

            real_t lcSpacing {2.1_r * diameter_SI * 0.5_r};
            auto localDomain = forest->begin()->getAABB();
            for (auto& blk : *forest) {
                localDomain.merge(blk.getAABB());
            }
            mesa_pd::data::LinkedCells lc{localDomain.getExtended(lcSpacing), lcSpacing};
            kernel::InsertParticleIntoLinkedCells insertIntoLinkedCells;










            uint_t randomSeed = 1; // uint_c(std::chrono::system_clock::now().time_since_epoch().count());
            //mpi::broadcastObject(randomSeed); // root process chooses seed and broadcasts it
            std::mt19937 randomNumberGenerator(static_cast<unsigned int>(randomSeed));

            ///particles
            initSpheresFromFile("logging_sphere1.txt", *particleStorage, *mesapdDomain, sphereShape);
            int64_t numParticles = int64_c(particleStorage->size());
            walberla::mpi::reduceInplace(numParticles, walberla::mpi::SUM);
            WALBERLA_LOG_INFO_ON_ROOT("Created " << numParticles << " particles.");





            // create bounding planes
            createPlane(*particleStorage, *shapeStorage, domainAABB.minCorner(), Vec3(real_t(1), real_t(0), real_t(0)));
            createPlane(*particleStorage, *shapeStorage, domainAABB.minCorner(), Vec3(real_t(0), real_t(1), real_t(0)));
            createPlane(*particleStorage, *shapeStorage, domainAABB.minCorner(), Vec3(real_t(0), real_t(0), real_t(1)));
            createPlane(*particleStorage, *shapeStorage, domainAABB.maxCorner(), Vec3(real_t(-1), real_t(0), real_t(0)));
            createPlane(*particleStorage, *shapeStorage, domainAABB.maxCorner(), Vec3(real_t(0), real_t(-1), real_t(0)));
            createPlane(*particleStorage, *shapeStorage, domainAABB.maxCorner(), Vec3(real_t(0), real_t(0), real_t(-1)));

            auto vtkDomainOutput = walberla::vtk::createVTKOutput_DomainDecomposition( forest, "domain_decomposition", 1, vtkOutputFolder, "simulation_step" );
            auto vtkOutput       = make_shared<mesa_pd::vtk::ParticleVtkOutput>(particleStorage) ; // pointer with ParticleVtkOutput type points to particlestorage
            auto vtkWriter       = walberla::vtk::createVTKOutput_PointData(vtkOutput, "Bodies", 1, vtkOutputFolder, "simulation_step", false, false);
            vtkOutput->addOutput<data::SelectParticleInteractionRadius>("radius"); //we can add another things like rotation and velocity
            vtkOutput->addOutput<data::SelectParticleLinearVelocity>("velocity");
            vtkDomainOutput->write();
            vtkWriter->write();



            mesa_pd::kernel::LinearSpringDashpot dem(2);
            /// Cohesion
            mesa_pd::kernel::Cohesion cohesion(timeStepSize_SI, E, damp, b_c);
            mesa_pd::kernel::CohesionInitialization cohesionInitialization;

            const double collisionTime_SI = collisionTime * timeStepSize_SI;
            const double poissonsRatio = double(0.22);
            const double kappa = 2. * ( 1. - poissonsRatio ) / ( 2. - poissonsRatio ) ;

            double volumeSphere_SI = math::pi / 6. * diameter_SI * diameter_SI * diameter_SI;
            double massSphere_SI = density_SI * volumeSphere_SI;
            double effectiveMass_SpherePlane = massSphere_SI;
            double effectiveMass_SphereSphere = massSphere_SI * massSphere_SI / (real_t(2) * massSphere_SI);

            //sphere- wall
            dem.setStiffnessAndDamping(0,1,restitutionCoefficient, collisionTime_SI, kappa, effectiveMass_SpherePlane);
            dem.setFrictionCoefficientDynamic(0,1,frictionCoefficient);

            //sphere-sphere
            dem.setStiffnessAndDamping(1,1,restitutionCoefficient, collisionTime_SI, kappa, effectiveMass_SphereSphere);
            dem.setFrictionCoefficientDynamic(1,1,frictionCoefficient);

            double gravitationalForce = gravity_SI * massSphere_SI;

            mpi::ReduceProperty reduceProperty;
            mpi::ReduceContactHistory reduceAndSwapContactHistory;
            kernel::ExplicitEuler particleIntegration(timeStepSize_SI);

            mpi::SyncNextNeighbors syncNextNeighbors;


            /// Initialize Cohesion
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
                                                       // initialize cohesion
                                                       cohesionInitialization(idx1, idx2, *accessor, acd.getPenetrationDepth());
                                                   }
                                               }
                                           }
                                       });

            reduceAndSwapContactHistory(*particleStorage);
            syncNextNeighbors(*particleStorage, *mesapdDomain);


            real_t maximumParticleHeightInSystem = getMaximumSphereHeightInSystem(*accessor, sphereShape);
            boxPosition[2] = boxEdgeLengths[2] * real_t(0.5) + maximumParticleHeightInSystem + diameter_SI;
            WALBERLA_LOG_INFO_ON_ROOT("Creating box at position " << boxPosition);

            auto boxShape = shapeStorage->create<data::Box>( boxEdgeLengths );//call boxShape with prefered dimensions
            auto box = createBox(*particleStorage, boxPosition, boxShape);// iterators are poinrers so we use ->
            (*box).setLinearVelocity(boxVelocity); //   box->setLinearVelocity(boxVelocity);
            auto boxUID = box->getUid();

            //vtk sphere and box
            auto sphereVtkOutput = make_shared<mesa_pd::vtk::ParticleVtkOutput>(particleStorage);
            sphereVtkOutput->addOutput<data::SelectParticleInteractionRadius>("radius");
            sphereVtkOutput->addOutput<data::SelectParticleLinearVelocity>("velocity");
            sphereVtkOutput->setParticleSelector( [sphereShape](const mesa_pd::data::ParticleStorage::iterator& pIt) {return pIt->getShapeID() == sphereShape;} ); //limit output to sphere
            auto sphereVtkWriter = walberla::vtk::createVTKOutput_PointData(sphereVtkOutput, "Spheres", 1, vtkOutputFolder, "simulation_step", false, false);

            auto boxVtkOutput = make_shared<mesa_pd::vtk::ParticleVtkOutput>(particleStorage);
            boxVtkOutput->addOutput<data::SelectParticleLinearVelocity>("velocity");
            boxVtkOutput->setParticleSelector( [boxShape](const mesa_pd::data::ParticleStorage::iterator& pIt) {return pIt->getShapeID() == boxShape;} ); //limit output to boxes
            auto boxVtkWriter = walberla::vtk::createVTKOutput_PointData(boxVtkOutput, "Box", 1, vtkOutputFolder, "simulation_step", false, false);

            uint_t timeStepsWithBox = uint_c( std::ceil( ( ( boxPosition[2] - boxEdgeLengths[2] * real_t(0.5) ) / -boxVelocity[2]) / timeStepSize_SI ) - 800 );
            WALBERLA_LOG_INFO_ON_ROOT("Starting simulation with box, running for " << timeStepsWithBox << " time steps!");
            for(uint_t t = 0; t < timeStepsWithBox; ++t)
            {
                // compute collision forces -> DEM
//                particleStorage->forEachParticlePairHalf(false, ExcludeGlobalGlobal(), *accessor,
//                                                         [&dem, &mesapdDomain, timeStepSize_SI]
//                                                                 (const size_t idx1, const size_t idx2, auto& ac)
//                                                         {
//                                                             mesa_pd::collision_detection::AnalyticContactDetection acd;
//                                                             mesa_pd::kernel::DoubleCast double_cast;
//                                                             mesa_pd::mpi::ContactFilter contact_filter;
//                                                             if (double_cast(idx1, idx2, ac, acd, ac ))
//                                                             {
//                                                                 if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *mesapdDomain))
//                                                                 {
//                                                                     dem(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), acd.getContactNormal(), acd.getPenetrationDepth(), timeStepSize_SI);
//                                                                 }
//                                                             }
//                                                         },
//                                                         *accessor );

                particleStorage->forEachParticlePairHalf(false, ExcludeGlobalGlobal(), *accessor,
                                                         [&]
                                                                 (const size_t idx1, const size_t idx2, auto& ac)
                                                         {
                                                             mesa_pd::collision_detection::AnalyticContactDetection acd;
                                                             mesa_pd::kernel::DoubleCast double_cast;
                                                             mesa_pd::mpi::ContactFilter contact_filter;
                                                             bool contactExists = double_cast(idx1, idx2, ac, acd, ac);
                                                             if(ac.getShape(idx1)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE &&
                                                                ac.getShape(idx2)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE) {
                                                                 Vector3<real_t> filteringPoint;
                                                                 if (contactExists)  {
                                                                     filteringPoint = acd.getContactPoint();
                                                                 } else{
                                                                     filteringPoint = (ac.getPosition(idx1) + ac.getPosition(idx2)) / real_t(2);
                                                                 }
                                                                 if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, filteringPoint, *mesapdDomain)) {
                                                                     bool cohesiveContactTreated = cohesion(idx1, idx2, ac,
                                                                                                            contactExists, acd.getContactNormal(), acd.getPenetrationDepth());
                                                                     if (cohesiveContactTreated){

                                                                     }else if(contactExists){
                                                                         cohesion(idx1, idx2, *accessor,
                                                                                  contactExists, acd.getContactNormal(), acd.getPenetrationDepth());
                                                                     }


                                                                 }

                                                             }else{
                                                                 if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *mesapdDomain))
                                                                 {
                                                                     dem(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), acd.getContactNormal(), acd.getPenetrationDepth(), timeStepSize_SI);

                                                                 }

                                                             }

                                                         },
                                                         *accessor );

                // output forces to file
                auto collisionForce = getCollisionForceOnBox(*accessor, boxUID);
                writeCollisionForceToFile(t, collisionForce, collisionForceLoggingFile);

                // synchronize collision information
                reduceAndSwapContactHistory(*particleStorage);

                // add gravitational force
                particleStorage->forEachParticle(false, kernel::SelectLocal(), *accessor, [gravitationalForce](const size_t idx, data::ParticleAccessorWithShape& ac){mesa_pd::addForceAtomic(idx, ac, Vec3(real_t(0), real_t(0),-gravitationalForce));},*accessor);

                // synchronize forces
                reduceProperty.operator()<mesa_pd::ForceTorqueNotification>(*particleStorage);

                // update position and velocity
                // .. of spheres
                particleStorage->forEachParticle(false, kernel::SelectLocal(), *accessor, particleIntegration, *accessor);
                // .. of box
                updateBoxPosition(*accessor, boxUID, timeStepSize_SI);

                // synchronize position and velocity
                syncNextNeighbors(*particleStorage, *mesapdDomain);

                if(t % vtkSpacing == 0)
                {
                    sphereVtkWriter->write();
                    boxVtkWriter->write();
                }

                if(t % 10 == 0)
                {
                    WALBERLA_LOG_INFO_ON_ROOT("Time step: " << t);
                }

            }




            return EXIT_SUCCESS;
        }

    } // namespace mesa_pd
} // namespace walberla

int main( int argc, char* argv[] )
{
    return walberla::mesa_pd::main( argc, argv );
}
