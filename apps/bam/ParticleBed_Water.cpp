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
//! \file TwoSettlingSpheres.cpp
//! \ingroup lbm_mesapd_coupling
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"
#include "blockforest/communication/UniformBufferedScheme.h"

#include "boundary/all.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/SharedFunctor.h"
#include "core/debug/Debug.h"
#include "core/debug/TestSubsystem.h"
#include "core/math/all.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/logging/all.h"
#include "core/mpi/MPIManager.h"
#include "core/mpi/Broadcast.h"
#include "core/mpi/MPITextFile.h"
#include <core/grid_generator/SCIterator.h>


#include "domain_decomposition/SharedSweep.h"

#include "field/AddToStorage.h"
#include "field/StabilityChecker.h"
#include "field/communication/PackInfo.h"

#include "lbm/boundary/all.h"
#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/field/PdfField.h"
#include "lbm/lattice_model/D3Q19.h"
#include "lbm/sweeps/CellwiseSweep.h"
#include "lbm/sweeps/SweepWrappers.h"

#include "lbm_mesapd_coupling/mapping/ParticleMapping.h"
#include "lbm_mesapd_coupling/momentum_exchange_method/MovingParticleMapping.h"
#include "lbm_mesapd_coupling/momentum_exchange_method/boundary/CurvedLinear.h"
#include "lbm_mesapd_coupling/momentum_exchange_method/reconstruction/Reconstructor.h"
#include "lbm_mesapd_coupling/momentum_exchange_method/reconstruction/ExtrapolationDirectionFinder.h"
#include "lbm_mesapd_coupling/momentum_exchange_method/reconstruction/PdfReconstructionManager.h"
#include "lbm_mesapd_coupling/utility/AddForceOnParticlesKernel.h"
#include "lbm_mesapd_coupling/utility/ParticleSelector.h"
#include "lbm_mesapd_coupling/DataTypes.h"
#include "lbm_mesapd_coupling/utility/AverageHydrodynamicForceTorqueKernel.h"
#include "lbm_mesapd_coupling/utility/AddHydrodynamicInteractionKernel.h"
#include "lbm_mesapd_coupling/utility/ResetHydrodynamicForceTorqueKernel.h"
#include "lbm_mesapd_coupling/utility/LubricationCorrectionKernel.h"
#include "lbm_mesapd_coupling/utility/OmegaBulkAdaption.h"

#include "mesa_pd/collision_detection/AnalyticContactDetection.h"
#include "mesa_pd/data/ParticleAccessorWithShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/ShapeStorage.h"
#include "mesa_pd/data/DataTypes.h"
#include "mesa_pd/data/shape/HalfSpace.h"
#include "mesa_pd/data/shape/Sphere.h"
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/kernel/DoubleCast.h"
#include "mesa_pd/kernel/ExplicitEulerWithShape.h"
#include "mesa_pd/kernel/ParticleSelector.h"
#include "mesa_pd/kernel/LinearSpringDashpot.h"
#include "mesa_pd/kernel/VelocityVerlet.h"
#include "mesa_pd/mpi/SyncNextNeighbors.h"
#include "mesa_pd/mpi/ReduceProperty.h"
#include "mesa_pd/mpi/ReduceContactHistory.h"
#include "mesa_pd/mpi/ContactFilter.h"
#include "mesa_pd/mpi/notifications/ForceTorqueNotification.h"
#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "timeloop/SweepTimeloop.h"

#include "vtk/all.h"
#include "field/vtk/all.h"
#include "lbm/vtk/all.h"

#include "Utility_water.h"

#include <mesa_pd/kernel/CohesionInitialization.h>
#include <mesa_pd/kernel/Cohesion.h>

#include <functional>

namespace two_settling_spheres
{

///////////
// USING //
///////////

    using namespace walberla;
    using walberla::uint_t;

    using LatticeModel_T = lbm::D3Q19< lbm::collision_model::SRT>;

    using Stencil_T = LatticeModel_T::Stencil;
    using PdfField_T = lbm::PdfField<LatticeModel_T>;

    using flag_t = walberla::uint8_t;
    using FlagField_T = FlagField<flag_t>;

    using ScalarField_T = GhostLayerField< real_t, 1>;

    const uint_t FieldGhostLayers = 1;

///////////
// FLAGS //
///////////

    const FlagUID Fluid_Flag( "fluid" );
    const FlagUID NoSlip_Flag( "no slip" );
    const FlagUID MO_Flag( "moving obstacle" );
    const FlagUID FormerMO_Flag( "former moving obstacle" );

/////////////////////////////////////
// BOUNDARY HANDLING CUSTOMIZATION //
/////////////////////////////////////
    template <typename ParticleAccessor_T>
    class MyBoundaryHandling
    {
    public:

        using NoSlip_T = lbm::NoSlip< LatticeModel_T, flag_t >;
        using MO_T = lbm_mesapd_coupling::CurvedLinear< LatticeModel_T, FlagField_T, ParticleAccessor_T >;
        using Type = BoundaryHandling< FlagField_T, Stencil_T, NoSlip_T, MO_T >;

        MyBoundaryHandling( const BlockDataID & flagFieldID, const BlockDataID & pdfFieldID,
                            const BlockDataID & particleFieldID, const shared_ptr<ParticleAccessor_T>& ac) :
                flagFieldID_( flagFieldID ), pdfFieldID_( pdfFieldID ), particleFieldID_( particleFieldID ), ac_( ac ) {}

        Type * operator()( IBlock* const block, const StructuredBlockStorage* const storage ) const
        {
            WALBERLA_ASSERT_NOT_NULLPTR( block );
            WALBERLA_ASSERT_NOT_NULLPTR( storage );

            auto * flagField     = block->getData< FlagField_T >( flagFieldID_ );
            auto *  pdfField     = block->getData< PdfField_T > ( pdfFieldID_ );
            auto * particleField = block->getData< lbm_mesapd_coupling::ParticleField_T > ( particleFieldID_ );

            const auto fluid = flagField->flagExists( Fluid_Flag ) ? flagField->getFlag( Fluid_Flag ) : flagField->registerFlag( Fluid_Flag );

            Type * handling = new Type( "moving obstacle boundary handling", flagField, fluid,
                                        NoSlip_T( "NoSlip", NoSlip_Flag, pdfField ),
                                        MO_T( "MO", MO_Flag, pdfField, flagField, particleField, ac_, fluid, *storage, *block ) );

            // Add other boundary conditions here -> get cell interval

            handling->fillWithDomain( FieldGhostLayers ); // initialize flag field with "Fluid" flag

            return handling;
        }

    private:

        const BlockDataID flagFieldID_;
        const BlockDataID pdfFieldID_;
        const BlockDataID particleFieldID_;

        shared_ptr<ParticleAccessor_T> ac_;
    };
//*******************************************************************************************************************


    template <typename Accessor_T>
    void writeSpherePropertiesToFile(Accessor_T & accessor, walberla::id_t sphereUid, std::string fileName, uint_t timestep, real_t dx_SI, real_t dt_SI)
    {

        auto sphereIdx = accessor.uidToIdx(sphereUid);
        if(sphereIdx != accessor.getInvalidIdx())
        {
            if(!isSet(accessor.getFlags(sphereIdx), mesa_pd::data::particle_flags::GHOST))
            {
                auto position = accessor.getPosition(sphereIdx);
                auto velocity = accessor.getLinearVelocity(sphereIdx);

                std::ofstream file;
                if(timestep == 0) file.open( fileName.c_str() );
                else file.open( fileName.c_str(), std::ofstream::app );

                position = position * dx_SI;
                velocity = velocity * dx_SI / dt_SI;

                file << real_c(timestep) * dt_SI << " " << position[0] << " " << position[1] << " " << position[2] << " "
                     << velocity[0] << " "  << velocity[1] << " "  << velocity[2] << "\n";

                file.close();
            }
        }
    }

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
            pIt->setInteractionRadius(radius);
            pIt->setOwner(rank);
            pIt->setType(1);

            WALBERLA_CHECK_EQUAL( iss.tellg(), -1);
        }
    }

//////////
// MAIN //
//////////

    int main( int argc, char **argv )
    {
        debug::enterTestMode();//?

        mpi::Environment env( argc, argv );

        ///////////////////
        // Customization //
        ///////////////////

        // simulation control
        bool usingLSD = false;
        std::string baseFolder = "vtk_out_editing";
        real_t cellsPerDiameter = real_t(10);
        real_t relaxationTime = real_t(0.65); // (0.5, \infty)
        Vector3<uint_t> numberOfBlocksPerDirection( uint_t(2), uint_t(2), uint_t(2) );
        uint_t vtkSpacing = uint_t(100);
        uint_t timeSteps = uint_t(4000);
        uint_t numberOfMesapdSubCycles = uint_t(1);//?

        bool usePabloStiffness = false;

        for( int i = 1; i < argc; ++i )
        {
            if( std::strcmp( argv[i], "--baseFolder" ) == 0 ) { baseFolder = argv[++i]; continue; }
            if( std::strcmp( argv[i], "--cellsPerDiameter" ) == 0 ) { cellsPerDiameter = real_c(std::atof(argv[++i])); continue; }
            if( std::strcmp( argv[i], "--relaxationTime" ) == 0 ) { relaxationTime = real_c(std::atof(argv[++i])); continue; }
            if( std::strcmp( argv[i], "--vtkSpacing" ) == 0 ) { vtkSpacing = uint_c( std::atof( argv[++i] ) ); continue; }
            if( std::strcmp( argv[i], "--timeSteps" ) == 0 ) { timeSteps = uint_c( std::atof( argv[++i] ) ); continue; }
            if( std::strcmp( argv[i], "--subCycles" ) == 0 ) { numberOfMesapdSubCycles = uint_c( std::atof( argv[++i] ) ); continue; }
            if( std::strcmp( argv[i], "--usePabloStiffness" ) == 0 ) { usePabloStiffness = true; continue; }
            WALBERLA_ABORT("Unrecognized command line argument found: " << argv[i]);
        }


        //////////////////////////////////////
        // SIMULATION PROPERTIES in SI units//
        //////////////////////////////////////

        real_t diameter_SI = real_t(3e-3);
        real_t densityParticle_SI = real_t(1010);
        Vector3<real_t> domainSize_SI(real_t(0.02),real_t(0.02),real_t(0.02));
        real_t densityFluid_SI = real_t(1000);
        real_t kinematicViscosity_SI = real_t(1e-6); // m**2 / s
        real_t frictionCoefficient = real_t(0);
        real_t gravitationalAcceleration_SI = real_t(9.81); // m / s**2
        real_t sphereGenerationSpacing_SI = 4e-3;
//        real_t initialHeightSphere1_SI = real_t(0.072);
//        real_t initialHeightSphere2_SI = real_t(0.068);
        real_t E_SI {1e5_r};
        real_t b_c {0.2_r};
        real_t en {0_r};
        real_t damp_SI {-log(en) / sqrt(log(en)*log(en) + math::pi*math::pi)};




        // Simulation properties in lattice units
        real_t dx_SI = diameter_SI / cellsPerDiameter; // m
        real_t omega = real_t(1) / relaxationTime;
        real_t kinematicViscosity = lbm::collision_model::viscosityFromOmega(omega);
        real_t dt_SI = kinematicViscosity / kinematicViscosity_SI * dx_SI * dx_SI; // s //?
        real_t gravitationalAcceleration = gravitationalAcceleration_SI * dt_SI * dt_SI / dx_SI;
        real_t densityRatio = densityParticle_SI / densityFluid_SI;
        real_t sphereGenerationSpacing = sphereGenerationSpacing_SI/dx_SI;
        real_t densityParticle = densityRatio;
        real_t densityFluid = real_t(1);

//        real_t initialHeightSphere1 = initialHeightSphere1_SI / dx_SI;
//        real_t initialHeightSphere2 = initialHeightSphere2_SI / dx_SI;
        real_t diameter = diameter_SI / dx_SI;
        real_t E = E_SI / densityFluid_SI * (dt_SI * dt_SI ) / (dx_SI * dx_SI * dx_SI);
        real_t damp = damp_SI / densityFluid_SI * (dt_SI * dt_SI ) / (dx_SI * dx_SI * dx_SI);


        real_t timeStepSize = real_t(1);


        Vector3<uint_t> domainSize( uint_c(domainSize_SI[0] / dx_SI), uint_c(domainSize_SI[1] / dx_SI), uint_c(domainSize_SI[2] / dx_SI));

        WALBERLA_LOG_INFO_ON_ROOT("dx_SI = " << dx_SI << " m");
        WALBERLA_LOG_INFO_ON_ROOT("dt_SI = " << dt_SI << " s");
        WALBERLA_LOG_INFO_ON_ROOT("gravitational acceleration lattice units = " << gravitationalAcceleration);
        WALBERLA_LOG_INFO_ON_ROOT("density ratio = " << densityRatio);
        WALBERLA_LOG_INFO_ON_ROOT("Domain size = " << domainSize);




        Vector3<uint_t> cellsPerBlockPerDirection( domainSize[0] / numberOfBlocksPerDirection[0],
                                                   domainSize[1] / numberOfBlocksPerDirection[1],
                                                   domainSize[2] / numberOfBlocksPerDirection[2] );
        for( uint_t i = 0; i < 3; ++i ) {
            WALBERLA_CHECK_EQUAL(cellsPerBlockPerDirection[i] * numberOfBlocksPerDirection[i], domainSize[i],
                                 "Unmatching domain decomposition in direction " << i << "!");
        }

        auto domainAABB = math::AABB{Vector3<real_t>{0_r}, domainSize};
        auto blocks = blockforest::createUniformBlockGrid( numberOfBlocksPerDirection[0], numberOfBlocksPerDirection[1], numberOfBlocksPerDirection[2],
                                                           cellsPerBlockPerDirection[0], cellsPerBlockPerDirection[1], cellsPerBlockPerDirection[2], real_t(1),
                                                           0, false, false,
                                                           false, false, false, //periodicity
                                                           false );

        WALBERLA_LOG_INFO_ON_ROOT("Domain decomposition:");
        WALBERLA_LOG_INFO_ON_ROOT(" - blocks per direction = " << numberOfBlocksPerDirection );
        WALBERLA_LOG_INFO_ON_ROOT(" - cells per block = " << cellsPerBlockPerDirection );

        //write domain decomposition to file
        if( vtkSpacing > 0 )
        {
            vtk::writeDomainDecomposition( blocks, "initial_domain_decomposition", baseFolder );
        }


        // MESA_PD parts

        auto mesapdDomain = std::make_shared<mesa_pd::domain::BlockForestDomain>(blocks->getBlockForestPointer());
        auto ps = walberla::make_shared<mesa_pd::data::ParticleStorage>(1);
        auto ss = walberla::make_shared<mesa_pd::data::ShapeStorage>();
        using ParticleAccessor_T = mesa_pd::data::ParticleAccessorWithShape;
        auto accessor = walberla::make_shared<ParticleAccessor_T >(ps, ss);

        // create bounding planes
        particle_erosion_utility::createPlane(*ps, *ss, Vector3<real_t>(real_t(0)), Vector3<real_t>(real_t(1), real_t(0), real_t(0)));
        particle_erosion_utility::createPlane(*ps, *ss, Vector3<real_t>(real_t(0)), Vector3<real_t>(real_t(0), real_t(1), real_t(0)));
        particle_erosion_utility::createPlane(*ps, *ss, Vector3<real_t>(real_t(0)), Vector3<real_t>(real_t(0), real_t(0), real_t(1)));
        particle_erosion_utility::createPlane(*ps, *ss, domainSize, Vector3<real_t>(real_t(-1), real_t(0), real_t(0)));
        particle_erosion_utility::createPlane(*ps, *ss, domainSize, Vector3<real_t>(real_t(0), real_t(-1), real_t(0)));
        particle_erosion_utility::createPlane(*ps, *ss, domainSize, Vector3<real_t>(real_t(0), real_t(0), real_t(-1)));




//        /// MESAPD Particles
        uint_t randomSeed = 1;
        std::mt19937 randomNumberGenerator{static_cast<unsigned int>(randomSeed)}; // rand()
        auto sphereShape = ss->create<mesa_pd::data::Sphere>( diameter * real_t(0.5) );
        WALBERLA_CHECK(sphereGenerationSpacing > diameter, "Spacing should be larger than diameter!");

        for (auto& iBlk : *blocks)
        {
            for (auto position : grid_generator::SCGrid{domainAABB,
                                                        Vector3<real_t>{sphereGenerationSpacing} * real_c(0.5),
                                                        sphereGenerationSpacing})
            {
                Vector3 positionOffset{math::realRandom<real_t>(-0.1_r, 0.1_r, randomNumberGenerator),
                                    math::realRandom<real_t>(-0.1_r, 0.1_r, randomNumberGenerator),
                                    math::realRandom<real_t>(-0.1_r, 0.1_r, randomNumberGenerator)};

                if(iBlk.getAABB().contains(position))
                {

                    mesa_pd::data::Particle&& p = *ps->create();
                    p.setPosition(position + positionOffset * sphereGenerationSpacing);
                    ss->shapes[sphereShape]->updateMassAndInertia(densityParticle);
                    p.setInteractionRadius(diameter * real_t(0.5));
                    p.setShapeID(sphereShape);
                    p.setOwner(mpi::MPIManager::instance()->rank());
                    p.setType(1);

                }
            }
        }
        int64_t numParticles = int64_c(ps->size());
        walberla::mpi::reduceInplace(numParticles, walberla::mpi::SUM);
        WALBERLA_LOG_INFO_ON_ROOT("Created " << numParticles << " particles.");




//        initSpheresFromFile("logging_sphere_edit.txt", *ps, *mesapdDomain, sphereShape);
//        int64_t numParticles = int64_c(ps->size());
//        walberla::mpi::reduceInplace(numParticles, walberla::mpi::SUM);
//        WALBERLA_LOG_INFO_ON_ROOT("Created " << numParticles << " particles.");







        if(!usingLSD){
            ps->forEachParticlePairHalf(false, mesa_pd::kernel::SelectAll(), *accessor,
                                       [&](const size_t idx1, const size_t idx2, auto& ac){
                                           // call the general contact detection kernel (gcd) for particles with idx1 and idx2
                                           mesa_pd::collision_detection::AnalyticContactDetection acd;
                                           mesa_pd::kernel::DoubleCast double_cast;
                                           mesa_pd::mpi::ContactFilter contact_filter;
                                           if(ac.getShape(idx1)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE &&
                                              ac.getShape(idx2)->getShapeType() == mesa_pd::data::Sphere::SHAPE_TYPE) {
                                               if (double_cast(idx1, idx2, ac, acd, ac)) {
                                                   // particles overlap
                                                   // check if the overlap should be treated on this process to avoid duplicate calculations
                                                   if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(),
                                                                      *mesapdDomain)) {
                                                       WALBERLA_LOG_INFO("Found cohesive particles");
                                                       // initialize cohesion to false
                                                       // contact history of particle 1 -> particle 2
                                                       auto& nch1 = ac.getNewContactHistoryRef(idx1)[ac.getUid(idx2)];
                                                       // contact history of particle 2 -> particle 1
                                                       auto& nch2 = ac.getNewContactHistoryRef(idx2)[ac.getUid(idx1)];
                                                       // save for each of the particles that they are bound to the other by cohesion
                                                       nch1.setCohesionBound(false);
                                                       nch2.setCohesionBound(false);
                                                   }
                                               }
                                           }
                                       },*accessor );
        }

        // LBM parts

        LatticeModel_T latticeModel = LatticeModel_T(omega);

        // PDF field
        BlockDataID pdfFieldID = lbm::addPdfFieldToStorage< LatticeModel_T >( blocks, "pdf field (fzyx)", latticeModel,
                                                                              Vector3< real_t >( real_t(0) ), real_t(1),
                                                                              FieldGhostLayers, field::fzyx );//?
        // flag field
        BlockDataID flagFieldID = field::addFlagFieldToStorage<FlagField_T>( blocks, "flag field" );

        // particle field -> for coupling
        BlockDataID particleFieldID = field::addToStorage<lbm_mesapd_coupling::ParticleField_T>( blocks, "particle field", accessor->getInvalidUid(), field::fzyx, FieldGhostLayers );

        // add boundary handling
        using BoundaryHandling_T = MyBoundaryHandling<ParticleAccessor_T>::Type;
        BlockDataID boundaryHandlingID = blocks->addStructuredBlockData< BoundaryHandling_T >(MyBoundaryHandling<ParticleAccessor_T>( flagFieldID, pdfFieldID, particleFieldID, accessor), "boundary handling" );

        // coupling kernels

        lbm_mesapd_coupling::AddHydrodynamicInteractionKernel addHydrodynamicInteraction;
        lbm_mesapd_coupling::ResetHydrodynamicForceTorqueKernel resetHydrodynamicForceTorque;
        lbm_mesapd_coupling::AverageHydrodynamicForceTorqueKernel averageHydrodynamicForceTorque;

        // set up RPD functionality
        std::function<void(void)> syncCall = [ps,mesapdDomain](){
            const real_t overlap = real_t( 1.5 );//?
            mesa_pd::mpi::SyncNextNeighbors syncNextNeighborFunc;
            syncNextNeighborFunc(*ps, *mesapdDomain, overlap);
        };
        syncCall();


        // initialize fields

        // map planes into the LBM simulation -> act as no-slip boundaries
        lbm_mesapd_coupling::ParticleMappingKernel<BoundaryHandling_T> particleMappingKernel(blocks, boundaryHandlingID);
        ps->forEachParticle(false, lbm_mesapd_coupling::GlobalParticlesSelector(), *accessor, particleMappingKernel, *accessor, NoSlip_Flag);


        // map particles into the LBM simulation
        lbm_mesapd_coupling::MovingParticleMappingKernel<BoundaryHandling_T> movingParticleMappingKernel(blocks, boundaryHandlingID, particleFieldID);
        lbm_mesapd_coupling::RegularParticlesSelector sphereSelector;
        ps->forEachParticle(false, sphereSelector, *accessor, movingParticleMappingKernel, *accessor, MO_Flag);


        // particle simulation kernels

        mesa_pd::kernel::LinearSpringDashpot dem(2);
        dem.setFrictionCoefficientDynamic(0,1,frictionCoefficient);
        dem.setFrictionCoefficientDynamic(1,1,frictionCoefficient);

        real_t volumeSphere = math::pi / real_t(6) * diameter * diameter * diameter;
        real_t massSphere = densityParticle * volumeSphere;

        if(usePabloStiffness)
        {

            real_t stiffnessN_SI = real_t(1.1e5); // N / m
            real_t stiffnessN = stiffnessN_SI / densityFluid_SI * (dt_SI * dt_SI ) / (dx_SI * dx_SI * dx_SI);

            dem.setStiffnessN(0,1,stiffnessN);
            dem.setStiffnessN(1,1,stiffnessN);
        } else
        {
            real_t collisionTime = real_t(60);
            real_t restitutionCoefficient = real_t(0.9);
            const real_t poissonsRatio = real_t(0.22);
            const real_t kappa = real_t(2) * ( real_t(1) - poissonsRatio ) / ( real_t(2) - poissonsRatio ) ;

            real_t effectiveMass_SpherePlane = massSphere;
            real_t effectiveMass_SphereSphere = massSphere * massSphere / (real_t(2) * massSphere);
            dem.setStiffnessAndDamping(0,1,restitutionCoefficient, collisionTime, kappa, effectiveMass_SpherePlane);
            dem.setStiffnessAndDamping(1,1,restitutionCoefficient, collisionTime, kappa, effectiveMass_SphereSphere);
        }

        WALBERLA_LOG_INFO_ON_ROOT("stiffness N = " << dem.getStiffnessN(1,1));





        Vector3<real_t> gravitationalForce(real_t(0), real_t(0), -gravitationalAcceleration * massSphere);
        Vector3<real_t> buoyancyForce(real_t(0), real_t(0), gravitationalAcceleration * densityFluid * volumeSphere);


        mesa_pd::mpi::ReduceProperty reduceProperty;
        mesa_pd::mpi::ReduceContactHistory reduceAndSwapContactHistory;

        real_t timeStepSizeMesapd = timeStepSize / real_c(numberOfMesapdSubCycles);
        mesa_pd::kernel::ExplicitEuler particleIntegration(timeStepSizeMesapd);


        /// new kernel:
        /// Cohesion
        mesa_pd::kernel::Cohesion cohesion(timeStepSizeMesapd, E, damp, b_c);
        mesa_pd::kernel::CohesionInitialization cohesionInitialization;


        // create the timeloop
        SweepTimeloop timeloop( blocks->getBlockStorage(), timeSteps );
        timeloop.addFuncBeforeTimeStep( RemainingTimeLogger( timeloop.getNrOfTimeSteps() ), "Remaining Time Logger" );

        // vtk output
        if( vtkSpacing != uint_t(0) )
        {
            // spheres
            auto particleVtkOutput = make_shared<mesa_pd::vtk::ParticleVtkOutput>(ps);
            particleVtkOutput->addOutput<mesa_pd::data::SelectParticleOwner>("owner");
            particleVtkOutput->addOutput<mesa_pd::data::SelectParticleInteractionRadius>("radius");
            particleVtkOutput->addOutput<mesa_pd::data::SelectParticleLinearVelocity>("velocity");
            auto particleVtkWriter = vtk::createVTKOutput_PointData(particleVtkOutput, "Particles", vtkSpacing, baseFolder, "simulation_step");
            timeloop.addFuncBeforeTimeStep( vtk::writeFiles( particleVtkWriter ), "VTK (sphere data)" );

            // fluid
            auto pdfFieldVTK = vtk::createVTKOutput_BlockData( blocks, "fluid_field", vtkSpacing, 0, false, baseFolder );
            field::FlagFieldCellFilter< FlagField_T > fluidFilter( flagFieldID );
            fluidFilter.addFlag( Fluid_Flag );
            pdfFieldVTK->addCellInclusionFilter( fluidFilter );
            pdfFieldVTK->addCellDataWriter( make_shared< lbm::VelocityVTKWriter< LatticeModel_T, float > >( pdfFieldID, "VelocityFromPDF" ) );
            pdfFieldVTK->addCellDataWriter( make_shared< lbm::DensityVTKWriter < LatticeModel_T, float > >( pdfFieldID, "DensityFromPDF" ) );
            timeloop.addFuncBeforeTimeStep( vtk::writeFiles( pdfFieldVTK ), "VTK (fluid field data)" );

        }


        blockforest::communication::UniformBufferedScheme< Stencil_T > optimizedPDFCommunicationScheme( blocks );//meaning?
        optimizedPDFCommunicationScheme.addPackInfo( make_shared< lbm::PdfFieldPackInfo< LatticeModel_T > >( pdfFieldID ) ); // optimized sync

        // add LBM communication function (updates ghost layers) and boundary handling sweep (does the hydro force calculations and the no-slip treatment)
        auto boundaryHandlingSweep = BoundaryHandling_T::getBlockSweep( boundaryHandlingID );
        timeloop.add() << BeforeFunction( optimizedPDFCommunicationScheme, "LBM Communication" )
                       << Sweep(boundaryHandlingSweep, "Boundary Handling" );

        // add LBM part (stream + collide)
        auto lbmSweep = lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >( pdfFieldID, flagFieldID, Fluid_Flag );
        timeloop.add() << Sweep( makeSharedSweep( lbmSweep ), "cell-wise LB sweep" );


        SweepTimeloop timeloopAfterParticle( blocks->getBlockStorage(), timeSteps );

        // update mapping: check if fluid -> moving obstacle, and moving obstacle -> former MO, when particles have moved
        timeloopAfterParticle.add() << Sweep( lbm_mesapd_coupling::makeMovingParticleMapping<PdfField_T, BoundaryHandling_T>(blocks, pdfFieldID, boundaryHandlingID, particleFieldID, accessor, MO_Flag, FormerMO_Flag, sphereSelector, false), "Particle Mapping" );

        // reconstruct PDFs in former MO flags (former MO -> fluid)
        timeloopAfterParticle.add() << Sweep( makeSharedSweep(lbm_mesapd_coupling::makePdfReconstructionManager<PdfField_T,BoundaryHandling_T>(blocks, pdfFieldID, boundaryHandlingID, particleFieldID, accessor, FormerMO_Flag, Fluid_Flag, false) ), "PDF Restore" );



        // time loop
        WcTimingPool timeloopTiming;//meaning?
        for (uint_t i = 0; i < timeSteps; ++i )
        {
            // LBM + boundary handling + coupling force evaluation
            timeloop.singleStep(timeloopTiming);

            // average hdydrodynamic force over two time steps to avoid oscillations
            ps->forEachParticle(false, mesa_pd::kernel::SelectAll(), *accessor, averageHydrodynamicForceTorque, *accessor );

            // add sub cycling for particle simulation -> increase temporal resolution of contact detection and resolving
            for(uint_t subCycle = 0; subCycle < numberOfMesapdSubCycles; ++subCycle )
            {

                // take stored Fhyd values and add onto particles as force
                ps->forEachParticle(false, sphereSelector, *accessor, addHydrodynamicInteraction, *accessor );

                // compute collision forces -> DEM
                if(usingLSD){
                    ps->forEachParticlePairHalf(false, mesa_pd::kernel::ExcludeInfiniteInfinite(), *accessor,
                                                [&dem, &mesapdDomain, timeStepSizeMesapd]
                                                        (const size_t idx1, const size_t idx2, auto& ac)
                                                {
                                                    mesa_pd::collision_detection::AnalyticContactDetection acd;
                                                    mesa_pd::kernel::DoubleCast double_cast;
                                                    mesa_pd::mpi::ContactFilter contact_filter;
                                                    if (double_cast(idx1, idx2, ac, acd, ac ))
                                                    {
                                                        if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *mesapdDomain))
                                                        {
                                                            dem(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), acd.getContactNormal(), acd.getPenetrationDepth(), timeStepSizeMesapd);
                                                        }
                                                    }
                                                },
                                                *accessor );


                    // synchronize collision information
                    reduceAndSwapContactHistory(*ps);

                }else{

                    ps->forEachParticlePairHalf(false, mesa_pd::kernel::ExcludeInfiniteInfinite(), *accessor,
                                                [&]
                                                        (const size_t idx1, const size_t idx2, auto& ac)
                                                {
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
                                                        if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, filteringPoint, *mesapdDomain)) {
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
                                                        if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *mesapdDomain)) {
                                                            dem(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(),
                                                                acd.getContactNormal(), acd.getPenetrationDepth(), timeStepSizeMesapd);
                                                        }
                                                    }
                                                },
                                                *accessor );



                    // synchronize collision information
                    reduceAndSwapContactHistory(*ps);
                }




                // add gravitational + buoyancy force
                ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor, [gravitationalForce](const size_t idx, ParticleAccessor_T& ac){mesa_pd::addForceAtomic(idx, ac, gravitationalForce);},*accessor);
                ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor, [buoyancyForce](const size_t idx, ParticleAccessor_T& ac){mesa_pd::addForceAtomic(idx, ac, buoyancyForce);},*accessor);

                // synchronize forces
                reduceProperty.operator()<mesa_pd::ForceTorqueNotification>(*ps);

                // update position and velocity
                ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor, particleIntegration, *accessor);

                // synchronize position and velocity
                syncCall();
            }

            // reset F hyd
            ps->forEachParticle(false, mesa_pd::kernel::SelectAll(), *accessor, resetHydrodynamicForceTorque, *accessor );

            // update mapping + PDF restore
            timeloopAfterParticle.singleStep(timeloopTiming);

            // logging to file
//            if(i % 10 == 0)
//            {
//                writeSpherePropertiesToFile(*accessor, sphere1Uid, "logging_sphere1.txt", i, dx_SI, dt_SI);
//                writeSpherePropertiesToFile(*accessor, sphere2Uid, "logging_sphere2.txt", i, dx_SI, dt_SI);
//            }

        }

        timeloopTiming.logResultOnRoot();
        writeSpherePropertiesToFile(*accessor, "logging_sphere_edit.txt", sphereShape);


        return EXIT_SUCCESS;
    }

} // namespace two_settling_spheres

int main( int argc, char **argv ){
    two_settling_spheres::main(argc, argv);
}
