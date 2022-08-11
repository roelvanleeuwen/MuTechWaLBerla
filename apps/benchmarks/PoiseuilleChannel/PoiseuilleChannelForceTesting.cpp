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
//! \file PoiseuilleChannelForceTesting.cpp
//! \author Florian Schornbaum <florian.schornbaum@fau.de>
//! \author Martin Bauer <martin.bauer@fau.de>
//! \author Jonas Plewinski <jonas.plewinski@fau.de>
//
//======================================================================================================================


#include "blockforest/all.h"
#include "core/all.h"
#include "domain_decomposition/all.h"
#include "field/all.h"
#include "geometry/all.h"
#include "lbm/all.h"
#include "timeloop/all.h"

#ifdef WALBERLA_BUILD_WITH_CODEGEN
   #include "GeneratedLBM.h"
#endif


namespace walberla {

//const uint_t FieldGhostLayers( 1 );
#define CODEGEN

//#define USE_SRT
#define USE_TRT

#define USE_GuoConstant
//#define USE_SimpleConstant
//#define USE_GuoField

#ifdef CODEGEN
#else
   #if defined(USE_GuoConstant)
      using ForceModel_T = lbm::force_model::GuoConstant;
   #elif defined(USE_SimpleConstant)
      using ForceModel_T = lbm::force_model::SimpleConstant;
   #endif
#endif

//using Vec3Field_T = field::GhostLayerField<Vector3<real_t>, 1>;
//using ForceModel_T = lbm::force_model::GuoField<Vec3Field_T>;
#ifdef CODEGEN
   using LatticeModel_T = lbm::GeneratedLBM;
#else
   #ifdef USE_SRT
      using CollisionModel_T = lbm::collision_model::SRT;
   #elif defined(USE_TRT)
      using CollisionModel_T = lbm::collision_model::TRT;
   #endif
   using LatticeModel_T = lbm::D2Q9<CollisionModel_T , true, ForceModel_T>;
#endif

using Stencil_T = LatticeModel_T::Stencil;
using CommunicationStencil_T = LatticeModel_T::CommunicationStencil;

using PdfField_T = lbm::PdfField<LatticeModel_T>;

using flag_t = walberla::uint8_t;
using FlagField_T = FlagField<flag_t>;


int main( int argc, char ** argv )
{
   walberla::Environment walberlaEnv( argc, argv );

   auto blocks = blockforest::createUniformBlockGridFromConfig( walberlaEnv.config() );

   // read parameters
   auto parameters = walberlaEnv.config()->getOneBlock( "Parameters" );

   const real_t          omega           = parameters.getParameter< real_t >         ( "omega",           real_c( 1.4 ) );
   const Vector3<real_t> initialVelocity = parameters.getParameter< Vector3<real_t> >( "initialVelocity", Vector3<real_t>() );
   const uint_t          timesteps       = parameters.getParameter< uint_t >         ( "timesteps",       uint_c( 10 )  );

   const double remainingTimeLoggerFrequency = parameters.getParameter< double >( "remainingTimeLoggerFrequency", 3.0 ); // in seconds
   const Vector3<real_t> bodyForce(real_t(1e-5), real_t(0), real_t(0));

   // create fields
   //> BlockDataID forceFieldID = field::addToStorage< Vec3Field_T >( blocks, "force field", Vector3<real_t>(real_t(0)), field::zyxf, FieldGhostLayers );
#ifdef CODEGEN
   #ifdef USE_SRT
      LatticeModel_T  latticeModel = LatticeModel_T (bodyForce[0], bodyForce[1], omega);
   #elif defined(USE_TRT)
      real_t lambda_e = lbm::collision_model::TRT::lambda_e( omega );
      real_t lambda_d = lbm::collision_model::TRT::lambda_d( omega, lbm::collision_model::TRT::threeSixteenth );
      std::cout << "   lambda_e = " << lambda_e << " | lambda_d = " << lambda_d << std::endl;
      LatticeModel_T  latticeModel = LatticeModel_T (bodyForce[0], bodyForce[1], lambda_e, lambda_d);
   #endif
#else
   #if defined(USE_SRT)
      LatticeModel_T latticeModel = LatticeModel_T(CollisionModel_T(omega),ForceModel_T(bodyForce));
   #elif defined(USE_TRT)
      real_t lambda_e = lbm::collision_model::TRT::lambda_e( omega );
      real_t lambda_d = lbm::collision_model::TRT::lambda_d( omega, lbm::collision_model::TRT::threeSixteenth );
      std::cout << "   lambda_e = " << lambda_e << " | lambda_d = " << lambda_d << std::endl;
      LatticeModel_T latticeModel =
         LatticeModel_T(CollisionModel_T(lambda_e, lambda_d), ForceModel_T(bodyForce));
      auto viscosity = lbm::collision_model::viscosityFromOmega(omega);
      std::cout << "   omega = " << omega << " | viscosity = " << viscosity << std::endl;
   #elif defined(USE_MRT)
      LatticeModel_T latticeModel = LatticeModel_T(CollisionModel_T(omega, omega, omega, omega, omega, omega),
                                                   ForceModel_T(bodyForce));
   #endif
#endif

   //> LatticeModel_T latticeModel = LatticeModel_T( lbm::collision_model::SRT( omega ), ForceModel_T( forceFieldID ));
   BlockDataID pdfFieldId = lbm::addPdfFieldToStorage( blocks, "pdf field", latticeModel, initialVelocity, real_t(1), field::fzyx );
   //> BlockDataID pdfFieldId = lbm::addPdfFieldToStorage( blocks, "pdf field (zyxf)", latticeModel, initialVelocity, real_t(1), FieldGhostLayers, field::zyxf );
   BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >( blocks, "flag field" );

   // create and initialize boundary handling
   const FlagUID fluidFlagUID( "Fluid" );

   auto boundariesConfig = walberlaEnv.config()->getOneBlock( "Boundaries" );

   using BHFactory = lbm::DefaultBoundaryHandlingFactory<LatticeModel_T, FlagField_T>;

   BlockDataID boundaryHandlingId = BHFactory::addBoundaryHandlingToStorage( blocks, "boundary handling", flagFieldId, pdfFieldId, fluidFlagUID,
                                                                             boundariesConfig.getParameter< Vector3<real_t> >( "velocity0", Vector3<real_t>() ),
                                                                             boundariesConfig.getParameter< Vector3<real_t> >( "velocity1", Vector3<real_t>() ),
                                                                             boundariesConfig.getParameter< real_t > ( "pressure0", real_c( 1.0 ) ),
                                                                             boundariesConfig.getParameter< real_t > ( "pressure1", real_c( 1.0 ) ) );

   geometry::initBoundaryHandling<BHFactory::BoundaryHandling>( *blocks, boundaryHandlingId, boundariesConfig );
   geometry::setNonBoundaryCellsToDomain<BHFactory::BoundaryHandling> ( *blocks, boundaryHandlingId );

   // create time loop
   SweepTimeloop timeloop( blocks->getBlockStorage(), timesteps );

   // create communication for PdfField
   blockforest::communication::UniformBufferedScheme< CommunicationStencil_T > communication( blocks );
   communication.addPackInfo( make_shared< lbm::PdfFieldPackInfo< LatticeModel_T > >( pdfFieldId ) );

   // add LBM sweep and communication to time loop
   timeloop.add() << BeforeFunction( communication, "communication" )
                  << Sweep( BHFactory::BoundaryHandling::getBlockSweep( boundaryHandlingId ), "boundary handling" );
#ifdef CODEGEN
   auto lbmSweep = LatticeModel_T::Sweep(pdfFieldId);
   timeloop.add() << Sweep( lbmSweep, "LB Sweep" );
#else
   auto lbmSweep = lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >( pdfFieldId, flagFieldId, fluidFlagUID );
   timeloop.add() << Sweep( makeSharedSweep( lbmSweep ), "LB stream & collide" );
#endif

   // LBM stability check
   timeloop.addFuncAfterTimeStep( makeSharedFunctor( field::makeStabilityChecker< PdfField_T, FlagField_T >( walberlaEnv.config(), blocks, pdfFieldId,
                                                                                                             flagFieldId, fluidFlagUID ) ),
                                  "LBM stability check" );

   // log remaining time
   timeloop.addFuncAfterTimeStep( timing::RemainingTimeLogger( timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency ), "remaining time logger" );

   // add VTK output to time loop
   lbm::VTKOutput< LatticeModel_T, FlagField_T >::addToTimeloop( timeloop, blocks, walberlaEnv.config(), pdfFieldId, flagFieldId, fluidFlagUID );

   // create adaptors, so that the GUI also displays density and velocity
   // adaptors are like fields with the difference that they do not store values
   // but calculate the values based on other fields ( here the PdfField )
   field::addFieldAdaptor<lbm::Adaptor<LatticeModel_T>::Density>       ( blocks, pdfFieldId, "DensityAdaptor" );
   field::addFieldAdaptor<lbm::Adaptor<LatticeModel_T>::VelocityVector>( blocks, pdfFieldId, "VelocityAdaptor" );

   timeloop.run();

   WALBERLA_LOG_INFO_ON_ROOT("Simulation was successful!")

   return EXIT_SUCCESS;
}
}

int main( int argc, char ** argv )
{
   walberla::main(argc, argv);
}