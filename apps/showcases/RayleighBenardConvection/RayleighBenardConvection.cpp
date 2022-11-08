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
//! \file RayleighBenardConvection.cpp
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

// Codegen includes
#include "RBCLatticeModel.h"
#include "RBCPackInfo.h"

namespace walberla {

const uint_t FieldGhostLayers( 1 );

////////////////////////////////////////
/// Macro for LBM collision operator ///
////////////////////////////////////////

#define USE_SRT
//#define USE_TRT

///////////////////////
/// Typedef Aliases ///
///////////////////////

// Typedef alias for the lattice model
typedef lbm::RBCLatticeModel LatticeModel_T;

// Communication pack info
typedef pystencils::RBCPackInfo PackInfo_T;

// Typedef aliases for the involved stencils
typedef LatticeModel_T::Stencil Stencil_T;
typedef LatticeModel_T::CommunicationStencil CommunicationStencil_T;

// Typedefs for the boundary handling, flag data type and flag field type
typedef walberla::uint8_t flag_t;
typedef FlagField<flag_t> FlagField_T;
typedef lbm::PdfField<LatticeModel_T> PdfField_T;
//using ScalarField_T = field::GhostLayerField< real_t, 1 >;
//using VectorField_T = field::GhostLayerField< math::Vector3< real_t >, 1 >;
typedef lbm::DefaultBoundaryHandlingFactory< LatticeModel_T, FlagField_T > BHFactory;

/////////////////////
/// Main Function ///
/////////////////////

int main( int argc, char ** argv )
{
   walberla::Environment walberlaEnv( argc, argv );
   auto configPtr = walberlaEnv.config();
   if(!configPtr) WALBERLA_ABORT("No configuration file specified!")
   WALBERLA_LOG_INFO_ON_ROOT(*configPtr);

   #ifdef USE_SRT
      WALBERLA_LOG_DEVEL_ON_ROOT("Using lbmpy generated SRT lattice model.")
   #elif defined( USE_TRT )
      WALBERLA_LOG_DEVEL_ON_ROOT("Using lbmpy generated TRT lattice model.")
   #endif

   ///////////////////////////////////////////////////////
   /// Block Storage Creation and Simulation Parameter ///
   ///////////////////////////////////////////////////////

   auto blocks = blockforest::createUniformBlockGridFromConfig( configPtr );

   // read parameters
   auto parameters = configPtr->getOneBlock( "Parameters" );

   const real_t          omega           = parameters.getParameter< real_t >         ( "omega",           real_c( 1.4 ) );
   const Vector3<real_t> initialVelocity = parameters.getParameter< Vector3<real_t> >( "initialVelocity", Vector3<real_t>() );
   const uint_t          timesteps       = parameters.getParameter< uint_t >         ( "timesteps",       uint_c( 10 )  );

   const double remainingTimeLoggerFrequency = parameters.getParameter< double >( "remainingTimeLoggerFrequency", 3.0 ); // in seconds
   const Vector3<real_t> bodyForce(real_t(1e-6), real_t(0), real_t(0));

   ///////////////////
   /// Field Setup ///
   ///////////////////

#if defined(USE_SRT)
   LatticeModel_T latticeModel = LatticeModel_T(bodyForce[0], bodyForce[1], bodyForce[2], omega);
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(latticeModel.omega_)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(latticeModel.force_0_)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(latticeModel.force_1_)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(latticeModel.force_2_)
#elif defined(USE_TRT)
   real_t lambda_e = lbm::collision_model::TRT::lambda_e( omega );
   real_t lambda_d = lbm::collision_model::TRT::lambda_d( omega, lbm::collision_model::TRT::threeSixteenth );
   std::cout << "   lambda_e = " << lambda_e << " | lambda_d = " << lambda_d << std::endl;
   LatticeModel_T latticeModel =
      LatticeModel_T(bodyForce[0], bodyForce[1], bodyForce[2], lambda_e, lambda_e);
   auto viscosity = lbm::collision_model::viscosityFromOmega(omega);
   std::cout << "   omega = " << omega << " | viscosity = " << viscosity << std::endl;
#endif
   BlockDataID pdfFieldId = lbm::addPdfFieldToStorage( blocks, "pdf field", latticeModel, field::fzyx );
   BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >( blocks, "flag field" );

   /////////////////////////
   /// Boundary Handling ///
   /////////////////////////

   // create and initialize boundary handling
   const FlagUID fluidFlagUID( "Fluid" );

   auto boundariesConfig = configPtr->getOneBlock( "Boundaries" );

   BlockDataID boundaryHandlingId =
      BHFactory::addBoundaryHandlingToStorage(blocks, "boundary handling", flagFieldId, pdfFieldId, fluidFlagUID,
                                              Vector3< real_t >(), Vector3< real_t >(), real_c(0.0), real_c(0.0));

   geometry::initBoundaryHandling<BHFactory::BoundaryHandling>( *blocks, boundaryHandlingId, boundariesConfig );
   geometry::setNonBoundaryCellsToDomain<BHFactory::BoundaryHandling> ( *blocks, boundaryHandlingId );

   /////////////////
   /// Time Loop ///
   /////////////////

   // create time loop
   SweepTimeloop timeloop( blocks->getBlockStorage(), timesteps );

   // create communication for PdfField
   blockforest::communication::UniformBufferedScheme< CommunicationStencil_T > communication( blocks );
   communication.addPackInfo( make_shared< PackInfo_T >( pdfFieldId ) );

   // add LBM sweep and communication to time loop
   timeloop.add() << BeforeFunction( communication, "communication" )
                  << Sweep( BHFactory::BoundaryHandling::getBlockSweep( boundaryHandlingId ), "boundary handling" );

   timeloop.add() << Sweep(LatticeModel_T::Sweep(pdfFieldId), "LB stream & collide");

   // LBM stability check
   timeloop.addFuncAfterTimeStep( makeSharedFunctor( field::makeStabilityChecker< PdfField_T, FlagField_T >(
                                  configPtr, blocks, pdfFieldId, flagFieldId, fluidFlagUID ) ),
                                  "LBM stability check" );

   // log remaining time
   timeloop.addFuncAfterTimeStep( timing::RemainingTimeLogger( timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency ), "remaining time logger" );

   // add VTK output to time loop
   lbm::VTKOutput< LatticeModel_T, FlagField_T >::addToTimeloop( timeloop, blocks, configPtr, pdfFieldId,
                                                                flagFieldId, fluidFlagUID );

   timeloop.run();

   WALBERLA_LOG_INFO_ON_ROOT("Simulation was successful!")
   return EXIT_SUCCESS;
}
}

int main( int argc, char ** argv )
{
   walberla::main(argc, argv);
}