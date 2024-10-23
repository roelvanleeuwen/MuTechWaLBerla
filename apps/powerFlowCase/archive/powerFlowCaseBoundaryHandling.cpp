#include "blockforest/all.h"

#include "core/all.h"

#include "domain_decomposition/all.h"

#include "field/all.h"

#include "geometry/all.h"

#include "gui/all.h"

#include "lbm/all.h"

#include "timeloop/all.h"

#include <cmath>
#include <iostream>
using namespace std;
using namespace walberla;

namespace walberla
{

using LatticeModel_T         = lbm::D2Q9< lbm::collision_model::SRT >;
using Stencil_T              = LatticeModel_T::Stencil;
using CommunicationStencil_T = LatticeModel_T::CommunicationStencil;

using PdfField_T = lbm::PdfField< LatticeModel_T >;

using flag_t      = walberla::uint16_t;
using FlagField_T = FlagField< flag_t >;

// unique identifiers for flags
const FlagUID FluidFlagUID("Fluid Flag");
const FlagUID NoSlipFlagUID("NoSlip Flag");
const FlagUID FreeSlipFlagUID("FreeSlip Flag");
const FlagUID SimpleUBBFlagUID("SimpleUBB Flag");
const FlagUID UBBFlagUID("UBB Flag");
// const FlagUID DynamicUBBFlagUID("DynamicUBB Flag");
// const FlagUID ParserUBBFlagUID("ParserUBB Flag");
const FlagUID SimplePressureFlagUID("SimplePressure Flag");
const FlagUID PressureFlagUID("Pressure Flag");
const FlagUID OutletFlagUID("Outlet Flag");
const FlagUID SimplePABFlagUID("SimplePAB Flag");
const FlagUID SimpleDiffusionDirichletFlagUID("SimpleDiffusionDirichlet Flag");

// number of ghost layers
const uint_t FieldGhostLayers = uint_t(4);

struct BoundarySetup
{
   std::string wallType;
   std::string inflowType;
   std::string outflowType;

   Vector3< real_t > inflowVelocity;
   real_t outflowPressure;

   // SimplePAB
   real_t omega;
};

using NoSlip_T   = lbm::NoSlip< LatticeModel_T, flag_t >;
using FreeSlip_T = lbm::FreeSlip< LatticeModel_T, FlagField_T >;

using SimpleUBB_T = lbm::SimpleUBB< LatticeModel_T, flag_t >;
using UBB_T       = lbm::UBB< LatticeModel_T, flag_t >;

using SimplePressure_T = lbm::SimplePressure< LatticeModel_T, flag_t >;
using Pressure_T       = lbm::Pressure< LatticeModel_T, flag_t >;
using Outlet_T         = lbm::Outlet< LatticeModel_T, FlagField_T >;
using SimplePAB_T      = lbm::SimplePAB< LatticeModel_T, FlagField_T >;

using SimpleDiffusionDirichlet_T = lbm::SimpleDiffusionDirichlet< LatticeModel_T, flag_t >;

using BoundaryHandling_T =
   BoundaryHandling< FlagField_T, Stencil_T, NoSlip_T, FreeSlip_T, SimpleUBB_T, UBB_T, SimplePressure_T, Pressure_T,
                     Outlet_T, SimplePAB_T, SimpleDiffusionDirichlet_T >;

class MyBoundaryHandling
{
 public:
   MyBoundaryHandling(const BlockDataID& flagFieldID, const BlockDataID& pdfFieldID, const BoundarySetup& setup,
                      const std::shared_ptr< lbm::TimeTracker >& timeTracker)
      : flagFieldID_(flagFieldID), pdfFieldID_(pdfFieldID), setup_(setup)
   {}

   BoundaryHandling_T* operator()(IBlock* const block, const StructuredBlockStorage* const storage) const;

 private:
   const BlockDataID flagFieldID_;
   const BlockDataID pdfFieldID_;

   BoundarySetup setup_;

}; // class MyBoundaryHandling

BoundaryHandling_T* MyBoundaryHandling::operator()(IBlock* const block,
                                                   const StructuredBlockStorage* const storage) const
{
   Vector3< real_t > domainSize(real_c(storage->getNumberOfXCells()), real_c(storage->getNumberOfYCells()),
                                real_c(storage->getNumberOfZCells()));

   real_t H = domainSize[1];

   WALBERLA_ASSERT_NOT_NULLPTR(block)

   //! [boundaryHandling_T fields]
   FlagField_T* flagField = block->getData< FlagField_T >(flagFieldID_);
   PdfField_T* pdfField   = block->getData< PdfField_T >(pdfFieldID_);

   const auto fluidFlag = flagField->getOrRegisterFlag(FluidFlagUID);
   //! [boundaryHandling_T fields]

   BoundaryHandling_T* handling = new BoundaryHandling_T(
      "Boundary Handling", flagField, fluidFlag,
      //! [handling_NoSlip]
      NoSlip_T("NoSlip", NoSlipFlagUID, pdfField),
      //! [handling_NoSlip]
      //! [handling_SimpleUBB]
      SimpleUBB_T("SimpleUBB", SimpleUBBFlagUID, pdfField, setup_.inflowVelocity),
      //! [handling_SimpleUBB]
      //! [handling_SimplePressure]
      SimplePressure_T("SimplePressure", SimplePressureFlagUID, pdfField, setup_.outflowPressure),
      //! [handling_SimplePressure]
      //! [handling_Outlet]
      Outlet_T("Outlet", OutletFlagUID, pdfField, flagField, fluidFlag),
      //! [handling_Outlet]
      SimpleDiffusionDirichlet_T("SimpleDiffusionDirichlet", SimpleDiffusionDirichletFlagUID, pdfField, flagField,
                                 fluidFlag));

   //! [domainBB]
   CellInterval domainBB = storage->getDomainCellBB();
   storage->transformGlobalToBlockLocalCellInterval(domainBB, *block);
   //! [domainBB]

   //! [westBoundary]
   cell_idx_t ghost = cell_idx_t(FieldGhostLayers);

   domainBB.xMin() -= ghost;
   domainBB.xMax() += ghost;

   // Define the cells which are on the boundary on the bottom plane (zMin). Each CellInterval is defined by two
   // corners of the interval. between the corners there is a line which is a domain boundary.
   CellInterval west(domainBB.xMin(), domainBB.yMin(), domainBB.zMin(), domainBB.xMin(), domainBB.yMax(),
                     domainBB.zMin());
   CellInterval east(domainBB.xMax(), domainBB.yMin(), domainBB.zMin(), domainBB.xMax(), domainBB.yMax(),
                     domainBB.zMin());

   domainBB.yMin() -= ghost;
   domainBB.yMax() += ghost;

   CellInterval south(domainBB.xMin(), domainBB.yMin(), domainBB.zMin(), domainBB.xMax(), domainBB.yMin(),
                      domainBB.zMin());
   CellInterval north(domainBB.xMin(), domainBB.yMax(), domainBB.zMin(), domainBB.xMax(), domainBB.yMax(),
                      domainBB.zMin());

   if (setup_.inflowType == "SimpleUBB")
   {
      //! [forceBoundary_SimpleUBB]
      handling->forceBoundary(SimpleUBBFlagUID, west);
      //! [forceBoundary_SimpleUBB]
   }
   else { WALBERLA_ABORT("Please specify a valid inflow type.") }

   if (setup_.outflowType == "SimplePressure")
   {
      //! [forceBoundary_SimplePressure]
      handling->forceBoundary(SimplePressureFlagUID, east);
      //! [forceBoundary_SimplePressure]
   }
   else if (setup_.outflowType == "Outlet")
   {
      //! [forceBoundary_Outlet]
      handling->forceBoundary(OutletFlagUID, east);
      //! [forceBoundary_Outlet]
   }
   else { WALBERLA_ABORT("Please specify a valid outflow type.") }

   //! [fillDomain]
   handling->fillWithDomain(domainBB);
   //! [fillDomain]

   return handling;
}

int main(int argc, char** argv)
{
   walberla::Environment walberlaEnv(argc, argv);

   auto blocks = blockforest::createUniformBlockGridFromConfig(walberlaEnv.config());

   // read parameters
   auto parameters = walberlaEnv.config()->getOneBlock("Parameters");

   const real_t omega = parameters.getParameter< real_t >("omega", real_c(1.4));
   const Vector3< real_t > initialVelocity =
      parameters.getParameter< Vector3< real_t > >("initialVelocity", Vector3< real_t >());
   const uint_t timesteps = parameters.getParameter< uint_t >("timesteps", uint_c(10));

   const real_t remainingTimeLoggerFrequency =
      parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(3.0)); // in seconds

   // create fields
   LatticeModel_T const latticeModel = LatticeModel_T(lbm::collision_model::SRT(omega));
   BlockDataID const pdfFieldID =
      lbm::addPdfFieldToStorage(blocks, "pdf field", latticeModel, initialVelocity, real_t(1));
   BlockDataID const flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field", FieldGhostLayers);

   // create and initialize boundary handling

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("Boundaries");

   BoundarySetup setup;
   setup.wallType        = boundariesConfig.getParameter< std::string >("wallType", "NoSlip");
   setup.inflowType      = boundariesConfig.getParameter< std::string >("inflowType", "SimpleUBB");
   setup.outflowType     = boundariesConfig.getParameter< std::string >("outflowType", "SimplePressure");
   setup.inflowVelocity  = boundariesConfig.getParameter< Vector3< real_t > >("inflowVelocity", Vector3< real_t >());
   setup.outflowPressure = boundariesConfig.getParameter< real_t >("outflowPressure", real_t(1));

   setup.period = boundariesConfig.getParameter< real_t >("period", real_t(100));

   if (setup.inflowType == "ParserUBB") setup.parser = boundariesConfig.getBlock("Parser");

   setup.omega = omega;

   //! [timeTracker]
   std::shared_ptr< lbm::TimeTracker > const timeTracker = std::make_shared< lbm::TimeTracker >();
   //! [timeTracker]

   //! [boundaryHandlingID]
   BlockDataID const boundaryHandlingID = blocks->addStructuredBlockData< BoundaryHandling_T >(
      MyBoundaryHandling(flagFieldID, pdfFieldID, setup, timeTracker), "boundary handling");
   //! [boundaryHandlingID]

   // create time loop
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   // create communication for PdfField
   blockforest::communication::UniformBufferedScheme< CommunicationStencil_T > communication(blocks);
   communication.addPackInfo(make_shared< lbm::PdfFieldPackInfo< LatticeModel_T > >(pdfFieldID));

   // add LBM sweep and communication to time loop
   //! [boundarySweep]
   timeloop.add() << BeforeFunction(communication, "communication")
                  << Sweep(BoundaryHandling_T::getBlockSweep(boundaryHandlingID), "boundary handling");
   //! [boundarySweep]
   timeloop.add() << Sweep(
      makeSharedSweep(lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >(pdfFieldID, flagFieldID, FluidFlagUID)),
      "LB stream & collide");

   // increment time step counter
   //! [timeTracker_coupling]
   timeloop.addFuncAfterTimeStep(makeSharedFunctor(timeTracker), "time tracking");
   //! [timeTracker_coupling]

   // LBM stability check
   timeloop.addFuncAfterTimeStep(makeSharedFunctor(field::makeStabilityChecker< PdfField_T, FlagField_T >(
                                    walberlaEnv.config(), blocks, pdfFieldID, flagFieldID, FluidFlagUID)),
                                 "LBM stability check");

   // log remaining time
   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
                                 "remaining time logger");

   // add VTK output to time loop
   //   lbm::VTKOutput< LatticeModel_T, FlagField_T >::addToTimeloop(timeloop, blocks, walberlaEnv.config(), pdfFieldID,
   //                                                                flagFieldID, FluidFlagUID);

   auto vtkConfig = walberlaEnv.config()->getBlock("VTK");

   uint_t const writeFrequency =
      vtkConfig.getBlock("fluid_field").getParameter< uint_t >("writeFrequency", uint_t(100));

   auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "fluid_field", writeFrequency, FieldGhostLayers, false,
                                                   "vtk_out", "simulation_step", false, true, true, false, 0);

   auto velocityWriter  = std::make_shared< lbm::VelocityVTKWriter< LatticeModel_T, float > >(pdfFieldID, "velocity");
   auto flagFieldWriter = std::make_shared< field::VTKWriter< FlagField_T > >(flagFieldID, "flag field");

   vtkOutput->addCellDataWriter(velocityWriter);
   vtkOutput->addCellDataWriter(flagFieldWriter);

   timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTKOutput");

   // create adaptors, so that the GUI also displays density and velocity
   // adaptors are like fields with the difference that they do not store values
   // but calculate the values based on other fields ( here the PdfField )
   field::addFieldAdaptor< lbm::Adaptor< LatticeModel_T >::Density >(blocks, pdfFieldID, "DensityAdaptor");
   field::addFieldAdaptor< lbm::Adaptor< LatticeModel_T >::VelocityVector >(blocks, pdfFieldID, "VelocityAdaptor");

   if (parameters.getParameter< bool >("useGui", false))
   {
      GUI gui(timeloop, blocks, argc, argv);
      lbm::connectToGui< LatticeModel_T >(gui);
      gui.run();
   }
   else { timeloop.run(); }

   return EXIT_SUCCESS;
}
} // namespace walberla

int main(int argc, char** argv) { walberla::main(argc, argv); }
