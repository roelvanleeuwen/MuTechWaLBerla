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
//! \file ViscousLDC.cpp
//! \author Jonas Plewinski <jonas.plewinski@fau.de>
//
// This showcase simulates a viscous mirrored lid driven cavity (velocity induced BC at the bottom), with a free surface
// at the top. The implementation uses an LBM kernel generated with lbmpy.
//======================================================================================================================

#include "blockforest/Initialization.h"

#include "core/Environment.h"

#include "field/Gather.h"

#include "geometry/InitBoundaryHandling.h"

#include "lbm/PerformanceLogger.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/free_surface/SurfaceMeshWriter.h"
#include "lbm/free_surface/TotalMassComputer.h"
#include "lbm/free_surface/VtkWriter.h"
#include "lbm/free_surface/bubble_model/Geometry.h"
#include "lbm/free_surface/dynamics/SurfaceDynamicsHandler.h"
#include "lbm/free_surface/surface_geometry/SurfaceGeometryHandler.h"
#include "lbm/free_surface/surface_geometry/Utility.h"
#include "lbm/lattice_model/D3Q19.h"

#include "SchwipSchwapLatticeModel.h"
//-#include "ViscousLDCLatticeModelThermal.h"

#include "GenDefines.h"

#include "core/perf_analysis/extern/likwid.h"

namespace walberla
{
namespace free_surface
{
namespace SchwipSchwapCodegen
{
using ScalarField_T          = GhostLayerField< real_t, 1 >;
using VectorField_T          = GhostLayerField< Vector3< real_t >, 1 >;
using VectorFieldFlattened_T = GhostLayerField< real_t, 3 >;
using VelocityField_T        = walberla::field::GhostLayerField<double, 3>;

// Fluid
using LatticeModelFluid_T        = lbm::SchwipSchwapLatticeModel;
using LatticeModelFluidStencil_T = LatticeModelFluid_T::Stencil;
using PdfFieldFluid_T            = lbm::PdfField< LatticeModelFluid_T >;
using PdfCommunicationFluid_T    = blockforest::SimpleCommunication< LatticeModelFluidStencil_T >;

// the geometry computations in SurfaceGeometryHandler require meaningful values in the ghost layers in corner
// directions (flag field and fill level field); this holds, even if the lattice model uses a D3Q19 stencil
using CommunicationStencilFluid_T =
   typename std::conditional< LatticeModelFluid_T::Stencil::D == uint_t(2), stencil::D2Q9, stencil::D3Q27 >::type;
using CommunicationFluid_T = blockforest::SimpleCommunication< CommunicationStencilFluid_T >;

// Thermal
//-using LatticeModelThermal_T        = lbm::ViscousLDCLatticeModelThermal;
//-using LatticeModelThermalStencil_T = LatticeModelThermal_T::Stencil;
//-using PdfFieldThermal_T            = lbm::PdfField< LatticeModelThermal_T >;
//-using PdfCommunicationThermal_T    = blockforest::SimpleCommunication< LatticeModelThermalStencil_T >;

//-using CommunicationStencilThermal_T =
//-   typename std::conditional< LatticeModelThermal_T::Stencil::D == uint_t(2), stencil::D2Q9, stencil::D3Q7 >::type;
//-using CommunicationThermal_T = blockforest::SimpleCommunication< CommunicationStencilThermal_T >;

using flag_t                        = uint32_t;
using FlagField_T                   = FlagField< flag_t >;
using FreeSurfaceBoundaryHandling_T = FreeSurfaceBoundaryHandling< LatticeModelFluid_T, FlagField_T, ScalarField_T >;

//const FlagUID NoSlipFlagUID("NoSlip Flag");
//using NoSlipThermal_T    = lbm::NoSlip< LatticeModelThermal_T, flag_t >;
//typedef BoundaryHandling< FlagField_T, LatticeModelThermalStencil_T, NoSlipThermal_T > BoundaryHandlingThermal_T;

//-typedef lbm::DefaultBoundaryHandlingFactory< LatticeModelThermal_T, FlagField_T > BHFactory;
const FlagUID FluidFlagUID("Fluid Flag");

// function describing the global initialization profile
inline real_t initializationProfile(real_t x, real_t amplitude, real_t offset, real_t wavelength)
{
   return amplitude * std::cos(x / wavelength * real_c(2) * math::pi + math::pi) + offset;
}

int main(int argc, char** argv)
{
   Environment walberlaEnv(argc, argv);

   if (argc < 2) { WALBERLA_ABORT("Please specify a parameter file as input argument.") }

   LIKWID_MARKER_INIT
   LIKWID_MARKER_THREADINIT
   //LIKWID_MARKER_REGISTER("fluid")

   // print content of parameter file
   WALBERLA_LOG_INFO_ON_ROOT(*walberlaEnv.config());

   // get block forest parameters from parameter file
   auto blockForestParameters              = walberlaEnv.config()->getOneBlock("BlockForestParameters");
   const Vector3< uint_t > cellsPerBlock   = blockForestParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
   const Vector3< bool > periodicity       = blockForestParameters.getParameter< Vector3< bool > >("periodicity");

   // get domain parameters from parameter file
   auto domainParameters         = walberlaEnv.config()->getOneBlock("DomainParameters");
   //const uint_t domainWidth      = domainParameters.getParameter< uint_t >("domainWidth");
   //const real_t liquidDepth      = domainParameters.getParameter< real_t >("liquidDepth");
   const real_t initialAmplitude = domainParameters.getParameter< real_t >("initialAmplitude");

   const uint_t nrOfProcesses = uint_c(MPIManager::instance()->numProcesses());
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(nrOfProcesses)

   Vector3< uint_t > blocksPerDimension;
   Vector3< uint_t > cellsPerBlockDummy;
   blockforest::calculateCellDistribution(cellsPerBlock, nrOfProcesses, blocksPerDimension, cellsPerBlockDummy);
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(blocksPerDimension)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellsPerBlockDummy)

   // define domain size
   Vector3< uint_t > domainSize;
   domainSize[0] = blocksPerDimension[0] * cellsPerBlock[0]; //domainWidth;
   domainSize[1] = blocksPerDimension[1] * cellsPerBlock[1]; //uint_c(liquidDepth * real_c(2));
   domainSize[2] = cellsPerBlock[2]; //uint_c(1);

   const real_t liquidDepth = real_c(domainSize[1]) / 2;

   // compute number of blocks as defined by domainSize and cellsPerBlock
   Vector3< uint_t > numBlocks;
   numBlocks[0] = uint_c(std::ceil(real_c(domainSize[0]) / real_c(cellsPerBlock[0])));
   numBlocks[1] = uint_c(std::ceil(real_c(domainSize[1]) / real_c(cellsPerBlock[1])));
   numBlocks[2] = uint_c(std::ceil(real_c(domainSize[2]) / real_c(cellsPerBlock[2])));

   // get number of (MPI) processes
   uint_t numProcesses = uint_c(MPIManager::instance()->numProcesses());
   WALBERLA_CHECK_LESS_EQUAL(numProcesses, numBlocks[0] * numBlocks[1] * numBlocks[2],
                             "The number of MPI processes is greater than the number of blocks as defined by "
                             "\"domainSize/cellsPerBlock\". This would result in unused MPI processes. Either decrease "
                             "the number of MPI processes or increase \"cellsPerBlock\".")

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numProcesses)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellsPerBlock)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(domainSize)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numBlocks)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(liquidDepth)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(initialAmplitude)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(periodicity)

   // get physics parameters from parameter file
   auto physicsParameters = walberlaEnv.config()->getOneBlock("PhysicsParameters");
   const uint_t timesteps = physicsParameters.getParameter< uint_t >("timesteps");

   const real_t relaxationRate = physicsParameters.getParameter< real_t >("relaxationRate");
   const real_t viscosity      = real_c(1) / real_c(3) * (real_c(1) / relaxationRate - real_c(0.5));

   //--
   //? const real_t omegaThermal = relaxationRate;
   //--

   const real_t reynoldsNumber = physicsParameters.getParameter< real_t >("reynoldsNumber");
   const real_t waveNumber     = real_c(2) * math::pi / real_c(domainSize[0]);
   const real_t waveFrequency  = reynoldsNumber * viscosity / real_c(domainSize[0]) / initialAmplitude;
   const real_t accelerationY  = -(waveFrequency * waveFrequency) / waveNumber / std::tanh(waveNumber * liquidDepth);
   std::shared_ptr< Vector3< real_t > > acceleration = std::make_shared< Vector3< real_t > >(-real_c(accelerationY/2), accelerationY, real_c(0));
   const Vector3< real_t > accelerationOrigin(-real_c(accelerationY), accelerationY, real_c(0));

   const bool enableWetting  = physicsParameters.getParameter< bool >("enableWetting");
   const real_t contactAngle = physicsParameters.getParameter< real_t >("contactAngle");

   const real_t amplitude = physicsParameters.getParameter< real_t >("amplitude", real_c(1.0));
   const real_t wavelength = physicsParameters.getParameter< real_t >("wavelength", real_c(8640));
   const real_t offset = physicsParameters.getParameter< real_t >("offset", real_c(0.0));

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(reynoldsNumber)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(relaxationRate)
   //? WALBERLA_LOG_DEVEL_VAR_ON_ROOT(omegaThermal)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(enableWetting)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(contactAngle)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(timesteps)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(viscosity)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(*acceleration)

   // read model parameters from parameter file
   const auto modelParameters               = walberlaEnv.config()->getOneBlock("ModelParameters");
   const std::string pdfReconstructionModel = modelParameters.getParameter< std::string >("pdfReconstructionModel");
   const std::string pdfRefillingModel      = modelParameters.getParameter< std::string >("pdfRefillingModel");
   const std::string excessMassDistributionModel =
      modelParameters.getParameter< std::string >("excessMassDistributionModel");
   const std::string curvatureModel          = modelParameters.getParameter< std::string >("curvatureModel");
   const bool useSimpleMassExchange          = modelParameters.getParameter< bool >("useSimpleMassExchange");
   const real_t cellConversionThreshold      = modelParameters.getParameter< real_t >("cellConversionThreshold");
   const real_t cellConversionForceThreshold = modelParameters.getParameter< real_t >("cellConversionForceThreshold");
   const bool enableBubbleModel              = modelParameters.getParameter< bool >("enableBubbleModel");
   const bool enableBubbleSplits             = modelParameters.getParameter< bool >("enableBubbleSplits");

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(pdfReconstructionModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(pdfRefillingModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(excessMassDistributionModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(curvatureModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(useSimpleMassExchange)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellConversionThreshold)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(cellConversionForceThreshold)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(enableBubbleModel)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(enableBubbleSplits)

   // read evaluation parameters from parameter file
   const auto evaluationParameters      = walberlaEnv.config()->getOneBlock("EvaluationParameters");
   const uint_t performanceLogFrequency = evaluationParameters.getParameter< uint_t >("performanceLogFrequency");
   const uint_t evaluationFrequency     = evaluationParameters.getParameter< uint_t >("evaluationFrequency");
   const std::string filename           = evaluationParameters.getParameter< std::string >("filename");

   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(performanceLogFrequency)
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(filename)

   // create non-uniform block forest (non-uniformity required for load balancing)
   //const std::shared_ptr< StructuredBlockForest > blockForest =
   //   createUniformBlockForest(domainSize, cellsPerBlock, numBlocks, periodicity);

   std::shared_ptr< StructuredBlockForest > blockForest = blockforest::createUniformBlockGrid(numBlocks[0], numBlocks[1], numBlocks[2],             // blocks
                                                     cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2], // cells
                                                     real_c(1.0),                                          // dx
                                                     true, // one block per process
                                                     periodicity[0], periodicity[1], periodicity[2]); // periodicity

   // add force field
   const BlockDataID forceDensityFieldID =
      field::addToStorage< VectorFieldFlattened_T >(blockForest, "Force field", real_c(0), field::fzyx, uint_c(1));

   // add velocity field -> necessary for temperature lbm step as input
   BlockDataID velocityFieldID =
      field::addToStorage< VelocityField_T >(blockForest, "velocity fluid", real_c(0.0), field::fzyx);

   // create lattice model
   LatticeModelFluid_T latticeModelFluid = LatticeModelFluid_T(forceDensityFieldID, velocityFieldID, relaxationRate);

   // add pdf field
   const BlockDataID pdfFieldFluidID = lbm::addPdfFieldToStorage(blockForest, "PDF field fluid", latticeModelFluid, field::fzyx);

   // add fill level field (initialized with 0, i.e., gas everywhere)
   const BlockDataID fillFieldID =
      field::addToStorage< ScalarField_T >(blockForest, "Fill level field", real_c(0.0), field::fzyx, uint_c(2));

   //--
   ////////////////////////
   // ADD DATA TO BLOCKS //
   ////////////////////////
   //? BlockDataID thermal_PDFs_ID =
   //?    field::addToStorage< PdfField_thermal_T >(blockForest, "LB PDF field thermal", real_c(0.0), field::fzyx);
   //? BlockDataID velocity_field_ID =
   //?    field::addToStorage< VelocityField_T >(blockForest, "velocity", real_c(0.0), field::fzyx);
   //? BlockDataID temperature_field_ID =
   //?    field::addToStorage< TemperatureField_T >(blockForest, "Temperature", real_c(0.0), field::fzyx);
   //--

   // add boundary handling
   const std::shared_ptr< FreeSurfaceBoundaryHandling_T > freeSurfaceBoundaryHandling =
      std::make_shared< FreeSurfaceBoundaryHandling_T >(blockForest, pdfFieldFluidID, fillFieldID);
   const BlockDataID flagFieldID                                      = freeSurfaceBoundaryHandling->getFlagFieldID();
   const typename FreeSurfaceBoundaryHandling_T::FlagInfo_T& flagInfo = freeSurfaceBoundaryHandling->getFlagInfo();

   // samples used in the Monte-Carlo like estimation of the fill level
   const uint_t fillLevelInitSamples = uint_c(100); // actually there will be 101 since 0 is also included

   const uint_t numTotalPoints = (fillLevelInitSamples + uint_c(1)) * (fillLevelInitSamples + uint_c(1));
   const real_t stepsize       = real_c(1) / real_c(fillLevelInitSamples);

   // initialize sine profile such that there is exactly one period in the domain, i.e., with wavelength=domainSize[0];
   // every length is normalized with domainSize[0]
   for (auto blockIt = blockForest->begin(); blockIt != blockForest->end(); ++blockIt)
   {
      ScalarField_T* const fillField = blockIt->getData< ScalarField_T >(fillFieldID);

      WALBERLA_FOR_ALL_CELLS(fillFieldIt, fillField, {
         // cell in block-local coordinates
         const Cell localCell = fillFieldIt.cell();

         // get cell in global coordinates
         Cell globalCell = fillFieldIt.cell();
         blockForest->transformBlockLocalToGlobalCell(globalCell, *blockIt, localCell);

         // Monte-Carlo like estimation of the fill level:
         // create uniformly-distributed sample points in each cell and count the number of points below the sine
         // profile; this fraction of points is used as the fill level to initialize the profile
         uint_t numPointsBelow = uint_c(0);

         for (uint_t xSample = uint_c(0); xSample <= fillLevelInitSamples; ++xSample)
         {
            // value of the sine-function
            const real_t functionValue = initializationProfile(real_c(globalCell[0]) + real_c(xSample) * stepsize,
                                                               initialAmplitude, liquidDepth, real_c(domainSize[0]));

            for (uint_t ySample = uint_c(0); ySample <= fillLevelInitSamples; ++ySample)
            {
               const real_t yPoint = real_c(globalCell[1]) + real_c(ySample) * stepsize;
               // with operator <, a fill level of 1 can not be reached when the line is equal to the cell's top border;
               // with operator <=, a fill level of 0 can not be reached when the line is equal to the cell's bottom
               // border
               if (yPoint < functionValue) { ++numPointsBelow; }
            }
         }

         // fill level is fraction of points below sine profile
         fillField->get(localCell) = real_c(numPointsBelow) / real_c(numTotalPoints);
      }) // WALBERLA_FOR_ALL_CELLS
   }

   // initialize domain boundary conditions from config file
   const auto boundaryParameters = walberlaEnv.config()->getOneBlock("BoundaryParameters");
   freeSurfaceBoundaryHandling->initFromConfig(boundaryParameters);

   // IMPORTANT REMARK: this must be only called after every solid flag has been set; otherwise, the boundary handling
   // might not detect solid flags correctly
   freeSurfaceBoundaryHandling->initFlagsFromFillLevel();

   // communication after initialization
   CommunicationFluid_T communication(blockForest, flagFieldID, fillFieldID, forceDensityFieldID);
   communication();

   PdfCommunicationFluid_T pdfCommunication(blockForest, pdfFieldFluidID);
   pdfCommunication();

   // add bubble model
   std::shared_ptr< bubble_model::BubbleModelBase > bubbleModel = nullptr;
   if (enableBubbleModel)
   {
      const std::shared_ptr< bubble_model::BubbleModel< LatticeModelFluidStencil_T > > bubbleModelDerived =
         std::make_shared< bubble_model::BubbleModel< LatticeModelFluidStencil_T > >(blockForest, enableBubbleSplits);
      bubbleModelDerived->initFromFillLevelField(fillFieldID);
      bubbleModelDerived->setAtmosphere(Cell(domainSize[0] - uint_c(1), domainSize[1] - uint_c(1), uint_c(0)),
                                        real_c(1));

      bubbleModel = std::static_pointer_cast< bubble_model::BubbleModelBase >(bubbleModelDerived);
   }
   else { bubbleModel = std::make_shared< bubble_model::BubbleModelConstantPressure >(real_c(1)); }

   // initialize hydrostatic pressure
   initHydrostaticPressure< PdfFieldFluid_T >(blockForest, pdfFieldFluidID, Vector3<real_t>(real_c(0), (*acceleration)[1], real_c(0)), liquidDepth);

   // initialize force density field
   initForceDensityFieldCodegen< PdfFieldFluid_T, FlagField_T, VectorFieldFlattened_T, ScalarField_T >(
      blockForest, forceDensityFieldID, fillFieldID, pdfFieldFluidID, flagFieldID, flagInfo, *acceleration);

   // set density in non-liquid or non-interface cells to 1 (after initializing with hydrostatic pressure)
   setDensityInNonFluidCellsToOne< FlagField_T, PdfFieldFluid_T >(blockForest, flagInfo, flagFieldID, pdfFieldFluidID);

   //----------------------------- Thermal Stuff -----------------------------
   /*-// add temperature field
   const BlockDataID temperatureFieldID =
      field::addToStorage< ScalarField_T >(blockForest, "Temperature field", real_c(0.0), field::fzyx, uint_c(1));

   // create lattice model
   //todo currently same relaxation rate as fluid
   LatticeModelThermal_T latticeModelThermal = LatticeModelThermal_T(temperatureFieldID, velocityFieldID, relaxationRate);

   // add pdf field
   const BlockDataID pdfFieldThermalID = lbm::addPdfFieldToStorage(blockForest, "PDF field", latticeModelFluid, field::fzyx);

   auto boundariesThermalConfig = walberlaEnv.config()->getOneBlock("BoundaryThermalParameters");
   BlockDataID boundaryHandlingID = BHFactory::addBoundaryHandlingToStorage(blockForest, "boundary handling thermal", flagFieldID, pdfFieldThermalID, FluidFlagUID, Vector3<real_t>{0., 0., 0.}, Vector3<real_t>{0., 0., 0.}, 0., 0.);
   geometry::initBoundaryHandling< BHFactory::BoundaryHandling >(*blockForest, boundaryHandlingID, boundariesThermalConfig);
   geometry::setNonBoundaryCellsToDomain< BHFactory::BoundaryHandling >(*blockForest, boundaryHandlingID);
   */
   //--

   //--
   //todo temperature initialization
   //? pystencils::initialize_thermal_field initializeThermalField(thermal_PDFs_ID, temperature_field_ID,
   //?                                                             velocity_field_ID);

   ///////////////////////////////
   // ADD THERMAL COMMUNICATION //
   ///////////////////////////////
   //? auto UniformBufferedSchemeThermalDistributions =
   //?    std::make_shared< blockforest::communication::UniformBufferedScheme< Stencil_thermal_T > >(blockForest);
   //? auto generatedPackInfo_thermal = std::make_shared< walberla::pystencils::PackInfo_thermal >(thermal_PDFs_ID);
   //? UniformBufferedSchemeThermalDistributions->addPackInfo(generatedPackInfo_thermal);
   //? auto Comm_thermal = std::function< void() >([&]() { UniformBufferedSchemeThermalDistributions->communicate(); });

   // initialize the two lattice Boltzmann fields
   //? WALBERLA_LOG_INFO_ON_ROOT("initialization of the distributions")
   //? for (auto& block : *blockForest)
   //? {
      //initializeFluidField(&block);
   //?    initializeThermalField(&block);
   //? }
   //? WALBERLA_LOG_INFO_ON_ROOT("initialization of the distributions done")
   //Comm_hydro();
   //? Comm_thermal();
   //--

   //-PdfCommunicationThermal_T pdfThermalCommunication(blockForest, pdfFieldThermalID);
   //-pdfCommunication();
   //-------------------------------------------------------------------------

   // create timeloop
   SweepTimeloop timeloop(blockForest, timesteps);

   const real_t surfaceTension = real_c(0);

   // Laplace pressure = 2 * surface tension * curvature; curvature computation is not necessary with no surface
   // tension
   bool computeCurvature = false;
   if (!realIsEqual(surfaceTension, real_c(0), real_c(1e-14))) { computeCurvature = true; }

   // add surface geometry handler
   const SurfaceGeometryHandler< LatticeModelFluid_T, FlagField_T, ScalarField_T, VectorField_T > geometryHandler(
      blockForest, freeSurfaceBoundaryHandling, fillFieldID, curvatureModel, computeCurvature, enableWetting,
      contactAngle);

   geometryHandler.addSweeps(timeloop);

   // get fields created by surface geometry handler
   const ConstBlockDataID curvatureFieldID = geometryHandler.getConstCurvatureFieldID();
   const ConstBlockDataID normalFieldID    = geometryHandler.getConstNormalFieldID();

   // add boundary handling for standard boundaries and free surface boundaries
   const SurfaceDynamicsHandler< LatticeModelFluid_T, FlagField_T, ScalarField_T, VectorField_T, true,
                                 VectorFieldFlattened_T >
      dynamicsHandler(blockForest, pdfFieldFluidID, flagFieldID, fillFieldID, forceDensityFieldID, normalFieldID,
                      curvatureFieldID, freeSurfaceBoundaryHandling, bubbleModel, pdfReconstructionModel,
                      pdfRefillingModel, excessMassDistributionModel, relaxationRate, acceleration, surfaceTension,
                      useSimpleMassExchange, cellConversionThreshold, cellConversionForceThreshold);

   dynamicsHandler.addSweeps(timeloop);

   // add evaluator for total and excessive mass (mass that is currently undistributed)
   const std::shared_ptr< real_t > totalMass  = std::make_shared< real_t >(real_c(0));
   const std::shared_ptr< real_t > excessMass = std::make_shared< real_t >(real_c(0));
   const TotalMassComputer< FreeSurfaceBoundaryHandling_T, PdfFieldFluid_T, FlagField_T, ScalarField_T > totalMassComputer(
      blockForest, freeSurfaceBoundaryHandling, pdfFieldFluidID, fillFieldID, dynamicsHandler.getConstExcessMassFieldID(),
      evaluationFrequency, totalMass, excessMass);
   timeloop.addFuncAfterTimeStep(totalMassComputer, "Evaluator: total mass");

   //-------------------------- Thermal Stuff --------------------------
   //? pystencils::thermal_lb_step thermal_lb_step(thermal_PDFs_ID, temperature_field_ID, velocity_field_ID, omegaThermal);

   // Boundaries Thermal
   //? BlockDataID flagFieldThermalID = field::addFlagFieldToStorage< FlagField_T >(blockForest, "flag field thermal");
   //? const FlagUID fluidFlagThermalUID("Thermal");
   //? const FlagUID TcoldUID("BC_thermal_Tcold");
   //? const FlagUID ThotUID("BC_thermal_Thot");
   //? const auto boundariesConfigThermal = walberlaEnv.config()->getOneBlock("BoundaryParametersThermal");
   //auto boundariesConfigThermal = config->getBlock("Boundaries_Thermal");
   //? if (boundariesConfigThermal)
   //? {
   //?    geometry::initBoundaryHandling< FlagField_T >(*blockForest, flagFieldThermalID, boundariesConfigThermal);
   //?    geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blockForest, flagFieldThermalID, fluidFlagThermalUID);
   //? }

   //? const real_t temperatureHot = real_t(0.5);
   //? const real_t temperatureCold = -temperatureHot;
   //? lbm::BC_thermal_Tcold thermal_Tcold(blockForest, thermal_PDFs_ID, temperatureCold);
   //? thermal_Tcold.fillFromFlagField< FlagField_T >(blockForest, flagFieldThermalID, TcoldUID, fluidFlagThermalUID);
   //? lbm::BC_thermal_Thot thermal_Thot(blockForest, thermal_PDFs_ID, temperatureHot);
   //? thermal_Thot.fillFromFlagField< FlagField_T >(blockForest, flagFieldThermalID, ThotUID, fluidFlagThermalUID);

   //? timeloop.add() << Sweep(thermal_Tcold, "Thermal Tcold boundary conditions");
   //? timeloop.add() << Sweep(thermal_Thot, "Thermal Thot boundary conditions")
   //?                << AfterFunction(Comm_thermal, "Communication of thermal PDFs");
   //? timeloop.add() << Sweep(thermal_lb_step, "Thermal LB Step");
   //-------------------------------------------------------------------

   //todo add thermal VTK output!!
   // add VTK output
   addVTKOutput< LatticeModelFluid_T, FreeSurfaceBoundaryHandling_T, PdfFieldFluid_T, FlagField_T, ScalarField_T, VectorField_T,
                 true, VectorFieldFlattened_T >(
      blockForest, timeloop, walberlaEnv.config(), flagInfo, pdfFieldFluidID, flagFieldID, fillFieldID, forceDensityFieldID,
      geometryHandler.getCurvatureFieldID(), geometryHandler.getNormalFieldID(),
      geometryHandler.getObstNormalFieldID());

   // add triangle mesh output of free surface
   SurfaceMeshWriter< ScalarField_T, FlagField_T > surfaceMeshWriter(
      blockForest, fillFieldID, flagFieldID, flagIDs::liquidInterfaceGasFlagIDs, real_c(0), walberlaEnv.config());
   surfaceMeshWriter(); // write initial mesh
   timeloop.addFuncAfterTimeStep(surfaceMeshWriter, "Writer: surface mesh");

   // add logging for computational performance
   const lbm::PerformanceLogger< FlagField_T > performanceLogger(
      blockForest, flagFieldID, flagIDs::liquidInterfaceFlagIDs, performanceLogFrequency);
   timeloop.addFuncAfterTimeStep(performanceLogger, "Evaluator: performance logging");

   WcTimingPool timingPool;

   //LIKWID_MARKER_START("fluid")
   for (uint_t t = uint_c(0); t != timesteps; ++t)
   {
      timeloop.singleStep(timingPool, true);

      (*acceleration)[0] = accelerationOrigin[0] * amplitude * std::sin(real_c(t) / wavelength * real_c(2) * math::pi + math::pi) + offset;

      WALBERLA_ROOT_SECTION()
      {
         // non-dimensionalize time and surface position
         const real_t tNonDimensional        = real_c(t) * waveFrequency;

         if (t % evaluationFrequency == uint_c(0))
         {
            WALBERLA_LOG_DEVEL("time step = " << t << "\n\t\ttNonDimensional = " << tNonDimensional
                                              << "\n\t\ttotal mass = "
                                              << *totalMass << "\n\t\texcess mass = " << *excessMass
                                              << "\n\t\taccelerationX = " << (*acceleration)[0])
         }
      }

      //if (t % performanceLogFrequency == uint_c(0) && t > uint_c(0)) { timingPool.logResultOnRoot(); }
   }
   //LIKWID_MARKER_STOP("fluid")

   WALBERLA_LOG_INFO_ON_ROOT("test")
   LIKWID_MARKER_CLOSE
   WALBERLA_LOG_INFO_ON_ROOT("test2")
   return EXIT_SUCCESS;
}

} // namespace GravityWave
} // namespace free_surface
} // namespace walberla

int main(int argc, char** argv) { return walberla::free_surface::SchwipSchwapCodegen::main(argc, argv); }