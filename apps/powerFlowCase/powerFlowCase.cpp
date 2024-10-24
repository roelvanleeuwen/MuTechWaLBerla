/*
================================================================================================================================================================
\file: powerFlowCase.cpp
Author: R. van Leeuwen
Contact: roelvanleeuwen8@gmail.com
Company: MuTech
Created: 23-09-2024


*/

// =====================================================================================================================
//! \file powerFlowCase.cpp
//! \author R. van Leeuwen
//! \ingroup apps
//! \date 23-09-2024
// ================================================================================================================================================================
// File description:
//
// This is the main file of the waLBerla implementation of the airfoil flow simulation ran in
// PowerFlow by
// Marlon van Crugten. The simulation consists of a NACA 0018 airfoil with the MuteSkin add-on. The aim
// of the simulation is to understand the working principle of the add-on better so de understanding can be used in
// optimising MuteSkin.

// The simulation runs a 2.5D airfoil with add-on. The airfoil and the flow have an angle of 0.0
// degrees.
// The flow has a velocity of 20 m/s. The ambient pressure is 101325 Pa.

// The domain must be large to prevent boundary effects. The domain is 10 times the chord length of the
// airfoil in the x and z direction. In the y direction

#ifndef __GNUC__
#endif
#include "blockforest/all.h"

#include "core/all.h"
#include "core/debug/Debug.h"

#include "domain_decomposition/all.h"

#include "field/AddToStorage.h"
#include "field/FlagField.h"
#include "field/GhostLayerField.h"
#include "field/all.h"
#include "field/communication/PackInfo.h"
#include "field/iterators/FieldPointer.h"
#include "field/iterators/IteratorMacros.h"

#include "geometry/all.h"

#include "gui/all.h"

#include "lbm/all.h"
#include "lbm/blockforest/communication/SimpleCommunication.h"
#include "lbm/lattice_model/SmagorinskyLES.h"

#include "mesh/blockforest/BlockExclusion.h"
#include "mesh/blockforest/BlockForestInitialization.h"
#include "mesh/blockforest/BlockWorkloadMemory.h"
#include "mesh/blockforest/RefinementSelection.h"
#include "mesh/boundary/BoundaryInfo.h"
#include "mesh/boundary/BoundaryLocation.h"
#include "mesh/boundary/BoundaryLocationFunction.h"
#include "mesh/boundary/BoundarySetup.h"
#include "mesh/boundary/BoundaryUIDFaceDataSource.h"
#include "mesh/boundary/ColorToBoundaryMapper.h"

#include "timeloop/SweepTimeloop.h"
#include "timeloop/all.h"

#include "vtk/ChainedFilter.h"
#include "vtk/all.h"

#include "mesh_common/DistanceComputations.h"
#include "mesh_common/DistanceFunction.h"
#include "mesh_common/MatrixVectorOperations.h"
#include "mesh_common/MeshIO.h"
#include "mesh_common/MeshOperations.h"
#include "mesh_common/TriangleMeshes.h"
#include "mesh_common/distance_octree/DistanceOctree.h"
#include "mesh_common/vtk/CommonDataSources.h"
#include "mesh_common/vtk/VTKMeshWriter.h"
// #include "spongeZone_2.h"
// #include "apps/powerFlowCase/myVTKOutput.h"
#include "unitConversion.cpp"
#include "xyAdjustment.cpp"

namespace walberla
{
namespace MuTech
{
namespace powerFlowCase
{

struct Setup
{
   // Flow parameters
   real_t kinViscositySI;               // physical kinematic viscosity
   real_t rhoSI;                        // physical density
   real_t temperatureSI;                // physical temperature
   real_t angleOfAttack;                // physical angle of attack
   real_t velocityMagnitudeSI;          // physical velocity magnitude
   real_t initialVelocityMagnitudeSI;   // initial velocity magnitude
   Vector3< real_t > initialVelocitySI; // physical initial flow velocity
   Vector3< real_t > flowVelocitySI;    // physical flow velocity
   real_t MachSI;                       // physical Mach number
   real_t ReChord;                      // physical Reynolds number

   // Space and time parameters
   real_t dxSI; // physical grid spacing
   real_t dtSI; // physical time step

   // Domain parameters
   std::string meshFile;                 // mesh file containing the complex geometry (airfoil)
   bool scalePowerFlowDomain;            // scale the domain equal to the PowerFlow case of Marlon van Crugten
   real_t decreasePowerFlowDomainFactor; // decrease the PowerFlow domain by a factor 'decreasePowerFlowDomainFactor'
   Vector3< real_t > domainScaling;      // scaling of the domain w.r.t the meshFile dimensions
   real_t meshZScaling; // adjust scaling of the mesh in the z direction (change the span of the airfoil)

   real_t xyAdjuster_x; // adjust the x dimension of the domain to perfectly fit the number of blocks in the x direction
   real_t xyAdjuster_y; // adjust the y dimension of the domain to perfectly fit the number of blocks in the y direction

   uint_t numLevels;
   uint_t numGhostLayers;
   Vector3< bool > periodicity;

   // Block data
   Vector3< uint_t > cellsPerBlock; // Number of cells in each block in the < x, y, z > directions. This is also called
                                    // blockSize in some codes. For refinement at least < 16, 16, 16 > is required
   real_t nBlocks_x;                // number of blocks in the x direction
   real_t nBlocks_y;                // number of blocks in the y direction
   real_t nBlocks_z;                // number of blocks in the z direction

   // Domain data in lattice units and physical units
   real_t domainLengthLU; // x dimension in latice units e.g. number of cells in x direction
   real_t domainHeightLU; // y dimension in latice units e.g. number of cells in y direction
   real_t domainWidthLU;  // z dimension in latice units e.g. number of cells in z direction

   real_t domainLengthSI; // physical x dimension of the domain in meters
   real_t domainHeightSI; // physical y dimension of the domain in meters
   real_t domainWidthSI;  // physical z dimension of the domain in meters

   // Airfoil data
   real_t airfoilXPositionSI; // x position of the airfoil origin (LE) in meters
   real_t airfoilYPositionSI; // y position of the airfoil origin (LE) in meters

   real_t airfoilChordLengthSI; // airfoil chord length in meters
   real_t airfoilThicknessSI;   // airfoil thickness in meters
   real_t airfoilSpanSI;        // airfoil span in meters

   real_t airfoilChordLengthLU; // airfoil chord length in lattice units
   real_t airfoilThicknessLU;   // airfoil thickness in lattice units
   real_t airfoilSpanLU;        // airfoil span in lattice units

   // LBM parameters
   real_t spongeInnerThicknessFactor; // inner radius factor of the sponge layer >> r_inner = factor * max(L/2, H/2)
   real_t spongeOuterThicknessFactor; // outer radius factor of the sponge layer >> r_outer = factor * max(L/2, H/2)
   real_t sponge_nuT_min; // minimum value of additional numerical viscosity in the sponge layer. This is a factor and
                          // multiplied with the base viscosity of the fluid
   real_t sponge_nuT_max; // maximum value of additional numerical viscosity in the sponge layer. This is a factor and
                          // multiplied with the base viscosity of the fluid

   real_t omegaEffective;      // chosen relaxation parameter
   real_t smagorinskyConstant; // Smagorinsky constant for LES

   // Output parameters
   uint_t timeSteps;                    // number of time steps
   real_t remainingTimeLoggerFrequency; // frequency of the remaining time logger in seconds

   // Define the operator<< for Setup
   friend std::ostream& operator<<(std::ostream& os, const Setup& setup)
   {
      os << "================= Setup ===============: \n";
      os << "Flow parameters:\n";
      os << "  kinViscositySI: " << setup.kinViscositySI << " m2/s \n";
      os << "  rhoSI: " << setup.rhoSI << " kg/m3 \n";
      os << "  temperatureSI: " << setup.temperatureSI << " K \n";
      os << "  angleOfAttack: " << setup.angleOfAttack << " deg \n";
      os << "  velocityMagnitudeSI: " << setup.velocityMagnitudeSI << " m/s \n";
      os << "  initialVelocityMagnitudeSI: " << setup.initialVelocityMagnitudeSI << " m/s \n";
      os << "  initialVelocitySI: " << setup.initialVelocitySI << " m/s \n";
      os << "  flowVelocitySI: " << setup.flowVelocitySI << " m/s \n";
      os << "  MachSI: " << setup.MachSI << " - \n";
      os << "  ReSI: " << setup.ReChord << " - \n";
      os << " \n";

      os << "Space and time parameters:\n";
      os << "  dxSI: " << setup.dxSI << " m \n";
      os << "  dtSI: " << setup.dtSI << " m \n";
      os << " \n";

      os << "Airfoil data:\n";
      os << "  meshFile: " << setup.meshFile << "\n";
      os << "  airfoilXPositionSI: " << setup.airfoilXPositionSI << " m \n";
      os << "  airfoilYPositionSI: " << setup.airfoilYPositionSI << " m \n";
      os << "  airfoilChordLengthSI: " << setup.airfoilChordLengthSI << " m \n";
      os << "  airfoilThicknessSI: " << setup.airfoilThicknessSI << " m \n";
      os << "  airfoilSpanSI: " << setup.airfoilSpanSI << " m \n";
      os << " \n";

      os << "Domain scaling:\n";
      os << "  scalePowerFlowDomain: " << setup.scalePowerFlowDomain << "\n";
      os << "  decreasePowerFlowDomainFactor: " << setup.decreasePowerFlowDomainFactor << "\n";
      os << "  domainScaling: " << setup.domainScaling << "\n";
      os << "  meshZScaling: " << setup.meshZScaling << "\n";
      os << "  xyAdjuster_x: " << setup.xyAdjuster_x << "\n";
      os << "  xyAdjuster_y: " << setup.xyAdjuster_y << "\n";
      os << "  numLevels: " << setup.numLevels << "\n";
      os << "  numGhostLayers: " << setup.numGhostLayers << "\n";

      os << "Domain data:\n";
      os << "  cellsPerBlock: " << setup.cellsPerBlock << "\n";
      os << "  nBlocks_x: " << setup.nBlocks_x << "\n";
      os << "  nBlocks_y: " << setup.nBlocks_y << "\n";
      os << "  nBlocks_z: " << setup.nBlocks_z << "\n";
      os << "  domainLengthSI: " << setup.domainLengthSI << " m \n";
      os << "  domainHeightSI: " << setup.domainHeightSI << " m\n";
      os << "  domainWidthSI: " << setup.domainWidthSI << " m \n";
      os << "  domainLengthLU: " << setup.domainLengthLU << " lu \n";
      os << "  domainHeightLU: " << setup.domainHeightLU << " lu \n";
      os << "  domainWidthLU: " << setup.domainWidthLU << " lu \n";
      os << " \n";

      os << "Boundary conditions:\n";
      os << "  periodicity: " << setup.periodicity << "\n";
      os << " \n";

      os << "LBM parameters:\n";
      os << "  spongeInnerThicknessFactor: " << setup.spongeInnerThicknessFactor << "\n";
      os << "  spongeOuterThicknessFactor: " << setup.spongeOuterThicknessFactor << "\n";
      os << "  sponge_nuT_min: " << setup.sponge_nuT_min << " - \n";
      os << "  sponge_nuT_max: " << setup.sponge_nuT_max << " - \n";
      os << "  omegaEffective: " << setup.omegaEffective << " - \n";
      os << "  smagorinskyConstant: " << setup.smagorinskyConstant << " - \n";
      os << " \n";

      os << "Output parameters:\n";
      os << "  timeSteps: " << setup.timeSteps << " ts \n";
      os << "  remainingTimeLoggerFrequency: " << setup.remainingTimeLoggerFrequency << " s \n";
      os << " \n";

      // Add other members as needed
      return os;
   }
};

using ScalarField_T = GhostLayerField< real_t, 1 >;
// using LatticeModel_T = lbm::D3Q19< lbm::collision_model::SRT >;
using LatticeModel_T = lbm::D3Q19< lbm::collision_model::SRTField< ScalarField_T >, true, lbm::force_model::None, 1 >;
using Stencil_T      = LatticeModel_T::Stencil;

using CommunicationStencil_T = LatticeModel_T::CommunicationStencil;

using PdfField_T = lbm::PdfField< LatticeModel_T >; // Probability density function field for the lattice model. The pdf
                                                    // describes the microscopic particle motion.

using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >; // Flag field for the lattice model. The flag field describes the boundary
                                         // conditions of the lattice model.

#pragma region MESH_OPERATIONS

template< typename MeshType >
void vertexToFaceColor(MeshType& mesh, const typename MeshType::Color& defaultColor)
{
   WALBERLA_CHECK(mesh.has_vertex_colors())
   mesh.request_face_colors();

   for (auto faceIt = mesh.faces_begin(); faceIt != mesh.faces_end(); ++faceIt)
   {
      typename MeshType::Color vertexColor;

      bool useVertexColor = true;

      auto vertexIt = mesh.fv_iter(*faceIt);
      WALBERLA_ASSERT(vertexIt.is_valid())

      vertexColor = mesh.color(*vertexIt);

      ++vertexIt;
      while (vertexIt.is_valid() && useVertexColor)
      {
         if (vertexColor != mesh.color(*vertexIt)) useVertexColor = false;
         ++vertexIt;
      }

      mesh.set_color(*faceIt, useVertexColor ? vertexColor : defaultColor);
   }
}

#pragma endregion MESH_OPERATIONS

#pragma region SPONGE_ZONE

// Function to calculate the psi value based on the given x and dxSI
real_t psi(real_t x, real_t dxSI) { return x > 0 ? std::exp(-dxSI / x) : 0; }

// Function to calculate the phi value based on the given x, x_min, and x_max
real_t phi(real_t x, real_t x_min = 0.0, real_t x_max = 1.0)
{
   real_t dxSI = x_max - x_min;
   if (x <= x_min) { return 0; }
   else if (x_min < x && x < x_max) { return psi(x - x_min, dxSI) / (psi(x - x_min, dxSI) + psi(x_max - x, dxSI)); }
   else { return 1; }
}

// Class to perform the Omega sweep operation
class OmegaSweep
{
 public:
   // Constructor to initialize the OmegaSweep class with the given parameters
   OmegaSweep(BlockDataID pdfFieldId, BlockDataID omegaFieldId, AABB& domain, Setup setup, Units units)
      : pdfFieldId_(pdfFieldId), omegaFieldId_(omegaFieldId), domain_(domain), setup_(setup), units_(units)
   {}

   // Operator to perform the sweep operation on the given block
   void operator()(IBlock* block)
   {
      auto pdfField   = block->getData< PdfField_T >(pdfFieldId_); // Get the PDF field of the block
      auto omegaField = block->getData< ScalarField_T >(omegaFieldId_);
      auto blockAABB  = block->getAABB(); // Get the AABB of the block

      WALBERLA_ASSERT_NOT_NULLPTR(pdfField);
      WALBERLA_ASSERT_NOT_NULLPTR(omegaField);

      WALBERLA_ASSERT_EQUAL(pdfField->xyzSizeWithGhostLayer(), omegaField->xyzSizeWithGhostLayer());

      // Calculate the inner and outer radius of the sponge zone
      real_t sponge_rmin = setup_.spongeInnerThicknessFactor * std::max(domain_.yMax(), domain_.xMax());
      real_t sponge_rmax = setup_.spongeOuterThicknessFactor * std::max(domain_.yMax(), domain_.xMax());

      // Iterate over all cells in the block
      for (auto it = pdfField->begin(); it != pdfField->end(); ++it)
      {
         // Calculate the global coordinates of the cell
         real_t xglobal = blockAABB.xMin() + (blockAABB.xMax() - blockAABB.xMin()) * it.x() /
                                                static_cast< real_t >(setup_.cellsPerBlock[0]);
         real_t yglobal = blockAABB.yMin() + (blockAABB.yMax() - blockAABB.yMin()) * it.y() /
                                                static_cast< real_t >(setup_.cellsPerBlock[1]);

         // Calculate the distance from the center of the domain to the cell
         real_t dist =
            std::sqrt(std::pow(xglobal - domain_.center()[0], 2.0) + std::pow(yglobal - domain_.center()[1], 2.0));

         // Calculate the additional kinematic viscosity to be added to the lattice kinematic viscosity
         real_t nuTAdd = setup_.sponge_nuT_min +
                         (setup_.sponge_nuT_max - setup_.sponge_nuT_min) * phi(dist, sponge_rmin, sponge_rmax);
         real_t nuAddFactor = nuTAdd * setup_.temperatureSI;

         // Calculate the lattice kinematic viscosity
         real_t viscosity_old = pdfField->latticeModel().collisionModel().viscosity(it.x(), it.y(), it.z());
         real_t nuLU          = units_.kinViscosityLU * nuAddFactor + viscosity_old;

         // Calculate the relaxation rate omega based on the lattice kinematic viscosity
         real_t omega = 1.0 / (std::pow(units_.pseudoSpeedOfSoundLU, 2.0) * nuLU + 0.5);

         // Ensure the relaxation rate omega is within the physical range
         WALBERLA_ASSERT(omega > 0.0 && omega < 2.0);
         // Set the relaxation rate omega in the collision model of the lattice model
         omegaField->get(it.x(), it.y(), it.z()) = omega;
         // pdfField->latticeModel().collisionModel().reset(it.x(), it.y(), it.z(), omega);
      }
   }

 private:
   const BlockDataID pdfFieldId_; // PDF field ID
   BlockDataID omegaFieldId_;
   AABB domain_; // Domain
   Setup setup_; // Setup parameters
   Units units_; // Units
};

#pragma endregion SPONGE_ZONE

#pragma region VTK_OUTPUT_FUNCTIONS

std::shared_ptr< walberla::vtk::VTKOutput >
   myAABBVTKOutput(const std::string identifier, const shared_ptr< StructuredBlockForest >& blocks,
                   const BlockDataID pdfFieldId, const BlockDataID omegaFieldId, const BlockDataID flagFieldId,
                   const FlagUID fluidFlagUID, const walberla::config::Config::BlockHandle& VTKconfig,
                   const Setup setup, const Units units)
{
   // zone specific parameters
   const auto zoneParams                  = VTKconfig.getBlock(identifier);
   const std::string zoneName             = zoneParams.getParameter("identifier", std::string("no_name"));
   uint_t zoneWriteFrequency              = zoneParams.getParameter("writeFrequency", uint_t(0));
   const uint_t zoneInitialExecutionCount = zoneParams.getParameter("initialExecutionCount", uint_t(0));
   const Vector3< real_t > resolutionLevel =
      zoneParams.getParameter("resolutionLevel", Vector3< real_t >(setup.numLevels));

   // Create the VTK output object. Many options are hard coded since they will not be changed.
   auto zoneOutput =
      vtk::createVTKOutput_BlockData(*blocks, zoneName, zoneWriteFrequency, 0, false, "results/vtk_out", "simulation_step",
                                     false, true, true, false, zoneInitialExecutionCount, false, false);

   // Set the sampling resolution of the VTK output. This is in SI units, based on the coarses grid resolution and the
   // desired level of resolution which is the same as the refinement levels.
   zoneOutput->setSamplingResolution(
      setup.dxSI * std::pow(1 / 2, resolutionLevel[0]), setup.dxSI * std::pow(1 / 2, resolutionLevel[1]),
      setup.dxSI * std::pow(1 / 2, resolutionLevel[2])); // Sampling resolution in SI dx, dy and dz

   // Specifiy the AABB of the zone and make a filter for this.
   Vector3< real_t > AABBMin = zoneParams.getParameter< Vector3< real_t > >("AABBMin", Vector3< real_t >(-1.0));
   Vector3< real_t > AABBMax = zoneParams.getParameter< Vector3< real_t > >("AABBMax", Vector3< real_t >(1.0));
   AABB AABBZone(AABBMin[0], AABBMin[1], AABBMin[2], AABBMax[0], AABBMax[1], AABBMax[2]);
   vtk::AABBCellFilter AABBZoneFilter(AABBZone);

   // Create a filter to only include the fluid cells in the zone
   field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagFieldId);
   fluidFilter.addFlag(fluidFlagUID);

   // Combine the filters. It is important to first add the AABB filter and then the fluid filter.
   vtk::ChainedFilter combinedFilter;
   combinedFilter.addFilter(AABBZoneFilter);
   combinedFilter.addFilter(fluidFilter);

   // Add the combined filter to the zone output.
   zoneOutput->addCellInclusionFilter(combinedFilter);

   // Add the cell data writers to the zone output. The data consists of the velocity, density, pressure, omega, etc.
   // You can choose which you want in the paramter file
   if (zoneParams.getParameter("writeVelocity", false))
   {
      auto velocitySIWriterZone = make_shared< lbm::VelocitySIVTKWriter< LatticeModel_T, float > >(
         pdfFieldId, units.xSI, units.tSI, "Velocity ");
      zoneOutput->addCellDataWriter(velocitySIWriterZone);
   }

   if (zoneParams.getParameter("writeDensity", false))
   {
      auto densitySIWriterZone =
         make_shared< lbm::DensitySIVTKWriter< LatticeModel_T, float > >(pdfFieldId, units.rhoSI, "Density");
      zoneOutput->addCellDataWriter(densitySIWriterZone);
   }

   if (zoneParams.getParameter("writeOmega", false))
   {
      auto omegaWriterZone = make_shared< field::VTKWriter< ScalarField_T > >(omegaFieldId, "Omega");
      zoneOutput->addCellDataWriter(omegaWriterZone);
   }

   if (zoneParams.getParameter("writePressure", false))
   {
      auto pressureSIWriterZone = make_shared< lbm::DensitySIVTKWriter< LatticeModel_T, float > >(
         pdfFieldId, units.rhoSI * std::pow(units.speedOfSoundSI, 2), "Pressure");
      zoneOutput->addCellDataWriter(pressureSIWriterZone);
   }

   return zoneOutput;

   // Example use:
   // auto VTKOutput = myAABBVTKOutput("zone1", blocks, pdfFieldId, omegaFieldId, flagFieldId, fluidFlagUID, VTKParams,
   // setup, units); vtk::writeFiles(VTKOutput)
}

// get the current velocity and density at the given coordinate "evalCoord"
template< typename PdfField_T >
class VelDensPointEvaluator
{
 public:
   VelDensPointEvaluator(const std::shared_ptr< StructuredBlockForest >& blocks, const BlockDataID& pdfFieldID,
                         const Vector3< real_t >& evalCoord, uint_t evalInterval,
                         const std::shared_ptr< std::vector< real_t > >& densityVelocity)
      : blocks_(blocks), pdfFieldID_(pdfFieldID), evalCoord_(evalCoord), evalInterval_(evalInterval),
        densityVelocity_(densityVelocity), executionCounter_(uint_c(0))
   {}

   void operator()()
   {
      ++executionCounter_;

      // only evaluate in given intervals
      if (executionCounter_ % evalInterval_ != uint_c(0)) { return; }

      // update density and velocity at evalCoord_
      updateDensityVelocity();
   }

 private:
   const std::shared_ptr< StructuredBlockForest > blocks_;
   const BlockDataID pdfFieldID_;
   const Vector3< real_t > evalCoord_;
   const uint_t evalInterval_;
   const std::shared_ptr< std::vector< real_t > > densityVelocity_;

   uint_t executionCounter_; // number of times operator() has been called

   void updateDensityVelocity()
   {
      real_t density = real_c(0);
      Vector3< real_t > velocity(real_c(0));
      
      for (auto blockIterator = blocks_->begin(); blockIterator != blocks_->end(); ++blockIterator)
      {
         if (blockIterator->getAABB().contains(evalCoord_, 1e-6))
         {
            Cell localCell       = blocks_->getBlockLocalCell(*blockIterator, evalCoord_);
            PdfField_T* pdfField = blockIterator->template getData< PdfField_T >(pdfFieldID_);

            velocity = pdfField->getVelocity(localCell);
            density  = pdfField->getDensity(localCell);

            *densityVelocity_ = { density, velocity[0], velocity[1], velocity[2], length(velocity) };
         }
      }
   }
}; // class VelDensPointEvaluator

template< typename T >
std::string toStringWithPrecision(const T a_value, const int n /*= 12*/)
{
   std::ostringstream out;
   out.precision(n);

   out << std::fixed << a_value;
   return out.str();
}

template< typename T >
void writeVector(const std::vector< T >& data, const uint_t& timestep, const std::string& filename)
{
   std::ofstream file;
   file.open(filename, std::ofstream::app);

   file << timestep;
   for (const auto i : data)
   {
      file << "\t" << toStringWithPrecision(i, 12);
   }
   file << "\n";
   file.close();
}

#pragma endregion VTK_OUTPUT_FUNCTIONS

int main(int argc, char** argv)
{
   walberla::Environment walberlaEnv(argc, argv);

   mpi::MPIManager::instance()->useWorldComm();

   // ================================================================================================================

#pragma region PARAMETER BLOCKS
   // Initialize the parameters blocks
   auto optionsParameters    = walberlaEnv.config()->getOneBlock("options");
   auto flowParameters       = walberlaEnv.config()->getOneBlock("flowParameters");
   auto simulationParameters = walberlaEnv.config()->getOneBlock("simulationParameters");
   auto domainParameters     = walberlaEnv.config()->getOneBlock("domainParameters");
   auto spongeZoneParameters = walberlaEnv.config()->getOneBlock("spongeZoneParameters");
   auto boudaryParameters    = walberlaEnv.config()->getOneBlock("boundaryConditions");
   auto VTKParams            = walberlaEnv.config()->getBlock("VTK");

#pragma endregion PARAMETER BLOCKS

   Setup setup; // Create a setup structure to store all the parameters

#pragma region PARAMETER_READ
   // Read the meaning of the parameters from the parameter blocks in the configuration file powerFlowCase.prm

   // Flow parameters
   setup.angleOfAttack              = flowParameters.getParameter< real_t >("angleOfAttack", real_c(0.0));
   setup.velocityMagnitudeSI        = flowParameters.getParameter< real_t >("velocityMagnitudeSI", real_c(20.0));
   setup.initialVelocityMagnitudeSI = flowParameters.getParameter< real_t >("initialVelocityMagnitudeSI", real_c(20.0));
   setup.kinViscositySI             = flowParameters.getParameter< real_t >("kinViscositySI", real_t(1.451e-6));
   setup.rhoSI                      = flowParameters.getParameter< real_t >("rhoSI", real_t(1.225));
   setup.temperatureSI              = flowParameters.getParameter< real_t >("temperatureSI", real_t(288.1));

   // Simulation parameters
   setup.omegaEffective = simulationParameters.getParameter< real_t >("omega", real_t(1.6));
   setup.timeSteps      = simulationParameters.getParameter< uint_t >("timeSteps", uint_c(10));
   setup.remainingTimeLoggerFrequency =
      simulationParameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_t(3.0));
   setup.smagorinskyConstant = simulationParameters.getParameter< real_t >("smagorinskyConstant", real_t(0.12));

   // Domain parameters
   setup.meshFile             = domainParameters.getParameter< std::string >("meshFile");
   setup.airfoilChordLengthSI = domainParameters.getParameter< real_t >("chord", real_t(1));
   setup.airfoilThicknessSI   = domainParameters.getParameter< real_t >("thickness", real_t(0.2));
   setup.airfoilSpanSI        = domainParameters.getParameter< real_t >("span", real_t(1));
   setup.scalePowerFlowDomain = domainParameters.getParameter< bool >("scalePowerFlowDomain", false);
   setup.decreasePowerFlowDomainFactor =
      domainParameters.getParameter< real_t >("decreasePowerFlowDomainFactor", real_t(1));
   setup.dxSI           = domainParameters.getParameter< real_t >("dxSI", real_t(1));
   setup.numLevels      = domainParameters.getParameter< uint_t >("numLevels", uint_t(1));
   setup.numLevels      = std::max(setup.numLevels, uint_t(1));
   setup.numGhostLayers = domainParameters.getParameter< uint_t >("numGhostLayers", uint_t(4));
   setup.meshZScaling   = domainParameters.getParameter< real_t >("meshZScaling", 1);

   // Sponge zone parameters
   setup.spongeInnerThicknessFactor =
      spongeZoneParameters.getParameter< real_t >("spongeInnerThicknessFactor", real_t(1));
   setup.spongeOuterThicknessFactor =
      spongeZoneParameters.getParameter< real_t >("spongeOuterThicknessFactor", real_t(1));
   setup.sponge_nuT_min = spongeZoneParameters.getParameter< real_t >("sponge_nuT_min", real_t(0.0));
   setup.sponge_nuT_max = spongeZoneParameters.getParameter< real_t >("sponge_nuT_max", real_t(0.5));

   // Boundary conditions
   setup.periodicity = boudaryParameters.getParameter< Vector3< bool > >("periodicity", Vector3< bool >(false));

   // Calculate the initial physical velocity [m/s] based on the angle of attack and the velocity magnitude
   setup.initialVelocitySI =
      Vector3< real_t >(setup.initialVelocityMagnitudeSI * std::cos(setup.angleOfAttack * M_PI / 180.0),
                        setup.initialVelocityMagnitudeSI * std::sin(setup.angleOfAttack * M_PI / 180.0), real_t(0));
   setup.flowVelocitySI =
      Vector3< real_t >(setup.velocityMagnitudeSI * std::cos(setup.angleOfAttack * M_PI / 180.0),
                        setup.velocityMagnitudeSI * std::sin(setup.angleOfAttack * M_PI / 180.0), real_t(0));

   real_t dxFineSI = setup.dxSI * std::pow(2.0, setup.numLevels);

   WALBERLA_LOG_INFO_ON_ROOT(" Checkpoint 1: Parameters read done ")

#pragma endregion PARAMETER_READ

#pragma region OBJECT_MESH_SETUP

   // read in mesh with vertex colors on a single process and broadcast it
   auto mesh = make_shared< mesh::TriangleMesh >();
   mesh->request_vertex_colors();

   // read the mesh file and broadcast it to all processes. This is necessary for parallel processing
   mesh::readAndBroadcast(setup.meshFile, *mesh);

   // Scale the mesh in the z direction to change the span of the airfoil
   for (auto vertexIt = mesh->vertices_begin(); vertexIt != mesh->vertices_end(); ++vertexIt)
   {
      auto point = mesh->point(*vertexIt);
      point[2] *= setup.meshZScaling; // Scale the z coordinate
      mesh->set_point(*vertexIt, point);
   }

   // color faces according to vertices
   vertexToFaceColor(*mesh, mesh::TriangleMesh::Color(255, 255, 255));

   // add information to mesh that is required for computing signed distances from a point to a triangle
   auto triDist = make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh);

   // building distance octree to determine which cells are inside the airfoil and which are outside
   auto distanceOctree = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(triDist);
   WALBERLA_LOG_INFO_ON_ROOT(" Checkpoint 2: Mesh setup done ")

   // write mesh info to file
   if (optionsParameters.getParameter("writeAirfoilMeshAndReturn", false))
   {
      mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriter(mesh, "airfoil_mesh", 1);
      meshWriter.addDataSource(make_shared< mesh::ColorFaceDataSource< mesh::TriangleMesh > >());
      meshWriter.addDataSource(make_shared< mesh::ColorVertexDataSource< mesh::TriangleMesh > >());
      meshWriter();
      WALBERLA_LOG_INFO_ON_ROOT("Mesh written to vtk_out/airfoil_mesh.vtu")
      WALBERLA_LOG_INFO_ON_ROOT("Stopping program")
      return EXIT_SUCCESS;
   }

#pragma endregion OBJECT_MESH_SETUP

#pragma region DOMAIN_SIZING
   // Make the axis-aligned bounding box (AABB) of a mesh object.
   auto aabb = computeAABB(*mesh);

   // What is the size of the mesh object in the x, y and z direction?
   setup.airfoilChordLengthSI = aabb.xSize();              // chord length of the airfoil in meters
   setup.airfoilThicknessSI   = aabb.ySize();              // thickness of the airfoil in meters
   setup.airfoilSpanSI        = aabb.zSize();              // span of the airfoil in meters
   setup.airfoilChordLengthLU = aabb.xSize() / setup.dxSI; // number of cells per chord length
   setup.airfoilThicknessLU   = aabb.ySize() / setup.dxSI; // number of cells per thickness
   setup.airfoilSpanLU        = aabb.zSize() / setup.dxSI; // number of cells per span

   // The x dimension is 100 times the airfoil chord. This is not to be changed. Therefore it is not in the parameter
   // file. Based on this the x and y dimension of the domain are adjusted so that the number of blocks in the x and y
   // direction are maximised to allow for refinement and that the blocks fit perfectly in the domain.
   AdjustmentResult adjustXYResult = xyAdjustment(100 * aabb.xSize(), setup.decreasePowerFlowDomainFactor, setup.dxSI);
   setup.xyAdjuster_x              = adjustXYResult.xyAdjustment;
   setup.xyAdjuster_y              = adjustXYResult.xyAdjustment;
   setup.cellsPerBlock             = Vector3< uint_t >(adjustXYResult.cellsPerBlock_x, adjustXYResult.cellsPerBlock_x,
                                                       16); // The z direction has 16 cells per block

   if (setup.scalePowerFlowDomain)
   {
      // The chord length of the airfoil is the x size of the mesh object
      setup.domainScaling =
         Vector3< real_t >(100 * setup.decreasePowerFlowDomainFactor * setup.xyAdjuster_x,
                           100 * aabb.xSize() / aabb.ySize() * setup.decreasePowerFlowDomainFactor * setup.xyAdjuster_y,
                           static_cast< real_t >(setup.cellsPerBlock[2]) * setup.dxSI / aabb.zSize());
   }
   else
   {
      setup.domainScaling = domainParameters.getParameter< Vector3< real_t > >("domainScaling", Vector3< real_t >(1));
   }

   // Blow up the domain and place the airfoil in the center of the domain
   aabb.scale(setup.domainScaling);
   aabb.setCenter(aabb.center());

   setup.domainLengthSI = aabb.xSize();              // physical x dimension of the domain in meters
   setup.domainHeightSI = aabb.ySize();              // physical y dimension of the domain in meters
   setup.domainWidthSI  = aabb.zSize();              // physical z dimension of the domain in meters
   setup.domainLengthLU = aabb.xSize() / setup.dxSI; // x dimension in latice units e.g. number of cells in x direction
   setup.domainHeightLU = aabb.ySize() / setup.dxSI; // y dimension in latice units e.g. number of cells in y direction
   setup.domainWidthLU  = aabb.zSize() / setup.dxSI; // z dimension in latice units e.g. number of cells in z direction

   // Position of the airfoil origin (LE) in meters
   setup.airfoilXPositionSI = aabb.center()[0];
   setup.airfoilYPositionSI = aabb.center()[1];

   // What will be the number of blocks in the x, y and z direction?
   setup.nBlocks_x = static_cast< real_t >(adjustXYResult.nBlocks_x);
   setup.nBlocks_y = static_cast< real_t >(adjustXYResult.nBlocks_x);
   setup.nBlocks_z = std::round(aabb.zSize() / setup.dxSI / static_cast< real_t >(setup.cellsPerBlock[2]));

   WALBERLA_ASSERT_LESS(
      std::abs(aabb.xSize() - setup.dxSI * setup.nBlocks_x * static_cast< real_t >(setup.cellsPerBlock[0])), 1e-6,
      "The blocks do not fit in the x direction")
   WALBERLA_ASSERT_LESS(
      std::abs(aabb.ySize() - setup.dxSI * setup.nBlocks_y * static_cast< real_t >(setup.cellsPerBlock[1])), 1e-6,
      "The blocks do not fit in the y direction")
   WALBERLA_ASSERT_LESS(
      std::abs(aabb.zSize() - setup.dxSI * setup.nBlocks_z * static_cast< real_t >(setup.cellsPerBlock[2])), 1e-6,
      "The blocks do not fit in the z direction")

   WALBERLA_LOG_INFO_ON_ROOT(" Checkpoint 3: Domain sizing done ")
#pragma endregion DOMAIN_SIZING

#pragma region BLOCK_FOREST_CREATION
   // create the structured block forest creator
   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, Vector3< real_t >(setup.dxSI));

   // exclude the object mesh interior with maximum error of dxSI and refine the maximum error to dxFineSI
   bfc.setRootBlockExclusionFunction(mesh::makeExcludeMeshInterior(distanceOctree, setup.dxSI));
   bfc.setBlockExclusionFunction(mesh::makeExcludeMeshInteriorRefinement(distanceOctree, dxFineSI));

   // set the workload memory assignment function
   auto meshWorkloadMemory = mesh::makeMeshWorkloadMemory(distanceOctree, setup.dxSI);
   meshWorkloadMemory.setInsideCellWorkload(1);
   meshWorkloadMemory.setOutsideCellWorkload(1);
   bfc.setWorkloadMemorySUIDAssignmentFunction(meshWorkloadMemory);
   bfc.setPeriodicity(setup.periodicity);
   bfc.setRefinementSelectionFunction(
      makeRefinementSelection(distanceOctree, setup.numLevels - 1, setup.dxSI, setup.dxSI * real_t(1)));

   // create block forest
   auto blocks = bfc.createStructuredBlockForest(setup.cellsPerBlock);
   WALBERLA_LOG_INFO_ON_ROOT(" Checkpoint 4: Block forest created ")

   // write setup forest and return. This shows all the blocks in the domain
   if (optionsParameters.getParameter< bool >("writeSetupForestAndReturn", false))
   {
      WALBERLA_LOG_INFO_ON_ROOT("")
      auto setupForest = bfc.createSetupBlockForest(setup.cellsPerBlock);
      WALBERLA_LOG_INFO_ON_ROOT("Writing SetupBlockForest to VTK file")
      WALBERLA_ROOT_SECTION() { setupForest->writeVTKOutput("vtk_out/SetupBlockForest"); }

      WALBERLA_LOG_INFO_ON_ROOT("Stopping program")
      return EXIT_SUCCESS;
   }

#pragma endregion BLOCK_FOREST_CREATION

#pragma region UNIT_CONVERSION_AND_SETUP
   // Calculate the parameters in lattice units
   Units inputUnits;

   inputUnits.xSI            = setup.dxSI;
   inputUnits.speedSI        = setup.velocityMagnitudeSI;
   inputUnits.kinViscositySI = setup.kinViscositySI;
   inputUnits.rhoSI          = setup.rhoSI;
   inputUnits.temperatureSI  = setup.temperatureSI;
   inputUnits.omegaEffective = setup.omegaEffective; // relaxation parameter
   inputUnits.omegaLevel     = 0;                    // level where omega is defined

   Units simulationUnits = convertToLatticeUnitsAcousticScaling(inputUnits); // convert the parameters to lattice units

   simulationUnits.initialVelocityLU =
      setup.initialVelocitySI * simulationUnits.thetaSpeed; // initial velocity in lattice units
   simulationUnits.flowVelocityLU = setup.flowVelocitySI * simulationUnits.thetaSpeed; // flow velocity in lattice units

   setup.dtSI = simulationUnits.tSI; // physical time step
   setup.ReChord =
      setup.airfoilChordLengthSI * setup.velocityMagnitudeSI / setup.kinViscositySI; // Reynolds number based on
                                                                                     // the airfoil chord length
   setup.MachSI = simulationUnits.MachSI;                                            // Mach number

   // std::vector< real_t > kinViscosityLUList(setup.numLevels);
   // std::vector< real_t > omegaList(setup.numLevels);
   // std::vector< real_t > flowVelocityLUList(setup.numLevels);
   // std::vector< real_t > dxList(setup.numLevels);
   // std::vector< real_t > dtList(setup.numLevels);
   // std::vector< real_t > speedOfSoundLUList(setup.numLevels);
   // std::vector< real_t > MachLUList(setup.numLevels);

   // for (uint_t i = 0; i < setup.numLevels; ++i)
   // {
   //    dxList[i]             = setup.dxSI / std::pow(2, i);
   //    kinViscosityLUList[i] = latticeModel.collisionModel().viscosity(i); // viscosity in lattice units
   //    omegaList[i]          = 1 / (3 * kinViscosityLUList[i] + 0.5);      // relaxation parameter
   //    dtList[i]             = kinViscosityLUList[i] * std::pow(dxList[i], 2) / setup.kinViscositySI; // time step
   //    flowVelocityLUList[i] = setup.velocityMagnitudeSI * dtList[i] / dxList[i];
   //    speedOfSoundLUList[i] = 1 / std::pow(3.0, 0.5);                        // speed of sound in lattice units
   //    MachLUList[i]         = flowVelocityLUList[i] / speedOfSoundLUList[i]; // Mach number in lattice units
   // }

   // WALBERLA_LOG_INFO_ON_ROOT("---------------------------------------------------------------")
   // WALBERLA_LOG_INFO_ON_ROOT(
   //    "| Level | omega   | kinViscosityLU | dxSI         | dtSI       | flowVelocityLU | speedOfSoundLU | MachLU |")
   // WALBERLA_LOG_INFO_ON_ROOT("---------------------------------------------------------------")
   // for (uint_t i = 0; i < setup.numLevels; ++i)
   // {
   //    WALBERLA_LOG_INFO_ON_ROOT("| " << std::setw(5) << i << " | " << std::setw(7) << omegaList[i] << " | "
   //                                   << std::setw(14) << kinViscosityLUList[i] << " | " << std::setw(8) <<
   //                                   dxList[i]
   //                                   << " | " << std::setw(8) << dtList[i] << " | " << std::setw(14)
   //                                   << flowVelocityLUList[i] << " | " << std::setw(14) << speedOfSoundLUList[i]
   //                                   << " | " << std::setw(8) << MachLUList[i] << " |")
   // }
   // WALBERLA_LOG_INFO_ON_ROOT("---------------------------------------------------------------")

   WALBERLA_LOG_INFO_ON_ROOT(" Checkpoint 5: Unit conversion done ")

   WALBERLA_LOG_INFO_ON_ROOT(" ======================== Initial Parameters ======================== "
                             "\n "
                             "\n ________________________ Flow parameters ________________________ "
                             "\n + Flow velocity: "
                             << setup.velocityMagnitudeSI
                             << " m/s"
                                "\n + Angle of attack: "
                             << setup.angleOfAttack
                             << " degrees"
                                "\n + Initial velocity < u, v, w >: "
                             << setup.initialVelocitySI
                             << " m/s"
                                " + Chord based Reynolds number: "
                             << setup.ReChord
                             << " - "
                                "\n + Mach number: "
                             << setup.MachSI
                             << " - "
                                "\n + Kinematic viscosity: "
                             << setup.kinViscositySI
                             << " m2/s"
                                "\n + Density: "
                             << setup.rhoSI
                             << " kg/m3"
                                "\n + Temperature: "
                             << setup.temperatureSI
                             << " K"
                                "\n "
                                "\n ________________________ Simulation parameters ________________________ "
                                "\n + Number of time steps: "
                             << setup.timeSteps
                             << ""
                                "\n + Relaxation parameter at the coarsest cells omega = 1/tau : "
                             << setup.omegaEffective
                             << " - "
                                "\n + Theoretical relaxation paramter based on acoustic scaling: "
                             << simulationUnits.omegaLUTheory
                             << "\n "
                                "\n ________________________ Domain parameters ________________________ "
                                "\n + Mesh file: "
                             << setup.meshFile
                             << ""
                                "\n + Coarse lattice spacing: "
                             << setup.dxSI
                             << " m"
                                "\n + Number of refinement levels: "
                             << setup.numLevels
                             << " "
                                "\n + Decrease PowerFlow domain factor: "
                             << setup.decreasePowerFlowDomainFactor
                             << ""
                                "\n + Airfoil size < x, y, z >: "
                             << setup.airfoilChordLengthSI << ", " << setup.airfoilThicknessSI << ", "
                             << setup.airfoilSpanSI
                             << " m "
                                "\n + Domain size < x, y, z >: "
                             << setup.domainLengthSI << ", " << setup.domainHeightSI << ", " << setup.domainWidthSI
                             << " m "
                                "\n + Number of blocks < x, y, z >: "
                             << setup.nBlocks_x << ", " << setup.nBlocks_y << ", " << setup.nBlocks_z
                             << "\n "
                                "\n ________________________ Boundary parameters ________________________ "
                                "\n + Periodicity < x_sides, y_sides, z_sides >: "
                             << setup.periodicity
                             << ""
                                "\n ");

   if (optionsParameters.getParameter< bool >("writeSimulationSetupAndUnits", false))
   {
      // Make a file with the simulation Setup struct to be used in the post pocessing
      WALBERLA_LOG_INFO_ON_ROOT("Writing simulation setup to file")
      std::ofstream outFile("simulation_log.txt");
      if (outFile.is_open())
      {
         outFile << setup;
         outFile << simulationUnits;
         outFile.close();
         WALBERLA_LOG_INFO_ON_ROOT("Simulation log written to file successfully")
      }
      else { WALBERLA_LOG_INFO_ON_ROOT("Failed to open file for writing simulation log") }
   }

#pragma endregion UNIT_CONVERSION_AND_SETUP

#pragma region FIELD_CREATION

   BlockDataID omegaFieldId    = field::addToStorage< ScalarField_T >(blocks, "Flag field", setup.omegaEffective,
                                                                      field::fzyx, setup.numGhostLayers);
   LatticeModel_T latticeModel = LatticeModel_T(lbm::collision_model::SRTField< ScalarField_T >(omegaFieldId));

   BlockDataID pdfFieldId = lbm::addPdfFieldToStorage(
      blocks, "pdf field", latticeModel, simulationUnits.initialVelocityLU, simulationUnits.rhoLU,
      setup.numGhostLayers); // Here the initialisation of the pdf field. This includes the
                             // initial velocity and density in lattice units
   BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field", setup.numGhostLayers);

   WALBERLA_LOG_INFO_ON_ROOT("Checkpoint 6: Fields created")

#pragma endregion FIELD_CREATION

#pragma region BOUNDARY_HANDLING
   // create and initialize boundary handling
   const FlagUID fluidFlagUID("Fluid");

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("boundaryConditions");

   typedef lbm::DefaultBoundaryHandlingFactory< LatticeModel_T, FlagField_T > BHFactory;
   BlockDataID boundaryHandlingId = BHFactory::addBoundaryHandlingToStorage(
      blocks, "boundary handling", flagFieldId, pdfFieldId, fluidFlagUID,
      boundariesConfig.getParameter< Vector3< real_t > >("velocity0", Vector3< real_t >()),
      boundariesConfig.getParameter< Vector3< real_t > >("velocity1", Vector3< real_t >()),
      boundariesConfig.getParameter< real_t >("pressure0", real_c(1.0)),
      boundariesConfig.getParameter< real_t >("pressure1", real_c(1.01)));

   // set NoSlip UID to boundaries that we colored
   mesh::ColorToBoundaryMapper< mesh::TriangleMesh > colorToBoundaryMapper(
      (mesh::BoundaryInfo(BHFactory::getNoSlipBoundaryUID())));

   // mark boundaries
   auto boundaryLocations = colorToBoundaryMapper.addBoundaryInfoToMesh(*mesh);

   // voxelize mesh
   mesh::BoundarySetup boundarySetup(blocks, makeMeshDistanceFunction(distanceOctree), setup.numGhostLayers);

   // write voxelisation to file. This shows all the voxels/cells that are in the domain
   if (optionsParameters.getParameter< bool >("writeVoxelfile", false))
   {
      WALBERLA_LOG_INFO_ON_ROOT("Waypoint 11: Writing Voxelisation");
      boundarySetup.writeVTKVoxelfile();
      WALBERLA_LOG_INFO_ON_ROOT("Voxelisation written to vtk_out/voxelisation.vtu")
      WALBERLA_LOG_INFO_ON_ROOT("Stopping program")
      return EXIT_SUCCESS;
   }

   // set fluid cells
   boundarySetup.setDomainCells< BHFactory::BoundaryHandling >(boundaryHandlingId, mesh::BoundarySetup::OUTSIDE);

   // set up obstacle boundaries from file
   // set up inflow/outflow/wall boundaries from DefaultBoundaryHandlingFactory
   // geometry::initBoundaryHandling< BHFactory::BoundaryHandling >(*blocks, boundaryHandlingId, boundariesConfig);
   boundarySetup.setBoundaries< BHFactory::BoundaryHandling >(
      boundaryHandlingId, makeBoundaryLocationFunction(distanceOctree, boundaryLocations), mesh::BoundarySetup::INSIDE);

   WALBERLA_LOG_INFO_ON_ROOT("Checkpoint 7: Boundary handling done")
#pragma endregion BOUNDARY_HANDLING

   // Log performance information at the end of the simulation
   lbm::BlockForestEvaluation< FlagField_T >(blocks, flagFieldId, fluidFlagUID).logInfoOnRoot();
   // lbm::PerformanceLogger< FlagField_T > perfLogger(blocks, flagFieldId, fluidFlagUID, 100);

#pragma region SWEEPS_AND_TIME_LOOP

   // create time loop
   SweepTimeloop timeloop(blocks->getBlockStorage(), setup.timeSteps);

   // add Smagorinsky LES model
   // Initialize the Smagorinsky LES model
   const lbm::SmagorinskyLES< LatticeModel_T > smagorinskySweep(
      blocks, pdfFieldId, omegaFieldId, simulationUnits.kinViscosityLU, setup.smagorinskyConstant);

   auto sweepBoundary = lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >(pdfFieldId, flagFieldId, fluidFlagUID);
   blockforest::communication::UniformBufferedScheme< CommunicationStencil_T > communication(blocks);
   timeloop.add()
      // Smagorinsky turbulence model
      << BeforeFunction(smagorinskySweep, "Sweep: Smagorinsky turbulence model")
      << Sweep(lbm::makeCollideSweep(sweepBoundary), "Sweep: collision after Smagorinsky sweep");
   // << AfterFunction(Communication_T(blocks, pdfFieldId),
   //               "Communication: after collision sweep with preceding Smagorinsky sweep");

   // OmegaSweep_new< LatticeModel_T > omegaSweepnew(blocks, pdfFieldId, omegaFieldId, aabb, setup,
   //                                                       simulationUnits);
   // timeloop.add() << BeforeFunction(omegaSweepnew, "OmegaSweep_new")
   //                << Sweep(lbm::makeCollideSweep(sweepBoundary), "Sweep: collision after OmegaSweep_new");

   timeloop.add() << Sweep(OmegaSweep(pdfFieldId, omegaFieldId, aabb, setup, simulationUnits), "OmegaSweep");

   auto refinementTimeStep = lbm::refinement::makeTimeStep< LatticeModel_T, BHFactory::BoundaryHandling >(
      blocks, sweepBoundary, pdfFieldId, boundaryHandlingId);

   // The refined cells need a smaller time step to ensure stability
   timeloop.addFuncBeforeTimeStep(makeSharedFunctor(refinementTimeStep), "Refinement time step");

   // log remaining time
   timeloop.addFuncAfterTimeStep(
      timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), setup.remainingTimeLoggerFrequency),
      "remaining time logger");

   // // create communication for PdfField
   // WALBERLA_LOG_INFO_ON_ROOT("Waypoint 20: Adding communication")
   // blockforest::communication::UniformBufferedScheme< CommunicationStencil_T > communication(blocks);
   // communication.addPackInfo(make_shared< lbm::PdfFieldPackInfo< LatticeModel_T > >(pdfFieldId));
   //
   // // add LBM sweep and communication to time loop
   // WALBERLA_LOG_INFO_ON_ROOT("Waypoint 21: Adding sweeps and communication to time loop")
   // timeloop.add() << BeforeFunction(communication, "communication")
   //                << Sweep(BHFactory::BoundaryHandling::getBlockSweep(boundaryHandlingId), "boundary handling");
   // timeloop.add() << Sweep(
   //    makeSharedSweep(lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >(pdfFieldId, flagFieldId, fluidFlagUID)),
   //    "LB stream & collide");

   // LBM stability check
   timeloop.addFuncAfterTimeStep(makeSharedFunctor(field::makeStabilityChecker< PdfField_T, FlagField_T >(
                                    walberlaEnv.config(), blocks, pdfFieldId, flagFieldId, fluidFlagUID)),
                                 "LBM stability check");

   WALBERLA_LOG_INFO_ON_ROOT(" Checkpoint 8: Sweeps and time loop done ")

#pragma endregion SWEEPS_AND_TIME_LOOP

#pragma region VTK_OUTPUT
   auto airfoilProximityOutput = myAABBVTKOutput("airfoil_proximity", blocks, pdfFieldId, omegaFieldId, flagFieldId,
                                                 fluidFlagUID, VTKParams, setup, simulationUnits);

   uint_t airfoilProximityWriteFrequency =
      VTKParams.getBlock("airfoil_proximity").getParameter("writeFrequency", uint_t(0));
   uint_t airfoilProximityInitialExecutionCount =
      VTKParams.getBlock("airfoil_proximity").getParameter("initialExecutionCount", uint_t(0));

   auto fullDomainOutput = myAABBVTKOutput("full_domain", blocks, pdfFieldId, omegaFieldId, flagFieldId, fluidFlagUID,
                                           VTKParams, setup, simulationUnits);

   uint_t fullDomainWriteFrequency = VTKParams.getBlock("full_domain").getParameter("writeFrequency", uint_t(0));
   uint_t fullDomainInitialExecutionCount =
      VTKParams.getBlock("full_domain").getParameter("initialExecutionCount", uint_t(0));

   auto InitDomainOutput = myAABBVTKOutput("initial_domain", blocks, pdfFieldId, omegaFieldId, flagFieldId,
                                           fluidFlagUID, VTKParams, setup, simulationUnits);


   // timeloop.addFuncAfterTimeStep(vtk::writeFiles(fullDomainOutput), "VTK Full Domain");
   // timeloop.addFuncAfterTimeStep(vtk::writeFiles(airfoilProximityOutput), "VTK Airfoil Proximity");

   // add sweep for getting velocity and density at predefined coordinates
   auto densityVelocityP1 = std::make_shared< std::vector< real_t > >();

   const auto probeParams                   = VTKParams.getBlock("probe1");
   const std::string probe1Name             = probeParams.getParameter("identifier", std::string("no_name"));
   uint_t probe1WriteFrequency              = probeParams.getParameter("writeFrequency", uint_t(0));
   const uint_t probe1InitialExecutionCount = probeParams.getParameter("initialExecutionCount", uint_t(0));
   const Vector3< real_t > probe1Coord      = probeParams.getParameter("coordinate", Vector3< real_t >(0));

   timeloop.addFuncAfterTimeStep(
      VelDensPointEvaluator< PdfField_T >(blocks, pdfFieldId, probe1Coord, probe1WriteFrequency, densityVelocityP1),
      "Evaluation at " + probe1Name);
   const std::string fileResultP1 =
      std::string("results/probes/") + probe1Name + ".txt";
   WALBERLA_LOG_INFO_ON_ROOT(fileResultP1)

   WALBERLA_LOG_INFO_ON_ROOT(" Checkpoint 9: VTK output added")
#pragma endregion VTK_OUTPUT

#pragma region RUN_SIMULATION

   WcTimingPool timingPool;
   WALBERLA_LOG_INFO_ON_ROOT("Starting timeloop")

   // Initial fields and data for checking purposes
   if (InitDomainOutput != nullptr) { InitDomainOutput->write(true, 0); }

   for (uint_t i = 0; i < setup.timeSteps + 1; ++i)
   {
      // perform a single simulation step
      timeloop.singleStep(timingPool, true);

      // After the time step, vtk output is added
      if (i > fullDomainInitialExecutionCount && fullDomainWriteFrequency > 0 && i % fullDomainWriteFrequency == 0)
      {
         if (fullDomainOutput != nullptr) { fullDomainOutput->write(true, 0); }
      }

      if (i > airfoilProximityInitialExecutionCount && airfoilProximityWriteFrequency > 0 &&
          i % airfoilProximityWriteFrequency == 0)
      {
         if (airfoilProximityOutput != nullptr) { airfoilProximityOutput->write(true, 0); }
      }

      // write current density and velocity at P1, P2, and P3 to files
      if (i > probe1InitialExecutionCount && probe1WriteFrequency > 0 && i % probe1WriteFrequency == uint_c(0))
      {
         if (!(*densityVelocityP1).empty()) 
         { 
            writeVector(*densityVelocityP1, i, fileResultP1); 
         }
      }
   }

   timeloop.run(timingPool);
   timingPool.unifyRegisteredTimersAcrossProcesses();
   timingPool.logResultOnRoot(timing::REDUCE_TOTAL, true);

#pragma endregion RUN_SIMULATION

   return EXIT_SUCCESS;
}

} // namespace powerFlowCase
} // namespace MuTech
} // namespace walberla

int main(int argc, char** argv) { walberla::MuTech::powerFlowCase::main(argc, argv); }
