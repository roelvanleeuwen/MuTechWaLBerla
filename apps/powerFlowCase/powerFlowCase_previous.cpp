/*
================================================================================================================================================================
File: powerFlowCase.cpp
Author: R. van Leeuwen
Contact: roelvanleeuwen8@gmail.com
Company: MuTech
Created: 23-09-2024
================================================================================================================================================================
Description:    This is the main file of the waLBerla implementation of the airfoil flow simulation ran in PowerFlow by
                Marlon van Crugten. The simulation consists of a NACA 0018 airfoil with the MuteSkin add-on. The aim of
the simulation is to understand the working principle of the add-on better so de understanding can be used in optimising
MuteSkin.

                The simulation runs a 2.5D airfoil with add-on. The airfoil and the flow have an angle of 7.8 degrees.
The flow has a velocity of 10 m/s. The ambient pressure is 101325 Pa.

                The domain must be large to prevent boundary effects. The domain is 10 times the chord length of the
airfoil in the x and z direction. In the y direction

*/

#include "blockforest/all.h"

#include "core/all.h"

#include "domain_decomposition/all.h"

#include "field/all.h"

#include "geometry/all.h"

#include "gui/all.h"

#include "lbm/all.h"

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

#include "timeloop/all.h"

#include "mesh_common/DistanceComputations.h"
#include "mesh_common/DistanceFunction.h"
#include "mesh_common/MatrixVectorOperations.h"
#include "mesh_common/MeshIO.h"
#include "mesh_common/MeshOperations.h"
#include "mesh_common/TriangleMeshes.h"
#include "mesh_common/distance_octree/DistanceOctree.h"
#include "mesh_common/vtk/CommonDataSources.h"
#include "mesh_common/vtk/VTKMeshWriter.h"
#include "xyAdjustment.cpp"

// #include "memoryLogger.cpp"

namespace walberla
{

struct Setup
{
   // Flow parameters
   real_t kinViscosity;                 // physical kinematic viscosity
   real_t kinViscosityLU;               // lattice kinematic viscosity
   real_t rho;                          // physical density
   real_t rhoLU = 1;                    // lattice density
   real_t angleOfAttack;                // physical angle of attack
   real_t velocityMagnitude;            // physical velocity magnitude
   Vector3< real_t > initialVelocity;   // physical inflow velocity
   Vector3< real_t > initialVelocityLU; // lattice inflow velocity
   Vector3< real_t > flowVelocity;      // physical flow velocity
   Vector3< real_t > flowVelocityLU;    // lattice flow velocity
   real_t temperature;                  // physical temperature
   real_t speedOfSound;                 // physical speed of sound
   real_t Mach;                         // physical Mach number
   real_t Re;                           // physical Reynolds number

   // Space and time parameters
   real_t dx;       // physical grid spacing
   real_t fineDX;   // physical grid spacing of the finest grid
   real_t dt;       // physical time step
   real_t dxLU = 1; // lattice grid spacing
   real_t dtLU = 1; // lattice time step

   // Domain parameters
   std::string meshFile;
   bool scalePowerFlowDomain;
   real_t decreasePowerFlowDomainFactor;
   Vector3< real_t > domainScaling;
   real_t meshZScaling;
   real_t xyAdjuster_x;
   real_t xyAdjuster_y;
   uint_t numLevels;
   uint_t numGhostLayers;
   Vector3< bool > periodicity;

   // Block data
   Vector3< uint_t > cellsPerBlock; // Number of cells in each block in the < x, y, z > directions. This is also called
                                    // blockSize in some codes. For refinement at least < 16, 16, 16 > is required
   real_t nBlocks_x;
   real_t nBlocks_y;
   real_t nBlocks_z;

   // Domain data
   real_t domainLengthLU; // x dimension in latice units e.g. number of cells in x direction
   real_t domainHeightLU; // y dimension in latice units e.g. number of cells in y direction
   real_t domainWidthLU;  // z dimension in latice units e.g. number of cells in z direction
   real_t domainLength;   // physical x dimension of the domain in meters
   real_t domainHeight;   // physical y dimension of the domain in meters
   real_t domainWidth;    // physical z dimension of the domain in meters

   // Airfoil data
   real_t airfoilXPosition;
   real_t airfoilYPosition;
   real_t airfoilChordLength;
   real_t airfoilThickness;
   real_t airfoilSpan;
   real_t airfoilChordLU;
   real_t airfoilThicknessLU;
   real_t airfoilSpanLU;

   // LBM parameters
   real_t omega; // relaxation parameter

   // Output parameters
   uint_t timeSteps;                    // number of time steps
   real_t remainingTimeLoggerFrequency; // in seconds

   // Define the operator<< for Setup
   friend std::ostream& operator<<(std::ostream& os, const Setup& setup)
   {
      os << "Setup: \n";
      os << "Flow parameters:\n";
      os << "  kinViscosity: " << setup.kinViscosity << "\n";
      os << "  kinViscosityLU: " << setup.kinViscosityLU << "\n";
      os << "  rho: " << setup.rho << "\n";
      os << "  rhoLU: " << setup.rhoLU << "\n";
      os << "  angleOfAttack: " << setup.angleOfAttack << "\n";
      os << "  velocityMagnitude: " << setup.velocityMagnitude << "\n";
      os << "  initialVelocity: " << setup.initialVelocity << "\n";
      os << "  initialVelocityLU: " << setup.initialVelocityLU << "\n";
      os << "  flowVelocity: " << setup.flowVelocity << "\n";
      os << "  flowVelocityLU: " << setup.flowVelocityLU << "\n";
      os << "  temperature: " << setup.temperature << "\n";
      os << "  speedOfSound: " << setup.speedOfSound << "\n";
      os << "  Mach: " << setup.Mach << "\n";
      os << "  Re: " << setup.Re << "\n";

      os << "Space and time parameters:\n";
      os << "  dx: " << setup.dx << "\n";
      os << "  fineDX: " << setup.fineDX << "\n";
      os << "  dt: " << setup.dt << "\n";
      os << "  dxLU: " << setup.dxLU << "\n";
      os << "  dtLU: " << setup.dtLU << "\n";

      os << "Domain parameters:\n";
      os << "  meshFile: " << setup.meshFile << "\n";
      os << "  scalePowerFlowDomain: " << setup.scalePowerFlowDomain << "\n";
      os << "  decreasePowerFlowDomainFactor: " << setup.decreasePowerFlowDomainFactor << "\n";
      os << "  domainScaling: " << setup.domainScaling << "\n";
      os << "  meshZScaling: " << setup.meshZScaling << "\n";
      os << "  xyAdjuster_x: " << setup.xyAdjuster_x << "\n";
      os << "  xyAdjuster_y: " << setup.xyAdjuster_y << "\n";
      os << "  numLevels: " << setup.numLevels << "\n";
      os << "  numGhostLayers: " << setup.numGhostLayers << "\n";
      os << "  periodicity: " << setup.periodicity << "\n";

      os << "Block data:\n";
      os << "  cellsPerBlock: " << setup.cellsPerBlock << "\n";
      os << "  nBlocks_x: " << setup.nBlocks_x << "\n";
      os << "  nBlocks_y: " << setup.nBlocks_y << "\n";
      os << "  nBlocks_z: " << setup.nBlocks_z << "\n";

      os << "Domain data:\n";
      os << "  domainLengthLU: " << setup.domainLengthLU << "\n";
      os << "  domainHeightLU: " << setup.domainHeightLU << "\n";
      os << "  domainWidthLU: " << setup.domainWidthLU << "\n";
      os << "  domainLength: " << setup.domainLength << "\n";
      os << "  domainHeight: " << setup.domainHeight << "\n";
      os << "  domainWidth: " << setup.domainWidth << "\n";

      os << "Airfoil data:\n";
      os << "  airfoilXPosition: " << setup.airfoilXPosition << "\n";
      os << "  airfoilYPosition: " << setup.airfoilYPosition << "\n";
      os << "  airfoilChordLength: " << setup.airfoilChordLength << "\n";
      os << "  airfoilThickness: " << setup.airfoilThickness << "\n";
      os << "  airfoilSpan: " << setup.airfoilSpan << "\n";
      os << "  airfoilChordLU: " << setup.airfoilChordLU << "\n";
      os << "  airfoilThicknessLU: " << setup.airfoilThicknessLU << "\n";
      os << "  airfoilSpanLU: " << setup.airfoilSpanLU << "\n";

      os << "LBM parameters:\n";
      os << "  omega: " << setup.omega << "\n";

      os << "Output parameters:\n";
      os << "  timeSteps: " << setup.timeSteps << "\n";
      os << "  remainingTimeLoggerFrequency: " << setup.remainingTimeLoggerFrequency << "\n";

      // Add other members as needed
      return os;
   }
};

// A function that outputs all

//! [typedefs]
using LatticeModel_T         = lbm::D3Q19< lbm::collision_model::SRT >;
using Stencil_T              = LatticeModel_T::Stencil;
using CommunicationStencil_T = LatticeModel_T::CommunicationStencil;
//! [typedefs]

using PdfField_T = lbm::PdfField< LatticeModel_T >;

using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

//! [vertexToFaceColor]
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

//! [vertexToFaceColor]

int main(int argc, char** argv)
{
   walberla::Environment walberlaEnv(argc, argv);

   mpi::MPIManager::instance()->useWorldComm();

   // ================================================================================================================

   ///////////////////////
   /// PARAMETER INPUT ///
   ///////////////////////
   // Initialize the parameters blocks
   auto optionsParameters    = walberlaEnv.config()->getOneBlock("options");
   auto flowParameters       = walberlaEnv.config()->getOneBlock("flowParameters");
   auto simulationParameters = walberlaEnv.config()->getOneBlock("simulationParameters");
   auto domainParameters     = walberlaEnv.config()->getOneBlock("domainParameters");

   Setup setup;
   // Read the flow parameters
   setup.angleOfAttack =
      flowParameters.getParameter< real_t >("angleOfAttack", real_c(7.8)); // angle of attack in degrees
   setup.velocityMagnitude =
      flowParameters.getParameter< real_t >("velocityMagnitude", real_c(10.0)); // velocity magnitude in m/s
   setup.kinViscosity =
      flowParameters.getParameter< real_t >("kinViscosity", real_t(1.0)); // physical kinematic viscosity in m^2/s
   setup.rho = flowParameters.getParameter< real_t >("rho", real_t(1.0)); // physical density in kg/m^3

   // Read the simulation parameters
   setup.omega     = simulationParameters.getParameter< real_t >("omega", real_t(1.6)); // relaxation parameter
   setup.timeSteps = simulationParameters.getParameter< uint_t >("timeSteps", uint_c(10));
   setup.remainingTimeLoggerFrequency =
      simulationParameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_t(3.0)); // in seconds

   // Read the domain parameters
   setup.meshFile = domainParameters.getParameter< std::string >("meshFile"); // mesh file containing the airfoil
   setup.airfoilChordLength =
      domainParameters.getParameter< real_t >("chord", real_t(1)); // chord length of the airfoil in meters
   setup.airfoilThickness =
      domainParameters.getParameter< real_t >("thickness", real_t(0.2));           // thickness of the airfoil in meters
   setup.airfoilSpan = domainParameters.getParameter< real_t >("span", real_t(1)); // span of the airfoil in meters
   setup.scalePowerFlowDomain = domainParameters.getParameter< bool >("scalePowerFlowDomain", false);
   setup.decreasePowerFlowDomainFactor =
      domainParameters.getParameter< real_t >("decreasePowerFlowDomainFactor", real_t(1));
   setup.dx = domainParameters.getParameter< real_t >("dx", real_t(1)); // grid spacing of coarsest cells in meters
   setup.numLevels = domainParameters.getParameter< uint_t >(
      "numLevels", uint_t(1)); // number of refinement levels // Ensure it is always at least 1
   setup.numLevels      = std::max(setup.numLevels, uint_t(1));
   setup.numGhostLayers = domainParameters.getParameter< uint_t >("numGhostLayers", uint_t(4));
   setup.periodicity    = domainParameters.getParameter< Vector3< bool > >("periodic", Vector3< bool >(true));
   setup.meshZScaling   = domainParameters.getParameter< real_t >("meshZScaling", 1);

   // Calculate the initial physical velocity [m/s] based on the angle of attack and the velocity magnitude
   setup.initialVelocity =
      Vector3< real_t >(setup.velocityMagnitude * std::cos(setup.angleOfAttack * M_PI / 180.0),
                        setup.velocityMagnitude * std::sin(setup.angleOfAttack * M_PI / 180.0), real_t(0));
   setup.flowVelocity = setup.initialVelocity;
   // Calculate the finest physical grid spacing [m] based on the coarsest grid spacing and the number of refinement
   // levels
   setup.fineDX = setup.dx / real_c(std::pow(2, setup.numLevels));

   // Log the parameters
   WALBERLA_LOG_INFO_ON_ROOT(" ======================== Initial Parameters ======================== "
                             "\n + Angle of attack: "
                             << setup.angleOfAttack
                             << " degrees"
                                "\n + Flow velocity: "
                             << setup.velocityMagnitude
                             << " m/s"
                                "\n + Dynamic viscosity: "
                             << setup.kinViscosity
                             << " m^2/s"
                                "\n + Density: "
                             << setup.rho
                             << " kg/m^3"
                                "\n + Inflow velocity < u, v, w >: "
                             << setup.initialVelocity
                             << " m/s"
                                "\n + Relaxation parameter omega = 1/tau : "
                             << setup.omega << "\n + Number of time steps: " << setup.timeSteps
                             << "\n + Remaining time logger frequency: " << setup.remainingTimeLoggerFrequency
                             << " s"
                                "\n + Mesh file: "
                             << setup.meshFile << "\n + Scale PowerFlow domain: " << setup.scalePowerFlowDomain
                             << "\n + Decrease PowerFlow domain factor: " << setup.decreasePowerFlowDomainFactor
                             << "\n + dx: " << setup.dx
                             << " m"
                                "\n + Fine dx: "
                             << setup.fineDX
                             << " m"
                                "\n + Number of refinement levels: "
                             << setup.numLevels << "\n + Number of ghost layers: " << setup.numGhostLayers
                             << "\n + Periodicity < x_sides, y_sides, z_sides >: " << setup.periodicity
                             << "\n + Mesh Z scaling: " << setup.meshZScaling << "\n");

   // ================================================================================================================
   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   WALBERLA_LOG_INFO_ON_ROOT(" ======================== Start of mesh and domain setup ======================== ")

   // read in mesh with vertex colors on a single process and broadcast it
   auto mesh = make_shared< mesh::TriangleMesh >();

   mesh->request_vertex_colors();
   mesh::readAndBroadcast(
      setup.meshFile,
      *mesh); // read the mesh file and broadcast it to all processes. This is necessary for parallel processing

   // Scale the mesh in the z direction to change the span of the airfoil
   for (auto vertexIt = mesh->vertices_begin(); vertexIt != mesh->vertices_end(); ++vertexIt)
   {
      auto point = mesh->point(*vertexIt);
      point[2] *= setup.meshZScaling; // Scale the y coordinate
      mesh->set_point(*vertexIt, point);
   }

   // color faces according to vertices
   vertexToFaceColor(*mesh, mesh::TriangleMesh::Color(255, 255, 255));
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 1: Mesh read and colored")

   // add information to mesh that is required for computing signed distances from a point to a triangle
   auto triDist = make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh);

   // building distance octree to determine which cells are inside the airfoil and which are outside
   auto distanceOctree = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(triDist);
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 2: Octree has height " << distanceOctree->height())

   // write distance octree to file
   WALBERLA_ROOT_SECTION()
   {
      distanceOctree->writeVTKOutput("vtk_out/distanceOctree");
      WALBERLA_LOG_INFO_ON_ROOT("Waypoint 2extra: Distance octree written to vtk_out/distanceOctree.vtk")
   }

   ///////////////////////////
   /// CREATE BLOCK FOREST ///
   ///////////////////////////

   // Make the axis-aligned bounding box (AABB) of a mesh object.
   auto aabb = computeAABB(*mesh);

   // What is the size of the mesh object in the x, y and z direction?
   WALBERLA_LOG_INFO_ON_ROOT("")
   WALBERLA_LOG_INFO_ON_ROOT(" ======================== Intermediate domain info ======================== ")
   WALBERLA_LOG_INFO_ON_ROOT("Mesh size in x, y, z: " << aabb.xSize() << ", " << aabb.ySize() << ", "
                                                      << aabb.zSize()) // in meters
   WALBERLA_LOG_INFO_ON_ROOT("Mesh size in latice units: " << aabb.xSize() / setup.dx << ", " << aabb.ySize() / setup.dx
                                                           << ", " << aabb.zSize() / setup.dx) // in latice units

   setup.airfoilChordLength = aabb.xSize();            // chord length of the airfoil
   setup.airfoilThickness   = aabb.ySize();            // thickness of the airfoil
   setup.airfoilSpan        = aabb.zSize();            // span of the airfoil
   setup.airfoilChordLU     = aabb.xSize() / setup.dx; // number of cells per chord length
   setup.airfoilThicknessLU = aabb.ySize() / setup.dx; // number of cells per thickness
   setup.airfoilSpanLU      = aabb.zSize() / setup.dx; // number of cells per span

   // The x dimension is 100 times the airfoil chord. This is not to be changed. Therefore it is not in the parameter
   // file.
   AdjustmentResult adjustXYResult = xyAdjustment(100 * aabb.xSize(), setup.decreasePowerFlowDomainFactor, setup.dx);
   setup.xyAdjuster_x              = adjustXYResult.xyAdjustment;
   setup.xyAdjuster_y              = adjustXYResult.xyAdjustment;
   setup.cellsPerBlock             = Vector3< uint_t >(adjustXYResult.cellsPerBlock_x, adjustXYResult.cellsPerBlock_x,
                                           16); // The z direction has 16 cells per block

   WALBERLA_LOG_INFO_ON_ROOT("xzAdjuster: " << setup.xyAdjuster_x)
   WALBERLA_LOG_INFO_ON_ROOT("Cells per block/block size  < nx, ny, nz >: " << setup.cellsPerBlock)

   if (setup.scalePowerFlowDomain)
   {
      // The chord length of the airfoil is the x size of the mesh object
      setup.domainScaling =
         Vector3< real_t >(100 * setup.decreasePowerFlowDomainFactor * setup.xyAdjuster_x,
                           100 * aabb.xSize() / aabb.ySize() * setup.decreasePowerFlowDomainFactor * setup.xyAdjuster_y,
                           static_cast< real_t >(setup.cellsPerBlock[2]) * setup.dx / aabb.zSize());
   }
   else
   {
      setup.domainScaling = domainParameters.getParameter< Vector3< real_t > >("domainScaling", Vector3< real_t >(1));
   }

   WALBERLA_LOG_INFO_ON_ROOT("Domain scaling: " << setup.domainScaling)
   aabb.scale(setup.domainScaling); // Scale the domain based on the x, y and z size of the mesh object.
   // aabb.setCenter(aabb.center() + Vector3< real_t >(0, 0.75 * thickness, 0)); // Place the airfoil in the center of
   // the domain
   aabb.setCenter(aabb.center()); // Place the airfoil in the center of the domain

   // Log the domain information
   setup.domainLength   = aabb.xSize();            // physical x dimension of the domain in meters
   setup.domainHeight   = aabb.ySize();            // physical y dimension of the domain in meters
   setup.domainWidth    = aabb.zSize();            // physical z dimension of the domain in meters
   setup.domainLengthLU = aabb.xSize() / setup.dx; // x dimension in latice units e.g. number of cells in x direction
   setup.domainHeightLU = aabb.ySize() / setup.dx; // y dimension in latice units e.g. number of cells in y direction
   setup.domainWidthLU  = aabb.zSize() / setup.dx; // z dimension in latice units e.g. number of cells in z direction

   setup.airfoilXPosition = aabb.center()[0];
   setup.airfoilYPosition = aabb.center()[1];

   // What will be the number of blocks in the x, y and z direction?
   setup.nBlocks_x = static_cast< real_t >(adjustXYResult.nBlocks_x);
   setup.nBlocks_y = static_cast< real_t >(adjustXYResult.nBlocks_x);
   setup.nBlocks_z = std::round(aabb.zSize() / setup.dx / static_cast< real_t >(setup.cellsPerBlock[2]));

   auto cell_dx = aabb.xSize() / static_cast< real_t >(setup.cellsPerBlock[0]) / setup.nBlocks_x;
   auto cell_dy = aabb.ySize() / static_cast< real_t >(setup.cellsPerBlock[1]) / setup.nBlocks_y;
   auto cell_dz = aabb.zSize() / static_cast< real_t >(setup.cellsPerBlock[2]) / setup.nBlocks_z;
   WALBERLA_LOG_INFO_ON_ROOT("Cell size in x, y, z: " << cell_dx << ", " << cell_dy << ", " << cell_dz)

   const real_t tolerance = 1e-6; // Define a tolerance for floating-point comparison

   // Check if the cells are cubic
   if (std::abs(cell_dx - cell_dy) > tolerance || std::abs(cell_dx - cell_dz) > tolerance ||
       std::abs(cell_dy - cell_dz) > tolerance)
   {
      WALBERLA_LOG_INFO_ON_ROOT(
         "The cells are not cubic. The cell size in the x, y and z direction is not equal within the tolerance.")
      WALBERLA_LOG_INFO_ON_ROOT("Check the domain generation documentation of the powerFlowCase simulation.")
      WALBERLA_LOG_INFO_ON_ROOT("")
      WALBERLA_LOG_INFO_ON_ROOT("First determine the number of blocks with a decreasePowerFlowDomainFactor so that the "
                                "number of cells in the x and y direction are just above 16")
      WALBERLA_LOG_INFO_ON_ROOT("Then set this number of cells in the x and y direction (z has per definitiion 16 "
                                "cells in this simulation) to this determinde number (just above 16)")
      WALBERLA_LOG_INFO_ON_ROOT("Then calculate the xyAdjuster as the (N_blocks * N_blockSize,x * dx)/(L_x,full * "
                                "decreasePowerFlowDomainFactor)")
      WALBERLA_LOG_INFO_ON_ROOT("")
      WALBERLA_LOG_INFO_ON_ROOT("Error: This simulation will end now.")
      return EXIT_SUCCESS;
   }
   else
   {
      WALBERLA_LOG_INFO_ON_ROOT(
         "The cells are cubic. The cell size in the x, y and z direction is equal within the tolerance.")
      WALBERLA_LOG_INFO_ON_ROOT(
         "Total number of unrefined cells: " << setup.cellsPerBlock[0] * static_cast< uint_t >(setup.nBlocks_x) *
                                                   setup.cellsPerBlock[1] * static_cast< uint_t >(setup.nBlocks_y) *
                                                   setup.cellsPerBlock[2] * static_cast< uint_t >(setup.nBlocks_z))
   }

   WALBERLA_LOG_INFO_ON_ROOT(" ======================== End of intermediate domain info ======================== ")
   WALBERLA_LOG_INFO_ON_ROOT("")

   // create and configure block forest creator
   mesh::ComplexGeometryStructuredBlockforestCreator bfc(
      aabb, Vector3< real_t >(setup.dx)); // create the structured block forest creator
   bfc.setRootBlockExclusionFunction(mesh::makeExcludeMeshInterior(
      distanceOctree, setup.dx)); // exclude the object mesh interior with maximum error of dx
   bfc.setBlockExclusionFunction(
      mesh::makeExcludeMeshInteriorRefinement(distanceOctree, setup.fineDX)); // refine the maximum error to fineDX

   auto meshWorkloadMemory = mesh::makeMeshWorkloadMemory(distanceOctree, setup.dx);
   meshWorkloadMemory.setInsideCellWorkload(1);
   meshWorkloadMemory.setOutsideCellWorkload(1);
   bfc.setWorkloadMemorySUIDAssignmentFunction(meshWorkloadMemory);
   bfc.setPeriodicity(setup.periodicity);
   bfc.setRefinementSelectionFunction(
      makeRefinementSelection(distanceOctree, setup.numLevels - 1, setup.dx, setup.dx * real_t(1)));
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 4: Refinement selection function set")

   // create block forest
   auto blocks = bfc.createStructuredBlockForest(setup.cellsPerBlock);
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 5: Block forest created")

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

   ////////////////////////////////////
   /// CREATE AND INITIALIZE FIELDS ///
   ////////////////////////////////////

   // create fields
   LatticeModel_T latticeModel = LatticeModel_T(lbm::collision_model::SRT(setup.omega));
   setup.kinViscosityLU        = latticeModel.collisionModel().viscosity(0);     // lattice kinematic viscosity
   setup.dt = setup.kinViscosityLU * std::pow(setup.dx, 2) / setup.kinViscosity; // time step in physical units
   setup.initialVelocityLU = setup.initialVelocity * setup.dt / setup.dx;  // initial velocity vector in lattice units
   setup.speedOfSound      = std::pow(1.4 * 287.15 * 281, 0.5);            // speed of sound in physical units
   setup.Mach              = setup.velocityMagnitude / setup.speedOfSound; // Mach number in physical units
   setup.Re                = setup.velocityMagnitude * setup.airfoilChordLength / setup.kinViscosity; // Reynolds number

   WALBERLA_LOG_INFO_ON_ROOT("Physical parameters: "
                             "\n + Speed of sound: "
                             << setup.speedOfSound
                             << " "
                                "\n + Physical Mach number: "
                             << setup.Mach
                             << " "
                                "\n + Reynolds number: "
                             << setup.Re << " ")

   std::vector< real_t > kinViscosityLUList(setup.numLevels);
   std::vector< real_t > omegaList(setup.numLevels);
   std::vector< real_t > flowVelocityLUList(setup.numLevels);
   std::vector< real_t > dxList(setup.numLevels);
   std::vector< real_t > dtList(setup.numLevels);
   std::vector< real_t > speedOfSoundLUList(setup.numLevels);
   std::vector< real_t > MachLUList(setup.numLevels);

   for (uint_t i = 0; i < setup.numLevels; ++i)
   {
      dxList[i]             = setup.dx / std::pow(2, i);
      kinViscosityLUList[i] = latticeModel.collisionModel().viscosity(i); // viscosity in lattice units
      omegaList[i]          = 1 / (3 * kinViscosityLUList[i] + 0.5);      // relaxation parameter
      dtList[i]             = kinViscosityLUList[i] * std::pow(dxList[i], 2) / setup.kinViscosity; // time step
      flowVelocityLUList[i] = setup.velocityMagnitude * dtList[i] / dxList[i];
      speedOfSoundLUList[i] = 1 / std::pow(3.0, 0.5);                        // speed of sound in lattice units
      MachLUList[i]         = flowVelocityLUList[i] / speedOfSoundLUList[i]; // Mach number in lattice units
   }

   WALBERLA_LOG_INFO_ON_ROOT("---------------------------------------------------------------")
   WALBERLA_LOG_INFO_ON_ROOT(
      "| Level | omega   | kinViscosityLU | dx         | dt       | flowVelocityLU | speedOfSoundLU | MachLU   |")
   WALBERLA_LOG_INFO_ON_ROOT("---------------------------------------------------------------")
   for (uint_t i = 0; i < setup.numLevels; ++i)
   {
      WALBERLA_LOG_INFO_ON_ROOT("| " << std::setw(5) << i << " | " << std::setw(7) << omegaList[i] << " | "
                                     << std::setw(14) << kinViscosityLUList[i] << " | " << std::setw(8) << dxList[i]
                                     << " | " << std::setw(8) << dtList[i] << " | " << std::setw(14)
                                     << flowVelocityLUList[i] << " | " << std::setw(14) << speedOfSoundLUList[i]
                                     << " | " << std::setw(8) << MachLUList[i] << " |")
   }
   WALBERLA_LOG_INFO_ON_ROOT("---------------------------------------------------------------")

   BlockDataID pdfFieldId =
      lbm::addPdfFieldToStorage(blocks, "pdf field", latticeModel, setup.initialVelocityLU, setup.rhoLU,
                                setup.numGhostLayers); // Here the initialisation of the pdf field. This includes the
                                                       // initial velocity and density in lattice units
   BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field", setup.numGhostLayers);
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 6: Fields created")

   /////////////////////////
   /// BOUNDARY HANDLING ///
   /////////////////////////

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

   // Log which boundaries are created
   // WALBERLA_LOG_INFO_ON_ROOT("Boundary handling created with the following boundaries: "
   //                            "\n + NoSlip: " << BHFactory::getNoSlipBoundaryUID() <<
   //                            "\n + Inflow: " << BHFactory::getInflowBoundaryUID() <<
   //                            "\n + Outflow: " << BHFactory::getOutflowBoundaryUID() <<
   //                            "\n + Wall: " << BHFactory::getWallBoundaryUID() <<
   //                            "\n + Pressure: " << BHFactory::getPressureBoundaryUID() <<
   //                            "\n + Periodic: " << BHFactory::getPeriodicBoundaryUID() <<
   //                            "\n");

   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 7: Boundary handling created")

   // set NoSlip UID to boundaries that we colored
   mesh::ColorToBoundaryMapper< mesh::TriangleMesh > colorToBoundaryMapper(
      (mesh::BoundaryInfo(BHFactory::getNoSlipBoundaryUID())));

   // mark boundaries
   auto boundaryLocations = colorToBoundaryMapper.addBoundaryInfoToMesh(*mesh);
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 8: Boundaries marked with color")

   // write mesh info to file
   if (optionsParameters.getParameter("writeAirfoilMeshAndReturn", false))
   {
      mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriter(mesh, "airfoil_mesh", 1);
      meshWriter.addDataSource(make_shared< mesh::ColorFaceDataSource< mesh::TriangleMesh > >());
      meshWriter.addDataSource(make_shared< mesh::ColorVertexDataSource< mesh::TriangleMesh > >());
      meshWriter();
      WALBERLA_LOG_INFO_ON_ROOT("Waypoint 9: Mesh written to vtk_out/airfoil_mesh.vtu")
      return EXIT_SUCCESS;
   }

   // voxelize mesh
   WALBERLA_LOG_DEVEL_ON_ROOT("Waypoint 10: Voxelizing mesh")
   mesh::BoundarySetup boundarySetup(blocks, makeMeshDistanceFunction(distanceOctree), setup.numGhostLayers);

   // write voxelisation to file. This shows all the voxels/cells that are in the domain
   if (optionsParameters.getParameter< bool >("writeVoxelfile", false))
   {
      WALBERLA_LOG_INFO_ON_ROOT("Waypoint 11: Writing Voxelisation");
      boundarySetup.writeVTKVoxelfile();
      return EXIT_SUCCESS;
   }
   else { WALBERLA_LOG_INFO_ON_ROOT("Waypoint 11: Voxelisation done") }

   // set fluid cells
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 12: Setting up fluid cells")
   boundarySetup.setDomainCells< BHFactory::BoundaryHandling >(boundaryHandlingId, mesh::BoundarySetup::OUTSIDE);

   // set up obstacle boundaries from file
   // set up inflow/outflow/wall boundaries from DefaultBoundaryHandlingFactory
   // geometry::initBoundaryHandling< BHFactory::BoundaryHandling >(*blocks, boundaryHandlingId, boundariesConfig);
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 13: Setting up boundaries")
   boundarySetup.setBoundaries< BHFactory::BoundaryHandling >(
      boundaryHandlingId, makeBoundaryLocationFunction(distanceOctree, boundaryLocations), mesh::BoundarySetup::INSIDE);
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 14: Boundaries set")
   //! [boundarySetup]

   // Log performance information at the end of the simulation
   lbm::BlockForestEvaluation< FlagField_T >(blocks, flagFieldId, fluidFlagUID).logInfoOnRoot();
   lbm::PerformanceLogger< FlagField_T > perfLogger(blocks, flagFieldId, fluidFlagUID, 100);

   // Write the simulation setup to a file
   std::ofstream outFile("vtk_out/simulation_setup.txt");
   if (outFile.is_open())
   {
      outFile << setup;
      outFile.close();
      WALBERLA_LOG_INFO_ON_ROOT("Simulation setup written to file successfully")
   }
   else { WALBERLA_LOG_INFO_ON_ROOT("Failed to open file for writing simulation setup") }

   //////////////////////////////////
   /// SET UP SWEEPS AND TIMELOOP ///
   //////////////////////////////////

   // create time loop
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 15: Setting up time loop")
   SweepTimeloop timeloop(blocks->getBlockStorage(), setup.timeSteps);

   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 16: Making sweeps")
   auto sweep = lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >(pdfFieldId, flagFieldId, fluidFlagUID);

   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 17: defining refinement time step")
   auto refinementTimeStep = lbm::refinement::makeTimeStep< LatticeModel_T, BHFactory::BoundaryHandling >(
      blocks, sweep, pdfFieldId, boundaryHandlingId);

   // The refined cells need a smaller time step to ensure stability
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 18: Adding refinement time step")
   timeloop.addFuncBeforeTimeStep(makeSharedFunctor(refinementTimeStep), "Refinement time step");

   // log remaining time
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 19: Adding remaining time logger")
   WALBERLA_LOG_INFO_ON_ROOT("number of time steps: " << timeloop.getNrOfTimeSteps())
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
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 21: Adding LBM stability check")
   timeloop.addFuncAfterTimeStep(makeSharedFunctor(field::makeStabilityChecker< PdfField_T, FlagField_T >(
                                    walberlaEnv.config(), blocks, pdfFieldId, flagFieldId, fluidFlagUID)),
                                 "LBM stability check");

   //////////////////
   /// VTK OUTPUT ///
   //////////////////

   // add VTK output to time loop
   auto VTKParams           = walberlaEnv.config()->getBlock("VTK");
   uint_t vtkWriteFrequency = VTKParams.getBlock("fluid_field").getParameter("writeFrequency", uint_t(0));
   auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "fluid_field", vtkWriteFrequency, setup.numGhostLayers,
                                                   false, "vtk_out", "simulation_step", false, true, true, false, 0);

   field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagFieldId);
   fluidFilter.addFlag(fluidFlagUID);
   vtkOutput->addCellInclusionFilter(fluidFilter);

   auto velocityWriter = make_shared< lbm::VelocityVTKWriter< LatticeModel_T, float > >(pdfFieldId, "Velocity");
   auto densityWriter  = make_shared< lbm::DensityVTKWriter< LatticeModel_T, float > >(pdfFieldId, "Density");
   vtkOutput->addCellDataWriter(velocityWriter);
   vtkOutput->addCellDataWriter(densityWriter);

   timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   timeloop.addFuncAfterTimeStep(perfLogger, "Evaluator: performance logging");
   WALBERLA_LOG_INFO_ON_ROOT("Waypoint 23: VTK output added")

   //////////////////////
   /// RUN SIMULATION ///
   //////////////////////
   WcTimingPool timingPool;
   WALBERLA_LOG_INFO_ON_ROOT("Starting timeloop")
   timeloop.run(timingPool);
   timingPool.unifyRegisteredTimersAcrossProcesses();
   timingPool.logResultOnRoot(timing::REDUCE_TOTAL, true);

   // Make a file with the simulation Setup struct to be used in the post processing
   WALBERLA_LOG_INFO_ON_ROOT("Writing simulation setup to file")

   return EXIT_SUCCESS;
}
} // namespace walberla

int main(int argc, char** argv) { walberla::main(argc, argv); }