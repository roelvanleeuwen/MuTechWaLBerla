/*
================================================================================================================================================================
File: powerFlowCase.cpp
Author: R. van Leeuwen
Contact: roelvanleeuwen8@gmail.com
Company: MuTech
Created: 23-09-2024
================================================================================================================================================================
Description:    This is the main file of the waLBerla implementation of the airfoil flow simulation ran in PowerFlow by 
                Marlon van Crugten. The simulation consists of a NACA 0018 airfoil with the MuteSkin add-on. The aim of the simulation 
                is to understand the working principle of the add-on better so de understanding can be used in optimising MuteSkin.   

                The simulation runs a 2.5D airfoil with add-on. The airfoil and the flow have an angle of 7.8 degrees. The flow has a 
                velocity of 10 m/s. The ambient pressure is 101325 Pa.

                The domain must be large to prevent boundary effects. The domain is 10 times the chord length of the airfoil in the x and z direction. In the y direction 

*/

#include "blockforest/all.h"

#include "core/all.h"

#include "domain_decomposition/all.h"

#include "field/all.h"

#include "geometry/all.h"

#include "gui/all.h"

#include "lbm/all.h"

#include "mesh_common/DistanceComputations.h"
#include "mesh_common/DistanceFunction.h"
#include "mesh_common/MatrixVectorOperations.h"
#include "mesh_common/MeshIO.h"
#include "mesh_common/MeshOperations.h"
#include "mesh_common/TriangleMeshes.h"
#include "mesh_common/distance_octree/DistanceOctree.h"
#include "mesh_common/vtk/CommonDataSources.h"
#include "mesh_common/vtk/VTKMeshWriter.h"
#include "mesh/blockforest/BlockExclusion.h"
#include "mesh/blockforest/BlockForestInitialization.h"
#include "mesh/boundary/BoundaryInfo.h"
#include "mesh/boundary/BoundaryLocation.h"
#include "mesh/boundary/BoundaryLocationFunction.h"
#include "mesh/boundary/BoundarySetup.h"
#include "mesh/boundary/BoundaryUIDFaceDataSource.h"
#include "mesh/boundary/ColorToBoundaryMapper.h"

#include "timeloop/all.h"

namespace walberla
{
uint_t numGhostLayers = uint_t(1);

//! [typedefs]
using LatticeModel_T         = lbm::D3Q27< lbm::collision_model::SRT >;
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

   /*
   Flow Parameters

   Parameters:
   - angleOfAttack: real_t
   - velocityMagnitude: real_t
   - initialVelocity: Vector3< real_t >

   Description: The flow parameters specify the initial state of the flow. For this there is an angle of attack and a velocity magnitude. The initial velocity is calculated based on the angle of attack and the velocity magnitude. 
   */
   auto parameters = walberlaEnv.config()->getOneBlock("flowParameters");

   // read the content from the block parameters. This contains: angleOfAttack, velocityMagnitude
   const real_t angleOfAttack = parameters.getParameter< real_t >("angleOfAttack", real_c(7.8)); // angle of attack in degrees
   const real_t flowVelocity = parameters.getParameter< real_t >("velocityMagnitude", real_c(10.0)); // velocity magnitude in m/s
   
   // calculate the initial velocity based on the angle of attack and the flow velocity
   const Vector3< real_t > initialVelocity = Vector3< real_t >(flowVelocity * cos(angleOfAttack*PI/180),
                                                               flowVelocity * sin(angleOfAttack*PI/180), Vector3< real_t >());
   WALBERLA_LOG_INFO_ON_ROOT("Angle of attack: " << angleOfAttack << " degrees")
   WALBERLA_LOG_INFO_ON_ROOT("Flow velocity: " << flowVelocity << " m/s")
   WALBERLA_LOG_INFO_ON_ROOT("Initial velocity < u, v, w >: " << initialVelocity << " m/s")
   // END of flow parameters


   /*
   Simulation Parameters

   Parameters:
   - timeSteps: uint_t
   - useGui: bool
   - remainingTimeLoggerFrequency: real_t

   Description: The simulation parameters are used for the simulation time, the use of the GUI and the terminal simulation logger. The GUI will not be used since Qt4 is not supported anymore by linux systems.
   */
   auto simulationParameters = walberlaEnv.config()->getOneBlock("simulationParameters");
   
   const uint_t timeSteps = parameters.getParameter< uint_t >("timeSteps", uint_c(10));
   const real_t remainingTimeLoggerFrequency =
      parameters.getParameter< real_t >("remainingTimeLoggerFrequency", real_c(3.0)); // in seconds
   
   WALBERLA_LOG_INFO_ON_ROOT("Number of time steps: " << timeSteps)
   // END of simulation parameters

   
   /*
   Domain Parameters

   Parameters:
   - meshFile: string
   - numLevels: uint_t
   - dx: real_t
   - domainScaling: Vector3< uint_t >
   - blocks: Vector3< uint_t >
   - cellsPerBlock: Vector3< uint_t >
   - periodic: Vector3< bool >

   Description: The domain parameters are used for the domain size and the block forest/refinement. 
   */
   auto domainParameters = walberlaEnv.config()->getOneBlock("domainParameters");

   std::string meshFile = domainParameters.getParameter< std::string >("meshFile");
   Vector3< uint_t > domainScaling =
      domainParameters.getParameter< Vector3< uint_t > >("domainScaling", Vector3< uint_t >(1));

   const real_t dx = domainParameters.getParameter< real_t >("dx", real_t(1));
   const real_t numLevels = domainParameters.getParameter< real_t >("numLevels", real_t(0));
   const real_t fineDX = dx / real_t(std::pow(2, numLevels)); // finest grid spacing based on octree refinement
   
   const Vector3< bool > periodicity =
      domainParameters.getParameter< Vector3< bool > >("periodic", Vector3< bool >(true));
   const Vector3< uint_t > cellsPerBlock = domainParameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
   
   WALBERLA_LOG_INFO_ON_ROOT("Mesh file: " << meshFile)
   WALBERLA_LOG_INFO_ON_ROOT("Domain scaling: " << domainScaling)
   WALBERLA_LOG_INFO_ON_ROOT("dx: " << dx)
   WALBERLA_LOG_INFO_ON_ROOT("Periodicity < x_sides, y_sides, z_sides: " << periodicity)
   WALBERLA_LOG_INFO_ON_ROOT("Cells per block < nx, ny, nz >: " << cellsPerBlock)
   WALBERLA_LOG_INFO_ON_ROOT("Number of refinement levels: " << numLevels)
   WALBERLA_LOG_INFO_ON_ROOT("Fine grid spacing: " << fineDX)
   

   // END of domain parameters

// ================================================================================================================     
   ////////////////////
   /// PROCESS MESH ///
   ////////////////////

   //! [readMesh]
   // read in mesh with vertex colors on a single process and broadcast it
   auto mesh = make_shared< mesh::TriangleMesh >();
   mesh->request_vertex_colors();
   mesh::readAndBroadcast(meshFile, *mesh);
   //! [readMesh]

   // color faces according to vertices
   vertexToFaceColor(*mesh, mesh::TriangleMesh::Color(255, 255, 255));

   //! [triDist]
   // add information to mesh that is required for computing signed distances from a point to a triangle
   auto triDist = make_shared< mesh::TriangleDistance< mesh::TriangleMesh > >(mesh);
   //! [triDist]

   //! [octree]
   // building distance octree
   auto distanceOctree = make_shared< mesh::DistanceOctree< mesh::TriangleMesh > >(triDist);
   //! [octree]

   WALBERLA_LOG_INFO_ON_ROOT("Octree has height " << distanceOctree->height())

   //! [octreeVTK]
   // write distance octree to file
   WALBERLA_ROOT_SECTION()
   {
      distanceOctree->writeVTKOutput("vtk_out/distanceOctree");
   }
   //! [octreeVTK]

   ///////////////////////////
   /// CREATE BLOCK FOREST ///
   ///////////////////////////

   //! [aabb]
   auto aabb = computeAABB(*mesh);
   aabb.scale(domainScaling); // Scale the domain based on the x, y and z size of the mesh object. 
   aabb.setCenter(aabb.center()); // Place the airfoil in the center of the domain
   //! [aabb]

   //! [bfc]
   // create and configure block forest creator
   mesh::ComplexGeometryStructuredBlockforestCreator bfc(aabb, Vector3< real_t >(dx));
   bfc.setRootBlockExclusionFunction(mesh::makeExcludeMeshInterior(distanceOctree, dx));
   bfc.setBlockExclusionFunction(mesh::makeExcludeMeshInteriorRefinement(distanceOctree, fineDX));

   auto meshWorkloadMemory = mesh::makeMeshWorkloadMemory(distanceOctree, dx);
   meshWorkloadMemory.setInsideCellWorkload(1);
   meshWorkloadMemory.setOutsideCellWorkload(1);
   bfc.setWorkloadMemorySUIDAssignmentFunction(meshWorkloadMemory);
   bfc.setPeriodicity(periodicity);
   bfc.setRefinementSelectionFunction(
      makeRefinementSelection(distanceOctree, numLevels - uint_t(1), dx, dx * real_t(1)));

    if (WriteSetupForestAndReturn)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Setting up SetupBlockForest")
      auto setupForest = bfc.createSetupBlockForest(blockSize);
      WALBERLA_LOG_INFO_ON_ROOT("Writing SetupBlockForest to VTK file")
      WALBERLA_ROOT_SECTION()
      {
         setupForest->writeVTKOutput("SetupBlockForest");
      }
      // WALBERLA_LOG_INFO_ON_ROOT("Stopping program")
      // return EXIT_SUCCESS;
   }
   
   //! [bfc]


   //! [blockForest]
   // create block forest
   auto blocks = bfc.createStructuredBlockForest(cellsPerBlock);
   //! [blockForest]

   ////////////////////////////////////
   /// CREATE AND INITIALIZE FIELDS ///
   ////////////////////////////////////

   // create fields
   LatticeModel_T latticeModel = LatticeModel_T(lbm::collision_model::SRT(omega));
   BlockDataID pdfFieldId =
      lbm::addPdfFieldToStorage(blocks, "pdf field", latticeModel, initialVelocity, real_t(1), numGhostLayers);
   BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field", numGhostLayers);

   /////////////////////////
   /// BOUNDARY HANDLING ///
   /////////////////////////

   //! [DefaultBoundaryHandling]
   // create and initialize boundary handling
   const FlagUID fluidFlagUID("Fluid");

   auto boundariesConfig = walberlaEnv.config()->getOneBlock("boundaryConditions");

   typedef lbm::DefaultBoundaryHandlingFactory< LatticeModel_T, FlagField_T > BHFactory;

   BlockDataID boundaryHandlingId = BHFactory::addBoundaryHandlingToStorage(
      blocks, "boundary handling", flagFieldId, pdfFieldId, fluidFlagUID,
      boundariesConfig.getParameter< Vector3< real_t > >("velocity0", Vector3< real_t >()),
      boundariesConfig.getParameter< Vector3< real_t > >("velocity1", Vector3< real_t >()),
      boundariesConfig.getParameter< real_t >("pressure0", real_c(1.0)),
      boundariesConfig.getParameter< real_t >("pressure1", real_c(1.0)));

   //! [DefaultBoundaryHandling]

   //! [colorToBoundary]
   // set NoSlip UID to boundaries that we colored
   mesh::ColorToBoundaryMapper< mesh::TriangleMesh > colorToBoundaryMapper(
      (mesh::BoundaryInfo(BHFactory::getNoSlipBoundaryUID())));
   colorToBoundaryMapper.set(mesh::TriangleMesh::Color(255, 255, 255),
                             mesh::BoundaryInfo(BHFactory::getNoSlipBoundaryUID()));

   // mark boundaries
   auto boundaryLocations = colorToBoundaryMapper.addBoundaryInfoToMesh(*mesh);
   //! [colorToBoundary]

   //! [VTKMesh]
   // write mesh info to file
   mesh::VTKMeshWriter< mesh::TriangleMesh > meshWriter(mesh, "meshBoundaries", 1);
   meshWriter.addDataSource(make_shared< mesh::BoundaryUIDFaceDataSource< mesh::TriangleMesh > >(boundaryLocations));
   meshWriter.addDataSource(make_shared< mesh::ColorFaceDataSource< mesh::TriangleMesh > >());
   meshWriter.addDataSource(make_shared< mesh::ColorVertexDataSource< mesh::TriangleMesh > >());
   meshWriter();
   //! [VTKMesh]

   //! [boundarySetup]
   // voxelize mesh
   WALBERLA_LOG_DEVEL_ON_ROOT("Voxelizing mesh")
   mesh::BoundarySetup boundarySetup(blocks, makeMeshDistanceFunction(distanceOctree), numGhostLayers);
   WALBERLA_LOG_DEVEL( "Writing Voxelisation" );
   boundarySetup.writeVTKVoxelfile();

   // set fluid cells
   WALBERLA_LOG_DEVEL_ON_ROOT("Setting up fluid cells")
   boundarySetup.setDomainCells< BHFactory::BoundaryHandling >(boundaryHandlingId, mesh::BoundarySetup::OUTSIDE);

   // set up inflow/outflow/wall boundaries from DefaultBoundaryHandlingFactory
   geometry::initBoundaryHandling< BHFactory::BoundaryHandling >(*blocks, boundaryHandlingId, boundariesConfig);
   
   // set up obstacle boundaries from file
   WALBERLA_LOG_DEVEL_ON_ROOT("Setting up boundaries")
   boundarySetup.setBoundaries< BHFactory::BoundaryHandling >(
      boundaryHandlingId, makeBoundaryLocationFunction(distanceOctree, boundaryLocations), mesh::BoundarySetup::INSIDE);
   WALBERLA_LOG_DEVEL_ON_ROOT("done")
   //! [boundarySetup]

   // Log performance information at the end of the simulation
   lbm::BlockForestEvaluation< FlagField_T >(blocks, flagFieldId, fluidFlagUID).logInfoOnRoot();
   lbm::PerformanceLogger< FlagField_T > perfLogger(blocks, flagFieldId, fluidFlagUID, 100);

   //////////////////////////////////
   /// SET UP SWEEPS AND TIMELOOP ///
   //////////////////////////////////

   // create time loop
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   auto sweep = lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >(pdfFieldId, flagFieldId, fluidFlagUID);
   auto refinementTimeStep = lbm::refinement::makeTimeStep< LatticeModel_T, BHFactory::BoundaryHandling >(
      blocks, sweep, pdfFieldId, boundaryHandlingId);

   // The refined cells need a smaller time step to ensure stability
   timeloop.addFuncBeforeTimeStep(makeSharedFunctor(refinementTimeStep), "Refinement time step");

   // create communication for PdfField
   blockforest::communication::UniformBufferedScheme< CommunicationStencil_T > communication(blocks);
   communication.addPackInfo(make_shared< lbm::PdfFieldPackInfo< LatticeModel_T > >(pdfFieldId));

   // add LBM sweep and communication to time loop
   timeloop.add() << BeforeFunction(communication, "communication")
                  << Sweep(BHFactory::BoundaryHandling::getBlockSweep(boundaryHandlingId), "boundary handling");
   timeloop.add() << Sweep(
      makeSharedSweep(lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >(pdfFieldId, flagFieldId, fluidFlagUID)),
      "LB stream & collide");

   // LBM stability check
   timeloop.addFuncAfterTimeStep(makeSharedFunctor(field::makeStabilityChecker< PdfField_T, FlagField_T >(
                                    walberlaEnv.config(), blocks, pdfFieldId, flagFieldId, fluidFlagUID)),
                                 "LBM stability check");

   // log remaining time
   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency),
                                 "remaining time logger");

   //////////////////
   /// VTK OUTPUT ///
   //////////////////

   // add VTK output to time loop
   auto VTKParams = walberlaEnv.config()->getBlock("VTK");
   uint_t vtkWriteFrequency = VTKParams.getBlock("fluid_field").getParameter("writeFrequency", uint_t(0));
   auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "fluid_field", vtkWriteFrequency, uint_t(0), false,
                                                   "vtk_out", "simulation_step", false, true, true, false, 0);

   field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagFieldId);
   fluidFilter.addFlag(fluidFlagUID);
   vtkOutput->addCellInclusionFilter(fluidFilter);

   auto velocityWriter = make_shared< lbm::VelocityVTKWriter< LatticeModel_T, float > >(pdfFieldId, "Velocity");
   auto densityWriter  = make_shared< lbm::DensityVTKWriter< LatticeModel_T, float > >(pdfFieldId, "Density");
   vtkOutput->addCellDataWriter(velocityWriter);
   vtkOutput->addCellDataWriter(densityWriter);

   timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   timeloop.addFuncAfterTimeStep(perfLogger, "Evaluator: performance logging");

   // create adaptors, so that the GUI also displays density and velocity
   // adaptors are like fields with the difference that they do not store values
   // but calculate the values based on other fields ( here the PdfField )
   field::addFieldAdaptor< lbm::Adaptor< LatticeModel_T >::Density >(blocks, pdfFieldId, "DensityAdaptor");
   field::addFieldAdaptor< lbm::Adaptor< LatticeModel_T >::VelocityVector >(blocks, pdfFieldId, "VelocityAdaptor");

   //////////////////////
   /// RUN SIMULATION ///
   //////////////////////
   timeloop.run();

   return EXIT_SUCCESS;
}
} // namespace walberla

int main(int argc, char** argv) { walberla::main(argc, argv); }
