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
//! \file Percolation.cpp
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/grid_generator/SCIterator.h"
#include "core/logging/all.h"
#include "core/timing/RemainingTimeLogger.h"

#include "field/AddToStorage.h"
#include "field/vtk/all.h"

#include "geometry/InitBoundaryHandling.h"

#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/communication/UniformGPUScheme.h"

#include "lbm/PerformanceLogger.h"
#include "lbm/vtk/all.h"

#include "lbm_mesapd_coupling/DataTypesGPU.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/gpu/PSMSweepCollectionGPU.h"
#include "lbm_mesapd_coupling/utility/ParticleSelector.h"

#include "mesa_pd/data/DataTypes.h"
#include "mesa_pd/data/ParticleAccessorWithShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/ShapeStorage.h"
#include "mesa_pd/data/shape/Sphere.h"
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/mpi/SyncNextNeighbors.h"
#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "vtk/all.h"

#include "LBMSweep.h"
#include "PSMPackInfo.h"
#include "PSMSweep.h"
#include "PSM_Density.h"
#include "PSM_InfoHeader.h"
#include "PSM_MacroGetter.h"

namespace percolation
{

///////////
// USING //
///////////

using namespace walberla;
using namespace lbm_mesapd_coupling::psm::gpu;
typedef pystencils::PSMPackInfo PackInfo_T;

using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

///////////
// FLAGS //
///////////

const FlagUID Fluid_Flag("Fluid");
const FlagUID Density_Flag("Density");
const FlagUID NoSlip_Flag("NoSlip");
const FlagUID Inflow_Flag("Inflow");

//////////
// MAIN //
//////////

//*******************************************************************************************************************
/*!\brief Benchmark of a percolation setup
 *
 * This code can be used as a percolation (useParticles=true) or as a channel flow (useParticles=false) benchmark.
 * A constant inflow from west is applied and a pressure boundary condition is set at the east.
 * For the percolation, mono-sized fixed spherical particles are generated on a structured grid with an offset for
 * every second particle layer in flow direction to avoid channels in flow direction. The flow is described by Darcy's
 * law. For the channel flow, the flow is described by the Hagen–Poiseuille equation.
 *
 * The domain is either periodic or bounded by (no slip) walls in the vertical directions (y and z).
 *
 * For the percolation, the PSM is used in combination with a two-way coupling, but no particle dynamics.
 * For the channel flow, only the LBM is used.
 *
 * The parameters can be changed via the input file.
 *
 */
//*******************************************************************************************************************
int main(int argc, char** argv)
{
   Environment env(argc, argv);
   auto cfgFile = env.config();
   if (!cfgFile) { WALBERLA_ABORT("Usage: " << argv[0] << " path-to-configuration-file \n"); }

   gpu::selectDeviceBasedOnMpiRank();

   WALBERLA_LOG_INFO_ON_ROOT("waLBerla revision: " << std::string(WALBERLA_GIT_SHA1).substr(0, 8));
   WALBERLA_LOG_INFO_ON_ROOT("compiler flags: " << std::string(WALBERLA_COMPILER_FLAGS));
   WALBERLA_LOG_INFO_ON_ROOT("build machine: " << std::string(WALBERLA_BUILD_MACHINE));
   WALBERLA_LOG_INFO_ON_ROOT(*cfgFile);

   // Read config file
   Config::BlockHandle numericalSetup = cfgFile->getBlock("NumericalSetup");
   const uint_t numXBlocks            = numericalSetup.getParameter< uint_t >("numXBlocks");
   const uint_t numYBlocks            = numericalSetup.getParameter< uint_t >("numYBlocks");
   const uint_t numZBlocks            = numericalSetup.getParameter< uint_t >("numZBlocks");
   WALBERLA_CHECK_EQUAL(numXBlocks * numYBlocks * numZBlocks, uint_t(MPIManager::instance()->numProcesses()),
                        "When using GPUs, the number of blocks ("
                           << numXBlocks * numYBlocks * numZBlocks << ") has to match the number of MPI processes ("
                           << uint_t(MPIManager::instance()->numProcesses()) << ")");
   const bool periodicInY                 = numericalSetup.getParameter< bool >("periodicInY");
   const bool periodicInZ                 = numericalSetup.getParameter< bool >("periodicInZ");
   const uint_t numXCellsPerBlock         = numericalSetup.getParameter< uint_t >("numXCellsPerBlock");
   const uint_t numYCellsPerBlock         = numericalSetup.getParameter< uint_t >("numYCellsPerBlock");
   const uint_t numZCellsPerBlock         = numericalSetup.getParameter< uint_t >("numZCellsPerBlock");
   const uint_t timeSteps                 = numericalSetup.getParameter< uint_t >("timeSteps");
   const bool useParticles                = numericalSetup.getParameter< bool >("useParticles");
   const real_t particleDiameter          = numericalSetup.getParameter< real_t >("particleDiameter");
   const real_t particleGenerationSpacing = numericalSetup.getParameter< real_t >("particleGenerationSpacing");
   const bool useParticleOffset           = numericalSetup.getParameter< bool >("useParticleOffset");
   const Vector3< uint_t > particleNumSubBlocks =
      numericalSetup.getParameter< Vector3< uint_t > >("particleNumSubBlocks");
   const real_t uInflow        = numericalSetup.getParameter< real_t >("uInflow");
   const real_t relaxationRate = numericalSetup.getParameter< real_t >("relaxationRate");
   if ((periodicInY && numYBlocks == 1) || (periodicInZ && numZBlocks == 1))
   {
      WALBERLA_LOG_WARNING_ON_ROOT("Using only 1 block in periodic dimensions can lead to unexpected behavior.")
   }
   const real_t viscosity = lbm::collision_model::viscosityFromOmega(relaxationRate);
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(viscosity)

   Config::BlockHandle outputSetup      = cfgFile->getBlock("Output");
   const uint_t vtkSpacing              = outputSetup.getParameter< uint_t >("vtkSpacing");
   const std::string vtkFolder          = outputSetup.getParameter< std::string >("vtkFolder");
   const uint_t performanceLogFrequency = outputSetup.getParameter< uint_t >("performanceLogFrequency");

   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   const bool periodicInX                     = false;
   shared_ptr< StructuredBlockForest > blocks = blockforest::createUniformBlockGrid(
      numXBlocks, numYBlocks, numZBlocks, numXCellsPerBlock, numYCellsPerBlock, numZCellsPerBlock, real_t(1), uint_t(0),
      false, false, periodicInX, periodicInY, periodicInZ, // periodicity
      false);

   auto simulationDomain = blocks->getDomain();

   ////////////
   // MesaPD //
   ////////////

   auto rpdDomain = std::make_shared< mesa_pd::domain::BlockForestDomain >(blocks->getBlockForestPointer());

   // Init data structures
   auto ps                  = walberla::make_shared< mesa_pd::data::ParticleStorage >(1);
   auto ss                  = walberla::make_shared< mesa_pd::data::ShapeStorage >();
   using ParticleAccessor_T = mesa_pd::data::ParticleAccessorWithShape;
   auto accessor            = walberla::make_shared< ParticleAccessor_T >(ps, ss);
   auto sphereShape         = ss->create< mesa_pd::data::Sphere >(particleDiameter * real_t(0.5));

   // Create spheres
   if (useParticles)
   {
      auto generationDomain =
         math::AABB::createFromMinMaxCorner(math::Vector3< real_t >(simulationDomain.xMax() * real_t(0.25),
                                                                    simulationDomain.yMin(), simulationDomain.zMin()),
                                            math::Vector3< real_t >(simulationDomain.xMax() * real_t(0.75),
                                                                    simulationDomain.yMax(), simulationDomain.zMax()));
      real_t particleOffset = particleGenerationSpacing / real_t(2);
      for (auto pt : grid_generator::SCGrid(generationDomain, generationDomain.center(), particleGenerationSpacing))
      {
         // Offset every second particle layer in flow direction to avoid channels in flow direction
         if (useParticleOffset &&
             uint_t(round(math::abs(generationDomain.center()[0] - pt[0]) / (particleGenerationSpacing))) % uint_t(2) !=
                uint_t(0))
         {
            pt = pt + Vector3(real_t(0), particleOffset, particleOffset);
         }
         if (rpdDomain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), pt))
         {
            mesa_pd::data::Particle&& p = *ps->create();
            p.setPosition(pt);
            p.setInteractionRadius(particleDiameter * real_t(0.5));
            p.setOwner(mpi::MPIManager::instance()->rank());
            p.setShapeID(sphereShape);
            p.setType(0);
         }
      }
   }

   ///////////////////////
   // ADD DATA TO BLOCKS //
   ////////////////////////

   // Setting initial PDFs to nan helps to detect bugs in the initialization/BC handling
   BlockDataID pdfFieldID =
      field::addToStorage< PdfField_T >(blocks, "pdf field (fzyx)", real_c(std::nan("")), field::fzyx);
   BlockDataID pdfFieldGPUID  = gpu::addGPUFieldToStorage< PdfField_T >(blocks, pdfFieldID, "pdf field GPU");
   BlockDataID densityFieldID = field::addToStorage< DensityField_T >(blocks, "density field", real_t(0), field::fzyx);
   BlockDataID velFieldID  = field::addToStorage< VelocityField_T >(blocks, "velocity field", real_t(0), field::fzyx);
   BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");
   BlockDataID BFieldID    = field::addToStorage< BField_T >(blocks, "B field", 0, field::fzyx, 1);

   // Synchronize particles between the blocks for the correct mapping of ghost particles
   mesa_pd::mpi::SyncNextNeighbors syncNextNeighborFunc;
   syncNextNeighborFunc(*ps, *rpdDomain);

   // Assemble boundary block string
   std::string boundariesBlockString = " Boundaries"
                                       "{"
                                       "Border { direction W;    walldistance -1;  flag Inflow; }"
                                       "Border { direction E;    walldistance -1;  flag Density; }";

   if (!periodicInY)
   {
      boundariesBlockString += "Border { direction S;    walldistance -1;  flag NoSlip; }"
                               "Border { direction N;    walldistance -1;  flag NoSlip; }";
   }

   if (!periodicInZ)
   {
      boundariesBlockString += "Border { direction T;    walldistance -1;  flag NoSlip; }"
                               "Border { direction B;    walldistance -1;  flag NoSlip; }";
   }

   boundariesBlockString += "}";
   WALBERLA_ROOT_SECTION()
   {
      std::ofstream boundariesFile("boundaries.prm");
      boundariesFile << boundariesBlockString;
      boundariesFile.close();
   }
   WALBERLA_MPI_BARRIER()

   auto boundariesCfgFile = Config();
   boundariesCfgFile.readParameterFile("boundaries.prm");
   auto boundariesConfig = boundariesCfgFile.getBlock("Boundaries");

   // map boundaries into the LBM simulation
   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldID, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldID, Fluid_Flag);
   lbm::PSM_Density density_bc(blocks, pdfFieldGPUID, real_t(1.0));
   density_bc.fillFromFlagField< FlagField_T >(blocks, flagFieldID, Density_Flag, Fluid_Flag);
   lbm::PSM_NoSlip noSlip(blocks, pdfFieldGPUID);
   noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldID, NoSlip_Flag, Fluid_Flag);
   lbm::PSM_UBB ubb(blocks, pdfFieldGPUID, uInflow, real_t(0), real_t(0));
   ubb.fillFromFlagField< FlagField_T >(blocks, flagFieldID, Inflow_Flag, Fluid_Flag);

   ///////////////
   // TIME LOOP //
   ///////////////

   // Map particles into the fluid domain
   ParticleAndVolumeFractionSoA_T< Weighting > particleAndVolumeFractionSoA(blocks, relaxationRate);
   PSMSweepCollectionGPU psmSweepCollection(blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(),
                                            particleAndVolumeFractionSoA, particleNumSubBlocks);
   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      psmSweepCollection.particleMappingSweep(&(*blockIt));
   }

   // Initialize PDFs
   pystencils::InitializeDomainForPSM pdfSetter(
      particleAndVolumeFractionSoA.BsFieldID, particleAndVolumeFractionSoA.BFieldID,
      particleAndVolumeFractionSoA.particleVelocitiesFieldID, pdfFieldGPUID, real_t(0), real_t(0), real_t(0),
      real_t(1.0), real_t(0), real_t(0), real_t(0));

   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      // pdfSetter requires particle velocities at cell centers
      psmSweepCollection.setParticleVelocitiesSweep(&(*blockIt));
      pdfSetter(&(*blockIt));
   }

   // Setup of the LBM communication for synchronizing the pdf field between neighboring blocks
   // TODO: set sendDirectlyFromGPU to true for performance measurements on cluster
   gpu::communication::UniformGPUScheme< Stencil_T > com(blocks, false, false);
   com.addPackInfo(make_shared< PackInfo_T >(pdfFieldGPUID));
   auto communication = std::function< void() >([&]() { com.communicate(); });

   SweepTimeloop timeloop(blocks->getBlockStorage(), timeSteps);

   timeloop.addFuncBeforeTimeStep(RemainingTimeLogger(timeloop.getNrOfTimeSteps()), "Remaining Time Logger");

   pystencils::PSM_MacroGetter getterSweep(BFieldID, densityFieldID, pdfFieldID, velFieldID, real_t(0.0), real_t(0.0),
                                           real_t(0.0));
   // VTK output
   if (vtkSpacing != uint_t(0))
   {
      // Spheres
      auto particleVtkOutput = make_shared< mesa_pd::vtk::ParticleVtkOutput >(ps);
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleUid >("uid");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleLinearVelocity >("velocity");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleInteractionRadius >("radius");
      // Limit output to process-local spheres
      particleVtkOutput->setParticleSelector([sphereShape](const mesa_pd::data::ParticleStorage::iterator& pIt) {
         return pIt->getShapeID() == sphereShape &&
                !(mesa_pd::data::particle_flags::isSet(pIt->getFlags(), mesa_pd::data::particle_flags::GHOST));
      });
      auto particleVtkWriter = vtk::createVTKOutput_PointData(particleVtkOutput, "particles", vtkSpacing, vtkFolder);
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(particleVtkWriter), "VTK (sphere data)");

      // Fields
      auto pdfFieldVTK = vtk::createVTKOutput_BlockData(blocks, "fluid", vtkSpacing, 0, false, vtkFolder);

      pdfFieldVTK->addBeforeFunction(communication);

      pdfFieldVTK->addBeforeFunction([&]() {
         gpu::fieldCpy< PdfField_T, gpu::GPUField< real_t > >(blocks, pdfFieldID, pdfFieldGPUID);
         gpu::fieldCpy< GhostLayerField< real_t, 1 >, BFieldGPU_T >(blocks, BFieldID,
                                                                    particleAndVolumeFractionSoA.BFieldID);
         for (auto& block : *blocks)
            getterSweep(&block);
      });

      pdfFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< VelocityField_T > >(velFieldID, "Velocity"));
      pdfFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< DensityField_T > >(densityFieldID, "Density"));
      pdfFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< BField_T > >(BFieldID, "OverlapFraction"));
      pdfFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< FlagField_T > >(flagFieldID, "FlagField"));

      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(pdfFieldVTK), "VTK (fluid field data)");
   }

   if (vtkSpacing != uint_t(0)) { vtk::writeDomainDecomposition(blocks, "domain_decomposition", vtkFolder); }

   // Add performance logging
   if (performanceLogFrequency > 0)
   {
      const lbm::PerformanceLogger< FlagField_T > performanceLogger(blocks, flagFieldID, Fluid_Flag,
                                                                    performanceLogFrequency);
      timeloop.addFuncAfterTimeStep(performanceLogger, "Evaluate performance logging");
   }

   // Add LBM communication function and boundary handling sweep
   // TODO: use split sweeps to hide communication
   timeloop.add() << BeforeFunction(communication, "LBM Communication")
                  << Sweep(deviceSyncWrapper(density_bc.getSweep()), "Boundary Handling (Density)");
   timeloop.add() << Sweep(deviceSyncWrapper(ubb.getSweep()), "Boundary Handling (UBB)");
   if (!periodicInY || !periodicInZ)
   {
      timeloop.add() << Sweep(deviceSyncWrapper(noSlip.getSweep()), "Boundary Handling (NoSlip)");
   }

   // PSM kernel
   pystencils::PSMSweep PSMSweep(particleAndVolumeFractionSoA.BsFieldID, particleAndVolumeFractionSoA.BFieldID,
                                 particleAndVolumeFractionSoA.particleForcesFieldID,
                                 particleAndVolumeFractionSoA.particleVelocitiesFieldID, pdfFieldGPUID, real_t(0.0),
                                 real_t(0.0), real_t(0.0), relaxationRate);
   pystencils::LBMSweep LBMSweep(pdfFieldGPUID, real_t(0.0), real_t(0.0), real_t(0.0), relaxationRate);

   if (useParticles) { addPSMSweepsToTimeloop(timeloop, psmSweepCollection, PSMSweep); }
   else { timeloop.add() << Sweep(deviceSyncWrapper(LBMSweep), "LBM sweep"); }

   WcTimingPool timeloopTiming;
   // TODO: maybe add warmup phase
   timeloop.run(timeloopTiming);
   timeloopTiming.logResultOnRoot();

   return EXIT_SUCCESS;
}

} // namespace percolation

int main(int argc, char** argv) { percolation::main(argc, argv); }