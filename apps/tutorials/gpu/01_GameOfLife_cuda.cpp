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
//! \file 03_GameOfLife.cpp
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

#include "01_GameOfLife_kernels.h"
#include "blockforest/Initialization.h"

#include "core/Environment.h"

#include "gpu/HostFieldAllocator.h"
#include "gpu/FieldCopy.h"
#include "gpu/GPUField.h"
#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/communication/GPUPackInfo.h"
#include "gpu/communication/UniformGPUScheme.h"

#include "field/AddToStorage.h"
#include "field/vtk/VTKWriter.h"

#include "geometry/initializer/ScalarFieldFromGrayScaleImage.h"
#include "geometry/structured/GrayScaleImage.h"

#include "stencil/D2Q9.h"

#include "timeloop/SweepTimeloop.h"

#if defined(WALBERLA_BUILD_WITH_SYCL)
#include "CL/sycl.hpp"
#endif


using namespace walberla;

using ScalarField = GhostLayerField<real_t, 1>;
using GPUField = gpu::GPUField<real_t>;
using CommScheme = gpu::communication::UniformGPUScheme<stencil::D2Q9 > ;
using Packing = gpu::communication::GPUPackInfo<GPUField> ;

int i = 0;

int main( int argc, char ** argv )
{

   WALBERLA_LOG_INFO("Hello World")

   walberla::Environment const env( argc, argv );
   WALBERLA_LOG_INFO("Create image")

   geometry::GrayScaleImage const image ("GosperGliderGun.png");

   WALBERLA_LOG_INFO("Create Blockforest")

   // Create blocks
   shared_ptr< StructuredBlockForest > const blocks = blockforest::createUniformBlockGrid (
            uint_t(1) ,              uint_t(2),                           uint_t(1), // number of blocks in x,y,z direction
            image.size( uint_t(0) ), image.size( uint_t(1) ) / uint_t(2), uint_t(1), // how many cells per block (x,y,z)
            real_t(1),                                                               // dx: length of one cell in physical coordinates
            false,                                                                   // one block per process - "false" means all blocks to one process
            false, false, false );                                                   // no periodicity

   WALBERLA_LOG_INFO("Create SYCL queue")

#if defined(WALBERLA_BUILD_WITH_SYCL)
   //auto syclQueue = make_shared<sycl::queue> (sycl::default_selector_v);
   auto syclQueue = make_shared<sycl::queue> (sycl::cpu_selector_v);
   std::cout << "Running on " << (*syclQueue).get_device().get_info<cl::sycl::info::device::name>() << "\n";
   blocks->setSYCLQueue(syclQueue);
#endif

   //auto hostFieldAllocator = make_shared< gpu::HostFieldAllocator<real_t> >();
   BlockDataID const cpuFieldID =field::addToStorage< ScalarField >(blocks, "CPU Field", real_c(0.0), field::fzyx, uint_c(1)/*, hostFieldAllocator*/);

   // Initializing the field from an image
   using geometry::initializer::ScalarFieldFromGrayScaleImage;
   ScalarFieldFromGrayScaleImage fieldInitializer ( *blocks, cpuFieldID ) ;
   fieldInitializer.init( image, uint_t(2), false );

   WALBERLA_LOG_INFO("Create GPU Fields")

   BlockDataID const gpuFieldSrcID = gpu::addGPUFieldToStorage<ScalarField>( blocks, cpuFieldID, "GPU Field Src", false );
   BlockDataID const gpuFieldDstID = gpu::addGPUFieldToStorage<ScalarField>( blocks, cpuFieldID, "GPU Field Dst", false );

   //const bool sendDirectlyFromGPU = false;
   //CommScheme commScheme(blocks, sendDirectlyFromGPU);
   //commScheme.addPackInfo( make_shared<Packing>(gpuFieldSrcID) );

   WALBERLA_LOG_INFO("Create Timeloops")

   // Create Timeloop
   const uint_t numberOfTimesteps = uint_t(1001); // number of timesteps for non-gui runs
   SweepTimeloop timeloop ( blocks, numberOfTimesteps );

   // Registering the sweep
   timeloop.add() /*<< BeforeFunction(  commScheme.getCommunicateFunctor(), "Communication" )*/
#if defined(WALBERLA_BUILD_WITH_SYCL)
                  << Sweep( GameOfLifeSweepCUDA(gpuFieldSrcID, gpuFieldDstID, syclQueue ), "GameOfLifeSweep" );
#else
                  << Sweep( GameOfLifeSweepCUDA(gpuFieldSrcID, gpuFieldDstID ), "GameOfLifeSweep" );
#endif
   // VTK Writer every vtkWriteFrequency timesteps
   const uint_t vtkWriteFrequency = 2;
   if (vtkWriteFrequency > 0)
   {
      // Create a vtkOutput object with standard arguments
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency);

      // Before the VTK output we need to sync the GPU data to the CPU memory
      vtkOutput->addBeforeFunction(gpu::fieldCpyFunctor<ScalarField, GPUField >(blocks, cpuFieldID, gpuFieldDstID));

      // Then create a dataWriter and write the output
      auto dataWriter = make_shared< field::VTKWriter< ScalarField > >(cpuFieldID, "output");
      vtkOutput->addCellDataWriter(dataWriter);
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   }

   WcTimer simTimer;
   //WALBERLA_GPU_CHECK( gpuDeviceSynchronize() )
   simTimer.start();
   timeloop.run();
   ////WALBERLA_GPU_CHECK( gpuDeviceSynchronize() )
   simTimer.end();
   WALBERLA_LOG_INFO_ON_ROOT("Simulation finished")
   auto time      = real_c(simTimer.last());
   WALBERLA_LOG_RESULT_ON_ROOT("Game of life tutorial finished. Elapsed time " << time)

   return EXIT_SUCCESS;
}
