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
//! \file ParticleAndVolumeFractionMapping.cpp
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/debug/TestSubsystem.h"

#include "cuda/AddGPUFieldToStorage.h"
#include "cuda/FieldCopy.h"
#include "cuda/FieldIndexing.h"
#include "cuda/GPUField.h"
#include "cuda/HostFieldAllocator.h"
#include "cuda/Kernel.h"

#include "field/AddToStorage.h"

#include "lbm_mesapd_coupling/DataTypes.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/cuda/ParticleAndVolumeFractionMappingGPU.h"
#include "lbm_mesapd_coupling/utility/ParticleSelector.h"

#include "mesa_pd/data/ParticleAccessorWithShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/ShapeStorage.h"
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/kernel/SemiImplicitEuler.h"
#include "mesa_pd/mpi/SyncNextNeighbors.h"

#include <functional>
#include <memory>

namespace particle_volume_fraction_check
{

///////////
// USING //
///////////

using namespace walberla;
using walberla::uint_t;
using namespace lbm_mesapd_coupling;

//*******************************************************************************************************************
/*!\brief Calculating the sum over all fraction values. This can be used as a sanity check since it has to be roughly
 * equal to the volume of all particles.
 *
 */
//*******************************************************************************************************************
class FractionFieldSum
{
 public:
   FractionFieldSum(const shared_ptr< StructuredBlockStorage >& blockStorage,
                    BlockDataID particleAndVolumeFractionFieldID)
      : blockStorage_(blockStorage), particleAndVolumeFractionFieldID_(particleAndVolumeFractionFieldID)
   {}

   real_t operator()()
   {
      real_t sum = 0.0;

      for (auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt)
      {
         psm::cuda::ParticleAndVolumeFractionField_T* particleAndVolumeFractionField =
            blockIt->getData< psm::cuda::ParticleAndVolumeFractionField_T >(particleAndVolumeFractionFieldID_);

         const cell_idx_t xSize = cell_idx_c(particleAndVolumeFractionField->xSize());
         const cell_idx_t ySize = cell_idx_c(particleAndVolumeFractionField->ySize());
         const cell_idx_t zSize = cell_idx_c(particleAndVolumeFractionField->zSize());

         for (cell_idx_t z = 0; z < zSize; ++z)
         {
            for (cell_idx_t y = 0; y < ySize; ++y)
            {
               for (cell_idx_t x = 0; x < xSize; ++x)
               {
                  for (uint_t n = 0; n < particleAndVolumeFractionField->get(x, y, z).index; ++n)
                  {
                     sum += particleAndVolumeFractionField->get(x, y, z).overlapFractions[n];
                  }
               }
            }
         }
      }

      WALBERLA_MPI_SECTION() { mpi::allReduceInplace(sum, mpi::SUM); }

      return sum;
   }

 private:
   shared_ptr< StructuredBlockStorage > blockStorage_;
   BlockDataID particleAndVolumeFractionFieldID_;
};

////////////////
// Parameters //
////////////////

struct Setup
{
   // domain size (in lattice cells) in x, y and z direction
   uint_t xlength;
   uint_t ylength;
   uint_t zlength;

   // number of block in x, y and z, direction
   Vector3< uint_t > nBlocks;

   // cells per block in x, y and z direction
   Vector3< uint_t > cellsPerBlock;

   real_t sphereDiam;

   uint_t timesteps;
};

psm::cuda::ParticleAndVolumeFractionField_T* createField(IBlock* const block, StructuredBlockStorage* const storage)
{
   return new psm::cuda::ParticleAndVolumeFractionField_T(
      storage->getNumberOfXCells(*block),                               // number of cells in x direction per block
      storage->getNumberOfYCells(*block),                               // number of cells in y direction per block
      storage->getNumberOfZCells(*block),                               // number of cells in z direction per block
      1,                                                                // one ghost layer
      psm::cuda::PSMCell_T(),                                           // initial value
      field::fzyx,                                                      // layout
      make_shared< cuda::HostFieldAllocator< psm::cuda::PSMCell_T > >() // allocator for host pinned memory
   );
}

//////////
// MAIN //
//////////

//*******************************************************************************************************************
/*!\brief Testcase that checks if ParticleAndVolumeFractionMapping.h works as intended
 *
 * A sphere particle is placed inside the domain and is moving with a constant velocity. The overlap fraction is
 * computed for all cells in each time step. If the mapping is correct, the sum over all fractions should be roughly
 * equivalent to the volume of the sphere.
 *
 */
//*******************************************************************************************************************

int main(int argc, char** argv)
{
   debug::enterTestMode();

   mpi::Environment env(argc, argv);

   auto processes = MPIManager::instance()->numProcesses();

   if (processes != 27)
   {
      std::cerr << "Number of processes must be 27!" << std::endl;
      return EXIT_FAILURE;
   }

   ///////////////////////////
   // SIMULATION PROPERTIES //
   ///////////////////////////

   Setup setup;

   setup.sphereDiam = real_c(12);
   setup.zlength    = uint_c(4 * setup.sphereDiam);
   setup.xlength    = setup.zlength;
   setup.ylength    = setup.zlength;

   const real_t sphereRadius = real_c(0.5) * setup.sphereDiam;
   const real_t dx           = real_c(1);

   setup.timesteps = 100;

   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   setup.nBlocks[0]       = uint_c(3);
   setup.nBlocks[1]       = uint_c(3);
   setup.nBlocks[2]       = uint_c(3);
   setup.cellsPerBlock[0] = setup.xlength / setup.nBlocks[0];
   setup.cellsPerBlock[1] = setup.ylength / setup.nBlocks[1];
   setup.cellsPerBlock[2] = setup.zlength / setup.nBlocks[2];

   auto blocks =
      blockforest::createUniformBlockGrid(setup.nBlocks[0], setup.nBlocks[1], setup.nBlocks[2], setup.cellsPerBlock[0],
                                          setup.cellsPerBlock[1], setup.cellsPerBlock[2], dx, true, true, true, true);

   ////////////
   // MesaPD //
   ////////////

   auto mesapdDomain        = std::make_shared< mesa_pd::domain::BlockForestDomain >(blocks->getBlockForestPointer());
   auto ps                  = std::make_shared< mesa_pd::data::ParticleStorage >(1);
   auto ss                  = std::make_shared< mesa_pd::data::ShapeStorage >();
   using ParticleAccessor_T = mesa_pd::data::ParticleAccessorWithShape;
   auto accessor            = walberla::make_shared< ParticleAccessor_T >(ps, ss);

   // set up synchronization
   std::function< void(void) > syncCall = [&]() {
      const real_t overlap = real_t(1.5) * dx;
      mesa_pd::mpi::SyncNextNeighbors syncNextNeighborFunc;
      syncNextNeighborFunc(*ps, *mesapdDomain, overlap);
   };

   // add the sphere in the center of the domain
   Vector3< real_t > position(real_c(setup.xlength) * real_c(0.5), real_c(setup.ylength) * real_c(0.5),
                              real_c(setup.zlength) * real_c(0.5));
   Vector3< real_t > velocity(real_c(-0.1), real_c(-0.1), real_c(-0.1));
   auto sphereShape = ss->create< mesa_pd::data::Sphere >(sphereRadius);

   if (mesapdDomain->isContainedInProcessSubdomain(uint_c(walberla::mpi::MPIManager::instance()->rank()), position))
   {
      auto sphereParticle = ps->create();

      sphereParticle->setShapeID(sphereShape);
      sphereParticle->setType(0);
      sphereParticle->setPosition(position);
      sphereParticle->setLinearVelocity(velocity);
      sphereParticle->setOwner(walberla::MPIManager::instance()->rank());
      sphereParticle->setInteractionRadius(sphereRadius);
   }

   Vector3< real_t > position2(real_c(0.0), real_c(0.0), real_c(0.0));
   Vector3< real_t > velocity2(real_c(0.1), real_c(0.1), real_c(0.1));

   if (mesapdDomain->isContainedInProcessSubdomain(uint_c(walberla::mpi::MPIManager::instance()->rank()), position2))
   {
      auto sphereParticle = ps->create();

      sphereParticle->setShapeID(sphereShape);
      sphereParticle->setType(0);
      sphereParticle->setPosition(position2);
      sphereParticle->setLinearVelocity(velocity2);
      sphereParticle->setOwner(walberla::MPIManager::instance()->rank());
      sphereParticle->setInteractionRadius(sphereRadius);
   }

   syncCall();

   ///////////////////////
   // ADD DATA TO BLOCKS //
   ////////////////////////

   // add particle and volume fraction field (needed for the PSM)
   BlockDataID particleAndVolumeFractionFieldID =
      blocks->addStructuredBlockData< psm::cuda::ParticleAndVolumeFractionField_T >(
         &createField, "particle and volume fraction field CPU");
   BlockDataID gpuFieldID = cuda::addGPUFieldToStorage< psm::cuda::ParticleAndVolumeFractionField_T >(
      blocks, particleAndVolumeFractionFieldID, "particle and volume fraction field GPU");

   // calculate fraction
   psm::cuda::ParticleAndVolumeFractionMappingGPU particleMapping(
      blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(), gpuFieldID, 4);
   particleMapping();

   FractionFieldSum fractionFieldSum(blocks, particleAndVolumeFractionFieldID);
   auto selector = mesa_pd::kernel::SelectMaster();
   mesa_pd::kernel::SemiImplicitEuler particleIntegration(1.0);

   for (uint_t i = 0; i < setup.timesteps; ++i)
   {
      // copy data back to perform the check on CPU
      for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
      {
         cuda::fieldCpySweepFunction< psm::cuda::ParticleAndVolumeFractionField_T,
                                      psm::cuda::ParticleAndVolumeFractionFieldGPU_T >(particleAndVolumeFractionFieldID,
                                                                                       gpuFieldID, &(*blockIt));
      }

      // check that the sum over all fractions is roughly the volume of the sphere
      real_t sum = fractionFieldSum();
      WALBERLA_CHECK_LESS(std::fabs(4.0 / 3.0 * math::pi * sphereRadius * sphereRadius * sphereRadius * 2 - sum),
                          real_c(3));

      // update position
      ps->forEachParticle(false, selector, *accessor, particleIntegration, *accessor);
      syncCall();

      // map particles into field
      particleMapping();
   }

   return EXIT_SUCCESS;
}

} // namespace particle_volume_fraction_check

int main(int argc, char** argv) { particle_volume_fraction_check::main(argc, argv); }