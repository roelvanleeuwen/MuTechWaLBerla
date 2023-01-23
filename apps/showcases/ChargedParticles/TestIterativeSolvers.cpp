#include "blockforest/Initialization.h"
#include "blockforest/communication/UniformBufferedScheme.h"

#include "boundary/all.h"

#include "core/Environment.h"
#include "core/grid_generator/SCIterator.h"
#include "core/logging/all.h"
#include "core/waLBerlaBuildInfo.h"

#include "field/AddToStorage.h"
#include "field/vtk/all.h"

#include "PoissonSolver.h"

namespace walberla {

using ScalarField_T = GhostLayerField< real_t, 1 >;

int main(int argc, char** argv)
{
   Environment env(argc, argv);

   WALBERLA_LOG_INFO_ON_ROOT("waLBerla revision: " << std::string(WALBERLA_GIT_SHA1).substr(0, 8));

   logging::Logging::instance()->setLogLevel(logging::Logging::LogLevel::DETAIL);
   logging::Logging::instance()->includeLoggingToFile( "log" );

   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   shared_ptr< StructuredBlockForest > blocks = blockforest::createUniformBlockGrid(
      math::AABB(0,0,0,1,1,1),
      uint_c( 1), uint_c( 1), uint_c( 1), // number of blocks in x,y,z direction
      uint_c( 125), uint_c( 50), uint_c( 250), // how many cells per block (x,y,z)
      true,                               // max blocks per process
      false, false, false,                // full periodicity
      false);

   BlockDataID rhs =
      field::addToStorage< ScalarField_T >(blocks, "rhs", 0, field::fzyx, 1);
   BlockDataID solution =
      field::addToStorage< ScalarField_T >(blocks, "solution", 0, field::fzyx, 1);
   BlockDataID solutionCpy =
      field::addCloneToStorage< ScalarField_T >(blocks, solution, "solution (copy)");

   // solver

   auto poissonSolverJacobi = PoissonSolver< true, false > (solution, solutionCpy, rhs, blocks, 20000, real_t(1e-4), 1000);
   auto poissonSolverSOR = PoissonSolver< false, false > (solution, solutionCpy, rhs, blocks, 20000, real_t(1e-4), 1000);

   // calc error

   auto computeMaxError = [&blocks, &solution]() {
      real_t error = real_c(0);

      for (auto block = blocks->begin(); block != blocks->end(); ++block) {
         ScalarField_T* solutionField = block->getData< ScalarField_T >(solution);

         WALBERLA_FOR_ALL_CELLS_XYZ_OMP(solutionField, omp parallel for schedule(static) reduction(max: error),
                                    const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x,y,z));
                                    auto cellCenter = cellAABB.center();
                                    real_t analyticalSol = cos ( real_c(2) * M_PI * cellCenter[0] ) *
                                                           cos ( real_c(2) * M_PI * cellCenter[1] ) *
                                                           cos ( real_c(2) * M_PI * cellCenter[2] );
                                    real_t currErr = fabs(solutionField->get(x, y, z) - analyticalSol);

                                    error = std::max (error, currErr);
         )
      }
      mpi::allReduceInplace( error, mpi::MAX );

      return error;
   };

   // init rhs

   for (auto block = blocks->begin(); block != blocks->end(); ++block) {
      ScalarField_T* rhsField = block->getData< ScalarField_T >(rhs);

         WALBERLA_FOR_ALL_CELLS_XYZ(rhsField,
                                    const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x,y,z));
                                    auto cellCenter = cellAABB.center();

                                 rhsField->get(x, y, z) = real_c(12) * M_PI * M_PI * cos ( real_c(2) * M_PI * cellCenter[0] ) *
                                                          cos ( real_c(2) * M_PI * cellCenter[1] ) * cos ( real_c(2) * M_PI * cellCenter[2] );
         )
   }

   WALBERLA_LOG_INFO_ON_ROOT("Initial error is: " << computeMaxError());

   // solve with jacobi
   poissonSolverJacobi();
   WALBERLA_LOG_INFO_ON_ROOT("Error after Jacobi solver is: " << computeMaxError());

   // solve with SOR
   for (auto block = blocks->begin(); block != blocks->end(); ++block) {
      ScalarField_T* solutionField = block->getData< ScalarField_T >(solution);
      ScalarField_T* solutionFieldCpy = block->getData< ScalarField_T >(solutionCpy);

      // reset fields
      solutionField->set(real_c(0));
      solutionFieldCpy->set(real_c(0));
   }

   poissonSolverSOR();
   WALBERLA_LOG_INFO_ON_ROOT("Error after SOR solver is: " << computeMaxError());

   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv) { walberla::main(argc, argv); }

