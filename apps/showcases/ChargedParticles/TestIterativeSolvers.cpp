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

template < typename PdeField >
void applyDirichletFunction(const shared_ptr< StructuredBlockStorage > & blocks, math::AABB domainAABB, const stencil::Direction& direction,
                            IBlock* block, PdeField* p, const CellInterval& interval, const cell_idx_t cx, const cell_idx_t cy, const cell_idx_t cz) {

   WALBERLA_FOR_ALL_CELLS_IN_INTERVAL_XYZ(
      interval, auto boundaryCoord_x = 0.; auto boundaryCoord_y = 0.; auto boundaryCoord_z = 0.;

      const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x, y, z));
      auto cellCenter     = cellAABB.center();

      // snap cell position to actual domain position
      switch (direction) {
         case stencil::W:
            boundaryCoord_x = domainAABB.xMin();
            boundaryCoord_y = cellCenter[1];
            boundaryCoord_z = cellCenter[2];
            break;
         case stencil::E:
            boundaryCoord_x = domainAABB.xMax();
            boundaryCoord_y = cellCenter[1];
            boundaryCoord_z = cellCenter[2];
            break;
         case stencil::S:
            boundaryCoord_x = cellCenter[0];
            boundaryCoord_y = domainAABB.yMin();
            boundaryCoord_z = cellCenter[2];
            break;
         case stencil::N:
            boundaryCoord_x = cellCenter[0];
            boundaryCoord_y = domainAABB.yMax();
            boundaryCoord_z = cellCenter[2];
            break;
         case stencil::B:
            boundaryCoord_x = cellCenter[0];
            boundaryCoord_y = cellCenter[1];
            boundaryCoord_z = domainAABB.zMin();
            break;
         case stencil::T:
            boundaryCoord_x = cellCenter[0];
            boundaryCoord_y = cellCenter[1];
            boundaryCoord_z = domainAABB.zMax();
            break;
         default:
            WALBERLA_ABORT("Unknown direction");
      }

      auto funcVal = sin(M_PI * boundaryCoord_x) * sin(M_PI * boundaryCoord_y) * sinh(sqrt(2.0) * M_PI * boundaryCoord_z);
      p->get(x, y, z) = real_c(2) * funcVal - p->get(x + cx, y + cy, z + cz);
   )
}

void resetSolution(const shared_ptr< StructuredBlockStorage > & blocks, BlockDataID & solution, BlockDataID & solutionCpy) {
   for (auto block = blocks->begin(); block != blocks->end(); ++block) {
      ScalarField_T* solutionField    = block->getData< ScalarField_T >(solution);
      ScalarField_T* solutionFieldCpy = block->getData< ScalarField_T >(solutionCpy);

      // reset fields
      solutionField->set(real_c(0));
      solutionFieldCpy->set(real_c(0));
   }
}

// solve two different scenarios (dirichlet scenario and neumann scenario) with different analytical solutions and setups
template < bool useDirichlet >
void solve(const shared_ptr< StructuredBlockForest > & blocks,
           math::AABB domainAABB, BlockDataID & solution, BlockDataID & solutionCpy, BlockDataID & rhs) {

   // set boundary handling depending on scenario
   std::function< void () > boundaryHandling = {};

   if constexpr (useDirichlet) {
      // set dirichlet function per domain face
      auto dirichletFunction = DirichletFunctionDomainBoundary< ScalarField_T >(*blocks, solution);

#define GET_BOUNDARY_LAMBDA(dir) \
         [&blocks, &domainAABB](IBlock* block, ScalarField_T* p, const CellInterval& interval, const cell_idx_t cx, const cell_idx_t cy, const cell_idx_t cz) { \
            applyDirichletFunction< ScalarField_T >(blocks, domainAABB, dir, block, p, interval, cx, cy, cz); \
         }

      dirichletFunction.setFunction(stencil::W, GET_BOUNDARY_LAMBDA(stencil::W));
      dirichletFunction.setFunction(stencil::E, GET_BOUNDARY_LAMBDA(stencil::E));
      dirichletFunction.setFunction(stencil::S, GET_BOUNDARY_LAMBDA(stencil::S));
      dirichletFunction.setFunction(stencil::N, GET_BOUNDARY_LAMBDA(stencil::N));
      dirichletFunction.setFunction(stencil::B, GET_BOUNDARY_LAMBDA(stencil::B));
      dirichletFunction.setFunction(stencil::T, GET_BOUNDARY_LAMBDA(stencil::T));

      boundaryHandling = dirichletFunction;
   }

   // solvers: Jacobi and SOR

   auto poissonSolverJacobi = PoissonSolver< true, useDirichlet > (solution, solutionCpy, rhs, blocks, 50000, real_t(1e-4), 1000, boundaryHandling);
   auto poissonSolverSOR = PoissonSolver< false, useDirichlet > (solution, solutionCpy, rhs, blocks, 50000, real_t(1e-4), 1000, boundaryHandling);

   // calc error depending on scenario

   auto computeMaxError = [&blocks, &solution]() {
      real_t error = real_c(0);

      for (auto block = blocks->begin(); block != blocks->end(); ++block) {
         ScalarField_T* solutionField = block->getData< ScalarField_T >(solution);

         WALBERLA_FOR_ALL_CELLS_XYZ_OMP(solutionField, omp parallel for schedule(static) reduction(max: error),
                                    const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x,y,z));
                                    auto cellCenter = cellAABB.center();

                                    // analytical solution of problem with neumann boundaries
                                    real_t analyticalSolNeumann = cos ( real_c(2) * M_PI * cellCenter[0] ) *
                                                           cos ( real_c(2) * M_PI * cellCenter[1] ) *
                                                           cos ( real_c(2) * M_PI * cellCenter[2] );

                                    // analytical solution of problem with dirichlet boundaries
                                    real_t analyticalSolDirichlet = sin( M_PI * cellCenter[0] ) *
                                                               sin ( M_PI * cellCenter[1] ) *
                                                               sinh ( sqrt(2.0) * M_PI * cellCenter[2] );

                                    real_t analyticalSol = (useDirichlet) ? analyticalSolDirichlet : analyticalSolNeumann;
                                    real_t currErr = fabs(solutionField->get(x, y, z) - analyticalSol);
                                    error = std::max (error, currErr);
         )
      }
      mpi::allReduceInplace( error, mpi::MAX );

      return error;
   };

   // init rhs depending on scenario

   for (auto block = blocks->begin(); block != blocks->end(); ++block) {
      ScalarField_T* rhsField = block->getData< ScalarField_T >(rhs);

      WALBERLA_FOR_ALL_CELLS_XYZ(
         rhsField,

         if constexpr (useDirichlet) {
            rhsField->get(x, y, z) = real_c(0);
         } else {
            const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x, y, z));
            auto cellCenter = cellAABB.center();

            rhsField->get(x, y, z) = real_c(12) * M_PI * M_PI * cos(real_c(2) * M_PI * cellCenter[0]) *
                                     cos(real_c(2) * M_PI * cellCenter[1]) * cos(real_c(2) * M_PI * cellCenter[2]);
         }
      )
   }

   WALBERLA_LOG_INFO_ON_ROOT("Initial error is: " << computeMaxError());

   // solve with jacobi
   poissonSolverJacobi();
   WALBERLA_LOG_INFO_ON_ROOT("Error after Jacobi solver is: " << computeMaxError());

   // solve with SOR
   resetSolution(blocks, solution, solutionCpy); // reset solutions and solve anew

   poissonSolverSOR();
   WALBERLA_LOG_INFO_ON_ROOT("Error after SOR solver is: " << computeMaxError());
}

int main(int argc, char** argv)
{
   Environment env(argc, argv);

   WALBERLA_LOG_INFO_ON_ROOT("waLBerla revision: " << std::string(WALBERLA_GIT_SHA1).substr(0, 8));

   logging::Logging::instance()->setLogLevel(logging::Logging::LogLevel::DETAIL);
   logging::Logging::instance()->includeLoggingToFile( "log" );

   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   auto domainAABB = math::AABB(0, 0, 0, 1, 1, 1);

   shared_ptr< StructuredBlockForest > blocks = blockforest::createUniformBlockGrid(
      domainAABB,
      uint_c( 1), uint_c( 1), uint_c( 1), // number of blocks in x,y,z direction
      uint_c( 125), uint_c( 50), uint_c( 250), // how many cells per block (x,y,z)
      true,                               // max blocks per process
      false, false, false,                // periodicity
      false);

   BlockDataID rhs =
      field::addToStorage< ScalarField_T >(blocks, "rhs", 0, field::fzyx, 1);
   BlockDataID solution =
      field::addToStorage< ScalarField_T >(blocks, "solution", 0, field::fzyx, 1);
   BlockDataID solutionCpy =
      field::addCloneToStorage< ScalarField_T >(blocks, solution, "solution (copy)");

   // first solve neumann problem...
   solve< false > (blocks, domainAABB, solution, solutionCpy, rhs);

   // ... then solve dirichlet problem
   resetSolution(blocks, solution, solutionCpy); // reset solutions and solve anew
   solve< true > (blocks, domainAABB, solution, solutionCpy, rhs);

   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv) { walberla::main(argc, argv); }

