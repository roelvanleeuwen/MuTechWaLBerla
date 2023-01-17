#ifndef WALBERLA_POISSONSOLVER_H
#define WALBERLA_POISSONSOLVER_H

#include "pde/all.h"

namespace walberla
{

// typedefs
using Stencil_T = stencil::D3Q7;
typedef GhostLayerField< real_t, 1 > ScalarField_T;

template< bool useJacobi, bool useDirichlet >
class PoissonSolver
{
 public:

   PoissonSolver(const BlockDataID& src, const BlockDataID& dst, const BlockDataID& rhs,
                 const std::shared_ptr< StructuredBlockForest >& blocks)
      : src_(src), dst_(dst), rhs_(rhs), blocks_(blocks) {

      // stencil weights

      weights_.resize(Stencil_T::Size, real_c(0));
      weights_[Stencil_T::idx[stencil::C]] = real_t( 2) / (blocks_->dx() * blocks_->dx()) +
                                             real_t( 2) / (blocks_->dy() * blocks_->dy()) +
                                             real_t( 2) / (blocks_->dz() * blocks_->dz());
      weights_[Stencil_T::idx[stencil::T]] = real_t(-1) / (blocks_->dz() * blocks_->dz());
      weights_[Stencil_T::idx[stencil::B]] = real_t(-1) / (blocks_->dz() * blocks_->dz());
      weights_[Stencil_T::idx[stencil::N]] = real_t(-1) / (blocks_->dy() * blocks_->dy());
      weights_[Stencil_T::idx[stencil::S]] = real_t(-1) / (blocks_->dy() * blocks_->dy());
      weights_[Stencil_T::idx[stencil::E]] = real_t(-1) / (blocks_->dx() * blocks_->dx());
      weights_[Stencil_T::idx[stencil::W]] = real_t(-1) / (blocks_->dx() * blocks_->dx());

      // communication

      commScheme_ = make_shared< blockforest::communication::UniformBufferedScheme< Stencil_T > >(blocks_);
      commScheme_->addPackInfo(make_shared< field::communication::PackInfo< ScalarField_T > >(src_));

      // boundary handling

      std::function< void () > boundaryHandling;
      if constexpr (useDirichlet) {
         // dirichlet BCs
         boundaryHandling = [&blocks, &src]() {
            for (auto block = blocks->begin(); block != blocks->end(); ++block) {
               ScalarField_T* field = block->getData< ScalarField_T >(src);
               CellInterval xyz     = field->xyzSizeWithGhostLayer();
               Cell offset = Cell();

               for (uint_t dim = 0; dim < Stencil_T::D; ++dim) {
                  switch (dim) {
                  case 0:
                     if (blocks->atDomainMinBorder(dim, *block)) {
                        xyz.xMax() = xyz.xMin();
                        offset     = Cell(-1, 0, 0);
                     } else if (blocks->atDomainMaxBorder(dim, *block)) {
                        xyz.xMin() = xyz.xMax();
                        offset     = Cell(1, 0, 0);
                     }
                     break;
                  case 1:
                     if (blocks->atDomainMinBorder(dim, *block)) {
                        xyz.yMax() = xyz.yMin();
                        offset     = Cell(0, -1, 0);
                     } else if (blocks->atDomainMaxBorder(dim, *block)) {
                        xyz.yMin() = xyz.yMax();
                        offset     = Cell(0, 1, 0);
                     }
                     break;
                  case 2:
                     if (blocks->atDomainMinBorder(dim, *block)) {
                        xyz.zMax() = xyz.zMin();
                        offset     = Cell(0, 0, -1);
                     } else if (blocks->atDomainMaxBorder(dim, *block)) {
                        xyz.zMin() = xyz.zMax();
                        offset     = Cell(0, 0, 1);
                     }
                     break;
                  }

                  // zero dirichlet BCs
                  for (auto cell = xyz.begin(); cell != xyz.end(); ++cell) {
                     field->get(*cell + offset) = -field->get(*cell);
                  }
               }
            }
         };
      } else {
         // neumann BCs
         boundaryHandling = pde::NeumannDomainBoundary< ScalarField_T > (*blocks, src_);
      }

      // iteration schemes

      auto residualNormThreshold = real_c(1e-4);
      auto residualCheckFrequency = uint_t(100);
      auto iterations = uint_t(1000);

      // res norm

      residualNorm_ = make_shared< pde::ResidualNorm< Stencil_T > >(blocks_->getBlockStorage(), src_, rhs_, weights_);

      // jacobi

      jacobiFixedSweep_ = make_shared < pde::JacobiFixedStencil< Stencil_T > >(src_, dst_, rhs_, weights_);

      jacobiIteration_ = std::make_unique< pde::JacobiIteration >(
         blocks_->getBlockStorage(), iterations,
         *commScheme_,
         *jacobiFixedSweep_,
         *residualNorm_, residualNormThreshold, residualCheckFrequency);

      jacobiIteration_->addBoundaryHandling(boundaryHandling);

      // SOR

      real_t omega = real_t(1.9);

      sorFixedSweep_ = make_shared< pde::SORFixedStencil< Stencil_T > >(blocks, src_, rhs_, weights_, omega);

      sorIteration_ = std::make_unique< pde::RBGSIteration >(
         blocks_->getBlockStorage(), iterations,
         *commScheme_,
         sorFixedSweep_->getRedSweep(), sorFixedSweep_->getBlackSweep(),
         *residualNorm_, residualNormThreshold, residualCheckFrequency);

      sorIteration_->addBoundaryHandling(boundaryHandling);
   }

   // get approximate solution of electric potential
   void operator()() {
      if constexpr (useJacobi) {
         (*jacobiIteration_)();
      } else {
         (*sorIteration_)();
      }
   }

 private:
   BlockDataID src_;
   BlockDataID dst_;
   BlockDataID rhs_;

   std::vector< real_t > weights_;
   std::shared_ptr< StructuredBlockForest > blocks_;
   std::shared_ptr< blockforest::communication::UniformBufferedScheme< Stencil_T > > commScheme_;

   std::shared_ptr < pde::ResidualNorm< Stencil_T > > residualNorm_;

   std::shared_ptr< pde::JacobiFixedStencil< Stencil_T > > jacobiFixedSweep_;
   std::unique_ptr< pde::JacobiIteration > jacobiIteration_;

   std::shared_ptr< pde::SORFixedStencil< Stencil_T  > > sorFixedSweep_;
   std::unique_ptr< pde::RBGSIteration > sorIteration_;
};

} // namespace walberla

#endif // WALBERLA_POISSONSOLVER_H
