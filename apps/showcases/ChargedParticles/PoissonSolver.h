#ifndef WALBERLA_POISSONSOLVER_H
#define WALBERLA_POISSONSOLVER_H

#include "pde/all.h"

namespace walberla
{

// typedefs
using Stencil_T = stencil::D3Q7;
typedef GhostLayerField< real_t, 1 > ScalarField_T;

template< bool useJacobi = true >
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

      commScheme_ = std::make_unique< blockforest::communication::UniformBufferedScheme< Stencil_T > >(blocks_);
      commScheme_->addPackInfo(make_shared< field::communication::PackInfo< ScalarField_T > >(src_));

      // boundary handling

      auto neumannBCs = pde::NeumannDomainBoundary< ScalarField_T > (*blocks, src_); // TODO: dirichlet?

      // iteration schemes

      auto residualNormThreshold = real_c(1e-6);
      auto residualCheckFrequency = uint_t(100);
      auto iterations = uint_t(1000);

      // jacobi

      jacobiIteration_ = std::make_unique< pde::JacobiIteration >(
         blocks_->getBlockStorage(), iterations, *commScheme_,
         pde::JacobiFixedStencil< Stencil_T >(src_, dst_, rhs_, weights_),
         pde::ResidualNorm< Stencil_T >(blocks_->getBlockStorage(), src_, rhs_, weights_), residualNormThreshold, residualCheckFrequency);

      jacobiIteration_->addBoundaryHandling(neumannBCs);

      // SOR

      real_t omega = real_t(1.9);

      auto SORFixedSweep = pde::SORFixedStencil< Stencil_T >(blocks, src_, rhs_, weights_, omega);

      sorIteration_ = std::make_unique< pde::RBGSIteration >(
         blocks_->getBlockStorage(), iterations, *commScheme_,
         SORFixedSweep.getRedSweep(), SORFixedSweep.getBlackSweep(),
         pde::ResidualNorm< Stencil_T >(blocks_->getBlockStorage(), src_, rhs_, weights_), residualNormThreshold, residualCheckFrequency);

      sorIteration_->addBoundaryHandling(neumannBCs);
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
   std::unique_ptr< blockforest::communication::UniformBufferedScheme< Stencil_T > > commScheme_;

   std::unique_ptr< pde::JacobiIteration > jacobiIteration_;
   std::unique_ptr< pde::RBGSIteration > sorIteration_;
};

} // namespace walberla

#endif // WALBERLA_POISSONSOLVER_H
