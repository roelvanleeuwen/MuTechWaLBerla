#pragma once

#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "domain_decomposition/StructuredBlockStorage.h"

#include "field/EvaluationFilter.h"
#include "field/GhostLayerField.h"
#include "field/iterators/IteratorMacros.h"

#include "lbm/field/PdfField.h"

#include "CollisionModel.h"

namespace walberla
{

template< typename LatticeModel_T >
// Class to perform the Omega sweep operation
class OmegaSweep_new
{
 public:
   using PdfField_T    = lbm::PdfField< LatticeModel_T >;
   using ScalarField_T = GhostLayerField< real_t, 1 >;

   // Constructor to initialize the OmegaSweep class with the given parameters
   OmegaSweep_new(weak_ptr< StructuredBlockStorage > blocks, BlockDataID pdfFieldId, BlockDataID omegaFieldId,
                  AABB& domain, Setup setup, Units units)
      : blocks_(blocks), pdfFieldId_(pdfFieldId), omegaFieldId_(omegaFieldId), domain_(domain), setup_(setup), units_(units)
   {}

   void operator()(IBlock* block);

 private:
   weak_ptr< StructuredBlockStorage > blocks_;
   const BlockDataID pdfFieldId_; // PDF field ID
   BlockDataID omegaFieldId_;
   AABB domain_; // Domain
   Setup setup_; // Setup parameters
   Units units_; // Units
};

template< typename LatticeModel_T >
void OmegaSweep_new< LatticeModel_T >::operator()(IBlock* block)
{
   const PdfField_T* pdfField = block->getData< PdfField_T >(pdfFieldId_);
   ScalarField_T* omegaField  = block->getData< ScalarField_T >(omegaFieldId_);

   WALBERLA_ASSERT_NOT_NULLPTR(pdfField);
   WALBERLA_ASSERT_NOT_NULLPTR(omegaField);

   WALBERLA_ASSERT_EQUAL(pdfField->xyzSizeWithGhostLayer(), omegaField->xyzSizeWithGhostLayer());

   WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(
      omegaField,

      // Calculate the inner and outer radius of the sponge zone
      real_t sponge_rmin = setup_.spongeInnerThicknessFactor * std::max(domain_.yMax(), domain_.xMax());
      real_t sponge_rmax = setup_.spongeOuterThicknessFactor * std::max(domain_.yMax(), domain_.xMax());

      // Calculate the distance from the center of the domain to the cell
      real_t dist = std::sqrt(std::pow(x - domain_.center()[0], 2.0) + std::pow(y - domain_.center()[1], 2.0));

      // Calculate the additional kinematic viscosity to be added to the lattice kinematic viscosity
      real_t nuTAdd =
         setup_.sponge_nuT_min + (setup_.sponge_nuT_max - setup_.sponge_nuT_min) * phi(dist, sponge_rmin, sponge_rmax);
      real_t nuAddFactor = nuTAdd * setup_.temperatureSI;

      // Calculate the lattice kinematic viscosity
      real_t viscosity_old = pdfField->latticeModel().collisionModel().viscosity(x, y, z);
      real_t nuLU          = units_.kinViscosityLU * nuAddFactor + viscosity_old;

      // Calculate the relaxation rate omega based on the lattice kinematic viscosity
      real_t omega = 1.0 / (std::pow(units_.pseudoSpeedOfSoundLU, 2.0) * nuLU + 0.5);

      // Ensure the relaxation rate omega is within the physical range
      WALBERLA_ASSERT(omega > 0.0 && omega < 2.0);
      // Set the relaxation rate omega in the collision model of the lattice model
      // pdfField->latticeModel().collisionModel().reset(x, y, z, omega);
      omegaField->get(x, y, z) = omega);
}

} // namespace walberla
