//
// Created by RichardAngersbach on 16.01.2023.
//

#ifndef WALBERLA_CHARGEFORCE_H
#define WALBERLA_CHARGEFORCE_H

namespace walberla
{
using Stencil_T = stencil::D3Q7;
typedef GhostLayerField< real_t, 1 > ScalarField_T;
typedef GhostLayerField< real_t, 3 > VectorField_T;

class ChargeForceUpdate
{
 public:
   ChargeForceUpdate(const BlockDataID& potential, const BlockDataID& chargeForce, const std::shared_ptr< StructuredBlockForest >& blocks)
      : potential_(potential), chargeForce_(chargeForce), blocks_(blocks) {}

   void operator()() {
      // get charge force with FD gradient from electric potential
      for (auto block = blocks_->begin(); block!=blocks_->end(); ++block) {
         VectorField_T* chargeForce = block->getData< VectorField_T >(chargeForce_);
         ScalarField_T* potential = block->getData< ScalarField_T >(potential_);

         WALBERLA_FOR_ALL_CELLS_XYZ(potential,
            chargeForce->get(x, y, z, 0) = (real_c(1) / (real_c(2) * (blocks_->dx()))) * (potential->get(x + 1, y, z) - potential->get(x - 1, y, z));
            chargeForce->get(x, y, z, 1) = (real_c(1) / (real_c(2) * (blocks_->dy()))) * (potential->get(x, y + 1, z) - potential->get(x, y - 1, z));
            chargeForce->get(x, y, z, 2) = (real_c(1) / (real_c(2) * (blocks_->dz()))) * (potential->get(x, y, z + 1) - potential->get(x, y, z - 1));
         )
      }
   }

 private:
   BlockDataID potential_;
   BlockDataID chargeForce_;

   std::shared_ptr< StructuredBlockForest > blocks_;
};

} // namespace walberla

#endif // WALBERLA_CHARGEFORCE_H
