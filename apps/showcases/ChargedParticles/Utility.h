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
//! \file   Utility.h
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "lbm_mesapd_coupling/DataTypes.h"

namespace walberla
{
namespace charged_particles
{

template< typename BlockStorage_T >
void computeChargeDensity(const shared_ptr< BlockStorage_T >& blocks,
                          const BlockDataID& particleAndVolumeFractionFieldID, const BlockDataID& chargeDensityFieldID)
{
   // TODO: compute physically correct charge density here using the particle charges, have a look at src/lbm_mesapd_coupling/partially_saturated_cells_method/ParticleAndVolumeFractionMapping.h for how to iterate over particles and cells
   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      lbm_mesapd_coupling::psm::ParticleAndVolumeFractionField_T* particleAndVolumeFractionField =
         blockIt->template getData< lbm_mesapd_coupling::psm::ParticleAndVolumeFractionField_T >(
            particleAndVolumeFractionFieldID);
      GhostLayerField< real_t, 1 >* chargeDensityField =
         blockIt->template getData< GhostLayerField< real_t, 1 > >(chargeDensityFieldID);

      WALBERLA_FOR_ALL_CELLS_XYZ(particleAndVolumeFractionField, chargeDensityField->get(x, y, z) = 0.0;
                                 for (auto& e
                                      : particleAndVolumeFractionField->get(x, y, z))
                                    chargeDensityField->get(x, y, z) -= e.second;) // rhs depends on the negative charge density
   }
}

} // namespace charged_particles
} // namespace walberla
