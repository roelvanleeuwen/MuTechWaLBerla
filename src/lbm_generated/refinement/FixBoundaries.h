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
//! \file FixBoundaries.h
//! \author Frederik Hennig <frederik.hennig@fau.de>
//
//======================================================================================================================

#pragma once

#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/StructuredBlockStorage.h"

#include "field/FlagField.h"

#include "stencil/D3Q6.h"

namespace walberla
{
namespace lbm_generated
{

/**
 * Fixes the boundary flag field for a refined-grid simulation using the lbm_generated backend.
 *
 * Due to the occasional need for boundary handling on the second ghost layer of the PDF field
 * whenever boundaries intersect with a refinement interface, the flag field requires three
 * ghost layers. The third ghost layer acts only as a buffer to avoid out-of-bounds accesses
 * during initialization, and must never store any flags. To make sure of this, this function
 * removes all flags from the third ghost layer.
 *
 * @tparam FlagField_T Type of the flag field
 * @param blocks The simulation's block storage
 * @param flagFieldId Block data ID of the flag field
 */
template< typename FlagField_T >
void fixBoundaryFlagFieldForRefinedGrid(StructuredBlockStorage& blocks, const BlockDataID& flagFieldId)
{
   using namespace stencil;

   for (auto& iblock : blocks)
   {
      FlagField_T* flagField = iblock.getData< FlagField_T >(flagFieldId);
      WALBERLA_CHECK_EQUAL(flagField->nrOfGhostLayers(), 3,
                           "For correct initialization of boundaries on refined grids with the lbm_generated backend, "
                           "the flag field needs three ghost layers.");

      CellInterval domainWithGls = flagField->xyzSizeWithGhostLayer();

      for(auto dIt = D3Q6::begin(); dIt != D3Q6::end(); ++dIt){
         const Direction dir = *dIt;
         Cell lower(domainWithGls.min());
         Cell upper(domainWithGls.max());

         for(uint_t i = 0; i < 3; ++i){
            if(c[i][dir] == 1){
               lower[i] = domainWithGls.max()[i];
            }
            if(c[i][dir] == -1){
               upper[i] = domainWithGls.min()[i];
            }
         }

         const CellInterval ci(lower, upper);
         for(auto cell: ci){
            flagField->get(cell) = 0;
         }
      }
   }
}

} // namespace lbm_generated
} // namespace walberla
