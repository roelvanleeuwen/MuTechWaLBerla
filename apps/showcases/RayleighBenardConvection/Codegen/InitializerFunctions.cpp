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
//! \file InitializerFunctions.cpp
//! \author Jonas Plewinski <jonas.plewinski@fau.de>
//
//======================================================================================================================
#include "core/logging/Initialization.h"
#include "core/math/Constants.h"
#include "core/math/Random.h"

#include "field/FlagField.h"
#include "field/communication/PackInfo.h"

namespace walberla
{
using FlagField_T     = FlagField< uint8_t >;
using TemperatureField_T = walberla::field::GhostLayerField<double, 1>;

// function describing the initialization profile (in global coordinates)
real_t initializationProfile(real_t x, real_t amplitude, real_t offset, real_t wavelength)
{
   return amplitude * std::cos(x / wavelength * real_c(2) * math::pi + math::pi) + offset;
}

// initialize sine profile such that there is exactly one period in the domain, i.e., with wavelength=domainSize[0];
// every length is normalized with domainSize[0]
void initTemperatureField(const shared_ptr< StructuredBlockStorage >& blocks, BlockDataID temperature_field_ID,
                          real_t amplitude, Vector3< uint_t > domainSize, real_t temperatureRange)
{
   for (auto& block : *blocks)
   {
      auto temperatureField    = block.getData< TemperatureField_T >(temperature_field_ID);

      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(temperatureField, {
         // cell in block-local coordinates
         Cell globalCell;
         blocks->transformBlockLocalToGlobalCell(globalCell, block, Cell(x, y, z));

         uint_t sampleSize = 100;
         real_t stepSize = real_c(1.) / real_c(sampleSize);
         real_t offset = real_c(domainSize[1] / 2);
         real_t wavelength = real_c(domainSize[0]);
         for (uint_t xSample = uint_c(0); xSample <= sampleSize; ++xSample)
         {
            // value of the sine-function
            const real_t functionValue = initializationProfile(real_c(globalCell[0]) + real_c(xSample) * stepSize, amplitude, offset, wavelength);
            for (uint_t ySample = uint_c(0); ySample <= sampleSize; ++ySample)
            {
               const real_t yPoint = real_c(globalCell[1]) + real_c(ySample) * stepSize;
               temperatureField->get(x, y, z) = (yPoint < functionValue) ?  temperatureRange/50 : -temperatureRange/50;
            }
         }
      }) // WALBERLA_FOR_ALL_CELLS
   }
}

} // namespace walberla
