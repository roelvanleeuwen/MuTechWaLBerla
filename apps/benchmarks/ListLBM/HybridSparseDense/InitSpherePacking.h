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
//! \file InitializerFunctions.h
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#include "core/Environment.h"
#include "core/logging/Logging.h"
#include "core/math/Constants.h"

#include "field/FlagField.h"

#include <cmath>
#pragma once

namespace walberla
{

void InitSpherePacking(const shared_ptr< StructuredBlockStorage >& blocks, BlockDataID flagFieldID,
                       const field::FlagUID boundaryFlagUID, const real_t Radius, const real_t Shift, const Vector3<real_t> fillIn);

} // namespace walberla
