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
//! \file   BoundaryCondition.h
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "mesa_pd/data/DataTypes.h"

namespace walberla
{
namespace piping
{

void assembleBoundaryBlock(const Vector3< uint_t >& domainSize, const mesa_pd::Vec3& boxPosition,
                           const mesa_pd::Vec3& boxEdgeLength, const bool periodicInY)
{
   // TODO: improve readability of boundary conditions
   std::string boundariesBlockString =
      " Boundaries"
      "{"
      "Border { direction W;    walldistance -1;  flag NoSlip; }"
      "Border { direction E;    walldistance -1;  flag NoSlip; }"
      "Border { direction B;    walldistance -1;  flag NoSlip; }"
      "CellInterval { min < 0,-1," +
      std::to_string(domainSize[2]) + ">; max < " + std::to_string(uint_t(boxPosition[0] - boxEdgeLength[0] / 2 - 1)) +
      "," + std::to_string(domainSize[1] + 1) + "," + std::to_string(domainSize[2] + 1) +
      ">; flag Density0; }"
      "CellInterval { min < " +
      std::to_string(uint_t(boxPosition[0] - boxEdgeLength[0] / 2)) + ",-1," + std::to_string(domainSize[2]) +
      ">; max < " + std::to_string(uint_t(boxPosition[0] + boxEdgeLength[0] / 2 - 1)) + "," +
      std::to_string(domainSize[1] + 1) + "," + std::to_string(domainSize[2] + 1) +
      ">; flag NoSlip; }"
      "CellInterval { min < " +
      std::to_string(uint_t(boxPosition[0] + boxEdgeLength[0] / 2)) + ",-1," + std::to_string(domainSize[2]) +
      ">; max < " + std::to_string(domainSize[0]) + "," + std::to_string(domainSize[1] + 1) + "," +
      std::to_string(domainSize[2] + 1) +
      ">; flag Density1; }"
      "Body { shape box; min <" +
      std::to_string(boxPosition[0] - boxEdgeLength[0] / 2) + "," +
      std::to_string(boxPosition[1] - boxEdgeLength[1] / 2) + "," +
      std::to_string(boxPosition[2] - boxEdgeLength[2] / 2) + ">; max <" +
      std::to_string(boxPosition[0] + boxEdgeLength[0] / 2) + "," +
      std::to_string(boxPosition[1] + boxEdgeLength[1] / 2) + "," +
      std::to_string(boxPosition[2] + boxEdgeLength[2] / 2) + ">; flag NoSlip; }";

   if (!periodicInY)
   {
      boundariesBlockString += "Border { direction S;    walldistance -1;  flag NoSlip; }"
                               "Border { direction N;    walldistance -1;  flag NoSlip; }";
   }

   boundariesBlockString += "}";

   WALBERLA_ROOT_SECTION()
   {
      std::ofstream boundariesFile("boundaries.prm");
      boundariesFile << boundariesBlockString;
      boundariesFile.close();
   }
   WALBERLA_MPI_BARRIER()
}

} // namespace piping
} // namespace walberla
