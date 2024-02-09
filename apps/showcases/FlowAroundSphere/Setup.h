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
//! \file Setup.h
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================
#pragma once

#include "core/DataTypes.h"
#include "core/math/Vector3.h"


namespace walberla
{
struct Setup
{
   uint_t xBlocks;
   uint_t yBlocks;
   uint_t zBlocks;

   uint_t xCells;
   uint_t yCells;
   uint_t zCells;

   Vector3< uint_t > cellsPerBlock;
   uint_t numGhostLayers;

   real_t sphereXPosition;
   real_t sphereYPosition;
   real_t sphereZPosition;
   real_t sphereRadius;

   bool evaluateForceComponents;
   uint_t nbrOfEvaluationPointsForCoefficientExtremas;

   bool evaluatePressure;
   Vector3< real_t > pAlpha;
   Vector3< real_t > pOmega;

   bool evaluateStrouhal;
   Vector3< real_t > pStrouhal;

   real_t viscosity;
   real_t rho;
   real_t inflowVelocity;
   real_t dx;
   real_t dt;
};

}