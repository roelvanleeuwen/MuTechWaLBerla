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
//! \file   ReadParticleBoundaiesFromFile.h
//! \author Philipp Suffa philipp.suffa@fau.de
//
//======================================================================================================================

#pragma once
#include "blockforest/StructuredBlockForest.h"
#include "core/logging/Logging.h"
#include "field/FlagField.h"

#include <algorithm>
#include <core/mpi/Broadcast.h>
#include <core/mpi/MPITextFile.h>
#include <core/mpi/Reduce.h>
#include <functional>
#include <iterator>

namespace walberla
{

void initSpheresFromFile(const std::string& filename, weak_ptr<StructuredBlockForest> blockForest, const BlockDataID flagFieldID, const field::FlagUID boundaryFlagUID, const real_t dx)
{
   using namespace walberla;
   auto forest = blockForest.lock();

   std::string textFile;

   WALBERLA_ROOT_SECTION()
   {
      std::ifstream t(filename.c_str());
      if (!t) { WALBERLA_ABORT("Invalid input file " << filename << "\n"); }
      std::stringstream buffer;
      buffer << t.rdbuf();
      textFile = buffer.str();
   }
   walberla::mpi::broadcastObject(textFile);

   std::istringstream fileIss(textFile);
   std::string line;

   // first line contains generation domain sizes
   std::getline(fileIss, line);
   Vector3< real_t > generationDomainSize_SI(0_r);
   std::istringstream firstLine(line);
   firstLine >> generationDomainSize_SI[0] >> generationDomainSize_SI[1] >> generationDomainSize_SI[2];
   WALBERLA_LOG_DEVEL_VAR_ON_ROOT(generationDomainSize_SI)

   std::vector<std::pair<Cell, cell_idx_t>> particleInfo;
   while (std::getline(fileIss, line))
   {
      std::istringstream iss(line);
      uint_t particleUID;
      Vector3< real_t > particlePos;
      real_t particleRadius;
      iss >> particleUID >> particlePos[0] >> particlePos[1] >> particlePos[2] >> particleRadius;
      Cell midPoint(cell_idx_c(particlePos[0] / dx), cell_idx_c(particlePos[1] / dx), cell_idx_c(particlePos[2] / dx));
      cell_idx_t radiusInCells = cell_idx_c(particleRadius / dx);
      std::pair particle(midPoint, radiusInCells);
      particleInfo.push_back(particle);
   }

   for (auto &block : *forest) {
      CellInterval BlockBB = forest->getBlockCellBB( block );
      BlockBB.expand(1);
      auto flagField    = block.template getData< FlagField_T >(flagFieldID);
      auto boundaryFlag = flagField->getFlag(boundaryFlagUID);

      for (auto particle : particleInfo)
      {
         CellInterval SphereBB(particle.first.x() - particle.second, particle.first.y() - particle.second,
                               particle.first.z() - particle.second, particle.first.x() + particle.second,
                               particle.first.y() + particle.second, particle.first.z() + particle.second);
         if (BlockBB.overlaps(SphereBB)) {
            SphereBB.intersect(BlockBB);
            Cell localCell;
            Cell localPoint;
            for(auto it = SphereBB.begin(); it != SphereBB.end(); ++it) {
               forest->transformGlobalToBlockLocalCell(localCell, block, Cell(it->x(), it->y(), it->z()));
               forest->transformGlobalToBlockLocalCell(localPoint, block, particle.first);
               real_t Ri = (localCell[0] - localPoint.x()) * (localCell[0] - localPoint.x()) +
                           (localCell[1] - localPoint.y()) * (localCell[1] - localPoint.y()) +
                           (localCell[2] - localPoint.z()) * (localCell[2] - localPoint.z());

               if(Ri < particle.second * particle.second)
               {
                  addFlag(flagField->get(localCell), boundaryFlag);
               }
            }
         }
      }
   }
}

} // namespace walberla
