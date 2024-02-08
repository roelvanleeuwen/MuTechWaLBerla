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

void initSpheresFromFile(const std::string& filename, weak_ptr<StructuredBlockForest> blockForest, const BlockDataID flagFieldID, const field::FlagUID boundaryFlagUID)
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

   std::vector<std::pair<Vector3< real_t >, real_t>> particleInfo;
   while (std::getline(fileIss, line))
   {
      std::istringstream iss(line);
      uint_t particleUID;
      Vector3< real_t > particlePos;
      real_t particleRadius;
      iss >> particleUID >> particlePos[0] >> particlePos[1] >> particlePos[2] >> particleRadius;
      std::pair particle(particlePos, particleRadius);
      particleInfo.push_back(particle);
   }

   auto domainAABB = forest->getDomain();
   auto blockDx = forest->dx(0);

   for (auto &block : *forest) {
      CellInterval BlockBB = forest->getBlockCellBB( block );
      BlockBB.expand(1);
      auto flagField    = block.template getData< FlagField_T >(flagFieldID);
      flagField->registerFlag(boundaryFlagUID);
      auto boundaryFlag = flagField->getFlag(boundaryFlagUID);
      for (auto particle : particleInfo) {
         auto particleBB = CellInterval(cell_idx_c((particle.first[0] - domainAABB.xMin())/blockDx - particle.second/blockDx),
                                        cell_idx_c((particle.first[1] - domainAABB.xMin())/blockDx - particle.second/blockDx),
                                        cell_idx_c((particle.first[2] - domainAABB.xMin())/blockDx - particle.second/blockDx),
                                        cell_idx_c((particle.first[0] - domainAABB.xMin())/blockDx + particle.second/blockDx + 1),
                                        cell_idx_c((particle.first[1] - domainAABB.xMin())/blockDx + particle.second/blockDx + 1),
                                        cell_idx_c((particle.first[2] - domainAABB.xMin())/blockDx + particle.second/blockDx + 1));

         if (BlockBB.overlaps(particleBB)) {
            particleBB.intersect(BlockBB);
            for(auto it = particleBB.begin(); it != particleBB.end(); ++it)
            {
               auto globalCell = Cell(it->x(), it->y(), it->z());
               Cell localCell;
               auto cellAABB = forest->getCellAABB(globalCell, 0);
               Vector3< real_t > cellCenter =
                  Vector3< real_t >(cellAABB.xMin() + 0.5 * blockDx, cellAABB.yMin() + 0.5 * blockDx, cellAABB.zMin() + 0.5 * blockDx);

               //forest->transformGlobalToBlockLocalCell(localCell, block, globalCell);
               //addFlag(flagField->get(localCell), boundaryFlag);

               //TODO sth is wrong here
               if ((cellCenter[0] - particle.first[0]) * (cellCenter[0] - particle.first[0]) +
                      (cellCenter[1] - particle.first[1]) * (cellCenter[1] - particle.first[1]) +
                      (cellCenter[2] - particle.first[2]) * (cellCenter[2] - particle.first[2]) <
                   particle.second * particle.second)
               {

                  forest->transformGlobalToBlockLocalCell(localCell, block, globalCell);
                  addFlag(flagField->get(localCell), boundaryFlag);
               }
            }
         }
      }
   }
}

} // namespace walberla
