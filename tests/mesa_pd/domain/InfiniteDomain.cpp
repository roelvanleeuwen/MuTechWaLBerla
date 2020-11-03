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
//! \file   InfiniteDomain.cpp
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

#include <mesa_pd/domain/InfiniteDomain.h>

#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <iostream>

namespace walberla {

using namespace walberla::mesa_pd;

void main( int argc, char ** argv )
{
   Environment env(argc, argv);
   WALBERLA_UNUSED(env);
   mpi::MPIManager::instance()->useWorldComm();
   auto rank = mpi::MPIManager::instance()->rank();

   domain::InfiniteDomain domain;
   WALBERLA_CHECK(domain.isContainedInProcessSubdomain(1, Vec3()));
   WALBERLA_CHECK_EQUAL(domain.findContainingProcessRank(Vec3()), rank);
   auto pt = Vec3(1.23_r,2.34_r,3.45_r);
   domain.periodicallyMapToDomain(pt);
   WALBERLA_CHECK_IDENTICAL(pt, Vec3(1.23_r,2.34_r,3.45_r));
   WALBERLA_CHECK_EQUAL(domain.getNeighborProcesses().size(), 0);
   WALBERLA_CHECK(domain.intersectsWithProcessSubdomain(0, Vec3(), 1_r));
   WALBERLA_CHECK(!domain.intersectsWithProcessSubdomain(1, Vec3(), 1_r));
   domain.correctParticlePosition(pt);
   WALBERLA_CHECK_IDENTICAL(pt, Vec3(1.23_r,2.34_r,3.45_r));
}

}

int main( int argc, char ** argv )
{
   walberla::main(argc, argv);
   return EXIT_SUCCESS;
}
