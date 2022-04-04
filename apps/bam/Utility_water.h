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
//! \author Mohammad
//
//======================================================================================================================

#pragma once

#include "core/mpi/MPIManager.h"
#include "core/mpi/Broadcast.h"
#include "mesa_pd/data/ParticleStorage.h"

#include <mesa_pd/common/ParticleFunctions.h>
#include <mesa_pd/data/ParticleAccessorWithShape.h>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/ShapeStorage.h>


namespace walberla {
    namespace particle_erosion_utility {

        using namespace mesa_pd;

        class ExcludeGlobalGlobal
        {
        public:
            template <typename Accessor>
            bool operator()(const size_t idx, const size_t jdx, Accessor& ac) const
            {
                using namespace walberla::mesa_pd::data::particle_flags;
                if (isSet(ac.getFlags(idx), GLOBAL) && isSet(ac.getFlags(jdx), GLOBAL)) return false;
                return true;
            }
        };


        data::ParticleStorage::iterator createPlane( data::ParticleStorage& ps,
                                                     data::ShapeStorage& ss,
                                                     const Vec3& position,
                                                     const Vec3& normal )
        {
            auto p0              = ps.create(true);
            p0->getPositionRef() = position;
            p0->getShapeIDRef()  = ss.create<data::HalfSpace>( normal );
            p0->getOwnerRef()    = walberla::mpi::MPIManager::instance()->rank();
            p0->getTypeRef()     = 0;
            data::particle_flags::set(p0->getFlagsRef(), data::particle_flags::INFINITE);
            data::particle_flags::set(p0->getFlagsRef(), data::particle_flags::FIXED);
            data::particle_flags::set(p0->getFlagsRef(), data::particle_flags::NON_COMMUNICATING);
            return p0;
        }


        data::ParticleStorage::iterator createBox( data::ParticleStorage& ps,
                                                   const Vec3& position,
                                                   size_t boxShape)
        {
            auto p0              = ps.create(true);
            p0->getPositionRef() = position;
            p0->getShapeIDRef()  = boxShape;
            p0->getOwnerRef()    = walberla::mpi::MPIManager::instance()->rank();
            p0->getTypeRef()     = 0;
            data::particle_flags::set(p0->getFlagsRef(), data::particle_flags::NON_COMMUNICATING);
            data::particle_flags::set(p0->getFlagsRef(), data::particle_flags::FIXED);
            return p0;
        }




    }// namespace particle_erosion_utility
} // namespace walberla