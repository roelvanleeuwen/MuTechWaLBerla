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
//! \file Squirmer.h
//! \author Michael Kuron <mkuron@icp.uni-stuttgart.de>
//! \brief Marshalling of objects for data transmission or storage.
//
//======================================================================================================================

#pragma once


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include "pe/communication/Instantiate.h"
#include "pe/communication/Marshalling.h"
#include "pe/rigidbody/Squirmer.h"
#include "pe/communication/rigidbody/Sphere.h"

namespace walberla {
namespace pe {
namespace communication {

struct SquirmerParameters : public SphereParameters {
   real_t squirmerVelocity_;
   real_t squirmerBeta_;
};

//*************************************************************************************************
/*!\brief Marshalling a squirmer primitive.
 *
 * \param buffer The buffer to be filled.
 * \param obj The object to be marshalled.
 * \return void
 */
void marshal( mpi::SendBuffer& buffer, const Squirmer& obj );
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Unmarshalling a squirmer primitive.
 *
 * \param buffer The buffer from where to read.
 * \param objparam The object to be reconstructed.
 * \return void
 */
void unmarshal( mpi::RecvBuffer& buffer, SquirmerParameters& objparam );
//*************************************************************************************************


inline SquirmerPtr instantiate( mpi::RecvBuffer& buffer, const math::AABB& domain, const math::AABB& block, SquirmerID& newBody )
{
   SquirmerParameters subobjparam;
   unmarshal( buffer, subobjparam );
   correctBodyPosition(domain, block.center(), subobjparam.gpos_);
   auto sq = std::make_unique<Squirmer>( subobjparam.sid_, subobjparam.uid_, subobjparam.gpos_, subobjparam.q_, subobjparam.radius_, subobjparam.squirmerVelocity_, subobjparam.squirmerBeta_, subobjparam.material_, false, subobjparam.communicating_, subobjparam.infiniteMass_ );
   sq->setLinearVel( subobjparam.v_ );
   sq->setAngularVel( subobjparam.w_ );
   sq->MPITrait.setOwner( subobjparam.mpiTrait_.owner_ );
   newBody = sq.get();
   return sq;
}

}  // namespace communication
}  // namespace pe
}  // namespace walberla
