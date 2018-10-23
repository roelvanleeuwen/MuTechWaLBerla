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
//! \file ConvexPolyhedronFactory.h
//! \author Christian Godenschwager <christian.godenschwager@fau.de>
//
//======================================================================================================================

#pragma once

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include "ConvexPolyhedron.h"

#include <boost/tuple/tuple.hpp>
#include "domain_decomposition/BlockStorage.h"
#include "mesh/decomposition/ConvexDecomposer.h"
#include "mesh/MeshOperations.h"
#include "pe/rigidbody/BodyStorage.h"
#include "pe/rigidbody/UnionFactory.h"
#include "pe/rigidbody/Union.h"
#include "pe/Materials.h"

namespace walberla {
namespace mesh {
namespace pe {

//=================================================================================================
//
//  CONVEXPOLYHEDRON SETUP FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/**
 * \ingroup pe
 * \brief Setup of a new ConvexPolyhedron.
 *
 * \param globalStorage process local global storage
 * \param blocks storage of all the blocks on this process
 * \param storageID BlockDataID of the BlockStorage block datum
 * \param uid The user-specific ID of the box.
 * \param gpos The global position of the center of the box.
 * \param pointCloud A point cloud which convex hull defines the polyhedron
 * \param material The material of the box.
 * \param global specifies if the box should be created in the global storage
 * \param communicating specifies if the box should take part in synchronization (syncNextNeighbour, syncShadowOwner)
 * \param infiniteMass specifies if the box has infinite mass and will be treated as an obstacle
 * \return Handle for the new box.
 * \exception std::invalid_argument Invalid box radius.
 * \exception std::invalid_argument Invalid global box position.
 *
 * This function creates a box primitive in the \b pe simulation system. The box with
 * user-specific ID \a uid is placed at the global position \a gpos, has the side lengths \a lengths,
 * and consists of the material \a material.
 *
 * The following code example illustrates the setup of a box:
 * \snippet PeDocumentationSnippets.cpp Create a Box
 *
 */
ConvexPolyhedronID createConvexPolyhedron( BodyStorage& globalStorage, BlockStorage& blocks, BlockDataID storageID,
                                           id_t uid, const Vec3& gpos, const std::vector< Vec3 > & pointCloud,
                                           MaterialID material = Material::find("iron"),
                                           bool global = false, bool communicating = true, bool infiniteMass = false );
//*************************************************************************************************


//*************************************************************************************************
/**
 * \ingroup pe
 * \brief Setup of a new ConvexPolyhedron.
 *
 * \param globalStorage process local global storage
 * \param blocks storage of all the blocks on this process
 * \param storageID BlockDataID of the BlockStorage block datum
 * \param uid The user-specific ID of the box.
 * \param gpos The global position of the center of the box.
 * \param mesh Surface mesh of convex polyhedron
 * \param material The material of the box.
 * \param global specifies if the box should be created in the global storage
 * \param communicating specifies if the box should take part in synchronization (syncNextNeighbour, syncShadowOwner)
 * \param infiniteMass specifies if the box has infinite mass and will be treated as an obstacle
 * \return Handle for the new box.
 * \exception std::invalid_argument Invalid box radius.
 * \exception std::invalid_argument Invalid global box position.
 *
 * This function creates a box primitive in the \b pe simulation system. The box with
 * user-specific ID \a uid is placed at the global position \a gpos, has the side lengths \a lengths,
 * and consists of the material \a material.
 *
 * The following code example illustrates the setup of a box:
 * \snippet PeDocumentationSnippets.cpp Create a Box
 *
 */
ConvexPolyhedronID createConvexPolyhedron( BodyStorage& globalStorage, BlockStorage& blocks, BlockDataID storageID,
                                           id_t uid, Vec3 gpos, TriangleMesh mesh,
                                           MaterialID material = Material::find("iron"),
                                           bool global = false, bool communicating = true, bool infiniteMass = false );
//*************************************************************************************************

//*************************************************************************************************
/**
 * \ingroup pe
 * \brief Setup of a new ConvexPolyhedron directly attached to a Union.
 *
 * \tparam BodyTypeTuple boost::tuple of all geometries the Union is able to contain
 * \exception std::runtime_error    Polyhedron TypeID not initalized!
 * \exception std::invalid_argument createSphere: Union argument is NULL
 * \exception std::logic_error      createSphere: Union is remote
 *
 * \see createConvexPolyhedron for more details
 */
template <typename BodyTypeTuple>
mesh::pe::ConvexPolyhedronID createConvexPolyhedron( Union<BodyTypeTuple>* un,
                       id_t uid, Vec3 gpos, mesh::TriangleMesh mesh,
                       MaterialID material = Material::find("iron"),
                       bool global = false, bool communicating = true, bool infiniteMass = false )
{
   if (mesh::pe::ConvexPolyhedron::getStaticTypeID() == std::numeric_limits<id_t>::max())
      throw std::runtime_error("Sphere TypeID not initalized!");

   // union not on this process/block -> terminate creation
   if (un == NULL)
      throw std::invalid_argument( "createSphere: Union argument is NULL" );

   // main union not on this process/block -> terminate creation
   if ( un->isRemote() )
      throw std::logic_error( "createSphere: Union is remote" );

   id_t sid(0);

   if (global)
   {
      sid = UniqueID<RigidBody>::createGlobal();
      WALBERLA_ASSERT_EQUAL(communicating, false);
      WALBERLA_ASSERT_EQUAL(infiniteMass, true);

   } else
   {
      sid = UniqueID<RigidBody>::create();
   }

   std::unique_ptr<mesh::pe::ConvexPolyhedron> cpolyhedron = std::make_unique<mesh::pe::ConvexPolyhedron>(sid, uid, gpos, Vec3(0,0,0), Quat(), mesh, material, global, communicating, infiniteMass);
   cpolyhedron->MPITrait.setOwner( un->MPITrait.getOwner() );

   if (cpolyhedron != NULL)
   {
      // Logging the successful creation of the sphere
      WALBERLA_LOG_DETAIL(
                "Created ConvexPolyhedron " << cpolyhedron->getSystemID() << " as part of union " << un->getSystemID() << "\n"
             << "   User-ID         = " << uid << "\n"
             << "   Global position = " << gpos << "\n"
             << "   LinVel          = " << cpolyhedron->getLinearVel() << "\n"
             << "   Material        = " << Material::getName( material )
               );
   }

   return static_cast<mesh::pe::ConvexPolyhedronID> (&(un->add( std::move(cpolyhedron) )));
}

//*************************************************************************************************
/**
 * \ingroup pe
 * \brief Setup of a new Non-ConvexPolyhedron as a union of its part. The mesh passed will be automatically decomposed.
 *
 * \tparam BodyTypeTuple boost::tuple of all geometries (including Union<ConvexPolyhedron> and ConvexPolyhedron)
 * \exception std::runtime_error    Polyhedron TypeID not initalized!
 * \exception std::invalid_argument createSphere: Union argument is NULL
 * \exception std::logic_error      createSphere: Union is remote
 *
 * \see createConvexPolyhedron for more details
 */
typedef boost::tuple<mesh::pe::ConvexPolyhedron> PolyhedronTuple;
typedef Union<PolyhedronTuple> TriangleMeshUnion;
TriangleMeshUnion* createNonConvexUnion( BodyStorage& globalStorage, BlockStorage& blocks, BlockDataID storageID,
                                                     id_t uid, Vec3 gpos, TriangleMesh mesh,
                                                     MaterialID material = Material::find("iron"),
                                                     bool global = false, bool communicating = true, bool infiniteMass = false );

} // namespace pe
} // namespace mesh
} // namespace walberla
