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
//! \file BoundaryFromMesh.impl.h
//! \ingroup geometry
//! \author Daniel Bauer <daniel.j.bauer@fau.de>
//
//======================================================================================================================

#include "boundary/Boundary.h"
#include "boundary/BoundaryUID.h"
#include "domain_decomposition/IBlock.h"
#include "field/FlagUID.h"
#include "mesh/boundary/BoundaryLocationFunction.h"
#include "mesh_common/DistanceFunction.h"

namespace walberla {
namespace geometry {
namespace initializer {

namespace internal {

// TODO: move this to central place
//**********************************************************************************************************************
/*! A small helper class that makes it possible to use mesh::boundary::BoundarySetup
*   to operate directly on the flag field instead of on a boundary handling.
*/
//**********************************************************************************************************************
template <typename FlagField_T>
class FlagFieldBoundaryHandling
{
public:
   using flag_t = typename FlagField_T::flag_t;

   static BlockDataID addToStorage( const shared_ptr<StructuredBlockStorage> & bs,
                                    const BlockDataID& flagFieldID )
   {
      return bs->addStructuredBlockData<FlagFieldBoundaryHandling>(
         [&] ( IBlock * const block, const StructuredBlockStorage * const /*blocks*/ )
         {
            FlagField_T * const flagField_ = block->getData<FlagField_T>(flagFieldID);
            FlagFieldBoundaryHandling * ffbh = new FlagFieldBoundaryHandling(flagField_);
            return ffbh;
         }
      );
   }

   FlagField_T * getFlagField()
   {
      return flagField_;
   }

   // map BoundaryUID to flag mask
   flag_t getBoundaryMask( const boundary::BoundaryUID & buid )
   {
      return flagField_->getOrRegisterFlag(FlagUID(buid.getIdentifier()));
   }

   // mark cell at (x, y, z) as boundary as indicated by flag
   void setBoundary( const flag_t flag,
                     const cell_idx_t x, const cell_idx_t y, const cell_idx_t z,
                     const BoundaryConfiguration & )
   {
      flagField_->addFlag(x, y, z, flag);
   }

   // For testing purposes, block data items must be comparable with operator "==".
   // Since instances of type "FlagFieldBoundaryHandling" are registered as block data items,
   // "FlagFieldBoundaryHandling" must implement operator "==". As of right now, comparing
   // two FlagFieldBoundaryHandling instances will always fail... :-) TODO: fixit?
   bool operator==( const FlagFieldBoundaryHandling & ) const
   {
       WALBERLA_CHECK(false, "You are trying to compare two FlagFieldBoundaryHandlings, however FlagFieldBoundaryHandling instances are not comparable!" );
       return false;
   }
   bool operator!=( const FlagFieldBoundaryHandling & rhs ) const
   {
       return !operator==(rhs);
   }

private:
   FlagField_T* const flagField_;

   FlagFieldBoundaryHandling( FlagField_T * const flagField )
      : flagField_(flagField)
   {}
};

} // namespace internal

//*******************************************************************************************************************
/*! Constructor for BoundaryFromMesh
*
* \param blocks         the structured block storage
* \param flagFieldID    the id of the flag field which type is the template parameter FlagField_T
* \param mesh           the mesh which describes the geometry of the simulation
* \param distanceOctree the distance octree which is obtained from the mesh
* \param numGhostLayers the number of ghost layers stored in the blocks
*/
//*******************************************************************************************************************
template<typename FlagField_T, typename Mesh_T>
BoundaryFromMesh<FlagField_T, Mesh_T>::BoundaryFromMesh(
   shared_ptr<StructuredBlockStorage> blocks,
   BlockDataID flagFieldID,
   shared_ptr<Mesh_T> mesh,
   shared_ptr<mesh::DistanceOctree<Mesh_T>> distanceOctree,
   const uint_t numGhostLayers )
   : flagFieldID_(flagFieldID),
     mesh_(mesh),
     distanceOctree_(distanceOctree),
     boundarySetup_(blocks, makeMeshDistanceFunction(distanceOctree), numGhostLayers)
{
   using namespace internal;

   flagFieldBoundarHandlingID_ = FlagFieldBoundaryHandling<FlagField_T>::addToStorage(blocks, flagFieldID);
}

//*******************************************************************************************************************
/*! Sets the fluid flag on the outside of the mesh, and boundary flags inside of the mesh
*
* \param colorToBoundaryMapper used to determine which boundary flag to set at which location
*                              based on the mesh's face colors
* \param fluidFlagID           the flag to set outside of the mesh, needs to be registered at the flag field
*
* In order to get the same bit representation for the same FlagUID on all blocks,
* you need to register all FlagUIDs at the flag field before calling this function.
* Doing so is adviced to ease debugging.
*/
//*******************************************************************************************************************
template<typename FlagField_T, typename Mesh_T>
void BoundaryFromMesh<FlagField_T, Mesh_T>::init(
   mesh::ColorToBoundaryMapper<Mesh_T> & colorToBoundaryMapper,
   FlagUID fluidFlagID )
{
   using namespace internal;

   boundarySetup_.setFlag<FlagField_T>(flagFieldID_, fluidFlagID, mesh::BoundarySetup::OUTSIDE);

   boundaryLocation_ = colorToBoundaryMapper.addBoundaryInfoToMesh(*mesh_);

   boundarySetup_.setBoundaries<FlagFieldBoundaryHandling<FlagField_T>>(
      flagFieldBoundarHandlingID_,
      makeBoundaryLocationFunction(distanceOctree_, boundaryLocation_),
      mesh::BoundarySetup::INSIDE );
}

template<typename FlagField_T, typename Mesh_T>
void BoundaryFromMesh<FlagField_T, Mesh_T>::init(
   BlockStorage &, const Config::BlockHandle & blockHandle )
{
   mesh::ColorToBoundaryMapper< Mesh_T > colorToBoundaryMapper{ blockHandle.getBlock( "ColorToBoundaryMapper") };
   FlagUID fluidFlag = FlagUID{ blockHandle.getParameter< std::string >( "fluidFlag") };

   init( colorToBoundaryMapper, fluidFlag );
}

//*******************************************************************************************************************
/*! Returns a null pointer if init has not been called yet
*/
//*******************************************************************************************************************
template<typename FlagField_T, typename Mesh_T>
shared_ptr<mesh::BoundaryLocation<Mesh_T>> BoundaryFromMesh<FlagField_T, Mesh_T>::getBoundaryLocation() const
{
   return boundaryLocation_;
}

} // namespace initializer
} // namespace geometry
} // namespace walberla
