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
//! \file PythonExports.cpp
//! \ingroup domain_decomposition
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

// Do not reorder includes - the include order is important
#include "python_coupling/PythonWrapper.h"

#ifdef WALBERLA_BUILD_WITH_PYTHON

#   include "blockforest/Initialization.h"
#   include "blockforest/SetupBlock.h"
#   include "blockforest/SetupBlockForest.h"
#   include "blockforest/StructuredBlockForest.h"
#   include "blockforest/communication/UniformBufferedScheme.h"
#   include "blockforest/loadbalancing/StaticCurve.h"
#   include "blockforest/python/Exports.h"

#   include "core/StringUtility.h"
#   include "core/logging/Logging.h"
#   include "core/mpi/MPIIO.h"

#   include "python_coupling/Manager.h"
#   include "python_coupling/helper/ConfigFromDict.h"

#   include "stencil/D3Q19.h"
#   include "stencil/D3Q27.h"
#   include "stencil/D3Q7.h"

#   include <memory>
#   include <pybind11/functional.h>
#   include <pybind11/numpy.h>
#   include <pybind11/pybind11.h>
#   include <pybind11/stl.h>
#   include <sstream>
#   include <utility>

namespace walberla
{
namespace blockforest
{
std::string printSetupBlock(const SetupBlock& b)
{
   std::stringstream out;
   out << "SetupBlock at " << b.getAABB();
   return out.str();
}

namespace py = pybind11;

//======================================================================================================================
//
//  StructuredBlockForest
//
//======================================================================================================================

#ifdef WALBERLA_BUILD_WITH_PYTHON

void NoSuchBlockData::translate(  const NoSuchBlockData & e ) {
   throw py::cast_error(e.what());
}

void BlockDataNotConvertible::translate(  const BlockDataNotConvertible & e ) {
   throw py::cast_error(e.what());
}
#else

void NoSuchBlockData::translate(  const NoSuchBlockData &  ) {}

void BlockDataNotConvertible::translate(  const BlockDataNotConvertible &  ) {}

#endif

BlockDataID blockDataIDFromString(BlockStorage& bs, const std::string& stringID)
{
   auto ids = bs.getBlockDataIdentifiers();

   for (uint_t i = 0; i < ids.size(); ++i)
      if (ids[i] == stringID) return BlockDataID(i);

   throw NoSuchBlockData();
}

BlockDataID blockDataIDFromString(IBlock& block, const std::string& stringID)
{
   return blockDataIDFromString(block.getBlockStorage(), stringID);
}

BlockDataID blockDataIDFromString(StructuredBlockForest& bs, const std::string& stringID)
{
   return blockDataIDFromString(bs.getBlockStorage(), stringID);
}

py::object StructuredBlockForest_iter(const shared_ptr< StructuredBlockForest >& bf) // NOLINT
{
   // shared_ptr<StructuredBlockForest> s = py::cast< shared_ptr<StructuredBlockForest> > ( StructuredBlockForest );

   std::vector< const IBlock* > blocks;
   bf->getBlocks(blocks);
   py::list resultList;

   for (auto it = blocks.begin(); it != blocks.end(); ++it)
   {
      py::object theObject = py::cast(*it);
      resultList.append(theObject);
   }

   return py::iter(resultList);
}

py::object* blockDataCreationHelper(IBlock* block, StructuredBlockForest* bs, py::object callable) // NOLINT
{
   py::object* res = new py::object(callable(block, bs));
   return res;
}

py::object StructuredBlockForest_getItem(const shared_ptr< StructuredBlockForest >& bf, uint_t i) // NOLINT
{
   if (i >= bf->size()) { throw py::value_error("Index out of bounds"); }

   std::vector< const IBlock* > blocks;
   bf->getBlocks(blocks);

   py::object theObject = py::cast(blocks[i]);
   return theObject;
}

py::list StructuredBlockForest_blocksOverlappedByAABB(StructuredBlockForest& s, const AABB& aabb)
{
   std::vector< IBlock* > blocks;
   s.getBlocksOverlappedByAABB(blocks, aabb);

   py::list resultList;
   for (auto it = blocks.begin(); it != blocks.end(); ++it)
      resultList.append(py::cast(*it));
   return resultList;
}

py::list StructuredBlockForest_blocksContainedWithinAABB(StructuredBlockForest& s, const AABB& aabb)
{
   std::vector< IBlock* > blocks;
   s.getBlocksContainedWithinAABB(blocks, aabb);

   py::list resultList;
   for (auto it = blocks.begin(); it != blocks.end(); ++it)
      resultList.append(py::cast(*it));
   return resultList;
}

py::object SbS_transformGlobalToLocal(StructuredBlockForest& s, IBlock& block, const py::object& global)
{
   if (py::isinstance< CellInterval >(global))
   {
      CellInterval ret;
      s.transformGlobalToBlockLocalCellInterval(ret, block, py::cast< CellInterval >(global));
      return py::cast(ret);
   }
   else if (py::isinstance< Cell >(global))
   {
      Cell ret;
      s.transformGlobalToBlockLocalCell(ret, block, py::cast< Cell >(global));
      return py::cast(ret);
   }

   throw py::value_error("Only CellIntervals and cells can be transformed");
}

py::object SbS_transformLocalToGlobal(StructuredBlockForest& s, IBlock& block, const py::object& local)
{
   if (py::isinstance< CellInterval >(local))
   {
      CellInterval ret;
      s.transformBlockLocalToGlobalCellInterval(ret, block, py::cast< CellInterval >(local));
      return py::cast(ret);
   }
   else if (py::isinstance< Cell >(local))
   {
      Cell ret;
      s.transformBlockLocalToGlobalCell(ret, block, py::cast< Cell >(local));
      return py::cast(ret);
   }
   throw py::value_error("Only CellIntervals and cells can be transformed");
}

void SbS_writeBlockData(StructuredBlockForest& s, const std::string& blockDataId, const std::string& file)
{
   mpi::SendBuffer buffer;
   s.serializeBlockData(blockDataIDFromString(s, blockDataId), buffer);
   mpi::writeMPIIO(file, buffer);
}

void SbS_readBlockData(StructuredBlockForest& s, const std::string& blockDataId, const std::string& file)
{
   mpi::RecvBuffer buffer;
   mpi::readMPIIO(file, buffer);

   s.deserializeBlockData(blockDataIDFromString(s, blockDataId), buffer);
   if (!buffer.isEmpty())
   { throw py::cast_error("Reading failed - file does not contain matching data for this type."); }
}

CellInterval SbS_getBlockCellBB(StructuredBlockForest& s, const IBlock* block) { return s.getBlockCellBB(*block); }

Vector3< real_t > SbS_mapToPeriodicDomain1(StructuredBlockForest& s, real_t x, real_t y, real_t z)
{
   Vector3< real_t > res(x, y, z);
   s.mapToPeriodicDomain(res);
   return res;
}
Vector3< real_t > SbS_mapToPeriodicDomain2(StructuredBlockForest& s, Vector3< real_t > in)
{
   s.mapToPeriodicDomain(in);
   return in;
}
Cell SbS_mapToPeriodicDomain3(StructuredBlockForest& s, Cell in, uint_t level = 0)
{
   s.mapToPeriodicDomain(in, level);
   return in;
}

py::object SbS_getBlock1(StructuredBlockForest& s, const real_t x, const real_t y, const real_t z)
{
   return py::cast(s.getBlock(x, y, z));
}

py::object SbS_getBlock2(StructuredBlockForest& s, const Vector3< real_t >& v) { return py::cast(s.getBlock(v)); }

py::tuple SbS_periodic(StructuredBlockForest& s)
{
   return py::make_tuple(s.isXPeriodic(), s.isYPeriodic(), s.isZPeriodic());
}

bool SbS_atDomainXMinBorder(StructuredBlockForest& s, const IBlock* b) { return s.atDomainXMinBorder(*b); }
bool SbS_atDomainXMaxBorder(StructuredBlockForest& s, const IBlock* b) { return s.atDomainXMaxBorder(*b); }
bool SbS_atDomainYMinBorder(StructuredBlockForest& s, const IBlock* b) { return s.atDomainYMinBorder(*b); }
bool SbS_atDomainYMaxBorder(StructuredBlockForest& s, const IBlock* b) { return s.atDomainYMaxBorder(*b); }
bool SbS_atDomainZMinBorder(StructuredBlockForest& s, const IBlock* b) { return s.atDomainZMinBorder(*b); }
bool SbS_atDomainZMaxBorder(StructuredBlockForest& s, const IBlock* b) { return s.atDomainZMaxBorder(*b); }

void exportBlockForest(py::module_& m)
{
   using namespace pybind11::literals;

   bool (StructuredBlockForest::*p_blockExists1)(const Vector3< real_t >&) const = &StructuredBlockForest::blockExists;
   bool (StructuredBlockForest::*p_blockExistsLocally1)(const Vector3< real_t >&) const =
      &StructuredBlockForest::blockExistsLocally;
   bool (StructuredBlockForest::*p_blockExistsRemotely1)(const Vector3< real_t >&) const =
      &StructuredBlockForest::blockExistsRemotely;

   bool (StructuredBlockForest::*p_blockExists2)(const real_t, const real_t, const real_t) const =
      &StructuredBlockForest::blockExists;
   bool (StructuredBlockForest::*p_blockExistsLocally2)(const real_t, const real_t, const real_t) const =
      &StructuredBlockForest::blockExistsLocally;
   bool (StructuredBlockForest::*p_blockExistsRemotely2)(const real_t, const real_t, const real_t) const =
      &StructuredBlockForest::blockExistsRemotely;

   py::class_< StructuredBlockForest, std::shared_ptr< StructuredBlockForest > >(m, "StructuredBlockForest")
      .def("getNumberOfLevels", &StructuredBlockStorage::getNumberOfLevels)
      .def_property_readonly("getDomain", &StructuredBlockStorage::getDomain)
      .def("mapToPeriodicDomain", &SbS_mapToPeriodicDomain1)
      .def("mapToPeriodicDomain", &SbS_mapToPeriodicDomain2)
      .def("mapToPeriodicDomain", &SbS_mapToPeriodicDomain3)
      .def("__getitem__", &StructuredBlockForest_getItem, py::keep_alive< 1, 2 >())
      .def("__len__", &StructuredBlockStorage::size)
      .def("getBlock", SbS_getBlock1)
      .def("getBlock", SbS_getBlock2)
      .def("containsGlobalBlockInformation", &StructuredBlockStorage::containsGlobalBlockInformation)
      .def("blocksOverlappedByAABB", &StructuredBlockForest_blocksOverlappedByAABB)
      .def("blocksContainedWithinAABB", &StructuredBlockForest_blocksContainedWithinAABB)
      .def("blockExists", p_blockExists1)
      .def("blockExists", p_blockExists2)
      .def("blockExistsLocally", p_blockExistsLocally1)
      .def("blockExistsLocally", p_blockExistsLocally2)
      .def("blockExistsRemotely", p_blockExistsRemotely1)
      .def("blockExistsRemotely", p_blockExistsRemotely2)
      .def("atDomainXMinBorder", &SbS_atDomainXMinBorder)
      .def("atDomainXMaxBorder", &SbS_atDomainXMaxBorder)
      .def("atDomainYMinBorder", &SbS_atDomainYMinBorder)
      .def("atDomainYMaxBorder", &SbS_atDomainYMaxBorder)
      .def("atDomainZMinBorder", &SbS_atDomainZMinBorder)
      .def("atDomainZMaxBorder", &SbS_atDomainZMaxBorder)
      .def("dx", &StructuredBlockStorage::dx)
      .def("dy", &StructuredBlockStorage::dy)
      .def("dz", &StructuredBlockStorage::dz)
      .def_property_readonly("getDomainCellBB", &StructuredBlockStorage::getDomainCellBB)
      .def("getBlockCellBB", &SbS_getBlockCellBB)
      .def("transformGlobalToLocal", &SbS_transformGlobalToLocal)
      .def("transformLocalToGlobal", &SbS_transformLocalToGlobal)
      .def("writeBlockData", &SbS_writeBlockData)
      .def("readBlockData", &SbS_readBlockData)
      .def("__iter__", &StructuredBlockForest_iter)
      .def_property_readonly("containsGlobalBlockInformation", &StructuredBlockStorage::containsGlobalBlockInformation)
      .def_property_readonly("periodic", &SbS_periodic);

   py::class_< SetupBlock, shared_ptr< SetupBlock > >(m, "SetupBlock")
      .def("get_level", &SetupBlock::getLevel)
      .def("set_workload", &SetupBlock::setWorkload)
      .def("get_workload", &SetupBlock::getWorkload)
      .def("set_memory", &SetupBlock::setMemory)
      .def("get_memory", &SetupBlock::getMemory)
      .def("get_aabb", &SetupBlock::getAABB)
      .def("__repr__", &printSetupBlock);

   m.def(
      "createUniformBlockGrid",
      [](std::array< uint_t, 3 > blocks, std::array< uint_t, 3 > cellsPerBlock, real_t dx,
         bool oneBlockPerProcess, std::array< bool, 3 > periodic, bool keepGlobalBlockInformation) {
         return blockforest::createUniformBlockGrid(blocks[0], blocks[1], blocks[2], cellsPerBlock[0], cellsPerBlock[1],
                                                    cellsPerBlock[2], dx, oneBlockPerProcess, periodic[0], periodic[1], periodic[2],
                                                    keepGlobalBlockInformation);
      },
      "blocks"_a, "cellsPerBlock"_a, "dx"_a = real_t(1), "oneBlockPerProcess"_a = true,
      "periodic"_a = std::array< bool, 3 >{ false, false, false }, "keepGlobalBlockInformation"_a = false);
}

} // namespace blockforest
} // namespace walberla

#endif // WALBERLA_BUILD_WITH_PYTHON