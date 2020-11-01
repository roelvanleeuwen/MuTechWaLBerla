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

#   include "core/StringUtility.h"
#   include "core/logging/Logging.h"

#   include "domain_decomposition/StructuredBlockStorage.h"

#   include "python_coupling/Manager.h"
#   include "python_coupling/helper/ConfigFromDict.h"

#   include "stencil/D3Q19.h"
#   include "stencil/D3Q27.h"
#   include "stencil/D3Q7.h"

#   include <memory>
#   include <pybind11/numpy.h>
#   include <pybind11/pybind11.h>
#   include <pybind11/stl.h>
#   include <pybind11/functional.h>
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

void exportBlockForest(py::module_& m)
{
   using namespace pybind11::literals;
   py::class_< StructuredBlockForest, std::shared_ptr< StructuredBlockForest > >(m, "StructuredBlockForest");

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
         std::array< uint_t, 3 > numberOfProcesses, std::array< bool, 3 > periodic, bool keepGlobalBlockInformation) {
         return blockforest::createUniformBlockGrid(blocks[0], blocks[1], blocks[2], cellsPerBlock[0], cellsPerBlock[1],
                                                    cellsPerBlock[2], dx, numberOfProcesses[0], numberOfProcesses[1],
                                                    numberOfProcesses[2], periodic[0], periodic[1], periodic[2],
                                                    keepGlobalBlockInformation);
      },
      "blocks"_a, "cellsPerBlock"_a, "dx"_a = real_t(1), "numberOfProcesses"_a = std::array< uint_t, 3 >{ 1, 1, 1 },
      "periodic"_a = std::array< bool, 3 >{ false, false, false }, "keepGlobalBlockInformation"_a = false);
}

} // namespace blockforest
} // namespace walberla

#endif // WALBERLA_BUILD_WITH_PYTHON
