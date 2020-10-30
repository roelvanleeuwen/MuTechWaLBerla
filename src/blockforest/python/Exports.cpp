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

#include "blockforest/StructuredBlockForest.h"
#include "blockforest/communication/UniformBufferedScheme.h"
#include "blockforest/Initialization.h"
#include "blockforest/SetupBlock.h"
#include "blockforest/SetupBlockForest.h"
#include "blockforest/loadbalancing/StaticCurve.h"

#include "core/logging/Logging.h"
#include "core/StringUtility.h"

#include "domain_decomposition/StructuredBlockStorage.h"

#include "python_coupling/Manager.h"
#include "python_coupling/helper/ConfigFromDict.h"

#include "stencil/D3Q7.h"
#include "stencil/D3Q19.h"
#include "stencil/D3Q27.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <memory>

#include <sstream>
#include <utility>


namespace walberla {
namespace blockforest {
namespace py = pybind11;

using walberla::blockforest::communication::UniformBufferedScheme;

bool checkForThreeTuple( py::object obj ) //NOLINT
{
   if( py::isinstance<py::tuple>(obj) )
      return false;

   py::tuple t = py::tuple ( obj );
   return py::len(t) == 3;
}


py::object python_createUniformBlockGrid(py::args args, py::kwargs kwargs) //NOLINT
{
   if( py::len(std::move(args)) > 0 ) {
      throw py::value_error("This function takes only keyword arguments");
   }

   py::dict kw = py::dict(kwargs);

   for (auto item : kw)
   {
      std::string key = py::str(item.first);
      if ( key != "cells" &&
           key != "cellsPerBlock" &&
           key != "blocks" &&
           key != "periodic" &&
           key != "dx" &&
           key != "oneBlockPerProcess"  )
      {
         throw py::value_error((std::string("Unknown Parameter: ") + key ).c_str());
      }
   }

   if( kw.contains("cells ") && ! checkForThreeTuple( kw["cells"] ) ) {
      throw py::value_error("Parameter 'cells' has to be tuple of length 3, indicating cells in x,y,z direction");
   }
   if( kw.contains("cellsPerBlock ") && ! checkForThreeTuple( kw["cellsPerBlock"] ) ) {
      throw py::value_error("Parameter 'cellsPerBlock' has to be tuple of length 3, indicating cells in x,y,z direction");
   }
   if( kw.contains("blocks ") && ! checkForThreeTuple( kw["blocks"] ) ) {
      throw py::value_error("Parameter 'blocks' has to be tuple of length 3, indicating cells in x,y,z direction");
   }

   bool keepGlobalBlockInformation = false;
   if ( kw.contains("keepGlobalBlockInformation") )
   {
      if ( py::isinstance<bool>(kw["keepGlobalBlockInformation"] ) )
         keepGlobalBlockInformation =  kw["keepGlobalBlockInformation"].cast<bool>() ;
      else
      {
         throw py::value_error("Parameter 'keepGlobalBlockInformation' has to be a boolean");
      }
   }

   shared_ptr<Config> cfg = python_coupling::configFromPythonDict( kw );

   try {
      shared_ptr< StructuredBlockForest > blocks = createUniformBlockGridFromConfig( cfg->getGlobalBlock(), nullptr, keepGlobalBlockInformation );
      return py::cast(blocks);
   }
   catch( std::exception & e)
   {
      PyErr_SetString( PyExc_ValueError, e.what() );
      throw py::error_already_set();
   }
}

shared_ptr<StructuredBlockForest> createStructuredBlockForest( Vector3<uint_t> blocks,
                                                               Vector3<uint_t> cellsPerBlock,
                                                               Vector3<bool> periodic,
                                                               py::object blockExclusionCallback = py::object(),
                                                               py::object workloadMemoryCallback = py::object(),
                                                               py::object refinementCallback = py::object(),
                                                               const real_t dx = 1.0,
                                                               memory_t processMemoryLimit = std::numeric_limits<memory_t>::max(),
                                                               const bool keepGlobalBlockInformation = false)
{
   using namespace blockforest;
   Vector3<real_t> bbMax;
   for( uint_t i=0; i < 3; ++i )
      bbMax[i] = real_c( blocks[i] * cellsPerBlock[i] ) * dx;
   AABB domainAABB ( Vector3<real_t>(0),  bbMax );

   SetupBlockForest sforest;

   auto blockExclusionFunc = [&blockExclusionCallback] ( std::vector<walberla::uint8_t>& excludeBlock, const SetupBlockForest::RootBlockAABB& aabb ) -> void
   {
      for( uint_t i = 0; i != excludeBlock.size(); ++i )
      {
         AABB bb = aabb(i);
         auto pythonReturnVal = blockExclusionCallback(bb);
         if( py::isinstance<bool>( pythonReturnVal ) ) {
            PyErr_SetString( PyExc_ValueError, "blockExclusionCallback has to return a boolean");
            throw py::error_already_set();
         }

         bool returnVal = bool(pythonReturnVal);
         if ( returnVal )
            excludeBlock[i] = 1;
      }
   };

   auto workloadMemoryFunc = [&workloadMemoryCallback] ( SetupBlockForest & forest )-> void
   {
      std::vector< SetupBlock* > blockVector;
      forest.getBlocks( blockVector );

      for( uint_t i = 0; i != blockVector.size(); ++i ) {
         blockVector[i]->setMemory( memory_t(1) );
         blockVector[i]->setWorkload( workload_t(1) );
         workloadMemoryCallback( blockVector[i] );
      }
   };

   auto refinementFunc = [&refinementCallback] ( SetupBlockForest & forest )-> void
   {
      for( auto block = forest.begin(); block != forest.end(); ++block )
      {
         SetupBlock * sb = &(*block);
         auto pythonRes = refinementCallback( sb );
         if( py::isinstance<bool>( pythonRes ) ) {
            throw py::value_error("refinementCallback has to return a boolean");
         }
         bool returnVal = bool( pythonRes );
         if( returnVal )
            block->setMarker( true );
      }
   };

   if ( blockExclusionCallback ) {
      if( !PyCallable_Check( blockExclusionCallback.ptr() ) ) {
         throw py::value_error("blockExclusionCallback has to be callable");
      }
      sforest.addRootBlockExclusionFunction( blockExclusionFunc );
   }

   if ( workloadMemoryCallback ) {
      if( !PyCallable_Check( workloadMemoryCallback.ptr() ) ) {
         throw py::value_error("workloadMemoryCallback has to be callable");
      }
      sforest.addWorkloadMemorySUIDAssignmentFunction( workloadMemoryFunc );
   }
   else
      sforest.addWorkloadMemorySUIDAssignmentFunction( uniformWorkloadAndMemoryAssignment );

   if ( refinementCallback ) {
      if( !PyCallable_Check( refinementCallback.ptr() ) ) {
         throw py::value_error("refinementCallback has to be callable");
      }
      sforest.addRefinementSelectionFunction( refinementFunc );
   }

   sforest.init( domainAABB, blocks[0], blocks[1], blocks[2], periodic[0], periodic[1], periodic[2] );

   // calculate process distribution
   sforest.balanceLoad( blockforest::StaticLevelwiseCurveBalanceWeighted(),
                        uint_c( MPIManager::instance()->numProcesses() ),
                        real_t(0), processMemoryLimit );

   if( !MPIManager::instance()->rankValid() )
      MPIManager::instance()->useWorldComm();

   // create StructuredBlockForest (encapsulates a newly created BlockForest)
   auto bf = std::make_shared< BlockForest >( uint_c( MPIManager::instance()->rank() ), sforest, keepGlobalBlockInformation );

   auto sbf = std::make_shared< StructuredBlockForest >( bf, cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2] );
   sbf->createCellBoundingBoxes();

   return sbf;
}

//py::object createUniformNeighborScheme(  const shared_ptr<StructuredBlockForest> & bf,
//                                         const std::string & stencil )
//{
//   if ( string_icompare(stencil, "D3Q7") == 0 )
//      return py::object ( make_shared< UniformBufferedScheme<stencil::D3Q7> > ( bf ) );
//   if ( string_icompare(stencil, "D3Q19") == 0 )
//      return py::object ( make_shared< UniformBufferedScheme<stencil::D3Q19> > ( bf ) );
//   if ( string_icompare(stencil, "D3Q27") == 0 )
//      return py::object ( make_shared< UniformBufferedScheme<stencil::D3Q27> > ( bf ) );
//   else {
//      PyErr_SetString( PyExc_RuntimeError, "Unknown stencil. Allowed values 'D3Q27', 'D3Q19', 'D3Q7'");
//      throw py::error_already_set();
//      return py::object();
//   }
//}

std::string printSetupBlock(const SetupBlock & b )
{
   std::stringstream out;
   out <<  "SetupBlock at " << b.getAABB();
   return out.str();
}


namespace py = pybind11;

void exportBlockForest(py::module_ &m)
{

//   py::class_< StructuredBlockForest, std::shared_ptr< StructuredBlockForest > >(m, "StructuredBlockForrest")
//      .def("thread_blocks", &getThreadBlocks)
//      .def("ghost_blocks", &getGhostBlocks)
//      .def_property_readonly("aabb", &StructuredBlockStorage::getDomain);


   py::class_< SetupBlock, shared_ptr<SetupBlock> > (m, "SetupBlock" )
            .def("get_level",       &SetupBlock::getLevel      )
            .def("set_workload",    &SetupBlock::setWorkload   )
            .def("get_workload",    &SetupBlock::getWorkload   )
            .def("set_memory",      &SetupBlock::setMemory     )
            .def("get_memory",      &SetupBlock::getMemory     )
            .def("get_aabb",        &SetupBlock::getAABB       )
            .def("__repr__", &printSetupBlock           )
            ;

   m.def( "createUniformBlockGrid", &python_createUniformBlockGrid );

//   m.def( "createCustomBlockGrid", &createStructuredBlockForest,
//                py::arg("blocks"), py::arg("cellsPerBlock"), py::arg("periodic"),
//                py::arg("blockExclusionCallback") = py::object(),
//                py::arg("workloadMemoryCallback") = py::object(),
//                py::arg("refinementCallback") = py::object() ,
//                py::arg("dx") = 1.0,
//                py::arg("processMemoryLimit") = std::numeric_limits<memory_t>::max(),
//                py::arg("keepGlobalBlockInformation") = false );
}

} // namespace domain_decomposition
} // namespace walberla


#endif //WALBERLA_BUILD_WITH_PYTHON
