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
//! \file GatherExport.impl.h
//! \ingroup field
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

#include "field/Gather.h"
#include "python_coupling/helper/MplHelpers.h"
#include "python_coupling/helper/BlockStorageExportHelpers.h"
#include "python_coupling/helper/ModuleScope.h"
#include "python_coupling/helper/SliceToCellInterval.h"


namespace walberla {
namespace field {


namespace internal {
namespace py = pybind11;
   //===================================================================================================================
   //
   //  Gather
   //
   //===================================================================================================================


   template<typename Field_T>
   Field_T gatherToObject( const shared_ptr<StructuredBlockStorage> & blocks, BlockDataID fieldID,
                                         CellInterval boundingBox = CellInterval(), int targetRank = 0 )
   {
      typedef Field< typename Field_T::value_type, Field_T::F_SIZE > ResultField;
      auto result = make_shared< ResultField > ( 0,0,0 );
      field::gather< Field_T, ResultField > ( *result, blocks, fieldID, boundingBox, targetRank, MPI_COMM_WORLD );

      if ( MPIManager::instance()->worldRank() == targetRank )
         return result;
      else
         return py::object();
   }

   FunctionExporterClass( gatherToObject,
                          py::object( const shared_ptr<StructuredBlockStorage> &,
                                                 BlockDataID, CellInterval,int ) );

   template<typename... FieldTypes>
   static py::object gatherWrapper (  const shared_ptr<StructuredBlockStorage> & blocks, const std::string & blockDataStr,
                                      const py::tuple & slice,  int targetRank = 0 )
   {


      auto fieldID = python_coupling::blockDataIDFromString( *blocks, blockDataStr );
      CellInterval boundingBox = python_coupling::globalPythonSliceToCellInterval( blocks, slice );

      if ( blocks->begin() == blocks->end() ) {
         // if no blocks are on this process the field::gather function can be called with any type
         // however we have to call it, otherwise a deadlock occurs
         gatherToObject< Field<real_t,1> > ( blocks, fieldID, boundingBox, targetRank );
         return py::object();
      }

      IBlock * firstBlock =  & ( * blocks->begin() );
      python_coupling::Dispatcher<Exporter_gatherToObject, FieldTypes... > dispatcher( firstBlock );
      auto func = dispatcher( fieldID );
      if ( !func )
      {
         PyErr_SetString( PyExc_RuntimeError, "This function cannot handle this type of block data.");
         throw py::error_already_set();
      }
      else
      {
         return func( blocks, fieldID, boundingBox, targetRank) ;
      }
   }

} // namespace internal


namespace py = pybind11;
template<typename... FieldTypes >
void exportGatherFunctions(py::module_ &m)
{
   // python_coupling::ModuleScope fieldModule( "field" );

   m.def( "gather",  &internal::gatherWrapper<FieldTypes...>, py::arg("blocks"), py::arg("blockDataName"), py::arg("slice"), py::arg("targetRank") = 0 );
}




} // namespace moduleName
} // namespace walberla


