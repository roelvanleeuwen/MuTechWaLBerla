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
//! \file Exports.impl.h
//! \ingroup geometry
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

// Do not reorder includes - the include order is important
#include "python_coupling/PythonWrapper.h"
#include "python_coupling/helper/ModuleScope.h"

#ifdef WALBERLA_BUILD_WITH_PYTHON
#include "python_coupling/Manager.h"
#include "python_coupling/helper/MplHelpers.h"
#include "python_coupling/helper/BlockStorageExportHelpers.h"
#include "python_coupling/helper/PythonIterableToStdVector.h"


#include "postprocessing/FieldToSurfaceMesh.h"

namespace py = pybind11;


namespace walberla {
namespace postprocessing {

FunctionExporterClass( realFieldToSurfaceMesh,
                       shared_ptr<geometry::TriangleMesh>( const shared_ptr<StructuredBlockStorage> &,
                                                           ConstBlockDataID, real_t, uint_t, bool, int, MPI_Comm  ) );

template<typename... FieldTypes>
static shared_ptr<geometry::TriangleMesh> exp_realFieldToSurfaceMesh
      ( const shared_ptr<StructuredBlockStorage> & bs, const std::string & blockDataStr, real_t threshold,
        uint_t fCoord = 0, bool calcNormals = false, int targetRank = 0 )
{
   if ( bs->begin() == bs->end() )
      return shared_ptr<geometry::TriangleMesh>();
   IBlock * firstBlock =  & ( * bs->begin() );

   auto fieldID = python_coupling::blockDataIDFromString( *bs, blockDataStr );

   python_coupling::Dispatcher<Exporter_realFieldToSurfaceMesh, FieldTypes... > dispatcher( firstBlock );
   return dispatcher( fieldID )( bs, fieldID, threshold, fCoord, calcNormals, targetRank, MPI_COMM_WORLD) ;
}


template< typename FField>
typename FField::value_type maskFromFlagList(  const shared_ptr<StructuredBlockStorage> & bs,
                                               ConstBlockDataID flagFieldID,
                                               const std::vector< std::string > & flagList )
{
   if ( bs->begin() == bs->end() )
      return 0;

   IBlock & firstBlock = *(  bs->begin() );
   const FField * flagField = firstBlock.getData< const FField > ( flagFieldID );

   typedef typename FField::flag_t flag_t;
   flag_t mask = 0;
   for( auto it = flagList.begin(); it != flagList.end(); ++it )
   {
      if ( ! flagField->flagExists( *it ) )
         throw python_coupling::BlockDataNotConvertible( "Unknown FlagID" );

      mask = flag_t( mask | flagField->getFlag( *it ) );
   }

   return mask;
}


template<typename FlagField_T>
py::object adaptedFlagFieldToSurfaceMesh( const shared_ptr<StructuredBlockStorage> & bs,
                                                     ConstBlockDataID fieldID, const std::vector< std::string > & flagList,
                                                     bool calcNormals = false, int targetRank = 0 )
{
   namespace py = pybind11;

   auto mask = maskFromFlagList<FlagField_T>( bs, fieldID, flagList );
   return py::object( flagFieldToSurfaceMesh<FlagField_T>(bs, fieldID, mask, calcNormals, targetRank ) );
}


FunctionExporterClass( adaptedFlagFieldToSurfaceMesh,
                       py::object( const shared_ptr<StructuredBlockStorage> &, ConstBlockDataID,
                                              const std::vector< std::string > &, bool,int  ) );


template<typename... FieldTypes>
static py::object exp_flagFieldToSurfaceMesh ( const shared_ptr<StructuredBlockStorage> & bs,
                                                          const std::string & blockDataName,
                                                          const py::list & flagList,
                                                          bool calcNormals = false, int targetRank = 0 )
{
   if ( bs->begin() == bs->end() )
      return py::object(); //TODO check if this is correct

   IBlock * firstBlock =  & ( * bs->begin() );

   auto fieldID = python_coupling::blockDataIDFromString( *bs, blockDataName );

   // auto flagVector = python_coupling::pythonIterableToStdVector<std::string>( flagList );
   python_coupling::Dispatcher<Exporter_adaptedFlagFieldToSurfaceMesh, FieldTypes... > dispatcher( firstBlock );
   return dispatcher( fieldID )( bs, fieldID, flagList, calcNormals, targetRank );
}


namespace internal {

template<typename FieldType, class Enable = void>
struct ExportFieldToPython;

template <typename FieldType>
struct ExportFieldToPython<FieldType, typename std::enable_if< ! std::is_same<FieldType, FlagField<typename FieldType::value_type>>::value >::type>
{
   static void exec(py::module_ &m)
   {
      m.def( "realFieldToMesh", &exp_realFieldToSurfaceMesh<FieldType>,
            py::arg("blocks"), py::arg("blockDataName"),  py::arg("fCoord")=0, py::arg("calcNormals") = false, py::arg("targetRank")=0 );
   }
};

template <typename FieldType>
struct ExportFieldToPython<FieldType, typename std::enable_if< std::is_same<FieldType, FlagField<typename FieldType::value_type>>::value >::type>
{
   static void exec(py::module_ &m)
   {
      m.def( "flagFieldToMesh", &exp_flagFieldToSurfaceMesh<FieldType>,
            py::arg("blocks"), py::arg("blockDataName"), py::arg("flagList"), py::arg("calcNormals") = false, py::arg("targetRank")=0  );
   }
};


template<typename... FieldTypes>
struct ExportFieldsToPython;

template<typename FieldType, typename... FieldTypes>
struct ExportFieldsToPython<FieldType, FieldTypes...>
{
   static void exec(py::module_ &m)
   {
      ExportFieldToPython<FieldType>::exec(m);
      ExportFieldsToPython<FieldTypes...>::exec(m);
   }
};

template<>
struct ExportFieldsToPython<>
{
   static void exec(py::module_ &m)
   {}
};

} // namespace internal



template<typename... FieldTypes>
void exportModuleToPython(py::module_ &m)
{
   python_coupling::ModuleScope fieldModule( "postprocessing" );
   
   internal::ExportFieldsToPython<FieldTypes...>::exec(m);
}



} // namespace postprocessing
} // namespace walberla


#endif //WALBERLA_BUILD_WITH_PYTHON
