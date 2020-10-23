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
//! \file CommunicationExport.impl.h
//! \ingroup field
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

#include "python_coupling/PythonWrapper.h"


#ifdef WALBERLA_BUILD_WITH_PYTHON


#include "field/communication/PackInfo.h"
#include "field/communication/StencilRestrictedPackInfo.h"
#include "field/communication/UniformMPIDatatypeInfo.h"

#include "python_coupling/helper/MplHelpers.h"
#include "python_coupling/helper/BoostPythonHelpers.h"
#include "python_coupling/helper/MplHelpers.h"

#include "stencil/D2Q9.h"
#include "stencil/D3Q7.h"
#include "stencil/D3Q15.h"
#include "stencil/D3Q19.h"
#include "stencil/D3Q27.h"


namespace walberla {
namespace field {


namespace internal {
   namespace py = pybind11;

   //===================================================================================================================
   //
   //  createStencilRestrictedPackInfo Export
   //
   //===================================================================================================================

   template< typename FieldType >
   typename std::enable_if<FieldType::F_SIZE == 27, py::object>::type
   createStencilRestrictedPackInfoObject( BlockDataID bdId )
   {
      typedef GhostLayerField<typename FieldType::value_type, 27> GlField_T;
      using field::communication::StencilRestrictedPackInfo;
      return py::object( make_shared< StencilRestrictedPackInfo<GlField_T, stencil::D3Q27> >( bdId) );
   }

   template< typename FieldType >
   typename std::enable_if<FieldType::F_SIZE == 19, py::object>::type
   createStencilRestrictedPackInfoObject( BlockDataID bdId )
   {
      typedef GhostLayerField<typename FieldType::value_type, 19> GlField_T;
      using field::communication::StencilRestrictedPackInfo;
      return py::object( make_shared< StencilRestrictedPackInfo<GlField_T, stencil::D3Q19> >( bdId) );
   }

   template< typename FieldType >
   typename std::enable_if<FieldType::F_SIZE == 15, py::object>::type
   createStencilRestrictedPackInfoObject( BlockDataID bdId )
   {
      typedef GhostLayerField<typename FieldType::value_type, 15> GlField_T;
      using field::communication::StencilRestrictedPackInfo;
      return py::object( make_shared< StencilRestrictedPackInfo<GlField_T, stencil::D3Q15> >( bdId) );
   }

   template< typename FieldType >
   typename std::enable_if<FieldType::F_SIZE == 7, py::object>::type
   createStencilRestrictedPackInfoObject( BlockDataID bdId )
   {
      typedef GhostLayerField<typename FieldType::value_type, 7> GlField_T;
      using field::communication::StencilRestrictedPackInfo;
      return py::object( make_shared< StencilRestrictedPackInfo<GlField_T, stencil::D3Q7> >( bdId) );
   }

   template< typename FieldType >
   typename std::enable_if<FieldType::F_SIZE == 9, py::object>::type
   createStencilRestrictedPackInfoObject( BlockDataID bdId )
   {
      typedef GhostLayerField<typename FieldType::value_type, 9> GlField_T;
      using field::communication::StencilRestrictedPackInfo;
      return py::object( make_shared< StencilRestrictedPackInfo<GlField_T, stencil::D2Q9> >( bdId) );
   }

   template< typename FieldType >
   typename std::enable_if<!(FieldType::F_SIZE == 9  ||
                             FieldType::F_SIZE == 7  ||
                             FieldType::F_SIZE == 15 ||
                             FieldType::F_SIZE == 19 ||
                             FieldType::F_SIZE == 27), py::object>::type
   createStencilRestrictedPackInfoObject( BlockDataID )
   {
      PyErr_SetString( PyExc_ValueError, "This works only for fields with fSize in 7, 9, 15, 19 or 27" );
      throw py::error_already_set();
   }

   FunctionExporterClass( createStencilRestrictedPackInfoObject, py::object( BlockDataID ) );

   template< typename... FieldTypes>
   py::object createStencilRestrictedPackInfo( const shared_ptr<StructuredBlockStorage> & bs,
                                                          const std::string & blockDataName )
   {
      auto bdId = python_coupling::blockDataIDFromString( *bs, blockDataName );
      if ( bs->begin() == bs->end() ) {
         // if no blocks are on this field an arbitrary PackInfo can be returned
         return createStencilRestrictedPackInfoObject< GhostLayerField<real_t,1> > ( bdId );
      }

      IBlock * firstBlock =  & ( * bs->begin() );
      python_coupling::Dispatcher<Exporter_createStencilRestrictedPackInfoObject, FieldTypes... > dispatcher( firstBlock );
      return dispatcher( bdId )( bdId ) ;
   }

   //===================================================================================================================
   //
   //  createPackInfo Export
   //
   //===================================================================================================================

   template< typename FieldType >
   py::object createPackInfoToObject( BlockDataID bdId, uint_t numberOfGhostLayers )
   {
      typedef typename FieldType::value_type T;
      const uint_t F_SIZE = FieldType::F_SIZE;
      typedef GhostLayerField<T, F_SIZE> GlField_T;
      if ( numberOfGhostLayers > 0  )
         return py::object( make_shared< field::communication::PackInfo<GlField_T> >( bdId, numberOfGhostLayers ) );
      else
         return py::object( make_shared< field::communication::PackInfo<GlField_T> >( bdId ) );
   }

   FunctionExporterClass( createPackInfoToObject, py::object( BlockDataID, uint_t  ) );

   template< typename... FieldTypes>
   py::object createPackInfo( const shared_ptr<StructuredBlockStorage> & bs,
                                         const std::string & blockDataName, uint_t numberOfGhostLayers )
   {
      auto bdId = python_coupling::blockDataIDFromString( *bs, blockDataName );
      if ( bs->begin() == bs->end() ) {
         // if no blocks are on this field an arbitrary PackInfo can be returned
         return createPackInfoToObject< GhostLayerField<real_t,1> > ( bdId, numberOfGhostLayers );
      }

      IBlock * firstBlock =  & ( * bs->begin() );
      python_coupling::Dispatcher<Exporter_createPackInfoToObject, FieldTypes... > dispatcher( firstBlock );
      return dispatcher( bdId )( bdId, numberOfGhostLayers ) ;
   }


   //===================================================================================================================
   //
   //  createMPIDatatypeInfo
   //
   //===================================================================================================================


   template< typename FieldType >
   py::object createMPIDatatypeInfoToObject( BlockDataID bdId, uint_t numberOfGhostLayers )
   {
      typedef typename FieldType::value_type T;
      const uint_t F_SIZE = FieldType::F_SIZE;
      typedef GhostLayerField<T, F_SIZE> GlField_T;
      using field::communication::UniformMPIDatatypeInfo;

      if ( numberOfGhostLayers > 0 )
         return py::object( make_shared< UniformMPIDatatypeInfo<GlField_T> >( bdId, numberOfGhostLayers ) );
      else
         return py::object( make_shared< UniformMPIDatatypeInfo<GlField_T> >( bdId ) );
   }

   FunctionExporterClass( createMPIDatatypeInfoToObject, py::object( BlockDataID, uint_t  ) );

   template< typename... FieldTypes>
   py::object createMPIDatatypeInfo( const shared_ptr<StructuredBlockStorage> & bs,
                                                const std::string & blockDataName,
                                                uint_t numberOfGhostLayers)
   {
      auto bdId = python_coupling::blockDataIDFromString( *bs, blockDataName );
      if ( bs->begin() == bs->end() ) {
         // if no blocks are on this field an arbitrary MPIDatatypeInfo can be returned
         return createMPIDatatypeInfoToObject< GhostLayerField<real_t,1> > ( bdId, numberOfGhostLayers );
      }

      IBlock * firstBlock =  & ( * bs->begin() );
      python_coupling::Dispatcher<Exporter_createMPIDatatypeInfoToObject, FieldTypes... > dispatcher( firstBlock );
      return dispatcher( bdId )( bdId, numberOfGhostLayers );
   }

   template< typename T>
   void exportStencilRestrictedPackInfo(py::module_ &m)
   {
      using field::communication::StencilRestrictedPackInfo;
      {
         typedef StencilRestrictedPackInfo<GhostLayerField<T, 9>, stencil::D2Q9> Pi;
         py::class_< Pi, shared_ptr<Pi>, walberla::communication::UniformPackInfo >(m, "StencilRestrictedPackInfo" );
      }
      {
         typedef StencilRestrictedPackInfo<GhostLayerField<T, 7>, stencil::D3Q7> Pi;
         py::class_< Pi, shared_ptr<Pi>, walberla::communication::UniformPackInfo >(m, "StencilRestrictedPackInfo" );
      }
      {
         typedef StencilRestrictedPackInfo<GhostLayerField<T, 15>, stencil::D3Q15> Pi;
         py::class_< Pi, shared_ptr<Pi>, walberla::communication::UniformPackInfo >(m, "StencilRestrictedPackInfo" );
      }
      {
         typedef StencilRestrictedPackInfo<GhostLayerField<T, 19>, stencil::D3Q19> Pi;
         py::class_< Pi, shared_ptr<Pi>, walberla::communication::UniformPackInfo >(m, "StencilRestrictedPackInfo" );
      }
      {
         typedef StencilRestrictedPackInfo<GhostLayerField<T, 27>, stencil::D3Q27> Pi;
         py::class_< Pi, shared_ptr<Pi>, walberla::communication::UniformPackInfo >(m, "StencilRestrictedPackInfo" );
      }

   }

} // namespace internal



namespace py = pybind11;

template<typename... FieldTypes>
void exportCommunicationClasses(py::module_ &m)
{

   internal::exportStencilRestrictedPackInfo<float>(m);
   internal::exportStencilRestrictedPackInfo<double>(m);

   m.def( "createMPIDatatypeInfo",&internal::createMPIDatatypeInfo<FieldTypes...>, py::arg("blocks"), py::arg("blockDataName"), py::arg("numberOfGhostLayers" ) =0  );
   m.def( "createPackInfo",       &internal::createPackInfo<FieldTypes...>,        py::arg("blocks"), py::arg("blockDataName"), py::arg("numberOfGhostLayers" ) =0 );
   m.def( "createStencilRestrictedPackInfo", &internal::createStencilRestrictedPackInfo<FieldTypes...>,
        py::arg("blocks"), py::arg("blockDataName") );
}


} // namespace moduleName
} // namespace walberla




#endif // WALBERLA_BUILD_WITH_PYTHON
