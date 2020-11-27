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
//! \file FieldExport.cpp
//! \ingroup cuda
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

// Do not reorder includes - the include order is important
#include "python_coupling/PythonWrapper.h"

#include "core/logging/Logging.h"
#include "cuda/GPUField.h"
#include "cuda/communication/GPUPackInfo.h"
#include "cuda/AddGPUFieldToStorage.h"
#include "cuda/FieldCopy.h"
#include "cuda/GPUField.h"
#include "field/communication/UniformMPIDatatypeInfo.h"
#include "field/AddToStorage.h"
#include "field/python/FieldExport.h"
#include "python_coupling/helper/MplHelpers.h"
#include "python_coupling/helper/BoostPythonHelpers.h"

#include <type_traits>
#include <iostream>

#include <pybind11/stl.h>
#include "pybind11/numpy.h"

namespace walberla {
namespace cuda {



namespace internal {
using namespace pybind11::literals;
   //===================================================================================================================
   //
   //  Field export
   //
   //===================================================================================================================

   template<typename GpuField_T>
   cudaPitchedPtr gpufield_ptr(const GpuField_T & gpuField)
   {
      return gpuField.pitchedPtr();
   }

   template<typename GpuField_T>
   std::string gpufield_dtypeStr(const GpuField_T & )
   {
      return std::string(field::internal::PythonFormatString<typename GpuField_T::value_type>::get());
   }

   struct GpuFieldExporter
   {
      GpuFieldExporter(py::module_& m) : m_(m) {}
      template< typename GpuField_T>
      void operator() ( python_coupling::NonCopyableWrap<GpuField_T> ) const
      {

         typedef typename GpuField_T::value_type T;
         std::string data_type_name = field::internal::PythonFormatString<T>::get();

         std::string class_name = "GpuField_" + data_type_name;
         py::class_<GpuField_T, shared_ptr<GpuField_T>>(m_, class_name.c_str() )
            .def("layout",              &field::internal::field_layout            < GpuField_T > )
            .def("size",                &field::internal::field_size              < GpuField_T > )
            .def("sizeWithGhostLayers", &field::internal::field_sizeWithGhostLayer< GpuField_T > )
            .def("allocSize",           &field::internal::field_allocSize         < GpuField_T > )
            .def("strides",             &field::internal::field_strides           < GpuField_T > )
            .def("offsets",             &field::internal::field_offsets           < GpuField_T > )
            .def("ptr",                 &gpufield_ptr                             < GpuField_T > )
            .def("dtypeStr",            &gpufield_dtypeStr                        < GpuField_T > )
            .def("isPitchedMem",        &GpuField_T::isPitchedMem )
            .def("swapDataPointers",    &field::internal::field_swapDataPointers  < GpuField_T > )
            .def("nrOfGhostLayers",     &GpuField_T::nrOfGhostLayers )
            .def("cloneUninitialized",  &GpuField_T::cloneUninitialized, py::return_value_policy::copy)
            ;


         using field::communication::PackInfo;
         using communication::GPUPackInfo;
         std::string GpuFieldPackInfoName = "GpuFieldPackInfo_" + data_type_name;
         py::class_< GPUPackInfo<GpuField_T>, shared_ptr< GPUPackInfo<GpuField_T> >>(m_, GpuFieldPackInfoName.c_str() );

         using field::communication::UniformMPIDatatypeInfo;
         std::string GpuFieldMPIDataTypeInfoName = "GpuFieldMPIDataTypeInfo_" + data_type_name;
         py::class_< UniformMPIDatatypeInfo<GpuField_T>, shared_ptr< UniformMPIDatatypeInfo<GpuField_T> >>(m_, GpuFieldMPIDataTypeInfoName.c_str() );

      }
      const py::module_& m_;
   };


   //===================================================================================================================
   //
   //  addToStorage
   //
   //===================================================================================================================

   class AddToStorageExporter
   {
   public:
      AddToStorageExporter( const shared_ptr<StructuredBlockStorage> & blocks,
                           const std::string & name, uint_t fs, uint_t gl, Layout layout,
                           bool usePitchedMem )
         : blocks_( blocks ), name_( name ), fs_( fs ),
           gl_(gl),layout_( layout), usePitchedMem_(usePitchedMem), found_(false)
      {}

      template< typename GpuField_T>
      void operator() ( python_coupling::NonCopyableWrap<GpuField_T> )
      {
         typedef typename GpuField_T::value_type T;
         addGPUFieldToStorage<GPUField<T> >(blocks_, name_, fs_, layout_, gl_, usePitchedMem_);
      }

      bool successful() const { return found_; }
   private:
      shared_ptr< StructuredBlockStorage > blocks_;
      std::string name_;
      uint_t fs_;
      uint_t gl_;
      Layout layout_;
      bool usePitchedMem_;
      bool found_;
   };

   template<typename... GpuFields>
   void addToStorage( const shared_ptr<StructuredBlockStorage> & blocks, const std::string & name,
                      uint_t fs, uint_t gl, Layout layout, bool usePitchedMem )
   {
      namespace py = pybind11;
      auto result = make_shared<py::object>();
      AddToStorageExporter exporter( blocks, name, fs, gl, layout, usePitchedMem );
      python_coupling::for_each_noncopyable_type<GpuFields...>( std::ref(exporter) );
   }


   //===================================================================================================================
   //
   //  createPackInfo Export
   //
   //===================================================================================================================

   template< typename GPUField_T >
   py::object createGPUPackInfoToObject( BlockDataID bdId, uint_t numberOfGhostLayers )
   {
      using cuda::communication::GPUPackInfo;
      if ( numberOfGhostLayers > 0  )
         return py::object( make_shared< GPUPackInfo<GPUField_T> >( bdId, numberOfGhostLayers ) );
      else
         return py::object( make_shared< GPUPackInfo<GPUField_T> >( bdId ) );
   }

   FunctionExporterClass( createGPUPackInfoToObject, py::object( BlockDataID, uint_t  ) );

   template< typename GpuFields>
   py::object createPackInfo( const shared_ptr<StructuredBlockStorage> & bs,
                                         const std::string & blockDataName, uint_t numberOfGhostLayers )
   {
      using cuda::communication::GPUPackInfo;

      auto bdId = python_coupling::blockDataIDFromString( *bs, blockDataName );
      if ( bs->begin() == bs->end() ) {
         // if no blocks are on this field an arbitrary PackInfo can be returned
         return createGPUPackInfoToObject< GPUField<real_t> > ( bdId, numberOfGhostLayers );
      }

      IBlock * firstBlock =  & ( * bs->begin() );
      python_coupling::Dispatcher<Exporter_createGPUPackInfoToObject, GpuFields > dispatcher( firstBlock );
      return dispatcher( bdId )( bdId, numberOfGhostLayers ) ;
   }


   //===================================================================================================================
   //
   //  createMPIDatatypeInfo
   //
   //===================================================================================================================


//   template< typename GpuField_T >
//   py::object createMPIDatatypeInfoToObject( BlockDataID bdId, uint_t numberOfGhostLayers )
//   {
//      using field::communication::UniformMPIDatatypeInfo;
//      if ( numberOfGhostLayers > 0 )
//         return py::object( make_shared< UniformMPIDatatypeInfo<GpuField_T> >( bdId, numberOfGhostLayers ) );
//      else
//         return py::object( make_shared< UniformMPIDatatypeInfo<GpuField_T> >( bdId ) );
//   }
//
//   FunctionExporterClass( createMPIDatatypeInfoToObject, py::object( BlockDataID, uint_t  ) );
//
//   template< typename GpuFields>
//   py::object createMPIDatatypeInfo( const shared_ptr<StructuredBlockStorage> & bs,
//                                                const std::string & blockDataName,
//                                                uint_t numberOfGhostLayers)
//   {
//      auto bdId = python_coupling::blockDataIDFromString( *bs, blockDataName );
//      if ( bs->begin() == bs->end() ) {
//         // if no blocks are on this field an arbitrary MPIDatatypeInfo can be returned
//         return createMPIDatatypeInfoToObject< GPUField<real_t> > ( bdId, numberOfGhostLayers );
//      }
//
//      IBlock * firstBlock =  & ( * bs->begin() );
//      python_coupling::Dispatcher<Exporter_createMPIDatatypeInfoToObject, GpuFields > dispatcher( firstBlock );
//      return dispatcher( bdId )( bdId, numberOfGhostLayers );
//   }


   //===================================================================================================================
   //
   //  fieldCopy
   //
   //===================================================================================================================

//struct copyFieldToGpu
//{
//   copyFieldToGpu(py::module_& m) : m_(m) {}
//   template< typename Field_T>
//   void operator() ( python_coupling::NonCopyableWrap<Field_T> ) const
//   {
//      typedef typename Field_T::value_type T;
//
//      std::string class_name = "Field_" + std::string(typeid(T).name()) + "_" + std::to_string(Field_T::F_SIZE);
//      m_.def(
//         class_name.c_str(),
//         [](const shared_ptr< StructuredBlockStorage > & blocks,
//            BlockDataID cpuFieldId, BlockDataID gpuFieldId, bool toGpu) {
//           typedef cuda::GPUField<typename Field_T::value_type> GpuField;
//           if(toGpu)
//              cuda::fieldCpy<GpuField, Field_T>(blocks, gpuFieldId, cpuFieldId);
//           else
//              cuda::fieldCpy<Field_T, GpuField>(blocks, cpuFieldId, gpuFieldId);
//         },
//         "blocks"_a, "cpuFieldId"_a, "gpuFieldId"_a, "toGpu"_a);
//
//   }
//   py::module_& m_;
//};


//   template<typename Field_T>
//   void copyFieldToGpuDispatch(const shared_ptr<StructuredBlockStorage> & bs,
//                               BlockDataID cpuFieldId, BlockDataID gpuFieldId, bool toGpu)
//   {
//      typedef cuda::GPUField<typename Field_T::value_type> GpuField;
//      if(toGpu)
//         cuda::fieldCpy<GpuField, Field_T>(bs, gpuFieldId, cpuFieldId);
//      else
//         cuda::fieldCpy<Field_T, GpuField>(bs, cpuFieldId, gpuFieldId);
//   }

//   struct Exporter_copyFieldToGpuDispatch
//   {
//      typedef std::function<  void( const shared_ptr<StructuredBlockStorage> &, BlockDataID, BlockDataID, bool ) > FunctionType;
//      Exporter_copyFieldToGpuDispatch( const IBlock * block, ConstBlockDataID id )
//         : block_( block ), blockDataID_( id )
//      {}
//      template< typename FieldType >
//      void operator()( walberla::python_coupling::NonCopyableWrap<FieldType> )
//      {
//         if ( block_->isDataClassOrSubclassOf< FieldType > ( blockDataID_ ) )
//             result = static_cast<FunctionType>( Exporter_copyFieldToGpuDispatch< FieldType > );
//      }
//      FunctionType result;
//      const IBlock * block_;
//      const ConstBlockDataID blockDataID_;
//   };
//
//   template< typename... FieldTypes >
//   void transferFields( const shared_ptr<StructuredBlockStorage> & bs,
//                        const std::string & gpuFieldId, const std::string & cpuFieldId, bool toGpu)
//   {
//      if( bs->begin() == bs->end()) {
//         return;
//      };
//
//      auto dstBdId = python_coupling::blockDataIDFromString( *bs, gpuFieldId );
//      auto srcBdId = python_coupling::blockDataIDFromString( *bs, cpuFieldId );
//
//      IBlock * firstBlock =  & ( * bs->begin() );
//      python_coupling::Dispatcher<Exporter_copyFieldToGpuDispatch, FieldTypes...> dispatcher( firstBlock );
//      dispatcher( srcBdId )( bs, srcBdId, dstBdId, toGpu );
//   }
//
//   template< typename FieldTypes>
//   void copyFieldToGpu(const shared_ptr<StructuredBlockStorage> & bs,
//                       const std::string & gpuFieldId, const std::string & cpuFieldId)
//   {
//      transferFields<FieldTypes>(bs, gpuFieldId, cpuFieldId, true);
//   }
//
//   template< typename FieldTypes>
//   void copyFieldToCpu(const shared_ptr<StructuredBlockStorage> & bs,
//                       const std::string & gpuFieldId, const std::string & cpuFieldId)
//   {
//      transferFields<FieldTypes>(bs, gpuFieldId, cpuFieldId, false);
//   }

} // namespace internal


using namespace pybind11::literals;

template<typename... GpuFields>
void exportModuleToPython(py::module_ &m)
{
   // python_coupling::ModuleScope fieldModule( "cuda" );

   // namespace py = pybind11;

   python_coupling::for_each_noncopyable_type<GpuFields...>( internal::GpuFieldExporter(m) );

//   m.def(
//      "addGpuFieldToStorage",
//      [](const shared_ptr< StructuredBlockStorage > & blocks, const std::string & name, uint_t values_per_cell,
//         uint_t ghost_layers, Layout layout, bool usePitchedMemory) {
//        return internal::addToStorage<GpuFields...>(blocks, name, values_per_cell, ghost_layers, layout, usePitchedMemory);
//      },
//      "blocks"_a, "name"_a, "values_per_cell"_a = 1, "ghost_layers"_a = uint_t(1), "layout"_a = zyxf, "alignment"_a = 0);

//
//   m.def( "createMPIDatatypeInfo",&internal::createMPIDatatypeInfo<GpuFields>, ( arg("blocks"), arg("blockDataName"), arg("numberOfGhostLayers" ) =0 ) );
//   m.def( "createPackInfo",       &internal::createPackInfo<GpuFields>,        ( arg("blocks"), arg("blockDataName"), arg("numberOfGhostLayers" ) =0 ) );
}

template<typename... CpuFields >
void exportCopyFunctionsToPython(py::module_ &m)
{
   // python_coupling::for_each_noncopyable_type<CpuFields...>( internal::copyFieldToGpu(m) );
//
//   m.def( "copyFieldToGpu", &internal::copyFieldToGpu<CpuFields>, (arg("blocks"), ("gpuFieldId"), ("cpuFieldId")));
//   m.def( "copyFieldToCpu", &internal::copyFieldToCpu<CpuFields>, (arg("blocks"), ("gpuFieldId"), ("cpuFieldId")));
}




} // namespace cuda
} // namespace walberla


