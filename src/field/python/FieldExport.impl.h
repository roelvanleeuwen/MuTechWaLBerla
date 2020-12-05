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
//! \ingroup field
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================
#include "core/VectorTrait.h"
#include "core/logging/Logging.h"

#include "field/AddToStorage.h"
#include "field/Field.h"
#include "field/FlagField.h"
#include "field/GhostLayerField.h"
#include "field/communication/PackInfo.h"
#include "field/communication/UniformMPIDatatypeInfo.h"
#include "field/python/GatherExport.h"
#include "field/vtk/FlagFieldMapping.h"
#include "field/vtk/VTKWriter.h"

#include "python_coupling/PythonWrapper.h"
#include "python_coupling/helper/MplHelpers.h"

#include <iostream>
#include <type_traits>

#include "pybind11/numpy.h"

namespace walberla
{
namespace field
{
namespace internal
{
namespace py = pybind11;

template<class T> struct PythonFormatString                    { inline static char * get() { static char value [] = "B"; return value; } };

template<>        struct PythonFormatString<double>            { inline static char * get() { static char value [] = "d"; return value; } };
template<>        struct PythonFormatString<float>             { inline static char * get() { static char value [] = "f"; return value; } };
template<>        struct PythonFormatString<unsigned short>    { inline static char * get() { static char value [] = "H"; return value; } };
template<>        struct PythonFormatString<int>               { inline static char * get() { static char value [] = "i"; return value; } };
template<>        struct PythonFormatString<unsigned int>      { inline static char * get() { static char value [] = "I"; return value; } };
template<>        struct PythonFormatString<long>              { inline static char * get() { static char value [] = "l"; return value; } };
template<>        struct PythonFormatString<unsigned long>     { inline static char * get() { static char value [] = "L"; return value; } };
template<>        struct PythonFormatString<long long>         { inline static char * get() { static char value [] = "q"; return value; } };
template<>        struct PythonFormatString<unsigned long long>{ inline static char * get() { static char value [] = "Q"; return value; } };
template<>        struct PythonFormatString<int8_t>            { inline static char * get() { static char value [] = "c"; return value; } };
template<>        struct PythonFormatString<int16_t>           { inline static char * get() { static char value [] = "h"; return value; } };
template<>        struct PythonFormatString<uint8_t>           { inline static char * get() { static char value [] = "C"; return value; } };

//===================================================================================================================
//
//  Aligned Allocation
//
//===================================================================================================================

template< typename T >
shared_ptr< field::FieldAllocator< T > > getAllocator(uint_t alignment)
{
   if (alignment == 0)
      return shared_ptr< field::FieldAllocator< T > >(); // leave to default - auto-detection of alignment
   else if (alignment == 16)
      return make_shared< field::AllocateAligned< T, 16 > >();
   else if (alignment == 32)
      return make_shared< field::AllocateAligned< T, 32 > >();
   else if (alignment == 64)
      return make_shared< field::AllocateAligned< T, 64 > >();
   else if (alignment == 128)
      return make_shared< field::AllocateAligned< T, 128 > >();
   else
   {
      throw py::value_error("Alignment parameter has to be one of 0, 16, 32, 64, 128.");
      return shared_ptr< field::FieldAllocator< T > >();
   }
}

template< typename GhostLayerField_T >
class GhostLayerFieldDataHandling : public field::BlockDataHandling< GhostLayerField_T >
{
 public:
   typedef typename GhostLayerField_T::value_type Value_T;

   GhostLayerFieldDataHandling(const weak_ptr< StructuredBlockStorage >& blocks, const uint_t nrOfGhostLayers,
                               const Value_T& initValue, const Layout layout, uint_t alignment = 0)
      : blocks_(blocks), nrOfGhostLayers_(nrOfGhostLayers), initValue_(initValue), layout_(layout),
        alignment_(alignment)
   {}

   GhostLayerField_T* allocate(IBlock* const block)
   {
      auto blocks = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(blocks, "Trying to access 'AlwaysInitializeBlockDataHandling' for a block "
                                         "storage object that doesn't exist anymore");
      GhostLayerField_T* field = new GhostLayerField_T(
         blocks->getNumberOfXCells(*block), blocks->getNumberOfYCells(*block), blocks->getNumberOfZCells(*block),
         nrOfGhostLayers_, initValue_, layout_, getAllocator< Value_T >(alignment_));
      return field;
   }

   GhostLayerField_T* reallocate(IBlock* const block) { return allocate(block); }

 private:
   weak_ptr< StructuredBlockStorage > blocks_;

   uint_t nrOfGhostLayers_;
   Value_T initValue_;
   Layout layout_;
   uint_t alignment_;
};

//===================================================================================================================
//
//  Field functions redefined for easier export
//
//===================================================================================================================

static inline Cell tupleToCell(py::tuple& tuple)
{
   return Cell(py::cast< cell_idx_t >(tuple[0]), py::cast< cell_idx_t >(tuple[1]), py::cast< cell_idx_t >(tuple[2]));
}

template< typename Field_T >
void field_setCellXYZ(Field_T& field, py::tuple args, const typename Field_T::value_type& value)
{
   using namespace py;

   if (len(args) < 3 || len(args) > 4)
   {
      throw py::value_error("3 or 4 indices required");
   }

   cell_idx_t f = 0;
   if (len(args) == 4) f = py::cast< cell_idx_t >(args[3]);

   Cell cell = tupleToCell(args);
   if (!field.coordinatesValid(cell[0], cell[1], cell[2], f))
   {
      throw py::value_error("Field indices out of bounds");
   }
   field(cell, f) = value;
}

template< typename Field_T >
typename Field_T::value_type field_getCellXYZ(Field_T& field, py::tuple args)
{
   using namespace py;
   if (len(args) < 3 || len(args) > 4)
   {
      throw py::value_error("3 or 4 indices required");
   }

   cell_idx_t f = 0;
   if (len(args) == 4) f = py::cast< cell_idx_t >(args[3]);

   Cell cell = tupleToCell(args);
   if (!field.coordinatesValid(cell[0], cell[1], cell[2], f))
   {
      throw py::value_error("Field indices out of bounds");
   }

   return field(cell, f);
}

template< typename Field_T >
py::object field_size(const Field_T& field)
{
   return py::make_tuple(field.xSize(), field.ySize(), field.zSize(), field.fSize());
}

template< typename GlField_T >
py::tuple field_sizeWithGhostLayer(const GlField_T& field)
{
   return py::make_tuple(field.xSizeWithGhostLayer(), field.ySizeWithGhostLayer(), field.zSizeWithGhostLayer(),
                         field.fSize());
}

template< typename Field_T >
py::tuple field_allocSize(const Field_T& field)
{
   return py::make_tuple(field.xAllocSize(), field.yAllocSize(), field.zAllocSize(), field.fAllocSize());
}

template< typename Field_T >
py::tuple field_strides(const Field_T& field)
{
   return py::make_tuple(field.xStride(), field.yStride(), field.zStride(), field.fStride());
}

template< typename Field_T >
py::tuple field_offsets(const Field_T& field)
{
   return py::make_tuple(field.xOff(), field.yOff(), field.zOff());
}

template< typename Field_T >
py::object field_layout(const Field_T& f)
{
   return py::cast(f.layout());
}

template< typename Field_T >
void field_swapDataPointers(Field_T& f1, Field_T& f2)
{
   if (!f1.hasSameAllocSize(f2) || !f1.hasSameSize(f2) || f1.layout() != f2.layout())
   {
      throw py::value_error("The data of fields with different sizes or layout cannot be swapped");
   }
   f1.swapDataPointers(f2);
}

template< typename Field_T >
py::object copyAdaptorToField(const Field_T& f)
{
   typedef GhostLayerField< typename Field_T::value_type, Field_T::F_SIZE > ResField;
   auto res = make_shared< ResField >(f.xSize(), f.ySize(), f.zSize(), f.nrOfGhostLayers());

   auto srcIt = f.beginWithGhostLayerXYZ();
   auto dstIt = res->beginWithGhostLayerXYZ();
   while (srcIt != f.end())
   {
      for (cell_idx_t fCoord = 0; fCoord < cell_idx_c(Field_T::F_SIZE); ++fCoord)
         dstIt.getF(fCoord) = srcIt.getF(fCoord);

      ++srcIt;
      ++dstIt;
   }
   return py::cast(res);
}

//===================================================================================================================
//
//  Field export
//
//===================================================================================================================
// ff.getFlagUID(flag).getIdentifier()
//template< typename T >
//void exportFlagField(const py::module_& m)
//{
//   py::class_< FlagField< T >, shared_ptr< FlagField< T > > >(m, "FlagField")
//      .def("registerFlag",
//           [](const std::string & flag_name){
//              return FlagField< T >::registerFlag(FlagUID(flag_name));
//           }, py::arg("flag_name"))
//      .def("flag",
//           [](const std::string& flag){
//             return FlagField< T >::getFlag(flag);
//           }, py::arg("flag"))
//      .def("flagName",
//           [](const std::string& flag){
//             return FlagField< T >::getFlagUID(flag).getIdentifier();
//           }, py::arg("flag"))
//      .def("flags",
//                    [](){
//                      std::vector< FlagUID > flags;
//                      FlagField< T >::getAllRegisteredFlags(flags);
//                      py::list result;
//
//                      for (auto i = flags.begin(); i != flags.end(); ++i)
//                         result.append(i->toString());
//                      return result;
//                    })
//      .def("flagMap",
//                    [](){
//                      std::vector< FlagUID > flags;
//                      FlagField< T >::getAllRegisteredFlags(flags);
//                      py::dict result;
//
//                      for (auto i = flags.begin(); i != flags.end(); ++i)
//                         result[i->toString().c_str()] = FlagField< T >::getFlag(*i);
//                      return result;
//                    });
//}

template< typename Field_T >
py::array_t< typename Field_T::value_type > toNumpyArray(const Field_T& field)
{
   using T    = typename Field_T::value_type;
   const T* ptr = field.dataAt(0, 0, 0, 0);

   if (field.fSize() == 1)
   {
      return pybind11::array_t< T, 0 >({ field.xSize(), field.ySize(), field.zSize() },
                                       { static_cast< size_t >(field.xStride()) * sizeof(T),
                                         static_cast< size_t >(field.yStride()) * sizeof(T),
                                         static_cast< size_t >(field.zStride()) * sizeof(T) },
                                       ptr, py::cast(field));
   }
   else
   {
      return pybind11::array_t< T, 0 >(
         { field.xSize(), field.ySize(), field.zSize(), field.fSize() },
         { static_cast< size_t >(field.xStride()) * sizeof(T), static_cast< size_t >(field.yStride()) * sizeof(T),
           static_cast< size_t >(field.zStride()) * sizeof(T), static_cast< size_t >(field.fStride()) * sizeof(T) },
         ptr, py::cast(field));
   }
}


struct FieldExporter
{
   FieldExporter(py::module_& m) : m_(m) {}
   template< typename FieldType >
   void operator()(python_coupling::NonCopyableWrap< FieldType >) const
   {
      typedef typename FieldType::value_type T;
      const uint_t F_SIZE = FieldType::F_SIZE;
      typedef GhostLayerField< T, F_SIZE > GlField_T;
      typedef Field< T, F_SIZE > Field_T;

      std::string data_type_name = PythonFormatString<T>::get();

      std::string class_name = "Field_" + data_type_name + "_" + std::to_string(FieldType::F_SIZE);

      py::class_< Field_T, shared_ptr< Field_T > >(m_, class_name.c_str())
         .def("layout", &field_layout< Field_T >)
         .def("size", &field_size< Field_T >)
         .def("allocSize", &field_allocSize< Field_T >)
         .def("strides", &field_strides< Field_T >)
         .def("offsets", &field_offsets< Field_T >)
         .def("clone", &Field_T::clone, py::return_value_policy::copy)
         .def("cloneUninitialized", &Field_T::cloneUninitialized, py::return_value_policy::copy)
         .def("swapDataPointers", &field_swapDataPointers< Field_T >)
         // .def("__getitem__", py::object([](){}&toNumpyArray< T, F_SIZE >))
         // .def("__setitem__", &field_setCellXYZ< Field_T >)
         .def("__array__", &toNumpyArray< Field_T >);

      std::string class_nameGL =
         "GhostLayerField_" + data_type_name + "_" + std::to_string(FieldType::F_SIZE);

      py::class_< GlField_T, shared_ptr< GlField_T >, Field_T >(m_, class_nameGL.c_str())
         .def("sizeWithGhostLayer", &GlField_T::xSizeWithGhostLayer)
         .def("nrOfGhostLayers", &GlField_T::nrOfGhostLayers);
//
//      exportFlagField< T >(m_);
//
//      // Field Buffer type
//
      using field::communication::PackInfo;
      std::string FieldPackInfo_name = "FieldPackInfo_" + data_type_name + "_" + std::to_string(FieldType::F_SIZE);
      py::class_< PackInfo< GlField_T >, shared_ptr< PackInfo< GlField_T > > >(m_, FieldPackInfo_name.c_str());

//      using field::communication::UniformMPIDatatypeInfo;
//      std::string FieldMPIDataTypeInfo_name = "FieldMPIDataTypeInfo_" + data_type_name + "_" + std::to_string(FieldType::F_SIZE);
//      py::class_< UniformMPIDatatypeInfo< GlField_T >, shared_ptr< UniformMPIDatatypeInfo< GlField_T > > >(
//         m_, FieldMPIDataTypeInfo_name.c_str());
   }
   const py::module_& m_;
};


struct FieldAllocatorExporter
{
   FieldAllocatorExporter(py::module_& m) : m_(m) {}
   template< typename T >
   void operator()(python_coupling::NonCopyableWrap< T >) const
   {
      std::string data_type_name = PythonFormatString<T>::get();
      std::string class_nameFieldAllocator = "FieldAllocator_" + data_type_name;
      py::class_< FieldAllocator< T >, shared_ptr< FieldAllocator< T > > >(m_, class_nameFieldAllocator.c_str())
         .def("incrementReferenceCount", &FieldAllocator< T >::incrementReferenceCount)
         .def("decrementReferenceCount", &FieldAllocator< T >::decrementReferenceCount);
   }
   const py::module_& m_;
};


struct GhostLayerFieldAdaptorExporter
{
   GhostLayerFieldAdaptorExporter(const py::module_& m, const std::string& name) : m_(m), name_(name) {}

   template< typename Adaptor >
   void operator()(python_coupling::NonCopyableWrap< Adaptor >) const
   {
      using namespace py;

      py::class_< Adaptor, shared_ptr< Adaptor > >(m_, name_.c_str())
         .def_readonly("size", &field_size< Adaptor >)
         .def_readonly("sizeWithGhostLayer", &field_sizeWithGhostLayer< Adaptor >)
         .def_readonly("nrOfGhostLayers", &Adaptor::nrOfGhostLayers)
         .def("__getitem__", &field_getCellXYZ< Adaptor >)
         .def("copyToField", &copyAdaptorToField< Adaptor >);
   }

   const py::module_& m_;
   std::string name_;
};

//===================================================================================================================
//
//  addToStorage
//
//===================================================================================================================

class AddToStorageExporter
{
 public:
   AddToStorageExporter(const shared_ptr< StructuredBlockForest >& blocks, const std::string& name, uint_t fs,
                        uint_t gl, Layout layout, uint_t alignment)
      : blocks_(blocks), name_(name), fs_(fs), gl_(gl), layout_(layout), alignment_(alignment), found_(true)
   {}

   template< typename FieldType >
   // TODO: Due to the NonCopyableWrap the operator needs to be set const thus found_ can not be changed to indicated
   // if there was an error when adding the field. Why do we need the NonCopyableWrap at all?
   void operator()(python_coupling::NonCopyableWrap<FieldType>) const
   {
      using namespace py;
      typedef typename FieldType::value_type T;
      const uint_t F_SIZE = FieldType::F_SIZE;

      if (F_SIZE != fs_) return;

      typedef internal::GhostLayerFieldDataHandling< GhostLayerField< T, F_SIZE > > DataHandling;
      auto dataHandling = walberla::make_shared< DataHandling >(blocks_, gl_, T(), layout_, alignment_);
      blocks_->addBlockData(dataHandling, name_);
   }

   bool successful() const { return found_; }

 private:
   shared_ptr< StructuredBlockStorage > blocks_;
   std::string name_;
   uint_t fs_;
   uint_t gl_;
   Layout layout_;
   uint_t alignment_;
   bool found_;
};

template< typename... FieldTypes >
void addToStorage(const shared_ptr< StructuredBlockForest >& blocks, const std::string& name,
                  uint_t fs, uint_t gl, Layout layout, uint_t alignment)
{
   using namespace py;

   auto result = make_shared< py::object >();
   AddToStorageExporter exporter(blocks, name, fs, gl, layout, alignment);
   python_coupling::for_each_noncopyable_type< FieldTypes... >(exporter);

   if (!exporter.successful())
   {
      throw py::value_error("Adding Field failed.");
   }
}

inline void addFlagFieldToStorage(const shared_ptr< StructuredBlockStorage >& blocks, const std::string& name,
                                  uint_t nrOfBits, uint_t gl)
{
   if (nrOfBits == 8)
      field::addFlagFieldToStorage< FlagField< uint8_t > >(blocks, name, gl);
   else if (nrOfBits == 16)
      field::addFlagFieldToStorage< FlagField< uint16_t > >(blocks, name, gl);
   else if (nrOfBits == 32)
      field::addFlagFieldToStorage< FlagField< uint32_t > >(blocks, name, gl);
   else if (nrOfBits == 64)
      field::addFlagFieldToStorage< FlagField< uint64_t > >(blocks, name, gl);
   else
   {
      PyErr_SetString(PyExc_ValueError, "Allowed values for number of bits are: 8,16,32,64");
      throw py::error_already_set();
   }
}

//===================================================================================================================
//
//  createVTKWriter
//
//===================================================================================================================

class CreateVTKWriterExporter
{
 public:
   CreateVTKWriterExporter(py::module_& m)
      : m_(m)
   {}

   template< typename FieldType >
   void operator()(python_coupling::NonCopyableWrap<FieldType>) const
   {
      typedef typename FieldType::value_type T;
      typedef field::VTKWriter< FieldType > VTKWriter;
      std::string data_type_name = PythonFormatString<T>::get();

      std::string class_name = "VTKWriter_" + data_type_name + "_" + std::to_string(FieldType::F_SIZE);

      py::class_< VTKWriter, shared_ptr<VTKWriter> >(m_, class_name.c_str() );

   }

 private:
   const py::module_ m_;
};


//===================================================================================================================
//
//  createBinarizationFieldWriter
//
//===================================================================================================================

class CreateBinarizationVTKWriterExporter
{
 public:
   CreateBinarizationVTKWriterExporter(py::module_& m)
      : m_(m)
   {}

   template< typename FieldType >
   void operator()()
   {
      typedef typename FieldType::value_type T;
      std::string data_type_name = PythonFormatString<T>::get();
      typedef field::BinarizationFieldWriter< FieldType > VTKWriter;
      std::string class_name = "BinarizationFieldWriter_" + data_type_name + "_" + std::to_string(FieldType::F_SIZE);

      py::class_< VTKWriter, shared_ptr<VTKWriter> >(m_, class_name.c_str() );
   }
 private:
   const py::module_ m_;
};

} // namespace internal

namespace py = pybind11;
template< typename... FieldTypes >
void exportFields(py::module_& m)
{
   using namespace py;

   py::enum_< Layout >(m, "Layout").value("fzyx", fzyx).value("zyxf", zyxf).export_values();

   python_coupling::for_each_noncopyable_type< FieldTypes... >(internal::FieldExporter(m));
   python_coupling::for_each_noncopyable_type< real_t, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t >(internal::FieldAllocatorExporter(m));

   m.def(
      "addToStorage",
      [](const shared_ptr< StructuredBlockForest > & blocks, const std::string & name, uint_t values_per_cell,
         uint_t ghost_layers, Layout layout, uint_t alignment) {
         return internal::addToStorage< FieldTypes... >(blocks, name, values_per_cell, ghost_layers, layout, alignment);
      },
      "blocks"_a, "name"_a, "values_per_cell"_a = 1, "ghost_layers"_a = uint_t(1), "layout"_a = zyxf, "alignment"_a = 0);
//
   python_coupling::for_each_noncopyable_type< FieldTypes... >(internal::CreateVTKWriterExporter(m));
//   python_coupling::for_each_noncopyable_type< FieldTypes... >(internal::CreateBinarizationVTKWriterExporter(m));
}

} // namespace field
} // namespace walberla
