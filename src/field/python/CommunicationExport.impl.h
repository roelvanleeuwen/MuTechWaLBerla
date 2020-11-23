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

#   include "field/communication/PackInfo.h"
#   include "field/communication/StencilRestrictedPackInfo.h"
#   include "field/communication/UniformMPIDatatypeInfo.h"

#   include "python_coupling/helper/BoostPythonHelpers.h"
#   include "python_coupling/helper/MplHelpers.h"

#   include "stencil/D2Q9.h"
#   include "stencil/D3Q15.h"
#   include "stencil/D3Q19.h"
#   include "stencil/D3Q27.h"
#   include "stencil/D3Q7.h"

#  include <typeinfo>

namespace walberla
{
namespace field
{
namespace internal
{
namespace py = pybind11;

//===================================================================================================================
//
//  createPackInfo Export
//
//===================================================================================================================

struct PackInfoExporter
{
   PackInfoExporter(py::module_& m) : m_(m) {}
   template< typename FieldType >
   void operator()(python_coupling::NonCopyableWrap< FieldType >) const
   {
      typedef typename FieldType::value_type T;
      const uint_t F_SIZE = FieldType::F_SIZE;
      typedef GhostLayerField< T, F_SIZE > GlField_T;
      typedef Field< T, F_SIZE > Field_T;

      typedef field::communication::PackInfo< GlField_T > PackInfo;
      std::string class_name = "PackInfo_" + std::string(typeid(T).name()) + "_" + std::to_string(F_SIZE);

      py::class_< PackInfo, shared_ptr< PackInfo > >(m_, class_name.c_str());
   }
   const py::module_ m_;
};

//===================================================================================================================
//
//  createMPIDatatypeInfo
//
//===================================================================================================================

struct UniformMPIDatatypeInfoExporter
{
   UniformMPIDatatypeInfoExporter(py::module_& m) : m_(m) {}
   template< typename FieldType >
   void operator()(python_coupling::NonCopyableWrap< FieldType >) const
   {
      typedef typename FieldType::value_type T;
      const uint_t F_SIZE = FieldType::F_SIZE;
      typedef GhostLayerField< T, F_SIZE > GlField_T;
      typedef Field< T, F_SIZE > Field_T;

      typedef field::communication::UniformMPIDatatypeInfo< GlField_T > MPIDataTypeInfo;
      std::string class_name =
         "UniformMPIDatatypeInfo_" + std::string(typeid(T).name()) + "_" + std::to_string(FieldType::F_SIZE);

      py::class_< MPIDataTypeInfo, shared_ptr< MPIDataTypeInfo > >(m_, class_name.c_str());
   }
   const py::module_ m_;
};

template< typename T >
void exportStencilRestrictedPackInfo(py::module_& m)
{
   using field::communication::StencilRestrictedPackInfo;
   {
      typedef StencilRestrictedPackInfo< GhostLayerField< T, 9 >, stencil::D2Q9 > Pi;
      py::class_< Pi, shared_ptr< Pi >, walberla::communication::UniformPackInfo >(m, "StencilRestrictedPackInfo_D2Q9");
   }
   {
      typedef StencilRestrictedPackInfo< GhostLayerField< T, 7 >, stencil::D3Q7 > Pi;
      py::class_< Pi, shared_ptr< Pi >, walberla::communication::UniformPackInfo >(m, "StencilRestrictedPackInfo_D3Q7");
   }
   {
      typedef StencilRestrictedPackInfo< GhostLayerField< T, 15 >, stencil::D3Q15 > Pi;
      py::class_< Pi, shared_ptr< Pi >, walberla::communication::UniformPackInfo >(m,
                                                                                   "StencilRestrictedPackInfo_D3Q15");
   }
   {
      typedef StencilRestrictedPackInfo< GhostLayerField< T, 19 >, stencil::D3Q19 > Pi;
      py::class_< Pi, shared_ptr< Pi >, walberla::communication::UniformPackInfo >(m,
                                                                                   "StencilRestrictedPackInfo_D3Q19");
   }
   {
      typedef StencilRestrictedPackInfo< GhostLayerField< T, 27 >, stencil::D3Q27 > Pi;
      py::class_< Pi, shared_ptr< Pi >, walberla::communication::UniformPackInfo >(m,
                                                                                   "StencilRestrictedPackInfo_D3Q27");
   }
}

} // namespace internal

namespace py = pybind11;

template< typename... FieldTypes >
void exportCommunicationClasses(py::module_& m)
{
   // internal::exportStencilRestrictedPackInfo< float >(m);
   internal::exportStencilRestrictedPackInfo< double >(m);

   python_coupling::for_each_noncopyable_type< FieldTypes... >(internal::UniformMPIDatatypeInfoExporter(m));
   python_coupling::for_each_noncopyable_type< FieldTypes... >(internal::PackInfoExporter(m));
}

} // namespace field
} // namespace walberla

#endif // WALBERLA_BUILD_WITH_PYTHON
