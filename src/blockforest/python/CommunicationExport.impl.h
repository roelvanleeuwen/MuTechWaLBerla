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
//! \ingroup blockforest
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

#include "blockforest/communication/UniformBufferedScheme.h"
#include "blockforest/communication/UniformDirectScheme.h"

#include "python_coupling/helper/MplHelpers.h"

#include <pybind11/pybind11.h>

namespace walberla
{
namespace blockforest
{
namespace py = pybind11;
namespace internal
{
//===================================================================================================================
//
//  UniformBufferedScheme
//
//===================================================================================================================

/// the purpose of this class could also be solved by adding return_internal_reference to "createUniformDirectScheme"
/// however this is not easily possible since it returns not a reference but an py::object
template< typename Stencil >
class UniformBufferedSchemeWrapper : public blockforest::communication::UniformBufferedScheme< Stencil >
{
 public:
   UniformBufferedSchemeWrapper(const shared_ptr< StructuredBlockForest >& bf, const int tag)
      : blockforest::communication::UniformBufferedScheme< Stencil >(bf, tag), blockforest_(bf)
   {}

 private:
   shared_ptr< StructuredBlockForest > blockforest_;
};

struct UniformBufferedSchemeExporter
{
   UniformBufferedSchemeExporter(py::module_& m) : m_(m) {}
   template< typename Stencil >
   void operator()(python_coupling::NonCopyableWrap< Stencil >) const
   {
      typedef UniformBufferedSchemeWrapper< Stencil > UBS;
      std::string class_name = "UniformBufferedScheme" + std::string(Stencil::NAME);

      py::class_< UBS, shared_ptr< UBS > >(m_, class_name.c_str())
         .def("__call__", &UBS::operator())
         .def("communicate", &UBS::communicate)
         .def("startCommunication", &UBS::startCommunication)
         .def("wait", &UBS::wait)
         .def("addPackInfo", &UBS::addPackInfo)
         .def("addDataToCommunicate", &UBS::addDataToCommunicate)
         .def("localMode", &UBS::localMode)
         .def("setLocalMode", &UBS::setLocalMode);
   }
   const py::module_& m_;
};

//===================================================================================================================
//
//  UniformDirectScheme
//
//===================================================================================================================

template< typename Stencil >
class UniformDirectSchemeWrapper : public blockforest::communication::UniformDirectScheme< Stencil >
{
 public:
   UniformDirectSchemeWrapper(const shared_ptr< StructuredBlockForest >& bf, const int tag)
      : blockforest::communication::UniformDirectScheme< Stencil >(
           bf, shared_ptr< walberla::communication::UniformMPIDatatypeInfo >(), tag),
        blockforest_(bf)
   {}

 private:
   shared_ptr< StructuredBlockForest > blockforest_;
};

struct UniformDirectSchemeExporter
{
   UniformDirectSchemeExporter(py::module_& m) : m_(m) {}
   template< typename Stencil >
   void operator()(python_coupling::NonCopyableWrap< Stencil >) const
   {
      typedef UniformDirectSchemeWrapper< Stencil > UDS;
      std::string class_name = "UniformDirectScheme_" + std::string(Stencil::NAME);

      py::class_< UDS, shared_ptr<UDS>>(m_, class_name.c_str() )
         .def("__call__", &UDS::operator())
         .def("communicate", &UDS::communicate)
         .def("startCommunication", &UDS::startCommunication)
         .def("wait", &UDS::wait)
         .def("addDataToCommunicate", &UDS::addDataToCommunicate);
   }
   const py::module_ m_;
};

} // namespace internal

template< typename... Stencils >
void exportUniformDirectScheme(py::module_& m)
{
   using namespace py;

   python_coupling::for_each_noncopyable_type< Stencils... >(internal::UniformDirectSchemeExporter(m));
}

template< typename... Stencils >
void exportUniformBufferedScheme(py::module_& m)
{
   py::enum_< LocalCommunicationMode >(m, "LocalCommunicationMode")
      .value("START", START)
      .value("WAIT", WAIT)
      .value("BUFFER", BUFFER)
      .export_values();

   python_coupling::for_each_noncopyable_type< Stencils... >(internal::UniformBufferedSchemeExporter(m));
}

} // namespace blockforest
} // namespace walberla