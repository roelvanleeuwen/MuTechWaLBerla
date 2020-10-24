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

namespace walberla {
namespace blockforest {
namespace py = pybind11;
namespace internal
{

//===================================================================================================================
//
//  UniformBufferedScheme
//
//===================================================================================================================

/// for details see documentation in core/WeakPtrWrapper.h
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
   template< typename Stencil >
   void operator()(py::module_& m)
   {
      typedef UniformBufferedSchemeWrapper< Stencil > UBS;

      py::class_< UBS, shared_ptr< UBS > >(m, "UniformBufferedScheme")
         .def("__call__", &UBS::operator())
         .def("communicate", &UBS::communicate)
         .def("startCommunication", &UBS::startCommunication)
         .def("wait", &UBS::wait)
         .def("addPackInfo", &UBS::addPackInfo)
         .def("addDataToCommunicate", &UBS::addDataToCommunicate)
         .def("localMode", &UBS::localMode)
         .def("setLocalMode", &UBS::setLocalMode);
   }
};

class UniformBufferedSchemeCreator
{
 public:
   UniformBufferedSchemeCreator(const shared_ptr< StructuredBlockForest >& bf, const std::string& stencilName,
                                const int tag)
      : blockforest_(bf), stencilName_(stencilName), tag_(tag)
   {}

   template< typename Stencil >
   void operator()()
   {
      if (std::string(Stencil::NAME) == stencilName_)
      { result_ = object(make_shared< UniformBufferedSchemeWrapper< Stencil > >(blockforest_, tag_)); }
   }

   py::object getResult() { return result_; }

 private:
   shared_ptr< StructuredBlockForest > blockforest_;
   std::string stencilName_;
   const int tag_;
   py::object result_;
};

template< typename Stencils >
py::object createUniformBufferedScheme(const shared_ptr< StructuredBlockForest >& bf, const std::string& stencil,
                                       const int tag)
{
   UniformBufferedSchemeCreator creator(bf, stencil, tag);
   python_coupling::for_each_noncopyable_type< Stencils >  ( std::ref(creator) );

   if (py::isinstance< py::object() >(creator.getResult()))
   {
      throw py::cast_error("Unknown stencil.");
   }
   return creator.getResult();
}

//===================================================================================================================
//
//  UniformDirectScheme
//
//===================================================================================================================

 template<typename Stencil>
 class UniformDirectSchemeWrapper : public blockforest::communication::UniformDirectScheme<Stencil>
{
 public:
   UniformDirectSchemeWrapper( const shared_ptr<StructuredBlockForest> & bf, const int tag )
      : blockforest::communication::UniformDirectScheme<Stencil>( bf, shared_ptr<walberla::communication::UniformMPIDatatypeInfo>(), tag),
        blockforest_( bf)
   {}
 private:
   shared_ptr< StructuredBlockForest > blockforest_;
};

 struct UniformDirectSchemeExporter
{
   template<typename Stencil>
   void operator() ( py::module_& m )
   {
      typedef UniformDirectSchemeWrapper<Stencil> UDS;

      py::class_< UDS, shared_ptr<UDS>>(m, "UniformDirectScheme" )
         .def( "__call__",             &UDS::operator()             )
         .def( "communicate",          &UDS::communicate            )
         .def( "startCommunication",   &UDS::startCommunication     )
         .def( "wait",                 &UDS::wait                   )
         .def( "addDataToCommunicate", &UDS::addDataToCommunicate   )
         ;
   }
};

 class UniformDirectSchemeCreator
{
 public:
   UniformDirectSchemeCreator( const shared_ptr<StructuredBlockForest> & bf,
                               const std::string & stencilName,
                               const int tag )
      : blockforest_( bf), stencilName_( stencilName ), tag_( tag )
   {}

   template<typename Stencil>
   void operator() ( )
   {
      if ( std::string(Stencil::NAME) == stencilName_ ) {
         result_ = py::object ( make_shared< UniformDirectSchemeWrapper<Stencil> > ( blockforest_, tag_ ) );
      }
   }

   py::object getResult() { return result_; }
 private:
   shared_ptr<StructuredBlockForest> blockforest_;
   std::string stencilName_;
   const int tag_;
   py::object result_;
};

 template<typename Stencils>
 py::object createUniformDirectScheme( const shared_ptr<StructuredBlockForest> & bf,
                                                 const std::string & stencil, const int tag )
{
   UniformDirectSchemeCreator creator( bf, stencil, tag );
   python_coupling::for_each_noncopyable_type< Stencils >  ( std::ref(creator) );

   if ( py::isinstance< py::object() >(creator.getResult()) )
   {
      throw py::cast_error("Unknown stencil.");
   }
   return creator.getResult();
}

}

template< typename Stencils >
void exportUniformBufferedScheme(py::module_& m)
{
   py::enum_< LocalCommunicationMode >(m, "LocalCommunicationMode")
      .value("START", START)
      .value("WAIT", WAIT)
      .value("BUFFER", BUFFER)
      .export_values();

   python_coupling::for_each_noncopyable_type< Stencils >  ( internal::UniformBufferedSchemeExporter() );

   m.def("createUniformBufferedScheme", &internal::createUniformBufferedScheme< Stencils >,
         py::arg("blockForest"), py::arg("stencilName"), py::arg("tag") = 778);
}

 template<typename Stencils>
 void exportUniformDirectScheme(py::module_& m)
{
   using namespace py;

   python_coupling::for_each_noncopyable_type< Stencils >  ( internal::UniformDirectSchemeExporter() );

   m.def( "createUniformDirectScheme", &internal::createUniformDirectScheme<Stencils>,
         py::arg("blockForest"), py::arg("stencilName"), py::arg("tag")=778 );
}


} // namespace blockforest
} // namespace walberla