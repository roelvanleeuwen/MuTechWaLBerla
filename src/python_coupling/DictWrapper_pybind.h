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
//! \file DictWrapper_pybind.h
//! \ingroup python_coupling
//! \author Martin Bauer <martin.bauer@fau.de>
//! \brief Wrapper to store and extract values from boost::python::dict
//
//! \warning: if you include this header you also have to include Python.h as first header in your
//!           cpp file
//
//======================================================================================================================

#pragma once

// #include "PythonWrapper.h"
#include <pybind11/pybind11.h>
#include "core/DataTypes.h"
#include "core/Abort.h"

#include <functional>


namespace walberla {
namespace python_coupling {
   namespace py = pybind11;


   class DictWrapper_pybind
   {
   public:

      //** Expose Data *************************************************************************************************
      /*! \name Expose Data */
      //@{
      template<typename T>  inline void exposePtr(const char* name, T * var );
      template<typename T>  inline void exposePtr(const char* name, const shared_ptr<T> & var );
      template<typename T>  void exposeValue     ( const char* name, const T & var );
      //@}
      //****************************************************************************************************************


      //** Get Data  ***************************************************************************************************
      /*! \name Get Data */
      //@{
      template<typename T> inline T    get( const char* name );
      template<typename T> inline bool has( const char* name );
      template<typename T> inline bool checkedGet( const char* name, T output );
      //@}
      //****************************************************************************************************************


#ifdef WALBERLA_BUILD_WITH_PYTHON
   public:
            py::dict & dict()        { return d_; }
      const py::dict & dict() const  { return d_; }
   protected:
            py::dict d_;
#endif
   };


} // namespace python_coupling
} // namespace walberla



#ifdef WALBERLA_BUILD_WITH_PYTHON

#include "DictWrapper_pybind.impl.h"

#else

// Stubs when Python is not built
namespace walberla {
namespace python_coupling {

   template<typename T> void DictWrapper_pybind::exposePtr( const std::string & , T *  ) {}
   template<typename T> void DictWrapper_pybind::exposePtr( const std::string & , const shared_ptr<T> & ) {}
   template<typename T> void DictWrapper_pybind::exposeValue( const std::string & , const T &  ) {}

   template<typename T> bool DictWrapper_pybind::has( const std::string &  )                      { return false;  }
   template<typename T> bool DictWrapper_pybind::checkedGet( const std::string & name, T output ) { return false; }

   template<typename T> T DictWrapper_pybind::get( const std::string & ) {
      WALBERLA_ABORT("Not available - waLBerla was built without Python suppport");
#ifdef __IBMCPP__
      return *(reinterpret_cast< T * >( NULL )); // silencing incorrect IBM compiler warning
#endif
   }


} // namespace python_coupling
} // namespace walberla

#endif

