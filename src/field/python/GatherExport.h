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
//! \file GatherExport.h
//! \ingroup field
//! \author Martin Bauer <martin.bauer@fau.de>
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#pragma once

#ifdef WALBERLA_BUILD_WITH_PYTHON
   #include "python_coupling/PythonWrapper.h"
#endif


namespace walberla {
namespace field {
   //*******************************************************************************************************************
   /*! Exports the gather functionality of waLberla
   *
   * With field.gather a corresponding field will the gathered to the specified process. This field can be viewed as a
   * numpy array with field.toArrayOn all other porcesses an empty pybind11::object will be returned.
   *
   * \hint For large scale simulations it is also possible to provide a slice to keep the gathered data low!
   */
   //*******************************************************************************************************************
   namespace py = pybind11;
   template<typename... FieldTypes >
   void exportGatherFunctions(py::module_ &m);



} // namespace field
} // namespace walberla


#include "GatherExport.impl.h"
