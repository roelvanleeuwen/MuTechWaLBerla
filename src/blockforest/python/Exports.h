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
//! \file PythonExports.h
//! \ingroup blockforest
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

#pragma once

#include "waLBerlaDefinitions.h"


#ifdef WALBERLA_BUILD_WITH_PYTHON

#include "blockforest/python/CommunicationExport.h"
#include <pybind11/pybind11.h>


namespace walberla {
namespace blockforest {

   namespace py = pybind11;

   void exportBlockForest(py::module_ &m);


   template<typename... Stencils>
   void exportModuleToPython(py::module_ &m)
   {
      exportBlockForest(m);
      exportUniformBufferedScheme<Stencils...>(m);
      exportUniformDirectScheme<Stencils...>(m);
   }

} // namespace blockforest
} // namespace walberla


#endif // WALBERLA_BUILD_WITH_PYTHON
