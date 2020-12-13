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
//! \file SliceToCellInterval.h
//! \ingroup python_coupling
//! \author Martin Bauer <martin.bauer@fau.de>
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/cell/CellInterval.h"

#include "domain_decomposition/StructuredBlockStorage.h"

#include "python_coupling/PythonWrapper.h"
namespace py = pybind11;
namespace walberla
{
namespace python_coupling
{
namespace internal
{
inline cell_idx_t normalizeIdx(py::object pyIndex, uint_t coordinateSize)
{
   cell_idx_t index;

   try {
      index = py::cast< cell_idx_t >(pyIndex);
   }
   catch (py::error_already_set & ){
      throw py::cast_error("Incompatible index data type");
   }

   if (index < 0)
      return cell_idx_c(coordinateSize) + 1 + index;
   else
      return index;
}

} // namespace internal

//*******************************************************************************************************************
/*! Creates a CellInterval as subset from the complete domain-cell-bounding-box based on a Python slice
 *
 *     Example: Python Slice: [ :, 3, -1 ]  and a domain size of ( 3,4,5 )
 *                 - x coordinate is the complete valid x-range indicated by the semicolon: i.e. [0,3)
 *                 - y coordinate is just a normal index i.e. the range from [3,4)
 *                 - z coordiante is the first valid coordinate from the back [4,5)
 *
 *     Python slices are tuples with slice classes as entry. Each slice has start, stop and step.
 *     Steps are not supported since they can not be encoded in a CellInterval
 */
//*******************************************************************************************************************
inline CellInterval globalPythonSliceToCellInterval(const shared_ptr< StructuredBlockStorage >& blocks,
                                                    py::tuple indexTuple)
{
   using internal::normalizeIdx;

   CellInterval bounds = blocks->getDomainCellBB();

   if (len(indexTuple) != 3)
   {
      throw py::index_error("Slice needs three components");
   }

   CellInterval interval;
   for (uint_t i = 0; i < 3; ++i)
   {
      if (!py::isinstance< py::slice >(indexTuple[i]))
      {
         cell_idx_t idx    = normalizeIdx(indexTuple[i], uint_c(bounds.max()[i]));
         interval.min()[i] = idx;
         interval.max()[i] = idx;
      }
      else if (py::isinstance< py::slice >(indexTuple[i]))
      {
         py::slice s = py::cast< py::slice >(indexTuple[i]);
         // Min
         if ( py::isinstance< py::none >(s.attr("start")) )
            interval.min()[i] = bounds.min()[i];
         else
            interval.min()[i] = normalizeIdx( s.attr("start"), uint_c( bounds.min()[i] ) );

         // Max
         if ( py::isinstance< py::none >(s.attr("stop")) )
            interval.max()[i] = bounds.max()[i];
         else
            interval.max()[i] = normalizeIdx( s.attr("stop"), uint_c( bounds.max()[i] ) );
      }
   }
   return interval;
}

//*******************************************************************************************************************
/*! Creates a CellInterval based on a Python Slice as subset of a field
 *
 *   Similar to globalPythonSliceToCellInterval() with the following additional features:
 *     - slice may have a forth component: [ :, 3, -1, 'g' ] with the only valid entry 'g' for ghost layers
 *     - if this ghost layer marker is present, coordinate 0 addresses the outermost ghost layer, otherwise the
 *       first inner cell is addressed
 */
//*******************************************************************************************************************
//TODO: Maybe this can be deleted. Was used in python bindings of boundary and lbm (Both is not exported anymore)
template< typename Field_T >
CellInterval localPythonSliceToCellInterval(const Field_T& field, py::tuple indexTuple)
{
   using internal::normalizeIdx;

   bool withGhostLayer = false;

   if (len(indexTuple) != 3)
   {
      if (len(indexTuple) == 4)
      {
         std::string marker = py::cast< std::string >(indexTuple[3]);
         if (marker == std::string("g"))
            withGhostLayer = true;
         else
         {
            throw py::index_error("Unknown marker in slice");
         }
      }
      else
      {
         throw py::index_error("Slice needs three components ( + optional ghost layer marker )");
      }
   }

   cell_idx_t gl = cell_idx_c(field.nrOfGhostLayers());

   CellInterval bounds;
   ;
   if (withGhostLayer)
   {
      bounds = field.xyzSizeWithGhostLayer();
      bounds.shift(gl, gl, gl);
   }
   else
      bounds = field.xyzSize();

   CellInterval interval;

   for (uint_t i = 0; i < 3; ++i)
   {
      if (!py::isinstance< py::slice >(indexTuple[i]))
      {
         interval.min()[i] = normalizeIdx(indexTuple[i], uint_c(bounds.max()[i]));
         interval.max()[i] = normalizeIdx(indexTuple[i], uint_c(bounds.max()[i]));
      }
      else
      {
         py::slice s = py::cast< py::slice >(indexTuple[i]);

         // Min
         if ( py::isinstance< py::none >(s.attr("start")) )
            interval.min()[i] = bounds.min()[i];
         else
            interval.min()[i] = normalizeIdx( s.attr("start"), uint_c( bounds.min()[i] ) );

         // Max
         if ( py::isinstance< py::none >(s.attr("stop")) )
            interval.max()[i] = bounds.max()[i];
         else
            interval.max()[i] = normalizeIdx( s.attr("stop"), uint_c( bounds.max()[i] ) );

      }
   }

   if (withGhostLayer) interval.shift(-gl, -gl, -gl);

   // Range check
   if (!field.xyzAllocSize().contains(interval))
   {
      throw py::index_error("Index out of bounds.");
   }

   return interval;
}

} // namespace python_coupling
} // namespace walberla
