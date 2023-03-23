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
//! \file SparseLayout.h
//! \ingroup lbm
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#pragma once

namespace walberla
{
namespace lbm
{

template< typename Stencil, typename Index_T = walberla::uint32_t >
struct LayoutAoS
{
   LayoutAoS() = delete;

   static uint_t alignment() { return uint_t(1); }
   static uint_t numFluidCellsToAllocate(const uint_t numFluidCellsRequired) { return numFluidCellsRequired; }

   static Index_T getPDFIdx(const uint_t idx, const uint_t f, const uint_t /*N*/)
   {
      return numeric_cast< Index_T >(idx * Stencil::Size + f);
   }
   static Index_T getPullIdxIdx(const uint_t idx, const stencil::Direction d, const uint_t /*N*/)
   {
      WALBERLA_ASSERT_UNEQUAL(d, stencil::C)
      WALBERLA_ASSERT_UNEQUAL(Stencil::idx[d], stencil::INVALID_DIR)

      const uint_t f = Stencil::idx[d] - Stencil::noCenterFirstIdx;
      return numeric_cast< Index_T >(idx * (Stencil::Size - Stencil::noCenterFirstIdx) + f);
   }
   static const real_t*& incPtr(const real_t*& p)
   {
      p += Stencil::Size;
      return p;
   }
   static real_t*& incPtr(real_t*& p)
   {
      p += Stencil::Size;
      return p;
   }

   static Index_T& incIdx(Index_T& idx)
   {
      idx += numeric_cast< Index_T >(Stencil::Size);
      return idx;
   }
};

template< typename Stencil, typename Index_T = walberla::uint32_t, uint_t ALIGNMENT = 64 >
struct LayoutSoA
{
   static_assert(((ALIGNMENT & (ALIGNMENT - uint_t(1))) == 0) && (ALIGNMENT >= uint_t(1)),
                 "The alignment for the SoA list layout has to a power of two!");

   LayoutSoA() = delete;

   static uint_t alignment() { return ALIGNMENT; }
   static uint_t numFluidCellsToAllocate(const uint_t numFluidCellsRequired)
   {
      uint_t alignedStepSize = std::max(uint_t(1), alignment() / sizeof(real_t));
      if ((numFluidCellsRequired % alignedStepSize) == 0)
         return numFluidCellsRequired;
      else
         return (numFluidCellsRequired / alignedStepSize + uint_t(1)) * alignedStepSize;
   }

   static Index_T getPDFIdx(const uint_t idx, const uint_t f, const uint_t N)
   {
      return numeric_cast< Index_T >(f * N + idx);
   }
   static Index_T getPullIdxIdx(const uint_t idx, const stencil::Direction d, const uint_t N)
   {
      WALBERLA_ASSERT_UNEQUAL(d, stencil::C)
      WALBERLA_ASSERT_UNEQUAL(Stencil::idx[d], stencil::INVALID_DIR)

      const uint_t f = Stencil::idx[d] - Stencil::noCenterFirstIdx;
      return numeric_cast< Index_T >(f * N + idx);
   }
   static const real_t*& incPtr(const real_t*& p)
   {
      ++p;
      return p;
   }
   static real_t*& incPtr(real_t*& p)
   {
      ++p;
      return p;
   }
   static Index_T& incIdx(Index_T& idx)
   {
      ++idx;
      return idx;
   }
};
} // namespace lbm
} // namespace walberla