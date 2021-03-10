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
//! \file AlignedPDFfieldTest.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#include "blockforest/all.h"

#include "core/all.h"

#include "field/allocation/FieldAllocator.h"

#include "lbm/all.h"

namespace walberla
{
using LatticeModel_T = lbm::D3Q19< lbm::collision_model::SRT, false >;
typedef lbm::PdfField< LatticeModel_T > PdfField_T;

int main(int argc, char** argv)
{
   debug::enterTestMode();
   walberla::Environment walberlaEnv(argc, argv);

   auto blocks = blockforest::createUniformBlockGridFromConfig(walberlaEnv.config());

   auto parameters    = walberlaEnv.config()->getOneBlock("Parameters");
   const real_t omega = parameters.getParameter< real_t >("omega", real_c(1.4));

   LatticeModel_T latticeModel = LatticeModel_T(omega);

   shared_ptr< field::FieldAllocator< real_t > > alloc = make_shared< field::AllocateAligned< real_t, 32 > >();

   BlockDataID pdfFieldId = lbm::addPdfFieldToStorage(blocks, "pdf field", latticeModel, field::fzyx, alloc);

   for (auto& block : *blocks)
   {
      auto pdfField = block.getData< PdfField_T >(pdfFieldId);
      std::cout << ((size_t) pdfField) % 32 << std::endl;
   }
}
} // namespace walberla

int main(int argc, char** argv) { return walberla::main(argc, argv); }