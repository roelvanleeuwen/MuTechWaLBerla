//
// Created by ed94aqyc on 11/9/22.
//

#pragma once
#include "core/DataTypes.h"
#include "blockforest/StructuredBlockForest.h"

#include "ListLBMList.h"
#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "domain_decomposition/StructuredBlockStorage.h"

#include "ListLBMInfoHeader.h"

namespace walberla {

class Lbm_Sweep_Pull
{
 public:
   Lbm_Sweep_Pull(BlockDataID listID_, double omega)
      : listID(listID_), omega_(omega)
                            {};

   void run( IBlock * block );

   void operator() ( IBlock * block )
   {
      run( block );
   }

   void odd_sweep(List_T* list);


   static std::function<void (IBlock*)> getSweep(const shared_ptr<Lbm_Sweep_Pull> & kernel) {
      return [kernel](IBlock * b){ kernel->run(b); };
   }

   std::function<void (IBlock *)> getSweep()
   {
      return [this ]
         (IBlock * b)
      { this->run(b ); };
   }
   BlockDataID listID;
   double omega_;
};

void Lbm_Sweep_Pull::odd_sweep(List_T* list)
{
   WALBERLA_ASSERT_NOT_NULLPTR(list);
   using LatticeModel_T = lbm::D3Q19<lbm::collision_model::SRT>; //TODO only used for Eqilibrium calc
   //real_t pdfs[Stencil_T::Size];
   const auto pullIndices = list->getidxbeginning();

   for (typename List_T::index_t idx = 0; idx < list->numFluidCells(); ++idx)
   {

      real_t pdfs[ Stencil_T::Size ];
      real_t rho = 0;
      Vector3<real_t> vel = Vector3<real_t>(0.0);

      auto pdf = list->get( idx, stencil::C );
      rho += pdf;
      pdfs[ 0 ] = pdf;

      for( auto d = Stencil_T::beginNoCenter(); d != Stencil_T::end(); ++d ) {

         pdf = list->get( pullIndices[list->getPullIdx( idx, d.toIdx() )] );
         pdfs[ d.toIdx() ] = pdf;

         vel[0] += numeric_cast<real_t>( d.cx() ) * pdf;
         vel[1] += numeric_cast<real_t>( d.cy() ) * pdf;
         vel[2] += numeric_cast<real_t>( d.cz() ) * pdf;
         rho += pdf;
      }
      rho += real_t( 1 );

      for (auto d = Stencil_T::begin(); d != Stencil_T::end(); ++d)
      {
         const real_t f_pulled = pdfs[d.toIdx()];
         auto f_eq = lbm::EquilibriumDistribution< LatticeModel_T >::get(*d, vel, rho);
         real_t f_new = f_pulled - omega_ * (f_pulled - f_eq);
         //WALBERLA_LOG_INFO("Cell "  <<  x << "," << y << ","  << z << ", Direction  "<< d.dirString() << " f_pulled " <<  f_pulled  << ", f_eq " << f_eq << ", f_new " << f_new);
         list->getTmp(idx, *d) = f_new;
      }
   }
   list->swapTmpPdfs();

}

void Lbm_Sweep_Pull::run(IBlock* block) {
   auto list = block->getData< List_T >(listID);
   odd_sweep(list);
}




class Lbm_Sweep_AA_Sparse
{
 public:
   Lbm_Sweep_AA_Sparse(BlockDataID listID_, double omega)
      : listID(listID_), omega_(omega)
                            {};

   void run( IBlock * block, uint8_t timestep );

   void operator() ( IBlock * block, uint8_t timestep )
   {
      run( block, timestep );
   }

   void odd_sweep(List_T* list);
   void even_sweep(List_T* list);


   static std::function<void (IBlock*)> getSweep(const shared_ptr<Lbm_Sweep_AA_Sparse> & kernel, uint8_t timestep) {
      return [kernel, timestep](IBlock * b){ kernel->run(b, timestep); };
   }

   std::function<void (IBlock *)> getSweep(std::shared_ptr<lbm::TimestepTracker> & tracker)
   {
      return [this, tracker ]
         (IBlock * b)
      { this->run(b, tracker->getCounter() ); };
   }
   BlockDataID listID;
   double omega_;
};

void Lbm_Sweep_AA_Sparse::odd_sweep(List_T* list)
{
   WALBERLA_ASSERT_NOT_NULLPTR(list);
   using LatticeModel_T = lbm::D3Q19<lbm::collision_model::SRT>; //TODO only used for Eqilibrium calc
   const auto pullIndices = list->getidxbeginning();

   for (typename List_T::index_t idx = 0; idx < list->numFluidCells(); ++idx)
   {
      real_t pdfs[ Stencil_T::Size ];
      real_t rho = 0;
      Vector3<real_t> vel = Vector3<real_t>(0.0);

      auto pdf = list->get( idx, stencil::C );
      rho += pdf;
      pdfs[ 0 ] = pdf;

      for( auto d = Stencil_T::beginNoCenter(); d != Stencil_T::end(); ++d ) {

         pdf = list->get( pullIndices[list->getPullIdx( idx, d.inverseDir() )] );
         pdfs[ d.toIdx() ] = pdf;

         vel[0] += numeric_cast<real_t>( d.cx() ) * pdf;
         vel[1] += numeric_cast<real_t>( d.cy() ) * pdf;
         vel[2] += numeric_cast<real_t>( d.cz() ) * pdf;
         rho += pdf;
      }
      rho += real_t( 1 );

      real_t f_pulled = pdfs[stencil::C];
      real_t f_eq = lbm::EquilibriumDistribution< LatticeModel_T >::get(stencil::C, vel, rho);
      real_t f_new = f_pulled - omega_ * (f_pulled - f_eq);
      list->get( idx, stencil::C ) = f_new;

      for (auto d = Stencil_T::beginNoCenter(); d != Stencil_T::end(); ++d)
      {
         f_pulled = pdfs[d.toIdx()];
         f_eq = lbm::EquilibriumDistribution< LatticeModel_T >::get(*d, vel, rho);
         f_new = f_pulled - omega_ * (f_pulled - f_eq);
         list->get( pullIndices[list->getPullIdx( idx, d.toIdx() )] ) = f_new;
      }
   }
   /*
   for (typename List_T::index_t idx = 0; idx < list->numFluidCells(); ++idx)
   {

      real_t pdfs[ Stencil_T::Size ];
      real_t rho = 0;
      Vector3<real_t> vel = Vector3<real_t>(0.0);

      auto pdf = list->get( idx, stencil::C );
      rho += pdf;
      pdfs[ 0 ] = pdf;

      for( auto d = Stencil_T::beginNoCenter(); d != Stencil_T::end(); ++d ) {

         pdf = list->get( pullIndices[list->getPullIdx( idx, d.inverseDir() )] );
         pdfs[ d.inverseDir() ] = pdf;

         vel[0] += numeric_cast<real_t>( d.cx() ) * pdf;
         vel[1] += numeric_cast<real_t>( d.cy() ) * pdf;
         vel[2] += numeric_cast<real_t>( d.cz() ) * pdf;
         rho += pdf;
      }
      rho += real_t( 1 );

      real_t f_pulled = pdfs[stencil::C];
      real_t f_eq = lbm::EquilibriumDistribution< LatticeModel_T >::get(stencil::C, vel, rho);
      real_t f_new = f_pulled - omega_ * (f_pulled - f_eq);
      list->get( idx, stencil::C ) = f_new;

      for (auto d = Stencil_T::beginNoCenter(); d != Stencil_T::end(); ++d)
      {
         f_pulled = pdfs[d.toIdx()];
         f_eq = lbm::EquilibriumDistribution< LatticeModel_T >::get(d.inverseDir(), vel, rho);
         f_new = f_pulled - omega_ * (f_pulled - f_eq);
         list->get( pullIndices[list->getPullIdx( idx, d.inverseDir() )] ) = f_new;
      }
   }*/
}




void Lbm_Sweep_AA_Sparse::even_sweep(List_T* list)
{
   WALBERLA_ASSERT_NOT_NULLPTR(list);
   using LatticeModel_T = lbm::D3Q19<lbm::collision_model::SRT>; //TODO only used for Eqilibrium calc

   for (typename List_T::index_t idx = 0; idx < list->numFluidCells(); ++idx)
   {

      real_t pdfs[ Stencil_T::Size ];
      real_t rho = 0;
      Vector3<real_t> vel = Vector3<real_t>(0.0);

      auto pdf = list->get( idx, stencil::C );
      rho += pdf;
      pdfs[ 0 ] = pdf;

      for( auto d = Stencil_T::beginNoCenter(); d != Stencil_T::end(); ++d ) {

         pdf = list->get(  idx, d.toIdx() );
         pdfs[ d.toIdx() ] = pdf;

         vel[0] += numeric_cast<real_t>( d.cx() ) * pdf;
         vel[1] += numeric_cast<real_t>( d.cy() ) * pdf;
         vel[2] += numeric_cast<real_t>( d.cz() ) * pdf;
         rho += pdf;
      }
      rho += real_t( 1 );

      for (auto d = Stencil_T::begin(); d != Stencil_T::end(); ++d)
      {
         const real_t f_pulled = pdfs[d.toIdx()];
         auto f_eq = lbm::EquilibriumDistribution< LatticeModel_T >::get(*d, vel, rho);
         real_t f_new = f_pulled - omega_ * (f_pulled - f_eq);
         list->get(idx, d.inverseDir()) = f_new;
      }
   }
}

void Lbm_Sweep_AA_Sparse::run(IBlock* block, uint8_t timestep) {
   auto list = block->getData< List_T >(listID);
   if(((timestep & 1) ^ 1)) {
      even_sweep(list);
   } else {
      odd_sweep(list);

   }
}







} // namespace walberla
