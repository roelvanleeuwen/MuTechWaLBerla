//
// Created by ed94aqyc on 11/9/22.
//

#pragma once
#include "blockforest/StructuredBlockForest.h"

#include "core/DataTypes.h"

#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "domain_decomposition/StructuredBlockStorage.h"

#include "lbm/lattice_model/EquilibriumDistribution.h"

#include "Lagoon_InfoHeader.h"

namespace walberla {

template< typename LatticeModel_T >
class Lbm_Sweep_Pull
{
 public:
   Lbm_Sweep_Pull(BlockDataID pdfFieldID, BlockDataID densityID, BlockDataID velocityID, double omega)
      : pdfFieldID_(pdfFieldID), densityID_(densityID), velocityID_(velocityID), omega_(omega)
           {};

   void run(IBlock* block, uint8_t timestep);

   void operator() (IBlock * block, uint8_t timestep)
   {
      run(block, timestep);
   }

   static std::function<void (IBlock *)> getSweep(const shared_ptr<Lbm_Sweep_Pull> & kernel, uint8_t timestep)
   {
      return [kernel, timestep]
         (IBlock * b)
      { kernel->run(b, timestep); };
   }

   std::function<void (IBlock *)> getSweep(std::shared_ptr<lbm::TimestepTracker> & tracker)
   {
      return [this, tracker]
         (IBlock * b)
      { this->run(b, tracker->getCounter()); };
   }

   void even_sweep(const PdfField_T * pdfField, PdfField_T * tmp_pdfField, ScalarField_T * density, VelocityField_T * velocity);

   BlockDataID pdfFieldID_;
   BlockDataID densityID_;
   BlockDataID velocityID_;

   double omega_;
};

template< typename LatticeModel_T >
void Lbm_Sweep_Pull<LatticeModel_T>::even_sweep(const PdfField_T * pdfField, PdfField_T * tmp_pdfField, ScalarField_T * density, VelocityField_T * velocity)
{
   WALBERLA_FOR_ALL_CELLS_XYZ(pdfField,

            real_t pdfs[ Stencil_T::Size ];
            real_t rho = 0;
            Vector3<real_t> vel = Vector3<real_t>(0.0);
            for( auto d = Stencil_T::begin(); d != Stencil_T::end(); ++d ) {

               auto pdf = pdfField->get(x-d.cx(),y-d.cy(),z-d.cz(),d.toIdx());
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
               tmp_pdfField->get(x,y,z,d.toIdx()) = f_new;
            }
            density->get(x,y,z) = rho;
            velocity->get(x,y,z,0) = vel[0];
            velocity->get(x,y,z,1) = vel[1];
            velocity->get(x,y,z,2) = vel[2];

   );
}

template< typename LatticeModel_T >
void Lbm_Sweep_Pull<LatticeModel_T>::run(IBlock* block, uint8_t timestep) {
   auto pdfField = block->getData<PdfField_T>(pdfFieldID_);
   auto density = block->getData< ScalarField_T >(densityID_);
   auto velocity = block->getData< VelocityField_T >(velocityID_);
   auto tmp_pdfField = pdfField->cloneUninitialized();


   even_sweep(pdfField, tmp_pdfField, density, velocity);
   pdfField->swapDataPointers(tmp_pdfField);

}



//------------------------------------------------------------------------------------------------------------------//


template< typename LatticeModel_T >
class Lbm_Sweep_AA
{
 public:
   Lbm_Sweep_AA(BlockDataID pdfFieldID, BlockDataID densityID, BlockDataID velocityID, double omega)
      : pdfFieldID_(pdfFieldID), densityID_(densityID), velocityID_(velocityID), omega_(omega)
                                                                                    {};

   void run(IBlock* block, uint8_t timestep);

   void operator() (IBlock * block, uint8_t timestep)
   {
      run(block, timestep);
   }

   static std::function<void (IBlock *)> getSweep(const shared_ptr<Lbm_Sweep_AA> & kernel, uint8_t timestep)
   {
      return [kernel, timestep]
         (IBlock * b)
      { kernel->run(b, timestep); };
   }

   std::function<void (IBlock *)> getSweep(std::shared_ptr<lbm::TimestepTracker> & tracker)
   {
      return [this, tracker]
         (IBlock * b)
      { this->run(b, tracker->getCounter()); };
   }

   void even_sweep(PdfField_T * pdfField, ScalarField_T * density, VelocityField_T * velocity);
   void odd_sweep(PdfField_T * pdfField, ScalarField_T * density, VelocityField_T * velocity);

   BlockDataID pdfFieldID_;
   BlockDataID densityID_;
   BlockDataID velocityID_;

   double omega_;
};

template< typename LatticeModel_T >
void Lbm_Sweep_AA<LatticeModel_T>::odd_sweep(PdfField_T * pdfField, ScalarField_T * density, VelocityField_T * velocity)
{
   //PULL ALL PDFS, BUT HANDLE THEM IN OTHER ORDER
   /*
   WALBERLA_FOR_ALL_CELLS_XYZ(pdfField,

                              real_t pdfs[ Stencil_T::Size ];
                                      real_t rho = 0;
                                      Vector3<real_t> vel = Vector3<real_t>(0.0);
                                      for( auto d = Stencil_T::begin(); d != Stencil_T::end(); ++d ) {

                                         auto pdf = pdfField->get(x-d.cx(),y-d.cy(),z-d.cz(),d.inverseDir());
                                         pdfs[ d.inverseDir() ] = pdf;

                                         vel[0] += numeric_cast<real_t>( d.cx() ) * pdf;
                                         vel[1] += numeric_cast<real_t>( d.cy() ) * pdf;
                                         vel[2] += numeric_cast<real_t>( d.cz() ) * pdf;
                                         rho += pdf;
                                      }
                                      rho += real_t( 1 );

                                      for (auto d = Stencil_T::begin(); d != Stencil_T::end(); ++d)
                                      {
                                         const real_t f_pulled = pdfs[d.toIdx()];
                                         auto f_eq = lbm::EquilibriumDistribution< LatticeModel_T >::get(d.inverseDir(), vel, rho);
                                         real_t f_new = f_pulled - omega_ * (f_pulled - f_eq);
                                         pdfField->get(x-d.cx(),y-d.cy(),z-d.cz(),d.inverseDir()) = f_new;
                                      }
                                      density->get(x,y,z) = rho;
                                      velocity->get(x,y,z,0) = vel[0];
                                      velocity->get(x,y,z,1) = vel[1];
                                      velocity->get(x,y,z,2) = vel[2];
   );
   *//*
 //PULL ALL PDFS, BUT HANDLE THEM IN OTHER ORDER
   WALBERLA_FOR_ALL_CELLS_XYZ(pdfField,

      real_t pdfs[ Stencil_T::Size ];
      real_t rho = 0;
      Vector3<real_t> vel = Vector3<real_t>(0.0);
      for( auto d = Stencil_T::begin(); d != Stencil_T::end(); ++d ) {

         auto pdf = pdfField->get(x-d.cx(),y-d.cy(),z-d.cz(),d.inverseDir());
         pdfs[ d.inverseDir() ] = pdf;

         vel[0] += numeric_cast<real_t>( d.cx() ) * pdf;
         vel[1] += numeric_cast<real_t>( d.cy() ) * pdf;
         vel[2] += numeric_cast<real_t>( d.cz() ) * pdf;
         rho += pdf;
      }
      rho += real_t( 1 );

      for (auto d = Stencil_T::begin(); d != Stencil_T::end(); ++d)
      {
         const real_t f_pulled = pdfs[d.toIdx()];
         auto f_eq = lbm::EquilibriumDistribution< LatticeModel_T >::get(d.inverseDir(), vel, rho);
         real_t f_new = f_pulled - omega_ * (f_pulled - f_eq);
         pdfField->get(x-d.cx(),y-d.cy(),z-d.cz(),d.inverseDir()) = f_new;
      }
      density->get(x,y,z) = rho;
      velocity->get(x,y,z,0) = vel[0];
      velocity->get(x,y,z,1) = vel[1];
      velocity->get(x,y,z,2) = vel[2];
   );
    */

    //PULL FROM LEFT AND WRITE TO RIGHT
   //Both versions are working
   WALBERLA_FOR_ALL_CELLS_XYZ(pdfField,

      real_t pdfs[ Stencil_T::Size ];
      real_t rho = 0;
      Vector3<real_t> vel = Vector3<real_t>(0.0);
      for( auto d = Stencil_T::begin(); d != Stencil_T::end(); ++d ) {

         auto pdf = pdfField->get(x-d.cx(),y-d.cy(),z-d.cz(),d.inverseDir());
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
         pdfField->get(x+d.cx(),y+d.cy(),z+d.cz(),d.toIdx()) = f_new;
      }
      density->get(x,y,z) = rho;
      velocity->get(x,y,z,0) = vel[0];
      velocity->get(x,y,z,1) = vel[1];
      velocity->get(x,y,z,2) = vel[2];
   );
}

template< typename LatticeModel_T >
void Lbm_Sweep_AA<LatticeModel_T>::even_sweep(PdfField_T * pdfField, ScalarField_T * density, VelocityField_T * velocity)
{
   WALBERLA_FOR_ALL_CELLS_XYZ(pdfField,

      real_t pdfs[ Stencil_T::Size ];
      real_t rho = 0;
      Vector3<real_t> vel = Vector3<real_t>(0.0);
      for( auto d = Stencil_T::begin(); d != Stencil_T::end(); ++d ) {

         auto pdf = pdfField->get(x,y,z,d.toIdx());
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
         pdfField->get(x,y,z,d.inverseDir()) = f_new;
         //pdfField->get(x,y,z,d.toIdx()) = f_new;
      }
      density->get(x,y,z) = rho;
      velocity->get(x,y,z,0) = vel[0];
      velocity->get(x,y,z,1) = vel[1];
      velocity->get(x,y,z,2) = vel[2];
   );
}

template< typename LatticeModel_T >
void Lbm_Sweep_AA<LatticeModel_T>::run(IBlock* block, uint8_t timestep) {
   auto pdfField = block->getData<PdfField_T>(pdfFieldID_);
   auto density = block->getData< ScalarField_T >(densityID_);
   auto velocity = block->getData< VelocityField_T >(velocityID_);

   if(((timestep & 1) ^ 1)) {
      even_sweep(pdfField, density, velocity);
   } else {
      odd_sweep(pdfField, density, velocity);
   }

}

} // namespace walberla
