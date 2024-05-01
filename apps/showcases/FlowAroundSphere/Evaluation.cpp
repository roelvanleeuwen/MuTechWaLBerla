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
//! \file Evaluation.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================
#include "Evaluation.h"


namespace walberla
{

void Evaluation::operator()()
{
   if (checkFrequency_ == uint_t(0)) return;
   ++executionCounter_;
   if (rampUpTime_ > executionCounter_) return;

   real_t cDRealArea(real_t(0));
   real_t cLRealArea(real_t(0));
   real_t cDDiscreteArea(real_t(0));
   real_t cLDiscreteArea(real_t(0));

   real_t pressureDifference_L(real_t(0));
   real_t pressureDifference(real_t(0));

   evaluate(cDRealArea, cLRealArea, cDDiscreteArea, cLDiscreteArea, pressureDifference_L, pressureDifference);

   if ((executionCounter_ - uint_c(1)) % checkFrequency_ != 0) return;

   auto blocks = blocks_.lock();
   WALBERLA_CHECK_NOT_NULLPTR(blocks)

   // Strouhal number (needs vortex shedding frequency)

   real_t vortexVelocity(real_t(0));

   if (setup_.evaluateStrouhal)
   {
      auto block = blocks->getBlock(setup_.pStrouhal);
      if (block != nullptr)
      {
         const VelocityField_T* const velocityField = block->template getData< VelocityField_T >(velocityFieldId_);
         const auto cell                            = blocks->getBlockLocalCell(*block, setup_.pStrouhal);
         WALBERLA_ASSERT(velocityField->xyzSize().contains(cell))
         vortexVelocity += velocityField->get(cell.x(), cell.y(), cell.z(), cell_idx_c(0));
      }
      mpi::reduceInplace(vortexVelocity, mpi::SUM);
   }

   WALBERLA_ROOT_SECTION()
   {
      coefficients_[0].push_back(cDRealArea);
      coefficients_[1].push_back(cLRealArea);
      coefficients_[2].push_back(cDDiscreteArea);
      coefficients_[3].push_back(cLDiscreteArea);

      if (coefficients_[0].size() > setup_.nbrOfEvaluationPointsForCoefficientExtremas)
      {
         for (uint_t i = uint_t(0); i < uint_t(4); ++i)
            coefficients_[i].pop_front();
      }

      for (uint_t i = uint_t(0); i < uint_t(4); ++i)
      {
         coefficientExtremas_[i] = std::make_pair(*(coefficients_[i].begin()), *(coefficients_[i].begin()));
         for (auto v = coefficients_[i].begin(); v != coefficients_[i].end(); ++v)
         {
            coefficientExtremas_[i].first  = std::min(coefficientExtremas_[i].first, *v);
            coefficientExtremas_[i].second = std::max(coefficientExtremas_[i].second, *v);
         }
      }

      std::ostringstream oss;

      if (logToStream_)
      {
         WALBERLA_LOG_RESULT_ON_ROOT(
            "force acting on cylinder (in dimensionless lattice units of the coarsest grid - evaluated in time step "
            << executionCounter_ - uint_c(1) << "):\n   " << force_ << oss.str()
            << "\ndrag and lift coefficients (including extrema of last " << (coefficients_[0].size() * checkFrequency_)
            << " time steps):"
               "\n   \"real\" area:"
               "\n      c_D: "
            << cDRealArea << " (min = " << coefficientExtremas_[0].first << ", max = " << coefficientExtremas_[0].second
            << ")"
            << "\n      c_L: " << cLRealArea << " (min = " << coefficientExtremas_[1].first
            << ", max = " << coefficientExtremas_[1].second << ")"
            << "\n   discrete area:"
               "\n      c_D: "
            << cDDiscreteArea << " (min = " << coefficientExtremas_[2].first
            << ", max = " << coefficientExtremas_[2].second << ")"
            << "\n      c_L: " << cLDiscreteArea << " (min = " << coefficientExtremas_[3].first
            << ", max = " << coefficientExtremas_[3].second << ")")
      }

      if (setup_.evaluatePressure && logToStream_)
      {
         WALBERLA_LOG_RESULT_ON_ROOT("pressure:\n   difference: " << pressureDifference << " Pa ("
                                                                  << pressureDifference_L << ")")
      }

      if (setup_.evaluateStrouhal)
      {
         // We evaluate the derivative (-> strouhalRising_) in order to find the local minima and maxima of the velocity
         // over time. If we know the time between a local minimum and a local maximum, we can calculate the frequency.
         // -> "Smooth noise-robust differentiators"
         // (http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/)

         if (strouhalVelocities_.size() < uint_t(11)) { strouhalVelocities_.push_back(vortexVelocity); }
         else
         {
            for (uint_t i = uint_t(0); i < uint_t(10); ++i)
               strouhalVelocities_[i] = strouhalVelocities_[i + 1];
            strouhalVelocities_[10] = vortexVelocity;

            const real_t f1 = strouhalVelocities_[6] - strouhalVelocities_[4];
            const real_t f2 = strouhalVelocities_[7] - strouhalVelocities_[3];
            const real_t f3 = strouhalVelocities_[8] - strouhalVelocities_[2];
            const real_t f4 = strouhalVelocities_[9] - strouhalVelocities_[1];
            const real_t f5 = strouhalVelocities_[10] - strouhalVelocities_[0];

            const real_t diff =
               (real_c(322) * f1 + real_c(256) * f2 + real_c(39) * f3 - real_c(32) * f4 - real_c(11) * f5) /
               real_c(1536);

            if ((diff > real_t(0)) != strouhalRising_)
            {
               strouhalRising_ = (diff > real_t(0));

               if (strouhalTimeStep_.size() < uint_t(3)) { strouhalTimeStep_.push_back(executionCounter_); }
               else
               {
                  strouhalTimeStep_[0] = strouhalTimeStep_[1];
                  strouhalTimeStep_[1] = strouhalTimeStep_[2];
                  strouhalTimeStep_[2] = executionCounter_;
               }
            }
         }

         if (strouhalTimeStep_.size() == uint_t(3))
         {
            strouhalNumberRealD_     = diameterSphere / (meanVelocity * real_c(strouhalTimeStep_[2] - strouhalTimeStep_[0]));
            strouhalEvaluationExecutionCount_ = executionCounter_ - uint_t(1);

            if (logToStream_)
            {
               WALBERLA_LOG_RESULT_ON_ROOT(
                  "Strouhal number (evaluated in time step "
                  << strouhalEvaluationExecutionCount_
                  << "):"
                     "\n   D/U (in lattice units): "
                  << (diameterSphere / meanVelocity) << " (\"real\" D), "
                  << "\n   T: "
                  << (real_c(strouhalTimeStep_[2] - strouhalTimeStep_[0]) * setup_.dt) << " s ("
                  << real_c(strouhalTimeStep_[2] - strouhalTimeStep_[0])
                  << ")"
                     "\n   f: "
                  << (real_t(1) / (real_c(strouhalTimeStep_[2] - strouhalTimeStep_[0]) * setup_.dt)) << " Hz ("
                  << (real_t(1) / real_c(strouhalTimeStep_[2] - strouhalTimeStep_[0]))
                  << ")"
                     "\n   St (\"real\" D):   "
                  << strouhalNumberRealD_ << "\n")
            }
         }
      }

      if (logToFile_)
      {
         std::ofstream file(filename_.c_str(), std::ios_base::app);
         file << executionCounter_ - uint_t(1) << " " << force_[0] << " " << force_[1] << " " << force_[2] << " "
              << cDRealArea << " " << cLRealArea << " " << cDDiscreteArea << " " << cLDiscreteArea << " "
              << pressureDifference_L << " " << pressureDifference << " " << vortexVelocity << " "
              << strouhalNumberRealD_ << '\n';
         file.close();
      }
   }

   // WALBERLA_MPI_WORLD_BARRIER();
}

void Evaluation::resetForce()
{
   if (!initialized_) refresh();
}

void Evaluation::forceCalculation(IBlock* block, const uint_t level)
{
   // Supposed to be used as a post boundary handling function on every level
   // if (checkFrequency_ == uint_t(0) || executionCounter_ % checkFrequency_ != 0)
   //    return;
   if (rampUpTime_ > executionCounter_) return;
   force_ += boundaryCollection_.ObstacleObject->getForce(block);
}

void Evaluation::prepareResultsForSQL()
{
   real_t cDRealArea(real_t(0));
   real_t cLRealArea(real_t(0));
   real_t cDDiscreteArea(real_t(0));
   real_t cLDiscreteArea(real_t(0));

   real_t pressureDifference_L(real_t(0));
   real_t pressureDifference(real_t(0));

   evaluate(cDRealArea, cLRealArea, cDDiscreteArea, cLDiscreteArea, pressureDifference_L, pressureDifference);

   WALBERLA_ROOT_SECTION()
   {
      sqlResults_["forceX_L"] = double_c(force_[0]);
      sqlResults_["forceY_L"] = double_c(force_[1]);
      sqlResults_["forceZ_L"] = double_c(force_[2]);
      sqlResults_["forceXMin_L"]    = 0.0;
      sqlResults_["forceXMax_L"]    = 0.0;
      sqlResults_["forceXAvg_L"]    = 0.0;
      sqlResults_["forceXMedian_L"] = 0.0;
      sqlResults_["forceXStdDev_L"] = 0.0;

      sqlResults_["forceYMin_L"]    = 0.0;
      sqlResults_["forceYMax_L"]    = 0.0;
      sqlResults_["forceYAvg_L"]    = 0.0;
      sqlResults_["forceYMedian_L"] = 0.0;
      sqlResults_["forceYStdDev_L"] = 0.0;

      sqlResults_["cDRealArea"]     = double_c(cDRealArea);
      sqlResults_["cLRealArea"]     = double_c(cLRealArea);
      sqlResults_["cDDiscreteArea"] = double_c(cDDiscreteArea);
      sqlResults_["cLDiscreteArea"] = double_c(cLDiscreteArea);

      sqlResults_["cDRealAreaMin"]     = double_c(coefficientExtremas_[0].first);
      sqlResults_["cDRealAreaMax"]     = double_c(coefficientExtremas_[0].second);
      sqlResults_["cLRealAreaMin"]     = double_c(coefficientExtremas_[1].first);
      sqlResults_["cLRealAreaMax"]     = double_c(coefficientExtremas_[1].second);
      sqlResults_["cDDiscreteAreaMin"] = double_c(coefficientExtremas_[2].first);
      sqlResults_["cDDiscreteAreaMax"] = double_c(coefficientExtremas_[2].second);
      sqlResults_["cLDiscreteAreaMin"] = double_c(coefficientExtremas_[3].first);
      sqlResults_["cLDiscreteAreaMax"] = double_c(coefficientExtremas_[3].second);

      sqlResults_["pressureDifference_L"] = double_c(pressureDifference_L);
      sqlResults_["pressureDifference"]   = double_c(pressureDifference);

      sqlResults_["strouhalNumberRealD"]     = double_c(strouhalNumberRealD_);
   }
}

void Evaluation::getResultsForSQLOnRoot(std::map< std::string, double >& realProperties,
                                        std::map< std::string, int >& integerProperties) const
{
   WALBERLA_ROOT_SECTION()
   {
      for (auto result = sqlResults_.begin(); result != sqlResults_.end(); ++result)
         realProperties[result->first] = result->second;

      integerProperties["forceEvaluationTimeStep"]    = int_c(executionCounter_ - uint_c(1));
      integerProperties["strouhalEvaluationTimeStep"] = int_c(strouhalEvaluationExecutionCount_);
   }
}


void Evaluation::refresh()
{
   auto blocks = blocks_.lock();
   WALBERLA_CHECK_NOT_NULLPTR(blocks)

   real_t AD(real_t(0));
   real_t AL(real_t(0));

   for (auto block = blocks->begin(); block != blocks->end(); ++block)
   {
      const FlagField_T* const flagField = block->template getData< FlagField_T >(flagFieldId_);

      auto fluid    = flagField->getFlag(fluid_);
      auto obstacle = flagField->getFlag(obstacle_);
      const real_t area  = real_c(1.0);

      auto xyzSize = flagField->xyzSize();
      for (auto z = xyzSize.zMin(); z <= xyzSize.zMax(); ++z)
      {
         for (auto y = xyzSize.yMin(); y <= xyzSize.yMax(); ++y)
         {
            for (auto x = xyzSize.xMin(); x <= xyzSize.xMax(); ++x)
            {
               if (flagField->isFlagSet(x, y, z, fluid))
               {
                  for (auto it = Stencil_T::beginNoCenter(); it != Stencil_T::end(); ++it)
                  {
                     auto nx = x + cell_idx_c(it.cx());
                     auto ny = y + cell_idx_c(it.cy());
                     auto nz = z + cell_idx_c(it.cz());

                     if (flagField->isFlagSet(nx, ny, nz, obstacle))
                     {
                        if (it.cx() == 1 && it.cy() == 0 && it.cz() == 0) { AD += area; }
                        else if (it.cx() == 0 && it.cz() == 0)
                        {
                           if (it.cy() == 1)
                           {
                              AL += area;
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   mpi::reduceInplace(AD, mpi::SUM);
   mpi::reduceInplace(AL, mpi::SUM);

   WALBERLA_ROOT_SECTION()
   {
      AD_ = AD;
      AL_ = AL;
   }

   // Check if points alpha and omega (required for evaluating the pressure difference) are located in fluid cells
   if (setup_.evaluatePressure)
   {
      auto block = blocks->getBlock(setup_.pAlpha);
      if (block != nullptr)
      {
         const FlagField_T* const flagField = block->template getData< FlagField_T >(flagFieldId_);

         const auto cell = blocks->getBlockLocalCell(*block, setup_.pAlpha);
         WALBERLA_ASSERT(flagField->xyzSize().contains(cell))

         const auto fluid = flagField->getFlag(fluid_);
         if (!flagField->isFlagSet(cell, fluid))
         {
            WALBERLA_ABORT("Cell for evaluating pressure difference at point alpha " << setup_.pAlpha << " is not a fluid cell!")
         }
      }

      block = blocks->getBlock(setup_.pOmega);
      if (block != nullptr)
      {
         const FlagField_T* const flagField = block->template getData< FlagField_T >(flagFieldId_);

         const auto cell = blocks->getBlockLocalCell(*block, setup_.pOmega);
         WALBERLA_ASSERT(flagField->xyzSize().contains(cell))

         const auto fluid = flagField->getFlag(fluid_);
         if (!flagField->isFlagSet(cell, fluid))
         {
            WALBERLA_ABORT("Cell for evaluating pressure difference at point omega " << setup_.pOmega << " is not a fluid cell!")
         }
      }
   }

   // Check if point for evaluating the Strouhal number is located inside a fluid cell

   if (setup_.evaluateStrouhal)
   {
      auto block = blocks->getBlock(setup_.pStrouhal);
      if (block != nullptr)
      {
         const FlagField_T* const flagField = block->template getData< FlagField_T >(flagFieldId_);

         const auto cell = blocks->getBlockLocalCell(*block, setup_.pStrouhal);
         WALBERLA_ASSERT(flagField->xyzSize().contains(cell))

         const auto fluid = flagField->getFlag(fluid_);

         if (!flagField->isFlagSet(cell, fluid))
            WALBERLA_ABORT("Cell for evaluating the Strouhal number at point " << setup_.pStrouhal << " is not a fluid cell!")
      }
   }

   initialized_ = true;
}

void Evaluation::evaluate(real_t& cDRealArea, real_t& cLRealArea, real_t& cDDiscreteArea, real_t& cLDiscreteArea,
                          real_t& pressureDifference_L, real_t& pressureDifference)
{
   if (!initialized_) refresh();
   // force on obstacle
   mpi::reduceInplace(force_, mpi::SUM);

   real_t pAlpha(real_t(0));
   real_t pOmega(real_t(0));

   auto blocks = blocks_.lock();
   WALBERLA_CHECK_NOT_NULLPTR(blocks)

   if (setup_.evaluatePressure)
   {
      auto block = blocks->getBlock(setup_.pAlpha);
      if (block != nullptr)
      {
         const ScalarField_T* const densityField = block->template getData< ScalarField_T >(densityFieldId_);
         const auto cell                         = blocks->getBlockLocalCell(*block, setup_.pAlpha);
         WALBERLA_ASSERT(densityField->xyzSize().contains(cell))
         pAlpha += densityField->get(cell) / real_c(3);
      }

      block = blocks->getBlock(setup_.pOmega);
      if (block != nullptr)
      {
         const ScalarField_T* const densityField = block->template getData< ScalarField_T >(densityFieldId_);
         const auto cell                         = blocks->getBlockLocalCell(*block, setup_.pOmega);
         WALBERLA_ASSERT(densityField->xyzSize().contains(cell))
         pOmega += densityField->get(cell) / real_c(3);
      }

      mpi::reduceInplace(pAlpha, mpi::SUM);
      mpi::reduceInplace(pOmega, mpi::SUM);
   }

   WALBERLA_ROOT_SECTION()
   {
      cDRealArea = (real_c(8.0) * force_[0]) / (meanVelocity * meanVelocity * surfaceAreaSphere);
      cLRealArea = (real_c(8.0) * force_[1]) / (meanVelocity * meanVelocity * surfaceAreaSphere);

      cDDiscreteArea = (real_c(8.0) * force_[0]) / (meanVelocity * meanVelocity * AD_);
      cLDiscreteArea = (real_c(8.0) * force_[1]) / (meanVelocity * meanVelocity * AL_);

      DragCoefficient DC(executionCounter_ - uint_c(1), force_, cDRealArea, cLRealArea, cDDiscreteArea, cLDiscreteArea);
      dragResults.push_back(DC);

      pressureDifference_L = pAlpha - pOmega;
      pressureDifference   = (pressureDifference_L * setup_.rho * setup_.dx * setup_.dx) / (setup_.dt * setup_.dt);
   }

   force_[0] = real_c(0.0);
   force_[1] = real_c(0.0);
   force_[2] = real_c(0.0);
}

}