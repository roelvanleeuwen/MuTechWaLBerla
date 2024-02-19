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
# pragma once

#include "core/config/Config.h"
#include "core/math/Sample.h"

#include "lbm_generated/field/PdfField.h"

#include "sqlite/SQLite.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "Setup.h"
#include "FlowAroundCylinderInfoHeader.h"
using namespace walberla;

using StorageSpecification_T = lbm::FlowAroundCylinderStorageSpecification;
using Stencil_T              = StorageSpecification_T::Stencil;
using PdfField_T             = lbm_generated::PdfField< StorageSpecification_T >;
using FlagField_T            = FlagField< uint8_t >;

using VoidFunction = std::function<void ()>;

namespace walberla
{

class Evaluation
{
 public:
   Evaluation(const weak_ptr< StructuredBlockStorage >& blocks, const uint_t checkFrequency, const uint_t rampUpTime, VoidFunction& getFields,
              const BlockDataID& pdfFieldId, const BlockDataID& densityFieldId, const BlockDataID& velocityFieldId,
              const BlockDataID& flagFieldId, const FlagUID& fluid, const FlagUID& obstacle, const Setup& setup,
              const bool logToStream = true, const bool logToFile = true,
              const std::string& filename = std::string("FlowAroundCylinder.txt"))
      : blocks_(blocks), executionCounter_(uint_t(0)), checkFrequency_(checkFrequency), rampUpTime_(rampUpTime),
        getFields_(getFields), pdfFieldId_(pdfFieldId), densityFieldId_(densityFieldId), velocityFieldId_(velocityFieldId),
        flagFieldId_(flagFieldId), fluid_(fluid), obstacle_(obstacle), setup_(setup), forceEvaluationExecutionCount_(uint_t(0)),
        strouhalNumberRealD_(real_t(0)), strouhalEvaluationExecutionCount_(uint_t(0)),
        logToStream_(logToStream), logToFile_(logToFile), filename_(filename)
   {
      forceSample_.resize(uint_t(2));
      coefficients_.resize(uint_t(4));
      coefficientExtremas_.resize(uint_t(4));

      WALBERLA_ROOT_SECTION()
      {
         if (logToFile_)
         {
            std::ofstream file(filename_.c_str());
            file << "# time step [1], force (x) [2], force (y) [3], force (z) [4], "
                    "cD (real area) [5], cL (real area) [6], cD (discrete area) [7], cL (discrete area) [8], "
                    "pressure difference (in lattice units) [9], pressure difference (in Pa) [10], vortex velocity (in "
                    "lattice units) [11], "
                    "Strouhal number (real D) [12], Strouhal number (discrete D) [13]"
                 << '\n';
            if (!setup_.evaluatePressure)
               file << "# ATTENTION: pressure was not evaluated, pressure difference is set to zero!" << '\n';
            if (!setup_.evaluateStrouhal)
               file << "# ATTENTION: vortex velocities were not evaluated, Strouhal number is set to zero!"
                    << '\n';
            file.close();
         }
      }
   }

   void operator()();
   void forceCalculation(IBlock* block, const uint_t level); // for calculating the force
   void resetForce();

   std::function<void (IBlock *, const uint_t)> forceCalculationFunctor()
   {
      return [this](IBlock* block, uint_t level) { forceCalculation(block, level); };
   }

   std::function<void()> resetForceFunctor()
   {
      return [this]() { resetForce(); };
   }


   void prepareResultsForSQL();
   void getResultsForSQLOnRoot(std::map< std::string, double >& realProperties,
                               std::map< std::string, int >& integerProperties) const;

   void check(const shared_ptr< Config >& config);

   void refresh();

 protected:
   void evaluate(real_t& cDRealArea, real_t& cLRealArea, real_t& cDDiscreteArea, real_t& cLDiscreteArea,
                 real_t& pressureDifference_L, real_t& pressureDifference);

   bool initialized_;

   weak_ptr< StructuredBlockStorage > blocks_;
   std::map< IBlock*, std::vector< std::pair< Cell, stencil::Direction > > > directions_;

   uint_t executionCounter_;
   uint_t checkFrequency_;
   uint_t rampUpTime_;

   VoidFunction & getFields_;

   BlockDataID pdfFieldId_;
   BlockDataID densityFieldId_;
   BlockDataID velocityFieldId_;
   BlockDataID flagFieldId_;

   FlagUID fluid_;
   FlagUID obstacle_;

   Setup setup_;

   uint_t D_;
   real_t AD_;
   real_t AL_;

   Vector3< real_t > force_;
   std::vector< math::Sample > forceSample_;
   uint_t forceEvaluationExecutionCount_;

   std::vector< std::deque< real_t > > coefficients_;
   std::vector< std::pair< real_t, real_t > > coefficientExtremas_;

   std::vector< real_t > strouhalVelocities_;
   std::vector< uint_t > strouhalTimeStep_;
   bool strouhalRising_;
   real_t strouhalNumberRealD_;
   real_t strouhalNumberDiscreteD_;
   uint_t strouhalEvaluationExecutionCount_;

   bool logToStream_;
   bool logToFile_;
   std::string filename_;

   std::map< std::string, double > sqlResults_;

}; // class Evaluation

class EvaluationRefresh
{
 public:
   EvaluationRefresh(const weak_ptr< Evaluation >& evaluation) : evaluation_(evaluation) {}

   void operator()()
   {
      auto evaluation = evaluation_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(evaluation);
      evaluation->refresh();
   }

   void operator()(BlockForest&, const PhantomBlockForest&)
   {
      auto evaluation = evaluation_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(evaluation);
      evaluation->refresh();
   }

 private:
   weak_ptr< Evaluation > evaluation_;
};

}