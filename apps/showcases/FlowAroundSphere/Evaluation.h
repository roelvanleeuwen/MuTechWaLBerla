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
#include "core/math/Constants.h"

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
#include "FlowAroundSphereInfoHeader.h"
using namespace walberla;

using StorageSpecification_T = lbm::FlowAroundSphereStorageSpecification;
using Stencil_T              = StorageSpecification_T::Stencil;
using PdfField_T             = lbm_generated::PdfField< StorageSpecification_T >;
using FlagField_T            = FlagField< uint8_t >;
using BoundaryCollection_T = lbm::FlowAroundSphereBoundaryCollection< FlagField_T >;

using VoidFunction = std::function<void ()>;

namespace walberla
{

struct DragCoefficient {
   uint_t timestep;
   double Fx;
   double Fy;
   double Fz;
   double cDRealArea;
   double cLRealArea;
   double cDDiscreteArea;
   double cLDiscreteArea;
   DragCoefficient(uint_t t, Vector3<double> f, double cdR, double clR, double cdD, double clD) : timestep(t), Fx(f[0]), Fy(f[1]), Fz(f[2]), cDRealArea(cdR), cLRealArea(clR), cDDiscreteArea(cdD), cLDiscreteArea(clD) {}
};

class Evaluation
{
 public:
   Evaluation(const weak_ptr< StructuredBlockStorage >& blocks, const uint_t checkFrequency, const uint_t rampUpTime,
              BoundaryCollection_T & boundaryCollection,
              const BlockDataID& pdfFieldId, const BlockDataID& densityFieldId, const BlockDataID& velocityFieldId,
              const BlockDataID& flagFieldId, const FlagUID& fluid, const FlagUID& obstacle, const Setup& setup,
              const bool logToStream = true, const bool logToFile = true,
              const std::string& filename = std::string("FlowAroundSphere.txt"))
      : blocks_(blocks), executionCounter_(uint_t(0)), checkFrequency_(checkFrequency), rampUpTime_(rampUpTime),
        boundaryCollection_(boundaryCollection), pdfFieldId_(pdfFieldId), densityFieldId_(densityFieldId), velocityFieldId_(velocityFieldId),
        flagFieldId_(flagFieldId), fluid_(fluid), obstacle_(obstacle), setup_(setup),
        strouhalNumberRealD_(real_t(0)), strouhalEvaluationExecutionCount_(uint_t(0)),
        logToStream_(logToStream), logToFile_(logToFile), filename_(filename)
   {
      coefficients_.resize(uint_t(4));
      coefficientExtremas_.resize(uint_t(4));

      diameterSphere    = real_c(2.0) * setup_.sphereRadius;
      surfaceAreaSphere = math::pi * diameterSphere * diameterSphere;
      meanVelocity      = setup_.inflowVelocity; // (real_c(4.0) * setup_.inflowVelocity) / real_c(9.0);

      WALBERLA_ROOT_SECTION()
      {
         if (logToFile_)
         {
            std::ofstream file(filename_.c_str());
            file << "# time step [1], force (x) [2], force (y) [3], force (z) [4], "
                    "cD [5], cL [6], "
                    "pressure difference (in lattice units) [7], pressure difference (in Pa) [8], vortex velocity (in "
                    "lattice units) [9], "
                    "Strouhal number (real D) [10]"
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
   void writeDrag()
   {
      filesystem::path dataFile( "dragCoefficient.csv" );
      if( filesystem::exists( dataFile ) )
         std::remove( dataFile.string().c_str() );

      std::ofstream outfile( "dragCoefficient.csv" );
      outfile << "SEP=," << "\n";
      outfile << "timestep," << "Fx," << "Fy," << "Fz," << "cDRealArea," << "cLRealArea," << "cDDiscreteArea," << "cLDiscreteArea" << "\n";

      for(auto it = dragResults.begin(); it != dragResults.end(); ++it)
      {
         outfile << it->timestep << ",";
         outfile << it->Fx << "," << it->Fy << "," << it->Fz << ",";
         outfile << it->cDRealArea << ",";
         outfile << it->cLRealArea << ",";
         outfile << it->cDDiscreteArea << ",";
         outfile << it->cLDiscreteArea;
         outfile << "\n";
      }
      outfile.close();
   };

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
   void refresh();

 protected:
   void evaluate(real_t& cDRealArea, real_t& cLRealArea, real_t& cDDiscreteArea, real_t& cLDiscreteArea,
                 real_t& pressureDifference_L, real_t& pressureDifference);

   bool initialized_{false};

   weak_ptr< StructuredBlockStorage > blocks_;

   uint_t executionCounter_;
   uint_t checkFrequency_;
   uint_t rampUpTime_;

   BoundaryCollection_T & boundaryCollection_;

   BlockDataID pdfFieldId_;
   BlockDataID densityFieldId_;
   BlockDataID velocityFieldId_;
   BlockDataID flagFieldId_;

   FlagUID fluid_;
   FlagUID obstacle_;

   Setup setup_;

   real_t diameterSphere;
   real_t surfaceAreaSphere;
   real_t meanVelocity;

   Vector3< real_t > force_;
   real_t AD_;
   real_t AL_;
   std::vector<DragCoefficient> dragResults;

   std::vector< std::deque< real_t > > coefficients_;
   std::vector< std::pair< real_t, real_t > > coefficientExtremas_;

   std::vector< real_t > strouhalVelocities_;
   std::vector< uint_t > strouhalTimeStep_;
   bool strouhalRising_{false};
   real_t strouhalNumberRealD_;
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
      WALBERLA_CHECK_NOT_NULLPTR(evaluation)
      evaluation->refresh();
   }

   void operator()(BlockForest&, const PhantomBlockForest&)
   {
      auto evaluation = evaluation_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(evaluation)
      evaluation->refresh();
   }

 private:
   weak_ptr< Evaluation > evaluation_;
};

}