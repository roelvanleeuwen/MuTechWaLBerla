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
//! \file
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//
//======================================================================================================================

#include <mesa_pd/collision_detection/AnalyticContactDetection.h>
#include <mesa_pd/data/HashGrids.h>
#include <mesa_pd/data/ParticleAccessor.h>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/ShapeStorage.h>
#include <mesa_pd/domain/BlockForestDomain.h>
#include <mesa_pd/kernel/DoubleCast.h>
#include <mesa_pd/kernel/ParticleSelector.h>
#include <mesa_pd/mpi/SyncNextNeighbors.h>

#include <blockforest/BlockForest.h>
#include <blockforest/Initialization.h>
#include <core/Environment.h>
#include <core/grid_generator/SCIterator.h>
#include <core/logging/Logging.h>
#include <core/math/Random.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>

namespace walberla {
namespace mesa_pd {

class ParticleAccessorWithShape : public data::ParticleAccessor
{
public:
   ParticleAccessorWithShape(std::shared_ptr<data::ParticleStorage>& ps, std::shared_ptr<data::ShapeStorage>& ss)
         : ParticleAccessor(ps)
         , ss_(ss)
   {}

   const auto& getInvMass(const size_t p_idx) const {return ss_->shapes[ps_->getShapeID(p_idx)]->getInvMass();}

   const auto& getInvInertiaBF(const size_t p_idx) const {return ss_->shapes[ps_->getShapeID(p_idx)]->getInvInertiaBF();}

   data::BaseShape* getShape(const size_t p_idx) const {return ss_->shapes[ps_->getShapeID(p_idx)].get();}
private:
   std::shared_ptr<data::ShapeStorage> ss_;
};

class comp
{
public:
   comp(std::vector<collision_detection::AnalyticContactDetection>& cs) : cs_(cs) {}
   bool operator()(const size_t& c1, const size_t& c2)
   {
      if (cs_[c1].getIdx1() == cs_[c2].getIdx1()) return cs_[c1].getIdx2() < cs_[c2].getIdx2();
      return cs_[c1].getIdx1() < cs_[c2].getIdx1();
   }
   std::vector<collision_detection::AnalyticContactDetection>& cs_;
};

/*
 * Generates particles randomly inside the domain.
 * Then checks if the Hash Grids find the same interaction pairs as the naive all-against-all check.
 * Similar to test in "LinkedCellsVsBruteForce.cpp"
 */
int main( int argc, char ** argv )
{
   Environment env(argc, argv);
   WALBERLA_UNUSED(env);
   walberla::mpi::MPIManager::instance()->useWorldComm();

   math::seedRandomGenerator( numeric_cast<std::mt19937::result_type>( 42 * walberla::mpi::MPIManager::instance()->rank() ) );

   //logging::Logging::instance()->setStreamLogLevel(logging::Logging::DETAIL);
   //logging::Logging::instance()->includeLoggingToFile("MESA_PD_Kernel_SyncNextNeighbor");
   //logging::Logging::instance()->setFileLogLevel(logging::Logging::DETAIL);

   //init domain partitioning
   auto forest = blockforest::createBlockForest( AABB(0,0,0,30,30,30), // simulation domain
                                                 Vector3<uint_t>(3,3,3), // blocks in each direction
                                                 Vector3<bool>(true, true, true) // periodicity
                                                 );
   domain::BlockForestDomain domain(forest);

   WALBERLA_CHECK_EQUAL(forest->size(), 1);
   const Block& blk = *static_cast<blockforest::Block*>(&*forest->begin());

   //init data structures
   auto ps = std::make_shared<data::ParticleStorage>(100);
   auto ss = std::make_shared<data::ShapeStorage>();
   data::HashGrids hg;
   std::vector<collision_detection::AnalyticContactDetection> csBF(100);
   std::vector<collision_detection::AnalyticContactDetection> csHG1(100);
   std::vector<collision_detection::AnalyticContactDetection> csHG2(100);
   std::vector<collision_detection::AnalyticContactDetection> csHG3(100);

   ParticleAccessorWithShape accessor(ps, ss);

   //initialize particles
   const real_t radius  = real_t(0.5);
   auto smallSphere = ss->create<data::Sphere>( radius );
   ss->shapes[smallSphere]->updateMassAndInertia(real_t(2707));

   for (int i = 0; i < 1000; ++i)
   {
      data::Particle&& p          = *ps->create();
      p.getPositionRef()          = Vec3( math::realRandom(blk.getAABB().xMin(), blk.getAABB().xMax()),
                                       math::realRandom(blk.getAABB().yMin(), blk.getAABB().yMax()),
                                       math::realRandom(blk.getAABB().zMin(), blk.getAABB().zMax()) );
      p.getInteractionRadiusRef() = radius;
      p.getShapeIDRef()           = smallSphere;
      p.getOwnerRef()             = walberla::mpi::MPIManager::instance()->rank();
   }

   //init kernels
   mpi::SyncNextNeighbors                 SNN;

   SNN(*ps, domain);

   ps->forEachParticlePairHalf(false,
                               kernel::SelectAll(),
                               accessor,
                               [&csBF](const size_t idx1, const size_t idx2, auto& ac)
   {
      collision_detection::AnalyticContactDetection         acd;
      kernel::DoubleCast               double_cast;
      if (double_cast(idx1, idx2, ac, acd, ac ))
      {
         csBF.push_back(acd);
      }
   },
   accessor );

   // insert into hash grids initially

   ps->forEachParticle(true, kernel::SelectAll(), accessor, hg, accessor);
   hg.forEachParticlePairHalf(false,
                              kernel::SelectAll(),
                              accessor,
                              [&csHG1](const size_t idx1, const size_t idx2, auto& ac)
   {
      collision_detection::AnalyticContactDetection         acd;
      kernel::DoubleCast               double_cast;
      if (double_cast(idx1, idx2, ac, acd, ac ))
      {
         csHG1.push_back(acd);
      }
   },
   accessor );

   WALBERLA_CHECK_EQUAL(csBF.size(), csHG1.size());
   WALBERLA_LOG_DEVEL(csBF.size() << " contacts detected");

   std::vector<size_t> csBF_idx(csBF.size());
   std::vector<size_t> csHG1_idx(csHG1.size());
   std::iota(csBF_idx.begin(), csBF_idx.end(), 0);
   std::iota(csHG1_idx.begin(), csHG1_idx.end(), 0);
   std::sort(csBF_idx.begin(), csBF_idx.end(), comp(csBF));
   std::sort(csHG1_idx.begin(), csHG1_idx.end(), comp(csHG1));

   for (size_t i = 0; i < csBF.size(); ++i)
   {
      WALBERLA_CHECK_EQUAL(csBF[csBF_idx[i]].getIdx1(), csHG1[csHG1_idx[i]].getIdx1());
      WALBERLA_CHECK_EQUAL(csBF[csBF_idx[i]].getIdx2(), csHG1[csHG1_idx[i]].getIdx2());
   }

   WALBERLA_LOG_DEVEL_ON_ROOT("Initial insertion checked");

   // redo to check clear
   hg.clear();
   ps->forEachParticle(true, kernel::SelectAll(), accessor, hg, accessor);
   hg.forEachParticlePairHalf(false,
                              kernel::SelectAll(),
                              accessor,
                              [&csHG2](const size_t idx1, const size_t idx2, auto& ac)
                              {
                                 collision_detection::AnalyticContactDetection         acd;
                                 kernel::DoubleCast               double_cast;
                                 if (double_cast(idx1, idx2, ac, acd, ac ))
                                 {
                                    csHG2.push_back(acd);
                                 }
                              },
                              accessor );

   WALBERLA_CHECK_EQUAL(csBF.size(), csHG2.size());

   std::vector<size_t> csHG2_idx(csHG2.size());
   std::iota(csHG2_idx.begin(), csHG2_idx.end(), 0);
   std::sort(csHG2_idx.begin(), csHG2_idx.end(), comp(csHG2));

   for (size_t i = 0; i < csBF.size(); ++i)
   {
      WALBERLA_CHECK_EQUAL(csBF[csBF_idx[i]].getIdx1(), csHG2[csHG2_idx[i]].getIdx1());
      WALBERLA_CHECK_EQUAL(csBF[csBF_idx[i]].getIdx2(), csHG2[csHG2_idx[i]].getIdx2());
   }

   WALBERLA_LOG_DEVEL_ON_ROOT("Insertion after clear checked");

   // redo to check clearAll
   hg.clearAll();
   ps->forEachParticle(true, kernel::SelectAll(), accessor, hg, accessor);
   hg.forEachParticlePairHalf(false,
                              kernel::SelectAll(),
                              accessor,
                              [&csHG3](const size_t idx1, const size_t idx2, auto& ac)
                              {
                                 collision_detection::AnalyticContactDetection         acd;
                                 kernel::DoubleCast               double_cast;
                                 if (double_cast(idx1, idx2, ac, acd, ac ))
                                 {
                                    csHG3.push_back(acd);
                                 }
                              },
                              accessor );

   WALBERLA_CHECK_EQUAL(csBF.size(), csHG3.size());

   std::vector<size_t> csHG3_idx(csHG3.size());
   csHG3_idx = std::vector<size_t>(csHG3.size());
   std::iota(csHG3_idx.begin(), csHG3_idx.end(), 0);
   std::sort(csHG3_idx.begin(), csHG3_idx.end(), comp(csHG3));

   for (size_t i = 0; i < csBF.size(); ++i)
   {
      WALBERLA_CHECK_EQUAL(csBF[csBF_idx[i]].getIdx1(), csHG3[csHG3_idx[i]].getIdx1());
      WALBERLA_CHECK_EQUAL(csBF[csBF_idx[i]].getIdx2(), csHG3[csHG3_idx[i]].getIdx2());
   }

   WALBERLA_LOG_DEVEL_ON_ROOT("Insertion after clear checked");

   return EXIT_SUCCESS;
}

} //namespace mesa_pd
} //namespace walberla

int main( int argc, char ** argv )
{
   return walberla::mesa_pd::main(argc, argv);
}
