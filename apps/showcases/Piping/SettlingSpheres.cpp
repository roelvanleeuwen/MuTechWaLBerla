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
//! \file   SettlingSpheres.cpp
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//! \brief Based on showcases/Antidunes/BedGeneration.cpp
//
//======================================================================================================================

#include "blockforest/Initialization.h"

#include "core/Environment.h"
#include "core/grid_generator/SCIterator.h"
#include "core/math/Random.h"
#include "core/mpi/Reduce.h"

#include "mesa_pd/collision_detection/AnalyticContactDetection.h"
#include "mesa_pd/data/DataTypes.h"
#include "mesa_pd/data/LinkedCells.h"
#include "mesa_pd/data/ParticleAccessorWithBaseShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/shape/Sphere.h"
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/kernel/AssocToBlock.h"
#include "mesa_pd/kernel/DoubleCast.h"
#include "mesa_pd/kernel/InsertParticleIntoLinkedCells.h"
#include "mesa_pd/kernel/LinearSpringDashpot.h"
#include "mesa_pd/kernel/ParticleSelector.h"
#include "mesa_pd/kernel/VelocityVerlet.h"
#include "mesa_pd/mpi/ContactFilter.h"
#include "mesa_pd/mpi/ReduceContactHistory.h"
#include "mesa_pd/mpi/ReduceProperty.h"
#include "mesa_pd/mpi/SyncNextNeighbors.h"
#include "mesa_pd/mpi/notifications/ForceTorqueNotification.h"
#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "vtk/VTKOutput.h"

#include "Utility.h"

namespace walberla
{
namespace piping
{

using namespace mesa_pd;

int main(int argc, char** argv)
{
   Environment env(argc, argv);
   walberla::mpi::MPIManager::instance()->useWorldComm();

   // Config
   auto cfg = env.config();
   if (cfg == nullptr) WALBERLA_ABORT("No config specified!");
   WALBERLA_LOG_INFO_ON_ROOT(*cfg);
   const Config::BlockHandle bedGenerationConf = cfg->getBlock("BedGeneration");

   Vec3 domainSize_SI            = bedGenerationConf.getParameter< Vec3 >("domainSize_SI");
   Vector3< int > blocks         = bedGenerationConf.getParameter< Vector3< int > >("blocks");
   real_t minDiameter_SI         = bedGenerationConf.getParameter< real_t >("minDiameter_SI");
   real_t maxDiameter_SI         = bedGenerationConf.getParameter< real_t >("maxDiameter_SI");
   real_t gravity_SI             = bedGenerationConf.getParameter< real_t >("gravity_SI");
   real_t densityFluid_SI        = bedGenerationConf.getParameter< real_t >("densityFluid_SI");
   real_t densityParticle_SI     = bedGenerationConf.getParameter< real_t >("densityParticle_SI");
   real_t generationSpacing_SI   = bedGenerationConf.getParameter< real_t >("generationSpacing_SI");
   real_t initialVelocity_SI     = bedGenerationConf.getParameter< real_t >("initialVelocity_SI");
   real_t dt_SI                  = bedGenerationConf.getParameter< real_t >("dt_SI");
   real_t frictionCoefficient    = bedGenerationConf.getParameter< real_t >("frictionCoefficient");
   real_t restitutionCoefficient = bedGenerationConf.getParameter< real_t >("restitutionCoefficient");
   real_t collisionTime_SI       = bedGenerationConf.getParameter< real_t >("collisionTime_SI");
   real_t poissonsRatio          = bedGenerationConf.getParameter< real_t >("poissonsRatio");
   uint_t timeSteps              = bedGenerationConf.getParameter< uint_t >("timeSteps");
   uint_t visSpacing             = bedGenerationConf.getParameter< uint_t >("visSpacing");
   std::string outFileName       = bedGenerationConf.getParameter< std::string >("outFileName");

   // BlockForest
   math::AABB simulationDomain_SI(real_t(0.0), real_t(0.0), real_t(0.0), domainSize_SI[0], domainSize_SI[1],
                                  domainSize_SI[2]);
   Vector3< bool > isPeriodic{ true, true, false };

   shared_ptr< BlockForest > forest = blockforest::createBlockForest(simulationDomain_SI, blocks, isPeriodic);
   auto domain                      = std::make_shared< mesa_pd::domain::BlockForestDomain >(forest);

   // MesaPD data structures
   auto ps = std::make_shared< data::ParticleStorage >(1);
   data::ParticleAccessorWithBaseShape accessor(ps);

   // Init spheres
   // Use offset to domain boundary to prevent the spheres from touching in the beginning
   real_t domainOffset = maxDiameter_SI / real_t(2);
   math::AABB generationDomain_SI(simulationDomain_SI.xMin() + domainOffset, simulationDomain_SI.yMin() + domainOffset,
                                  simulationDomain_SI.zMin() + domainOffset, simulationDomain_SI.xMax() - domainOffset,
                                  simulationDomain_SI.yMax() - domainOffset, simulationDomain_SI.zMax() - domainOffset);

   for (auto pt :
        grid_generator::SCGrid(generationDomain_SI, Vec3(generationSpacing_SI) * real_c(0.5), generationSpacing_SI))
   {
      auto diameter = math::realRandom< real_t >(minDiameter_SI, maxDiameter_SI);

      if (!domain->isContainedInLocalSubdomain(pt, real_t(0))) continue;
      auto p                       = ps->create();
      p->getPositionRef()          = pt;
      p->getInteractionRadiusRef() = diameter * real_t(0.5);
      p->getBaseShapeRef()         = std::make_shared< data::Sphere >(p->getInteractionRadius());
      p->getBaseShapeRef()->updateMassAndInertia(densityParticle_SI);

      p->setLinearVelocity(Vec3(real_t(0.1) * math::realRandom(-initialVelocity_SI, initialVelocity_SI),
                                real_t(0.1) * math::realRandom(-initialVelocity_SI, initialVelocity_SI),
                                -initialVelocity_SI));
      p->getOwnerRef() = walberla::mpi::MPIManager::instance()->rank();
      p->getTypeRef()  = 0;
   }

   uint_t numParticles = ps->size();
   walberla::mpi::reduceInplace(numParticles, walberla::mpi::SUM);

   createPlane(*ps, simulationDomain_SI.minCorner(), Vec3(real_t(0), real_t(0), real_t(1)));
   createPlane(*ps, simulationDomain_SI.maxCorner(), Vec3(real_t(0), real_t(0), real_t(-1)));

   // VTK
   auto vtkDomainOutput = walberla::vtk::createVTKOutput_DomainDecomposition(forest, "domain_decomposition", 1,
                                                                             "vtk_settling_spheres", "simulation_step");
   vtkDomainOutput->write();

   auto particleVtkOutput = make_shared< mesa_pd::vtk::ParticleVtkOutput >(ps);
   particleVtkOutput->addOutput< mesa_pd::data::SelectParticleLinearVelocity >("velocity");
   particleVtkOutput->addOutput< mesa_pd::data::SelectParticleInteractionRadius >("radius");
   particleVtkOutput->setParticleSelector([](const data::ParticleStorage::iterator& pIt) {
      using namespace walberla::mesa_pd::data::particle_flags;
      return (pIt->getBaseShape()->getShapeType() == data::Sphere::SHAPE_TYPE) && !isSet(pIt->getFlags(), GHOST);
   });
   auto vtkWriter = walberla::vtk::createVTKOutput_PointData(particleVtkOutput, "Particles", 1, "vtk_settling_spheres",
                                                             "simulation_step", false, false);

   // Init kernels
   mesa_pd::kernel::VelocityVerletPreForceUpdate vvIntegratorPreForce(dt_SI);
   mesa_pd::kernel::VelocityVerletPostForceUpdate vvIntegratorPostForce(dt_SI);
   kernel::LinearSpringDashpot dem(1);
   dem.setFrictionCoefficientDynamic(0, 0, frictionCoefficient);
   real_t kappa = real_t(2) * (real_t(1) - poissonsRatio) / (real_t(2) - poissonsRatio); // from Thornton et al

   kernel::AssocToBlock assoc(forest);
   mesa_pd::mpi::ReduceProperty RP;
   mesa_pd::mpi::ReduceContactHistory reduceAndSwapContactHistory;
   mesa_pd::mpi::SyncNextNeighbors SNN;

   ps->forEachParticle(false, kernel::SelectLocal(), accessor, assoc, accessor);

   // initial sync
   SNN(*ps, *domain);

   real_t linkedCellWidth = 1.01_r * maxDiameter_SI;
   data::LinkedCells linkedCells(domain->getUnionOfLocalAABBs().getExtended(linkedCellWidth), linkedCellWidth);
   kernel::InsertParticleIntoLinkedCells ipilc;

   WcTimer timer;
   timer.start();
   for (uint_t i = 0; i < timeSteps; ++i)
   {
      if (i % visSpacing == 0) { vtkWriter->write(); }

      ps->forEachParticle(false, kernel::SelectLocal(), accessor, assoc, accessor);

      ps->forEachParticle(false, kernel::SelectLocal(), accessor, vvIntegratorPreForce, accessor);

      SNN(*ps, *domain);

      // gravity - buoyancy
      ps->forEachParticle(
         false, kernel::SelectLocal(), accessor,
         [densityParticle_SI, densityFluid_SI, gravity_SI](const size_t idx, auto& ac) {
            mesa_pd::addForceAtomic(
               idx, ac, Vec3(0, 0, -(densityParticle_SI - densityFluid_SI) * ac.getVolume(idx) * gravity_SI));
         },
         accessor);

      linkedCells.clear();
      ps->forEachParticle(false, kernel::SelectAll(), accessor, ipilc, accessor, linkedCells);
      linkedCells.forEachParticlePairHalf(
         false, kernel::ExcludeInfiniteInfinite(), accessor,
         [restitutionCoefficient, collisionTime_SI, kappa, domain, &dem, dt_SI](const size_t idx1, const size_t idx2,
                                                                                auto& ac) {
            kernel::DoubleCast double_cast;
            mesa_pd::mpi::ContactFilter contact_filter;
            collision_detection::AnalyticContactDetection acd;

            if (double_cast(idx1, idx2, ac, acd, ac))
            {
               if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *domain))
               {
                  auto meff = real_t(1) / (ac.getInvMass(idx1) + ac.getInvMass(idx2));
                  dem.setStiffnessAndDamping(0, 0, restitutionCoefficient, collisionTime_SI, kappa, meff);
                  dem(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), acd.getContactNormal(),
                      acd.getPenetrationDepth(), dt_SI);
               }
            }
         },
         accessor);

      reduceAndSwapContactHistory(*ps);

      RP.operator()< ForceTorqueNotification >(*ps);

      ps->forEachParticle(false, kernel::SelectLocal(), accessor, vvIntegratorPostForce, accessor);
   }
   timer.end();

   auto timer_reduced = walberla::timing::getReduced(timer, timing::REDUCE_TOTAL, 0);
   WALBERLA_ROOT_SECTION()
   {
      WALBERLA_LOG_INFO_ON_ROOT(*timer_reduced);
      real_t PUpS = real_t(numParticles) * real_t(timeSteps) / real_t(timer_reduced->max());
      WALBERLA_LOG_INFO_ON_ROOT("PUpS: " << PUpS);
   }

   writeSphereInformationToFile(outFileName, *ps, domainSize_SI);

   return EXIT_SUCCESS;
}
} // namespace piping
} // namespace walberla

int main(int argc, char** argv) { return walberla::piping::main(argc, argv); }
