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
//! \file SQLProperties.cpp
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

#include "SQLProperties.h"

#include <blockforest/StructuredBlockForest.h>
#include <blockforest/loadbalancing/DynamicDiffusive.h>
#include <blockforest/loadbalancing/DynamicParMetis.h>
#include <core/DataTypes.h>
#include <core/waLBerlaBuildInfo.h>

namespace walberla {

void addBuildInfoToSQL( std::map< std::string, int64_t > &       /*integerProperties*/,
                        std::map< std::string, double > &        /*realProperties*/,
                        std::map< std::string, std::string > & stringProperties )
{
   stringProperties["walberla_sha"]    = walberla::core::buildinfo::gitSHA1();
   stringProperties["build_type"]      = walberla::core::buildinfo::buildType();
   stringProperties["compiler_flags"]  = walberla::core::buildinfo::compilerFlags();
   stringProperties["build_machine"]   = walberla::core::buildinfo::buildMachine();
}

void addDomainPropertiesToSQL( const std::shared_ptr< StructuredBlockForest > & forest,
                               std::map< std::string, int64_t > &          integerProperties,
                               std::map< std::string, double > &           /*realProperties*/,
                               std::map< std::string, std::string > &      /*stringProperties*/ )
{
   integerProperties["numMPIProcesses"]  = ::walberla::mpi::MPIManager::instance()->numProcesses();

   integerProperties[ "domainXMin" ] = ::walberla::int_c(forest->getDomain().xMin());
   integerProperties[ "domainXMax" ] = ::walberla::int_c(forest->getDomain().xMax());

   integerProperties[ "domainYMin" ] = ::walberla::int_c(forest->getDomain().yMin());
   integerProperties[ "domainYMax" ] = ::walberla::int_c(forest->getDomain().yMax());

   integerProperties[ "domainZMin" ] = ::walberla::int_c(forest->getDomain().zMin());
   integerProperties[ "domainZMax" ] = ::walberla::int_c(forest->getDomain().zMax());

   integerProperties[ "xBlocks" ] = ::walberla::int_c( forest->getXSize() );
   integerProperties[ "yBlocks" ] = ::walberla::int_c( forest->getYSize() );
   integerProperties[ "zBlocks" ] = ::walberla::int_c( forest->getZSize() );

   integerProperties["xCellsPerBlock"] = ::walberla::int_c(forest->getNumberOfXCellsPerBlock());
   integerProperties["yCellsPerBlock"] = ::walberla::int_c(forest->getNumberOfYCellsPerBlock());
   integerProperties["zCellsPerBlock"] = ::walberla::int_c(forest->getNumberOfZCellsPerBlock());

   integerProperties[ "xPeriodic" ] = ( forest->isXPeriodic() ? 1 : 0 );
   integerProperties[ "yPeriodic" ] = ( forest->isYPeriodic() ? 1 : 0 );
   integerProperties[ "zPeriodic" ] = ( forest->isZPeriodic() ? 1 : 0 );
}

void addLoadBalancingPropertiesToSQL( const ::walberla::blockforest::BlockForest& forest,
                                      std::map< std::string, int64_t > &          integerProperties,
                                      std::map< std::string, double > &         /*realProperties*/,
                                      std::map< std::string, std::string > &    /*stringProperties*/ )
{
   integerProperties[ "recalculateBlockLevelsInRefresh" ]                = ( forest.recalculateBlockLevelsInRefresh() ? 1 : 0 );
   integerProperties[ "alwaysRebalanceInRefresh" ]                       = ( forest.alwaysRebalanceInRefresh() ? 1 : 0 );
   integerProperties[ "reevaluateMinTargetLevelsAfterForcedRefinement" ] = ( forest.reevaluateMinTargetLevelsAfterForcedRefinement() ? 1 : 0 );
   integerProperties[ "allowRefreshChangingDepth" ]                      = ( forest.allowRefreshChangingDepth() ? 1 : 0 );
   integerProperties[ "allowMultipleRefreshCycles" ]                     = ( forest.allowMultipleRefreshCycles() ? 1 : 0 );
   integerProperties[ "checkForEarlyOutInRefresh" ]                      = ( forest.checkForEarlyOutInRefresh() ? 1 : 0 );
   integerProperties[ "checkForLateOutInRefresh" ]                       = ( forest.checkForLateOutInRefresh() ? 1 : 0 );
}

void addParMetisPropertiesToSQL( const ::walberla::blockforest::DynamicParMetis&         dpm,
                                 std::map< std::string, int64_t > &        /*integerProperties*/,
                                 std::map< std::string, double > &         realProperties,
                                 std::map< std::string, std::string > &    stringProperties )
{
   stringProperties["metisAlgorithm"]    = dpm.algorithmToString();
   stringProperties["metisEdgeSource"]   = dpm.edgeSourceToString();
   stringProperties["metisWeightsToUse"] = dpm.weightsToUseToString();
   realProperties["metisipc2redist"]     = dpm.getipc2redist();
}

std::string envToString(const char* env)
{
   return env != nullptr ? std::string(env) : "";
}

void addSlurmPropertiesToSQL( std::map< std::string, int64_t > &        /*integerProperties*/,
                              std::map< std::string, double > &         /*realProperties*/,
                              std::map< std::string, std::string > &    stringProperties )
{
   stringProperties["SLURM_CLUSTER_NAME"]       = envToString(std::getenv( "SLURM_CLUSTER_NAME" ));
   stringProperties["SLURM_CPUS_ON_NODE"]       = envToString(std::getenv( "SLURM_CPUS_ON_NODE" ));
   stringProperties["SLURM_CPUS_PER_TASK"]      = envToString(std::getenv( "SLURM_CPUS_PER_TASK" ));
   stringProperties["SLURM_JOB_ACCOUNT"]        = envToString(std::getenv( "SLURM_JOB_ACCOUNT" ));
   stringProperties["SLURM_JOB_ID"]             = envToString(std::getenv( "SLURM_JOB_ID" ));
   stringProperties["SLURM_JOB_CPUS_PER_NODE"]  = envToString(std::getenv( "SLURM_JOB_CPUS_PER_NODE" ));
   stringProperties["SLURM_JOB_NAME"]           = envToString(std::getenv( "SLURM_JOB_NAME" ));
   stringProperties["SLURM_JOB_NUM_NODES"]      = envToString(std::getenv( "SLURM_JOB_NUM_NODES" ));
   stringProperties["SLURM_NTASKS"]             = envToString(std::getenv( "SLURM_NTASKS" ));
   stringProperties["SLURM_NTASKS_PER_CORE"]    = envToString(std::getenv( "SLURM_NTASKS_PER_CORE" ));
   stringProperties["SLURM_NTASKS_PER_NODE"]    = envToString(std::getenv( "SLURM_NTASKS_PER_NODE" ));
   stringProperties["SLURM_NTASKS_PER_SOCKET"]  = envToString(std::getenv( "SLURM_NTASKS_PER_SOCKET" ));
   stringProperties["SLURM_CPU_BIND_TYPE"]      = envToString(std::getenv( "SLURM_CPU_BIND_TYPE" ));
}

} //namespace walberla
