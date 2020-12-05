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
//! \file BasicExports.cpp
//! \ingroup python_coupling
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

// Do not reorder includes - the include order is important
#include "python_coupling/PythonWrapper.h"

#ifdef WALBERLA_BUILD_WITH_PYTHON

#include "BasicExports.h"
#include "MPIExport.h"
#include "core/waLBerlaBuildInfo.h"
#include "core/logging/Logging.h"
#include "core/Abort.h"
#include "core/cell/CellInterval.h"
#include "core/math/AABB.h"
#include "core/mpi/MPIIO.h"
#include "core/timing/ReduceType.h"
#include "core/timing/TimingPool.h"
#include "core/timing/TimingTree.h"
#include "communication/UniformPackInfo.h"
#include "communication/UniformMPIDatatypeInfo.h"
#include "domain_decomposition/StructuredBlockStorage.h"
#include "python_coupling/Manager.h"
#include "python_coupling/helper/BlockStorageExportHelpers.h"
#include "stencil/Directions.h"

#include "field/GhostLayerField.h"

#include <functional>

// specialize operator== since == is deprecated in pybind11
template<>
bool walberla::domain_decomposition::internal::BlockData::Data< pybind11::object >::operator==(
   const BlockData::DataBase& rhs) const
{
   const Data< pybind11::object >* rhsData = dynamic_cast< const Data< pybind11::object >* >(&rhs);
   return (rhsData == &rhs) && (data_->is(*(rhsData->data_)));
}

namespace py = pybind11;
namespace walberla {
namespace python_coupling {

//======================================================================================================================
//
//  Helper Functions
//
//======================================================================================================================

void checkForThreeSequence( const py::object & o, const char * message )
{
   // strange construct required because also the len function throws , if object has no len
   try {
      if ( py::len(o ) != 3 )  throw py::error_already_set();
   }
   catch( py::error_already_set & ) {
      throw py::cast_error(message);
   }
}

//======================================================================================================================
//
//  CellInterval
//
//======================================================================================================================


void cellInterval_setMin( CellInterval & ci, const Cell & min ) {
   ci.min() = min;
}
void cellInterval_setMax( CellInterval & ci, const Cell & max ) {
   ci.max() = max;
}
void cellInterval_shift( CellInterval & ci, cell_idx_t xShift, cell_idx_t yShift, cell_idx_t zShift ) {
   ci.shift( xShift, yShift, zShift );
}

py::tuple cellInterval_size( CellInterval & ci ) {
   return py::make_tuple( ci.xSize(), ci.ySize(), ci.zSize() );
}

CellInterval cellInterval_getIntersection( CellInterval & ci1, CellInterval & ci2 )
{
   CellInterval result ( ci1 );
   result.intersect( ci2 );
   return result;
}

CellInterval cellInterval_getShifted( CellInterval & ci1, cell_idx_t xShift, cell_idx_t yShift, cell_idx_t zShift )
{
   CellInterval result ( ci1 );
   result.shift( xShift, yShift, zShift  );
   return result;
}

CellInterval cellInterval_getExpanded1( CellInterval & ci1, cell_idx_t expandVal )
{
   CellInterval result ( ci1 );
   result.expand( expandVal  );
   return result;
}

CellInterval cellInterval_getExpanded2( CellInterval & ci1, cell_idx_t xExpand, cell_idx_t yExpand, cell_idx_t zExpand )
{
   CellInterval result ( ci1 );
   result.expand( Cell(xExpand, yExpand, zExpand)  );
   return result;
}

void exportCellInterval(py::module_ &m)
{
   const Cell & ( CellInterval::*p_getMin )( ) const = &CellInterval::min;
   const Cell & ( CellInterval::*p_getMax )( ) const = &CellInterval::max;

   bool ( CellInterval::*p_contains1) ( const Cell         & ) const = &CellInterval::contains;
   bool ( CellInterval::*p_contains2) ( const CellInterval & ) const = &CellInterval::contains;

   void ( CellInterval::*p_expand1) ( const cell_idx_t ) = &CellInterval::expand;
   void ( CellInterval::*p_expand2) ( const Cell &     ) = &CellInterval::expand;

   bool          ( CellInterval::*p_overlaps ) ( const CellInterval & ) const = &CellInterval::overlaps;

   py::class_<CellInterval>(m, "CellInterval")
      .def( py::init<const Cell&, const Cell&>() )
      .def( py::init<cell_idx_t, cell_idx_t, cell_idx_t, cell_idx_t, cell_idx_t, cell_idx_t>() )
      .def_property( "min", p_getMin, &cellInterval_setMin )
      .def_property( "max", p_getMax, &cellInterval_setMax )
      .def_property_readonly( "size", &cellInterval_size  )
      .def( "empty", &CellInterval::empty )
      .def( "positiveIndicesOnly", &CellInterval::positiveIndicesOnly )
      .def( "contains",        p_contains1 )
      .def( "contains",        p_contains2 )
      .def( "overlaps",        p_overlaps )
      .def( "shift",           &cellInterval_shift )
      .def( "getShifted",      &cellInterval_getShifted )
      .def( "expand",          p_expand1 )
      .def( "expand",          p_expand2 )
      .def( "getExpanded",     &cellInterval_getExpanded1 )
      .def( "getExpanded",     &cellInterval_getExpanded2 )
      .def( "intersect",       &CellInterval::intersect )
      .def( "getIntersection", &cellInterval_getIntersection )
      .def("__eq__",           &CellInterval::operator==)
      .def("__ne__",           &CellInterval::operator!=)
      .def_property_readonly( "numCells",  &CellInterval::numCells  )
      ;
}

//======================================================================================================================
//
//  AABB
//
//======================================================================================================================


py::tuple aabb_getMin( const AABB & domainBB ) {
   return py::make_tuple( domainBB.xMin(), domainBB.yMin(), domainBB.zMin() );
}
py::tuple aabb_getMax( const AABB & domainBB ) {
   return py::make_tuple( domainBB.xMax(), domainBB.yMax(), domainBB.zMax() );
}

void aabb_setMin( AABB & domainBB, const py::tuple& min )
{
   checkForThreeSequence(min, "Error assigning minimum of AABB - Sequence of length 3 required" );
   real_t min0 = py::cast<real_t>( min[0] );
   real_t min1 = py::cast<real_t>( min[1] );
   real_t min2 = py::cast<real_t>( min[2] );

   domainBB = AABB( domainBB.max(), AABB::vector_type ( min0, min1, min2 ) );
}

void aabb_setMax( AABB & domainBB, const py::tuple& max )
{
   checkForThreeSequence( max, "Error assigning maximum of AABB - Sequence of length 3 required" );

   real_t max0 = py::cast<real_t>( max[0] );
   real_t max1 = py::cast<real_t>( max[1] );
   real_t max2 = py::cast<real_t>( max[2] );

   domainBB = AABB( domainBB.min(), AABB::vector_type ( max0, max1, max2 ) );
}


void exportAABB(py::module_ &m)
{
   bool ( AABB::*p_containsBB  )( const AABB & bb           )     const = &AABB::contains;
   bool ( AABB::*p_containsVec )( const Vector3<real_t> & point ) const = &AABB::contains;


   bool ( AABB::*p_containsClosedInterval1 ) ( const Vector3<real_t> & ) const               = &AABB::containsClosedInterval;
   bool ( AABB::*p_containsClosedInterval2 ) ( const Vector3<real_t> &, const real_t ) const = &AABB::containsClosedInterval;

   AABB ( AABB::*p_getExtended1 ) ( const real_t            ) const = &AABB::getExtended;
   AABB ( AABB::*p_getExtended2 ) ( const Vector3<real_t> & ) const = &AABB::getExtended;

   AABB ( AABB::*p_getScaled1 ) ( const real_t            ) const = &AABB::getScaled;
   AABB ( AABB::*p_getScaled2 ) ( const Vector3<real_t> & ) const = &AABB::getScaled;

   AABB ( AABB::*p_getMerged1 ) ( const Vector3<real_t> & ) const = &AABB::getMerged;
   AABB ( AABB::*p_getMerged2 ) ( const AABB            & ) const = &AABB::getMerged;

   bool ( AABB::*p_intersects1  )( const AABB & bb            ) const = &AABB::intersects;
   bool ( AABB::*p_intersects2  )( const AABB & bb, real_t dx ) const = &AABB::intersects;

   bool ( AABB::*p_intersectsClosed1  )( const AABB & bb            ) const = &AABB::intersectsClosedInterval;
   bool ( AABB::*p_intersectsClosed2  )( const AABB & bb, real_t dx ) const = &AABB::intersectsClosedInterval;

   void ( AABB::*p_extend1 ) ( const real_t            ) = &AABB::extend;
   void ( AABB::*p_extend2 ) ( const Vector3<real_t> & ) = &AABB::extend;

   void  ( AABB::*p_scale1 ) ( const real_t            )  = &AABB::scale;
   void  ( AABB::*p_scale2 ) ( const Vector3<real_t> & )  = &AABB::scale;

   void  ( AABB::*p_merge1 ) ( const Vector3<real_t> & )  = &AABB::merge;
   void  ( AABB::*p_merge2 ) ( const AABB            & )  = &AABB::merge;

   real_t  ( AABB::*p_sqDistance1 ) ( const AABB & )            const = &AABB::sqDistance;
   real_t  ( AABB::*p_sqDistance2 ) ( const Vector3<real_t> & ) const = &AABB::sqDistance;

   real_t  ( AABB::*p_sqMaxDistance1 ) ( const AABB & )            const = &AABB::sqMaxDistance;
   real_t  ( AABB::*p_sqMaxDistance2 ) ( const Vector3<real_t> & ) const = &AABB::sqMaxDistance;


   py::class_<AABB>(m, "AABB")
      .def( py::init<real_t,real_t,real_t,real_t,real_t,real_t>() )
      .def( py::init<Vector3<real_t>,Vector3<real_t> >() )
      .def("__eq__",     &walberla::math::operator==<real_t, real_t > )
      .def("__ne__",     &walberla::math::operator!=<real_t, real_t > )
      .def_property( "min",  &aabb_getMin, &aabb_setMin )
      .def_property( "max",  &aabb_getMax, &aabb_setMax )
      .def_property_readonly( "size", &AABB::sizes )
      .def( "empty",    &AABB::empty )
      .def( "volume", &AABB::volume )
      .def( "center", &AABB::center )
      .def( "contains", p_containsBB )
      .def( "contains", p_containsVec )
      .def( "containsClosedInterval", p_containsClosedInterval1 )
      .def( "containsClosedInterval", p_containsClosedInterval2 )
      .def( "getExtended", p_getExtended1 )
      .def( "getExtended", p_getExtended2 )
      .def( "getTranslated", &AABB::getTranslated )
      .def( "getScaled", p_getScaled1 )
      .def( "getScaled", p_getScaled2 )
      .def( "getMerged", p_getMerged1 )
      .def( "getMerged", p_getMerged2 )
      .def( "intersects", p_intersects1 )
      .def( "intersects", p_intersects2 )
      .def( "intersectsClosedInterval", p_intersectsClosed1 )
      .def( "intersectsClosedInterval", p_intersectsClosed2 )
      .def( "intersectionVolume", &AABB::intersectionVolume )
      .def( "getIntersection", &AABB::getIntersection )
      .def( "isIdentical", &AABB::isIdentical )
      .def( "isEqual",     &AABB::isEqual )
      .def( "sqDistance",       p_sqDistance1 )
      .def( "sqDistance",       p_sqDistance2 )
      .def( "sqMaxDistance",    p_sqMaxDistance1 )
      .def( "sqMaxDistance",    p_sqMaxDistance2 )
      .def( "sqSignedDistance", &AABB::sqSignedDistance)
      .def( "distance"        , &AABB::distance)
      .def( "signedDistance"  , &AABB::signedDistance)
      .def( "maxDistance"     , &AABB::maxDistance)
      .def( "extend", p_extend1 )
      .def( "extend", p_extend2 )
      .def( "translate", &AABB::translate )
      .def( "scale", p_scale1 )
      .def( "scale", p_scale2 )
      .def( "merge", p_merge1 )
      .def( "merge", p_merge2 )
      .def( "intersect", &AABB::intersect )
      ;
}

//======================================================================================================================
//
//  Timing
//
//======================================================================================================================

py::dict buildDictFromTimingNode(const WcTimingNode & tn)
{
   py::dict result;

   result["all"] = tn.timer_;
   for ( auto it = tn.tree_.begin(); it != tn.tree_.end(); ++it)
   {
      std::string key = it->first;
      if (it->second.tree_.empty())
      {
         result[key.c_str()] = it->second.timer_;
      } else
      {
         result[key.c_str()] = buildDictFromTimingNode(it->second);
      }
   }

   return result;
}

py::dict buildDictFromTimingTree(const WcTimingTree & tt)
{
   return buildDictFromTimingNode( tt.getRawData() );
}

void timingTreeStopWrapper(WcTimingTree & tt, const std::string& name)
{
   if (!tt.isTimerRunning(name))
   {
      throw py::value_error(("Timer '" + name + "' is currently not running!").c_str());
   }
   tt.stop(name);
}

void exportTiming(py::module_ &m)
{
   py::class_<WcTimer> (m, "Timer")
      .def( py::init<>() )
      .def( "start",  &WcTimer::start )
      .def( "stop",   &WcTimer::end   )
      .def( "reset",  &WcTimer::reset )
      .def( "merge",  &WcTimer::merge )
      .def_property_readonly( "counter",      &WcTimer::getCounter   )
      .def_property_readonly( "total",        &WcTimer::total        )
      .def_property_readonly( "sumOfSquares", &WcTimer::sumOfSquares )
      .def_property_readonly( "average",      &WcTimer::average      )
      .def_property_readonly( "variance",     &WcTimer::variance     )
      .def_property_readonly( "min",          &WcTimer::min          )
      .def_property_readonly( "max",          &WcTimer::max          )
      .def_property_readonly( "last",         &WcTimer::last         )
      ;


   WcTimer & ( WcTimingPool::*pGetItem ) ( const std::string & ) = &WcTimingPool::operator[];

   {
      py::scope classScope =
      py::class_<WcTimingPool, shared_ptr<WcTimingPool> > (m, "TimingPool")
         .def( py::init<>() )
         .def_property_readonly( "__getitem__",     pGetItem)
         .def( "__contains__",    &WcTimingPool::timerExists )
         .def( "getReduced",      &WcTimingPool::getReduced)
         .def( "merge",           &WcTimingPool::merge)
         .def( "clear",           &WcTimingPool::clear )
         .def( "unifyRegisteredTimersAcrossProcesses", &WcTimingPool::unifyRegisteredTimersAcrossProcesses )
         .def( "logResultOnRoot", &WcTimingPool::logResultOnRoot)
         ;
      WALBERLA_UNUSED( classScope );

      py::enum_<timing::ReduceType>(m, "ReduceType")
          .value("min"  , timing::REDUCE_MIN)
          .value("avg"  , timing::REDUCE_AVG)
          .value("max"  , timing::REDUCE_MAX)
          .value("total", timing::REDUCE_TOTAL)
          .export_values()
          ;
   }

   const WcTimer & ( WcTimingTree::*pTimingTreeGet ) ( const std::string & ) const = &WcTimingTree::operator[];
   py::class_<WcTimingTree, shared_ptr<WcTimingTree> > (m, "TimingTree")
         .def( py::init<>() )
         .def_property_readonly( "__getitem__",  pTimingTreeGet )
         .def( "start",        &WcTimingTree::start )
         .def( "stop",         &timingTreeStopWrapper )
         .def( "getReduced",   &WcTimingTree::getReduced )
         .def( "toDict",       &buildDictFromTimingTree )
    ;
}




//======================================================================================================================
//
//  IBlock
//
//======================================================================================================================

py::object IBlock_getData( py::object iblockObject, const std::string & stringID ) //NOLINT
{
   typedef GhostLayerField< double, 1 > GlField_T;
   IBlock * block = py::cast<IBlock*>( iblockObject );

   BlockDataID id = blockDataIDFromString( *block, stringID );

   auto manager = python_coupling::Manager::instance();
   py::object res =  manager->pythonObjectFromBlockData( *block, id );

   //if ( res.is(py::object()) )
   //   throw BlockDataNotConvertible();

   // py::cast(block->getData<GlField_T>(id))

   return manager->pythonObjectFromBlockData( *block, id );
}


py::list IBlock_blockDataList( py::object iblockObject ) //NOLINT
{
   IBlock * block = py::cast<IBlock*>( iblockObject );

   const std::vector<std::string> & stringIds = block->getBlockStorage().getBlockDataIdentifiers();

   py::list resultList;

   for( auto it = stringIds.begin(); it != stringIds.end(); ++it ) {
      try {
         resultList.append( py::make_tuple( *it, IBlock_getData( iblockObject, *it) ) );
      }
      catch( BlockDataNotConvertible & /*e*/ ) {
      }
   }

   return resultList;
}

py::object IBlock_iter(  py::object iblockObject )
{
   py::list resultList = IBlock_blockDataList( iblockObject ); //NOLINT
   return resultList.attr("__iter__");
}

py::tuple IBlock_atDomainMinBorder( IBlock & block )
{
   return py::make_tuple( block.getBlockStorage().atDomainXMinBorder(block),
                                     block.getBlockStorage().atDomainYMinBorder(block),
                                     block.getBlockStorage().atDomainZMinBorder(block) );
}

py::tuple IBlock_atDomainMaxBorder( IBlock & block )
{
   return py::make_tuple( block.getBlockStorage().atDomainXMaxBorder(block),
                                     block.getBlockStorage().atDomainYMaxBorder(block),
                                     block.getBlockStorage().atDomainZMaxBorder(block) );
}

IBlockID::IDType IBlock_getIntegerID( IBlock & block )
{
   return block.getId().getID();
}

bool IBlock_equals( IBlock & block1, IBlock & block2 )
{
   return block1.getId() == block2.getId();
}

std::string IBlock_str( IBlock & b ) {
   std::stringstream out;
   out <<  "Block at " << b.getAABB();
   return out.str();

}

void exportIBlock(py::module_ &m)
{
   // TODO: register the exceptions: Does not work as done below
   // py::register_exception_translator( & NoSuchBlockData::translate );
   // py::register_exception_translator( & BlockDataNotConvertible::translate );

   py::class_<IBlock, std::unique_ptr<IBlock, py::nodelete>> (m, "Block")
         .def         ( "__getitem__",                   &IBlock_getData, py::keep_alive<1, 2>()   )
         .def_property_readonly( "atDomainMinBorder",    &IBlock_atDomainMinBorder                 )
         .def_property_readonly( "atDomainMaxBorder",    &IBlock_atDomainMaxBorder                 )
         .def_property_readonly( "items",                &IBlock_blockDataList                     )
         .def_property_readonly( "id",                   &IBlock_getIntegerID                      )
         .def         ( "__hash__",                      &IBlock_getIntegerID                      )
         .def         ( "__eq__",                        &IBlock_equals                            )
         .def         ( "__repr__",               &IBlock_str                               )
         .def_property_readonly( "__iter__",             &IBlock_iter                              )
         .def_property_readonly("aabb",            &IBlock::getAABB                          )
         ;

}

//======================================================================================================================
//
//  Logging & Abort
//
//======================================================================================================================


static void wlb_log_devel              ( const std::string & msg ) { WALBERLA_LOG_DEVEL          ( msg ); }
static void wlb_log_devel_on_root      ( const std::string & msg ) { WALBERLA_LOG_DEVEL_ON_ROOT  ( msg ); }

static void wlb_log_result             ( const std::string & msg ) { WALBERLA_LOG_RESULT         ( msg ); }
static void wlb_log_result_on_root     ( const std::string & msg ) { WALBERLA_LOG_RESULT_ON_ROOT ( msg ); }

static void wlb_log_warning            ( const std::string & msg ) { WALBERLA_LOG_WARNING         ( msg ); }
static void wlb_log_warning_on_root    ( const std::string & msg ) { WALBERLA_LOG_WARNING_ON_ROOT ( msg ); }

#ifdef WALBERLA_LOGLEVEL_INFO
static void wlb_log_info               ( const std::string & msg ) { WALBERLA_LOG_INFO            ( msg ); }
static void wlb_log_info_on_root       ( const std::string & msg ) { WALBERLA_LOG_INFO_ON_ROOT    ( msg ); }
#else
static void wlb_log_info               ( const std::string & ) {}
static void wlb_log_info_on_root       ( const std::string & ) {}
#endif

#ifdef WALBERLA_LOGLEVEL_PROGRESS
static void wlb_log_progress           ( const std::string & msg ) { WALBERLA_LOG_PROGRESS        ( msg ); }
static void wlb_log_progress_on_root   ( const std::string & msg ) { WALBERLA_LOG_PROGRESS_ON_ROOT( msg ); }
#else
static void wlb_log_progress           ( const std::string & ) {}
static void wlb_log_progress_on_root   ( const std::string & ) {}
#endif

#ifdef WALBERLA_LOGLEVEL_DETAIL
static void wlb_log_detail             ( const std::string & msg ) { WALBERLA_LOG_DETAIL          ( msg ); }
static void wlb_log_detail_on_root     ( const std::string & msg ) { WALBERLA_LOG_DETAIL_ON_ROOT  ( msg ); }
#else
static void wlb_log_detail             ( const std::string & ) {}
static void wlb_log_detail_on_root     ( const std::string & ) {}
#endif

static void wlb_abort                  ( const std::string & msg ) { WALBERLA_ABORT_NO_DEBUG_INFO ( msg ); }

void exportLogging(py::module_ &m)
{
   m.def ( "log_devel"         ,  wlb_log_devel           );
   m.def ( "log_devel_on_root" ,  wlb_log_devel_on_root   );
   m.def ( "log_result",          wlb_log_result          );
   m.def ( "log_result_on_root",  wlb_log_result_on_root  );
   m.def ( "log_warning",         wlb_log_warning         );
   m.def ( "log_warning_on_root", wlb_log_warning_on_root );
   m.def ( "log_info",            wlb_log_info            );
   m.def ( "log_info_on_root",    wlb_log_info_on_root    );
   m.def ( "log_progress",        wlb_log_progress        );
   m.def ( "log_progress_on_root",wlb_log_progress_on_root);
   m.def ( "log_detail",          wlb_log_detail          );
   m.def ( "log_detail_on_root",  wlb_log_detail_on_root  );

   m.def ( "abort", wlb_abort );
}

//======================================================================================================================
//
//  Communication
//
//======================================================================================================================

void exportCommunication(py::module_ &m)
{
   using communication::UniformPackInfo;
   py::class_< UniformPackInfo, shared_ptr<UniformPackInfo>> //NOLINT
      (m, "UniformPackInfo" );

   using communication::UniformMPIDatatypeInfo;
   py::class_< UniformMPIDatatypeInfo, shared_ptr<UniformMPIDatatypeInfo>>
      (m, "UniformMPIDatatypeInfo" );

}

//======================================================================================================================
//
//  Stencil Directions
//
//======================================================================================================================

void exportStencilDirections(py::module_ &m)
{
      py::enum_< stencil::Direction >(m, "Direction")
         .value("C", stencil::C)
         .value("N", stencil::N)
         .value("S", stencil::S)
         .value("W", stencil::W)
         .value("E", stencil::E)
         .value("T", stencil::T)
         .value("B", stencil::B)
         .value("NW", stencil::NW)
         .value("NE", stencil::NE)
         .value("SW", stencil::SW)
         .value("SE", stencil::SE)
         .value("TN", stencil::TN)
         .value("TS", stencil::TS)
         .value("TW", stencil::TW)
         .value("TE", stencil::TE)
         .value("BN", stencil::BN)
         .value("BS", stencil::BS)
         .value("BW", stencil::BW)
         .value("BE", stencil::BE)
         .value("TNE", stencil::TNE)
         .value("TNW", stencil::TNW)
         .value("TSE", stencil::TSE)
         .value("TSW", stencil::TSW)
         .value("BNE", stencil::BNE)
         .value("BNW", stencil::BNW)
         .value("BSE", stencil::BSE)
         .value("BSW", stencil::BSW)
         .export_values();
      py::list cx;

      py::list cy;

      py::list cz;

      py::list dirStrings;
      for (uint_t i = 0; i < stencil::NR_OF_DIRECTIONS; ++i)
      {
         cx.append(stencil::cx[i]);
         cy.append(stencil::cy[i]);
         cz.append(stencil::cz[i]);
         dirStrings.append(stencil::dirToString[i]);
      }
      py::list c;
      c.append(cx);
      c.append(cy);
      c.append(cz);

      m.attr("cx") = cx;
      m.attr("cy") = cy;
      m.attr("cz") = cz;
      m.attr("c") = c;
      m.attr("dirStrings") = dirStrings;
}


//======================================================================================================================
//
//  Build Info
//
//======================================================================================================================


void exportBuildInfo(py::module_ &m)
{
//   ModuleScope build_info( "build_info");
//   using py::scope;
   m.attr("version")         = WALBERLA_GIT_SHA1;
   m.attr("type" )           = WALBERLA_BUILD_TYPE;
   m.attr("compiler_flags" ) = WALBERLA_COMPILER_FLAGS;
   m.attr("build_machine" )  = WALBERLA_BUILD_MACHINE;
   m.attr("source_dir")      = WALBERLA_SOURCE_DIR;
   m.attr("build_dir")       = WALBERLA_BUILD_DIR;
}




void exportBasicWalberlaDatastructures(py::module_ &m)
{

//   NumpyIntConversion<uint8_t>();
//   NumpyIntConversion<int32_t>();
//   NumpyIntConversion<int64_t>();
//   NumpyIntConversion<uint32_t>();
//   NumpyIntConversion<uint64_t>();
//   NumpyIntConversion<size_t>();
//   NumpyIntConversion<bool>();
//   NumpyFloatConversion<float>();
//   NumpyFloatConversion<double>();
//   NumpyFloatConversion<long double>();


   exportMPI(m);

   exportBuildInfo(m);
//   exportVector3();
//   exportCell();
   exportCellInterval(m);
   exportAABB(m);

   exportTiming(m);

   exportIBlock(m);
   exportCommunication(m);

   exportLogging(m);
   exportStencilDirections(m);

}

} // namespace python_coupling
} // namespace walberla

#endif

