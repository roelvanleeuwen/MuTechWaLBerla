#include "python_coupling/PythonWrapper.h"

#ifdef WALBERLA_BUILD_WITH_PYTHON

#include "python_coupling/helper/PythonIterableToStdVector.h"

#include "core/mpi/MPIManager.h"
#include "core/mpi/Reduce.h"
#include "core/mpi/Gather.h"
#include "core/mpi/Broadcast.h"

#include <vector>

namespace py = pybind11;




namespace walberla {
namespace python_coupling {

   typedef std::vector<int64_t>      IntStdVector;
   typedef std::vector<real_t>       RealStdVector;
   typedef std::vector<std::string>  StringStdVector;


   //===================================================================================================================
   //
   //  MPIManager
   //
   //===================================================================================================================


   static int  rank()              { return MPIManager::instance()->rank();               }
   static int  worldRank()         { return MPIManager::instance()->worldRank();          }
   static int  numProcesses()      { return MPIManager::instance()->numProcesses();       }
   static bool hasCartesianSetup() { return MPIManager::instance()->hasCartesianSetup();  }
   static bool rankValid()         { return MPIManager::instance()->rankValid();          }



   //===================================================================================================================
   //
   //  Broadcast
   //
   //===================================================================================================================

   static py::object broadcast_string( py::object value, int sendRank ) //NOLINT
   {
      if ( py::isinstance<std::string>(value) )
      {
         std::string extractedValue = py::cast< std::string >(value);
         mpi::broadcastObject( extractedValue , sendRank );
         return py::cast( extractedValue );
      }
      StringStdVector extractedValue = pythonIterableToStdVector< StringStdVector::value_type >( value );
      mpi::broadcastObject( extractedValue, sendRank );
      return py::cast( extractedValue );
   }

   static py::object broadcast_int( py::object value, int sendRank ) //NOLINT
   {
      if ( py::isinstance<int64_t>(value) )
      {
         int64_t extractedValue = py::cast< int64_t >(value);
         mpi::broadcastObject( extractedValue , sendRank );
         return py::cast( extractedValue );
      }
      IntStdVector extractedValue = pythonIterableToStdVector< IntStdVector::value_type >( value );
      mpi::broadcastObject( extractedValue, sendRank );
      return py::cast( extractedValue );
   }

   static py::object broadcast_real( py::object value, int sendRank ) //NOLINT
   {
      if ( py::isinstance<real_t>(value) )
      {
         real_t extractedValue = py::cast< real_t  >(value);
         mpi::broadcastObject( extractedValue , sendRank);
         return py::cast( extractedValue );
      }
      RealStdVector extractedValue = pythonIterableToStdVector< RealStdVector::value_type >( value );
      mpi::broadcastObject( extractedValue , sendRank);
      return py::cast( extractedValue );
   }


   //===================================================================================================================
   //
   //  Reduce
   //
   //===================================================================================================================


   static py::object reduce_int( py::object value, mpi::Operation op, int recvRank ) //NOLINT
   {
      if ( py::isinstance<int64_t>(value) )
      {
         int64_t extractedValue = py::cast< int64_t >(value);
         mpi::reduceInplace( extractedValue , op, recvRank );
         return py::cast( extractedValue );
      }
      IntStdVector extractedValue = pythonIterableToStdVector< IntStdVector::value_type >( value );
      mpi::reduceInplace( extractedValue, op, recvRank );
      return py::cast( extractedValue );
   }

   static py::object reduce_real( py::object value, mpi::Operation op, int recvRank ) //NOLINT
   {
      if ( py::isinstance<real_t>(value) )
      {
         real_t extractedValue = py::cast< real_t  >(value);
         mpi::reduceInplace( extractedValue , op, recvRank);
         return py::cast( extractedValue );
      }
      py::array test = py::cast<py::array>(value);
      RealStdVector extractedValue = pythonIterableToStdVector< RealStdVector::value_type >( value );
      mpi::reduceInplace( extractedValue , op, recvRank);
      return py::cast( extractedValue );
   }


   static py::object allreduce_int( py::object value, mpi::Operation op ) //NOLINT
   {
      if ( py::isinstance<int64_t>(value) )
      {
         int64_t extractedValue = py::cast< int64_t >(value);
         mpi::allReduceInplace( extractedValue , op );
         return py::cast( extractedValue );
      }
      IntStdVector extractedValue = pythonIterableToStdVector< IntStdVector::value_type >( value );
      mpi::allReduceInplace( extractedValue, op );
      return py::cast( extractedValue );
   }

   static py::object allreduce_real( py::object value, mpi::Operation op ) //NOLINT
   {
      if ( py::isinstance<real_t>(value) )
      {
         real_t extractedValue = py::cast< real_t  >(value);
         mpi::allReduceInplace( extractedValue , op );
         return py::cast( extractedValue );
      }
      RealStdVector extractedValue = pythonIterableToStdVector< RealStdVector::value_type >( value );
      mpi::allReduceInplace( extractedValue , op );
      return py::cast( extractedValue );
   }


   //===================================================================================================================
   //
   //  Gather
   //
   //===================================================================================================================

   static IntStdVector gather_int( py::object value, int recvRank ) //NOLINT
   {
      if ( ! py::isinstance<int64_t>(value) )
      {
         PyErr_SetString( PyExc_RuntimeError, "Could not gather the given value - unknown type");
         throw py::error_already_set();
      }
      int64_t extractedValue = py::cast< int64_t >(value);
      return mpi::gather( extractedValue , recvRank );
   }

   static RealStdVector gather_real( py::object value, int recvRank ) //NOLINT
   {
      if ( ! py::isinstance<real_t>(value) )
      {
         PyErr_SetString( PyExc_RuntimeError, "Could not gather the given value - unknown type");
         throw py::error_already_set();
      }
      real_t extractedValue = py::cast< real_t  >(value);
      return mpi::gather( extractedValue , recvRank);
   }


   static IntStdVector allgather_int( py::object value ) //NOLINT
   {
      if ( ! py::isinstance<int64_t>(value) )
      {
         PyErr_SetString( PyExc_RuntimeError, "Could not gather the given value - unknown type");
         throw py::error_already_set();
      }
      int64_t extractedValue = py::cast< int64_t >(value);
      return mpi::allGather( extractedValue );
   }

   static RealStdVector allgather_real( py::object value ) //NOLINT
   {
      if ( ! py::isinstance<real_t>(value) )
      {
         PyErr_SetString( PyExc_RuntimeError, "Could not gather the given value - unknown type");
         throw py::error_already_set();
      }
      real_t extractedValue = py::cast< real_t  >(value);
      return mpi::allGather( extractedValue );
   }



   //===================================================================================================================
   //
   //  Export
   //
   //===================================================================================================================

   static void worldBarrier()
   {
      WALBERLA_MPI_WORLD_BARRIER();
   }


   void exportMPI(py::module_ &m)
   {
//      object mpiModule( handle<>( borrowed(PyImport_AddModule("walberla_cpp.mpi"))));
//      scope().attr("mpi") = mpiModule;
//      scope mpiScope = mpiModule;

      m.def( "rank"             , &rank             );
      m.def( "worldRank"        , &worldRank        );
      m.def( "numProcesses"     , &numProcesses     );
      m.def( "hasCartesianSetup", &hasCartesianSetup);
      m.def( "rankValid"        , &rankValid        );
      m.def( "worldBarrier"     , &worldBarrier     );

      py::enum_<mpi::Operation>(m, "Operation")
              .value("MIN"    ,      mpi::MIN )
              .value("MAX"    ,      mpi::MAX )
              .value("SUM"    ,      mpi::SUM )
              .value("PRODUCT",      mpi::PRODUCT )
              .value("LOGICAL_AND",  mpi::LOGICAL_AND )
              .value("BITWISE_AND",  mpi::BITWISE_AND )
              .value("LOGICAL_OR",   mpi::LOGICAL_OR  )
              .value("BITWISE_OR",   mpi::BITWISE_OR  )
              .value("LOGICAL_XOR",  mpi::LOGICAL_XOR )
              .value("BITWISE_XOR",  mpi::BITWISE_XOR )
              .export_values();

      m.def( "broadcastInt",   &broadcast_int);
      m.def( "broadcastReal",  &broadcast_real);
      m.def( "broadcastString",&broadcast_string);

      m.def( "reduceInt",     &reduce_int);
      m.def( "reduceReal",    &reduce_real);
      m.def( "allreduceInt",  &allreduce_int  );
      m.def( "allreduceReal", &allreduce_real );


//      py::class_< IntStdVector>   (m, "IntStdVector") .def(vector_indexing_suite<IntStdVector>()  );
//      py::class_< RealStdVector>  (m, "RealStdVector").def(vector_indexing_suite<RealStdVector>() );
//      py::class_< StringStdVector> (m, "StringStdVector").def(vector_indexing_suite<StringStdVector>() );

      m.def( "gatherInt",     &gather_int);
      m.def( "gatherReal",    &gather_real);
      m.def( "allgatherInt",  &allgather_int  );
      m.def( "allgatherReal", &allgather_real );
   }



} // namespace python_coupling
} // namespace walberla


#endif
