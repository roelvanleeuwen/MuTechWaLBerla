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
//! \file MPIWrapper.cpp
//! \ingroup core
//! \author Michael Zikeli <michael.zikeli@fau.de>
//
//======================================================================================================================

#include "MPIWrapper.h"

#include <set>

#include "MPIManager.h"

namespace walberla
{
#ifdef WALBERLA_BUILD_WITH_HALF_PRECISION_SUPPORT
namespace mpi
{
namespace
{
/// These functions than can be used by self defined mpi operations, e.g. by using CustomMPIOperation.
using float16_t = walberla::float16;
// The signature of MPI_User_function looks like this
// typedef void MPI_User_function( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype);

void sum(void* mpiRHSArray, void* mpiLHSArray, int* len, MPI_Datatype*)
{
   // cast mpi type to target c++ type
   auto* rhs = (float16_t*) mpiRHSArray;
   auto* lhs = (float16_t*) mpiLHSArray;
   for (int i = 0; i < *len; ++i)
   {
      *lhs += *rhs;
   }
}

void min(void* mpiRHSArray, void* mpiLHSArray, int* len, MPI_Datatype*)
{
   // cast mpi type to target c++ type
   auto* rhs = (float16_t*) mpiRHSArray;
   auto* lhs = (float16_t*) mpiLHSArray;
   for (int i = 0; i < *len; ++i)
   {
      *lhs = (*rhs >= *lhs) ? *lhs : *rhs;
   }
}

void max(void* mpiRHSArray, void* mpiLHSArray, int* len, MPI_Datatype*)
{
   // cast mpi type to target c++ type
   auto* rhs = (float16_t*) mpiRHSArray;
   auto* lhs = (float16_t*) mpiLHSArray;
   for (int i = 0; i < *len; ++i)
   {
      *lhs = (*rhs <= *lhs) ? *lhs : *rhs;
   }
}

MPI_User_function* returnMPIUserFctPointer(const Operation op)
{
   switch (op)
   {
   case SUM:
      return &sum;
   case MIN:
      return &min;
   case MAX:
      return &max;
   default:
      WALBERLA_ABORT("The chosen operation " << typeid(op).name() << " is not implemented for float16 yet.");
   }
}

}
}

/// Here some MPI_Datatypes and MPI_Ops are initialized that are not part of the MPI Standard and therefore have to be
/// define yourself. This is done in the MPIManager, since they need to be freed before MPIFinalize is called and this
/// way it is easiest to keep track of them.
///     For more information about this feature compare MR !647 (
///     https://i10git.cs.fau.de/walberla/walberla/-/merge_requests/647 )

/*!
 *  \brief Specialization of MPITrait for float16
 *
 *  The initialization of the self defined MPI_Datatype and MPI_Op is done in the MPIManager so that it can be freed
 * before MPI is finalized.
 */
MPI_Datatype MPITrait< walberla::float16 >::type()
{

#ifdef WALBERLA_BUILD_WITH_MPI
   // Since this type should be created only once, a static variable is used as safeguard.
   static bool initializedType = false;
   if (!initializedType)
   {
      // Since float16 consists of two Bytes, a continuous datatype with size of two byte is created.
      mpi::MPIManager::instance()->commitCustomType< walberla::float16, const int >(2);
      initializedType = true;
   }
   return mpi::MPIManager::instance()->getCustomType< walberla::float16 >();
#else
   return mpistubs::MPI_FLOAT16;
#endif
}

MPI_Op MPITrait< walberla::float16 >::operation(const mpi::Operation& op)
{
   WALBERLA_MPI_SECTION()
   {
      // mpi::Operation is an enum and not an enum class, thus, it is not sufficient to make a just a bool variable as
      // safeguard, since all operations are of type mpi::Operation and only the first one would pass the safeguard.
      // Therefore, a set is created and each operation that is called the first time, will be initialized.
      static std::set< mpi::Operation > operationInitializationRegister;
      const bool needsInitialization = std::get< 1 >(operationInitializationRegister.emplace(op));
      if (needsInitialization)
      {
         mpi::MPIManager::instance()->commitCustomOperation< walberla::float16 >(
            op, mpi::returnMPIUserFctPointer(op));
      }
      return MPIManager::instance()->getCustomOperation< walberla::float16 >(op);
   }
   WALBERLA_NON_MPI_SECTION() { WALBERLA_ABORT("If MPI is not used, a custom operator should never be called."); }
}
#endif // WALBERLA_BUILD_WITH_HALF_PRECISION_SUPPORT

void maxMagnitude(double* in, double* inout, int* len, MPI_Datatype*)
{
   for (int i = 0; i < *len; ++i)
   {
      *inout = (fabs(*inout) > fabs(*in)) ? fabs(*inout) : fabs(*in);
      in++;
      inout++;
   }
}

/*!
 *  \brief Specialization of the static operation() method of MPITrait.
 *
 *  It chooses a MPI_Op depending on the value type of the object the operation is performed on.
 *
 *  \param op The operation to be performed (op is an element of the enum mpi::Operation).
 */
template< typename T >
MPI_Op MPITrait< T >::operation(const mpi::Operation& op)
{
   switch (op)
   {
   case mpi::MIN:
      return MPI_MIN;
   case mpi::MAX:
      return MPI_MAX;
   case mpi::SUM:
      return MPI_SUM;
   case mpi::PRODUCT:
      return MPI_PROD;
   case mpi::LOGICAL_AND:
      return MPI_LAND;
   case mpi::BITWISE_AND:
      return MPI_BAND;
   case mpi::LOGICAL_OR:
      return MPI_LOR;
   case mpi::BITWISE_OR:
      return MPI_BOR;
   case mpi::LOGICAL_XOR:
      return MPI_LXOR;
   case mpi::BITWISE_XOR:
      return MPI_BXOR;
   case mpi::MAG_MAX: {
      static std::set< mpi::Operation > operationInitializationRegister;
      const bool needsInitialization = std::get< 1 >(operationInitializationRegister.emplace(op));
      if (needsInitialization)
      {
         mpi::MPIManager::instance()->commitCustomOperation< T >(
            op, reinterpret_cast< void (*)(void*, void*, int*, MPI_Datatype*) >(maxMagnitude));
      }
      return MPIManager::instance()->getCustomOperation< T >(op);
   }
   default:
      WALBERLA_ABORT("Unknown operation!");
   }
#ifdef __IBMCPP__
   return MPI_SUM; // never reached, helps to suppress a warning from the IBM compiler
#endif
}

template
MPI_Op MPITrait< char >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< signed char >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< signed short int >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< signed int >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< signed long int >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< signed long long >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< unsigned char >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< unsigned short int >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< unsigned int >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< unsigned long int >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< unsigned long long >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< float >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< double >::operation(const mpi::Operation& op);
template
MPI_Op MPITrait< long double >::operation(const mpi::Operation& op);

/// Macro for specialization of the MPI supported data types in MPITrait::type().
#define WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(CPP_TYPE, MPI_TYPE) \
   template<> \
   MPI_Datatype MPITrait< CPP_TYPE >::type() \
   { \
      return (MPI_TYPE); \
   }

// MPITRAIT SPECIALIZATIONS

WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(char, MPI_CHAR)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(signed char, MPI_CHAR)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(signed short int, MPI_SHORT)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(signed int, MPI_INT)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(signed long int, MPI_LONG)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(signed long long, MPI_LONG_LONG)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(unsigned char, MPI_UNSIGNED_CHAR)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(unsigned short int, MPI_UNSIGNED_SHORT)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(unsigned int, MPI_UNSIGNED)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(unsigned long int, MPI_UNSIGNED_LONG)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(unsigned long long, MPI_UNSIGNED_LONG_LONG)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(float, MPI_FLOAT)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(double, MPI_DOUBLE)
WALBERLA_CREATE_MPITRAIT_TYPE_SPECIALIZATION(long double, MPI_LONG_DOUBLE)
#ifdef WALBERLA_BUILD_WITH_HALF_PRECISION_SUPPORT
template<>
struct MPITrait< float16 >
{
   static MPI_Datatype type();
   static MPI_Op operation(const mpi::Operation&);
};
#endif
} // namespace walberla
