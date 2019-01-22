# - Find CGAL
# Find the CGAL includes and client library
# This module defines
#  CGAL_INCLUDE_DIR, where to find CGAL.h
#  CGAL_LIBRARIES, the libraries needed to use CGAL.
#  GMP_LIBRARIES, libraries needed for gmp, dependency of CGAL
#  MPFR_LIBRARIEs, libraries needed for mprf, dependency of CGAL
#  CGAL_FOUND, If false, do not try to use CGAL.

if(CGAL_INCLUDE_DIR AND CGAL_LIBRARIES AND Boost_FOUND AND GMP_LIBRARIES AND MPFR_LIBRARIES)
   set(CGAL_FOUND TRUE)

else(CGAL_INCLUDE_DIR AND CGAL_LIBRARIES AND Boost_FOUND AND GMP_LIBRARIES AND MPFR_LIBRARIES)

 FIND_PATH(CGAL_INCLUDE_DIR CGAL/basic.h
      ${CGAL_ROOT}/include
      /usr/include
      /usr/local/include
      $ENV{ProgramFiles}/CGAL/*/include
      $ENV{SystemDrive}/CGAL/*/include
      NO_DEFAULT_PATH
      )

  find_library(CGAL_LIBRARIES NAMES CGAL libCGAL
     PATHS
      ${CGAL_ROOT}/lib
     /usr/lib
     /usr/local/lib
     /usr/lib/CGAL
     /usr/lib64
     /usr/local/lib64
     /usr/lib64/CGAL
     $ENV{ProgramFiles}/CGAL/*/lib
     $ENV{SystemDrive}/CGAL/*/lib
     NO_DEFAULT_PATH
     )

  find_library(GMP_LIBRARIES NAMES gmp libgmp
     PATHS
      ${GMP_ROOT}/lib
     /usr/lib
     /usr/local/lib
     /usr/lib/gmp
     /usr/lib64
     /usr/local/lib64
     /usr/lib64/gmp
     $ENV{ProgramFiles}/gmp/*/lib
     $ENV{SystemDrive}/gmp/*/lib
     )

  find_library(MPFR_LIBRARIES NAMES mpfr libmpfr
     PATHS
      ${MPFR_ROOT}/lib
     /usr/lib
     /usr/local/lib
     /usr/lib/mpfr
     /usr/lib64
     /usr/local/lib64
     /usr/lib64/mpfr
     $ENV{ProgramFiles}/mpfr/*/lib
     $ENV{SystemDrive}/mpfr/*/lib
     )

  if(CGAL_INCLUDE_DIR AND CGAL_LIBRARIES AND Boost_FOUND AND GMP_LIBRARIES AND MPFR_LIBRARIES)
    set(CGAL_FOUND TRUE)
    message(STATUS "Found CGAL: ${CGAL_INCLUDE_DIR}, ${CGAL_LIBRARIES}")
    INCLUDE_DIRECTORIES(${CGAL_INCLUDE_DIR} $ENV{CGAL_CFG})
  else(CGAL_INCLUDE_DIR AND CGAL_LIBRARIES AND Boost_FOUND AND GMP_LIBRARIES AND MPFR_LIBRARIES)
    set(CGAL_FOUND FALSE)
    message(STATUS "CGAL not found.")
  endif(CGAL_INCLUDE_DIR AND CGAL_LIBRARIES AND Boost_FOUND AND GMP_LIBRARIES AND MPFR_LIBRARIES)

  mark_as_advanced(CGAL_INCLUDE_DIR CGAL_LIBRARIES GMP_LIBRARIES MPFR_LIBRARIES)

endif(CGAL_INCLUDE_DIR AND CGAL_LIBRARIES AND Boost_FOUND AND GMP_LIBRARIES AND MPFR_LIBRARIES)
