# - Find vHACD library for hierachical approximate complex decompositon
# Find the vHACD includes and client library
# This module defines
#  VHACD_INCLUDE_DIR, where to find CGAL.h
#  VHACD_LIBRARIES, the libraries needed to use CGAL.
#  VHACD_FOUND, If false, do not try to use CGAL.

if( VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )
   set(VHACD_FOUND TRUE)

else( VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )

 

 FIND_PATH(VHACD_INCLUDE_DIR_TMP VHACD.h
      ${VHACD_ROOT}/include
      /usr/include
      /usr/local/include
      NO_DEFAULT_PATH
      )

  set(VHACD_INCLUDE_DIR ${VHACD_INCLUDE_DIR_TMP} CACHE PATH "Path to search for vHACD include files.")

  
  find_library(VHACD_LIBRARIES_TMP NAMES libvhacd.a
     PATHS
      ${VHACD_ROOT}/lib
     /usr/lib
     /usr/local/lib
     NO_DEFAULT_PATH
     )
  set(VHACD_LIBRARIES ${VHACD_LIBRARIES_TMP} CACHE PATH "Path to search for vHACD libraries.")

message(STATUS "VHACD_INCLUDE_DIR=${VHACD_INCLUDE_DIR}")
message(STATUS "VHACD_LIBRARIES=${VHACD_LIBRARIES}")
  if(VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )
    set(VHACD_FOUND TRUE)
    INCLUDE_DIRECTORIES(${VHACD_INCLUDE_DIR} $ENV{CGAL_CFG})
  else(VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )
    set(VHACD_FOUND FALSE)
    message(STATUS "vHACD not found.")
  endif(VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )

  mark_as_advanced(VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )

endif(VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )
