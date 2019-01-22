# - Find vHACD library for hierachical approximate complex decompositon
# Find the vHACD includes and client library
# This module defines
#  VHACD_INCLUDE_DIR, where to find CGAL.h
#  VHACD_LIBRARIES, the libraries needed to use CGAL.
#  VHACD_FOUND, If false, do not try to use CGAL.

if( VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )
   set(VHACD_FOUND TRUE)

else( VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )

 message(STATUS "VHACD_ROOT=${VHACD_ROOT}")
 FIND_PATH(VHACD_INCLUDE_DIR VHACD.h
      ${VHACD_ROOT}/include
      /usr/include
      /usr/local/include
      NO_DEFAULT_PATH
      )

  
  find_library(VHACD_LIBRARIES NAMES libvhacd.a
     PATHS
      ${VHACD_ROOT}/lib
     /usr/lib
     /usr/local/lib
     NO_DEFAULT_PATH
     )

  if(VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )
    set(VHACD_FOUND TRUE)
    message(STATUS "Found V-HACD: ${VHACD_INCLUDE_DIR}, ${VHACD_LIBRARIES}")
    INCLUDE_DIRECTORIES(${VHACD_INCLUDE_DIR} )
  else(VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )
    set(VHACD_FOUND FALSE)
    message(STATUS "vHACD not found.")
  endif(VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )

  mark_as_advanced(VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )

endif(VHACD_INCLUDE_DIR AND VHACD_LIBRARIES )
