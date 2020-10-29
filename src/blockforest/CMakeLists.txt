
OPTION( WALBERLA_BLOCKFOREST_PRIMITIVE_BLOCKID "BlockForest: enables/disables BlockID implementation as a primitive data type (uint_t)" ON )

mark_as_advanced( WALBERLA_BLOCKFOREST_PRIMITIVE_BLOCKID )

configure_file ( CMakeDefs.in.h  CMakeDefs.h )

waLBerla_add_module( DEPENDS communication core domain_decomposition python_coupling stencil )

if ( WALBERLA_BUILD_WITH_PYTHON )
   # this is need to correctly integrate pybind
   target_link_libraries( blockforest PUBLIC python_coupling )
   set_target_properties( blockforest PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif ()
