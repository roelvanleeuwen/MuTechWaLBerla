if (WALBERLA_BUILD_WITH_PYTHON)
   waLBerla_add_module(DEPENDS core communication domain_decomposition stencil)
   target_link_libraries(python_coupling PRIVATE pybind)
endif()