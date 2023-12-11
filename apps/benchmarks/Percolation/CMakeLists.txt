waLBerla_link_files_to_builddir("*.prm")

if (WALBERLA_BUILD_WITH_GPU_SUPPORT AND WALBERLA_BUILD_WITH_CODEGEN AND CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60)
    waLBerla_add_executable(NAME Percolation FILES Percolation.cpp
            DEPENDS blockforest core field geometry gpu lbm lbm_mesapd_coupling mesa_pd vtk PSMCodegenPython_trt_sc3)
    target_compile_definitions(Percolation PRIVATE Weighting=2)
endif ()
