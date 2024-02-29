add_subdirectory(setups)

waLBerla_add_executable(NAME SettlingSpheres
        FILES SettlingSpheres.cpp
        DEPENDS blockforest core field lbm_mesapd_coupling mesa_pd vtk)

if (WALBERLA_BUILD_WITH_GPU_SUPPORT AND WALBERLA_BUILD_WITH_CODEGEN AND CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60)
    waLBerla_add_executable(NAME Piping FILES Piping.cpp
            DEPENDS blockforest core field geometry gpu lbm lbm_mesapd_coupling mesa_pd vtk PSMCodegenPython_trt_sc3)
    target_compile_definitions(Piping PRIVATE Weighting=2)
endif ()
