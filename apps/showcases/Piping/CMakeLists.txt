waLBerla_link_files_to_builddir(*.prm)

waLBerla_add_executable(NAME SettlingSpheres
        FILES SettlingSpheres.cpp
        DEPENDS blockforest core mesa_pd vtk)

if (WALBERLA_BUILD_WITH_CUDA AND WALBERLA_BUILD_WITH_CODEGEN AND CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60)
    waLBerla_add_executable(NAME Piping FILES Piping.cpp
            DEPENDS blockforest core field geometry gpu lbm lbm_mesapd_coupling mesa_pd vtk PSMCodegenPython_srt_sc1_s0)
endif ()
