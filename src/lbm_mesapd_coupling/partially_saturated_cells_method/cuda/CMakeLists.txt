if( WALBERLA_BUILD_WITH_CUDA )
    target_sources(lbm_mesapd_coupling
            PRIVATE
            ParticleAndVolumeFractionMappingGPU.h
            )
endif()

if( WALBERLA_BUILD_WITH_CUDA AND WALBERLA_BUILD_WITH_CODEGEN )
    walberla_generate_target_from_python( NAME PSMSolidCodegenPython
            FILE PSMSolidCodegen.py
            OUT_FILES   SolidKernel.cuh )

    target_sources(lbm_mesapd_coupling
            PRIVATE
            PSMKernel.cu
            PSMKernel.h
            PSMSweepGPU.h
            )
    add_dependencies(lbm_mesapd_coupling PSMSolidCodegenPython)
    target_include_directories(lbm_mesapd_coupling PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/default_codegen)
endif()