if( WALBERLA_BUILD_WITH_CUDA )
    target_sources(lbm_mesapd_coupling
            PRIVATE
            ParticleAndVolumeFractionMappingGPU.h
            )
    if( CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60 )
        target_sources(lbm_mesapd_coupling
                PRIVATE
                PSMKernel.cu
                PSMKernel.h
                PSMUtilityGPU.cuh
                PSMWrapperSweepsGPU.h
                )
    endif()
endif()
