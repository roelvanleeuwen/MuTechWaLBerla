if( WALBERLA_BUILD_WITH_CUDA )
    target_sources(lbm_mesapd_coupling
            PRIVATE
            ParticleAndVolumeFractionMappingGPU.h
            )
endif()

if( WALBERLA_BUILD_WITH_CUDA )
    target_sources(lbm_mesapd_coupling
            PRIVATE
            PSMKernel.cu
            PSMKernel.h
            PSMSweepGPU.h
            )
endif()