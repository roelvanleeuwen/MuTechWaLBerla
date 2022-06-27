if( WALBERLA_BUILD_WITH_CUDA AND WALBERLA_BUILD_WITH_CODEGEN )
    target_sources(lbm_mesapd_coupling
            PRIVATE
            OverlapFraction.h
            ParticleAndVolumeFractionMappingKernel.cu
            ParticleAndVolumeFractionMappingKernel.h
            )
endif()
