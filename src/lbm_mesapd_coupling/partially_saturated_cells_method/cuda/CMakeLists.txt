target_sources(lbm_mesapd_coupling
        PRIVATE
        ParticleAndVolumeFractionMappingGPU.h
        ParticleAndVolumeFractionMappingKernel.cu
        ParticleAndVolumeFractionMappingKernel.h
        )
