target_sources(lbm_mesapd_coupling
        PRIVATE
        HydrodynamicForcesSweepGPU.h
        ParticleAndVolumeFractionMappingGPU.h
        PSMKernel.cu
        PSMKernel.h
        PSMSweepGPU.h
        )
