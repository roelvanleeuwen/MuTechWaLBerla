target_sources( lbm_mesapd_coupling
    PRIVATE
    PSMSweep.h
    ParticleAndVolumeFractionMapping.h
    PSMUtility.h
    )

if( WALBERLA_BUILD_WITH_GPU_SUPPORT )
    add_subdirectory(gpu)
endif()
