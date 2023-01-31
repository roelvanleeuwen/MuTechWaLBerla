if( WALBERLA_BUILD_WITH_CUDA )
    target_sources(lbm_mesapd_coupling
            PRIVATE
            ParticleAndVolumeFractionMappingGPU.h
            PSMSweepCollectionGPU.h
            )
    if( CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60 )
        target_sources(lbm_mesapd_coupling
                PRIVATE
                PSMWrapperKernels.cu
                PSMWrapperKernels.h
                PSMUtilityGPU.cuh
                PSMWrapperSweepsGPU.h
                )
        if( WALBERLA_BUILD_WITH_CODEGEN )
            walberla_generate_target_from_python( NAME PSMCodegenPython
                    FILE PSMCodegen.py
                    OUT_FILES PSMSweep.cu PSMSweep.h PSMSweepSplit.cu PSMSweepSplit.h
                    PSMPackInfo.cu PSMPackInfo.h InitialPDFsSetter.cu InitialPDFsSetter.h PSM_NoSlip.cu PSM_NoSlip.h PSM_UBB.cu PSM_UBB.h PSM_Density.cu PSM_Density.h )
        endif()
    endif()
endif()
