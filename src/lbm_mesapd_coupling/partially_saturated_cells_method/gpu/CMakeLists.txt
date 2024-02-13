target_sources(lbm_mesapd_coupling
        PRIVATE
        ParticleAndVolumeFractionMappingGPU.h
        PSMSweepCollectionGPU.h
)
if (CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60)
    target_sources(lbm_mesapd_coupling
            PRIVATE
            PSMWrapperKernels.${CODEGEN_FILE_SUFFIX}
            PSMWrapperKernels.h
            PSMUtilityGPU.h
            PSMWrapperSweepsGPU.h
    )
    if (WALBERLA_BUILD_WITH_CODEGEN)
        foreach (collision_setup srt trt mrt cumulant smagorinsky)
            foreach (solid_collision 1 2 3)
                set(config ${collision_setup}_sc${solid_collision})
                walberla_generate_target_from_python(NAME PSMCodegenPython_${config}
                        FILE PSMCodegen.py
                        CODEGEN_CFG ${config}
                        OUT_FILES LBMSweep.${CODEGEN_FILE_SUFFIX} LBMSweep.h LBMSplitSweep.${CODEGEN_FILE_SUFFIX} LBMSplitSweep.h
                        PSMSweep.${CODEGEN_FILE_SUFFIX} PSMSweep.h PSMSweepSplit.${CODEGEN_FILE_SUFFIX} PSMSweepSplit.h
                        PSMPackInfo.${CODEGEN_FILE_SUFFIX} PSMPackInfo.h InitializeDomainForPSM.${CODEGEN_FILE_SUFFIX} InitializeDomainForPSM.h
                        PSM_NoSlip.${CODEGEN_FILE_SUFFIX} PSM_NoSlip.h PSM_UBB.${CODEGEN_FILE_SUFFIX} PSM_UBB.h PSM_Density.${CODEGEN_FILE_SUFFIX} PSM_Density.h
                        PSM_FreeSlip.${CODEGEN_FILE_SUFFIX} PSM_FreeSlip.h PSM_InfoHeader.h PSM_MacroGetter.cpp PSM_MacroGetter.h
                        PSM_MacroSetter.cpp PSM_MacroSetter.h)
            endforeach ()
        endforeach ()
    endif ()
endif ()
