waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )
waLBerla_link_files_to_builddir( *.obj )

walberla_generate_target_from_python( NAME PSM_TestCodeGen
        FILE PSM_Test.py
        OUT_FILES   LBMSweep.${CODEGEN_FILE_SUFFIX} LBMSweep.h
        PSMSweep.${CODEGEN_FILE_SUFFIX} PSMSweep.h
        PackInfo.${CODEGEN_FILE_SUFFIX} PackInfo.h
        NoSlip.${CODEGEN_FILE_SUFFIX} NoSlip.h
        UBB.${CODEGEN_FILE_SUFFIX} UBB.h
        FixedDensity.${CODEGEN_FILE_SUFFIX} FixedDensity.h
        MacroSetter.cpp MacroSetter.h
        MacroGetter.cpp MacroGetter.h
        )


if(WALBERLA_BUILD_WITH_GPU_SUPPORT)
    walberla_add_executable ( NAME PSM_Test
            FILES PSM_Test.cpp ObjectRotator.h
            DEPENDS blockforest gpu core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk PSM_TestCodeGen )
else()
    walberla_add_executable ( NAME PSM_Test
            FILES PSM_Test.cpp ObjectRotator.h
            DEPENDS blockforest core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk PSM_TestCodeGen )
endif()