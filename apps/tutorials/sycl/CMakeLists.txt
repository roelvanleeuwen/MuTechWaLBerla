waLBerla_link_files_to_builddir( *.prm )


if( WALBERLA_BUILD_WITH_CODEGEN )

    walberla_generate_target_from_python( NAME SYCLTestCodegenPython
        FILE SYCLTest.py
        OUT_FILES   #SYCLTestSweep.${CODEGEN_FILE_SUFFIX} SYCLTestSweep.h
                    #SYCLTestPackInfo.${CODEGEN_FILE_SUFFIX} SYCLTestPackInfo.h
                    InitialPDFsSetter.cpp InitialPDFsSetter.h

            )

    walberla_add_executable ( NAME SYCLTest
        FILES SYCLTest.cpp modified_codeGen/SYCLTestSweep.h modified_codeGen/SYCLTestSweep.cpp modified_codeGen/SYCLTestPackInfo.h modified_codeGen/SYCLTestPackInfo.cpp
        DEPENDS blockforest gpu core domain_decomposition field geometry timeloop lbm stencil vtk SYCLTestCodegenPython )

endif()