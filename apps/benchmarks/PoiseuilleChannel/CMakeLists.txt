
waLBerla_link_files_to_builddir( "*.dat" )                 
waLBerla_link_files_to_builddir( "*.prm" )


if( WALBERLA_BUILD_WITH_CODEGEN )

    waLBerla_generate_target_from_python(NAME PoiseuilleChannelForceGeneratedLBM FILE GeneratedLBM.py
            OUT_FILES GeneratedLBM.cpp GeneratedLBM.h )

    waLBerla_add_executable( NAME PoiseuilleChannelForceTesting
            FILES PoiseuilleChannelForceTesting.cpp
            DEPENDS blockforest core field lbm geometry timeloop PoiseuilleChannelForceGeneratedLBM )

else()

    waLBerla_add_executable( NAME PoiseuilleChannel
                             FILES PoiseuilleChannel.cpp
                             DEPENDS blockforest boundary core field lbm postprocessing stencil timeloop vtk sqlite )

    waLBerla_add_executable( NAME PoiseuilleChannelForceTesting
                             FILES PoiseuilleChannelForceTesting.cpp
                             DEPENDS blockforest core field lbm geometry timeloop )

    ##############
    # Some tests #
    ##############
    waLBerla_execute_test( NO_MODULE_LABEL NAME PoiseuilleChannelTestParallelPlatesNoCheckRelease COMMAND $<TARGET_FILE:PoiseuilleChannel> ${CMAKE_CURRENT_SOURCE_DIR}/TestParallelPlatesNoCheck.dat --trt --linear-exp PROCESSES 4 CONFIGURATIONS Release RelWithDbgInfo DEPENDS_ON_TARGETS PoiseuilleChannel )
    waLBerla_execute_test( NO_MODULE_LABEL NAME PoiseuilleChannelTestPipeNoCheckRelease           COMMAND $<TARGET_FILE:PoiseuilleChannel> ${CMAKE_CURRENT_SOURCE_DIR}/TestPipeNoCheck.dat           --trt --linear-exp PROCESSES 4 CONFIGURATIONS Release RelWithDbgInfo DEPENDS_ON_TARGETS PoiseuilleChannel )

    waLBerla_execute_test( NO_MODULE_LABEL NAME PoiseuilleChannelTestParallelPlatesNoCheckDebug COMMAND $<TARGET_FILE:PoiseuilleChannel> ${CMAKE_CURRENT_SOURCE_DIR}/TestParallelPlatesNoCheck.dat --trt --linear-exp PROCESSES 4 LABELS longrun CONFIGURATIONS Debug DebugOptimized DEPENDS_ON_TARGETS PoiseuilleChannel )
    waLBerla_execute_test( NO_MODULE_LABEL NAME PoiseuilleChannelTestPipeNoCheckDebug           COMMAND $<TARGET_FILE:PoiseuilleChannel> ${CMAKE_CURRENT_SOURCE_DIR}/TestPipeNoCheck.dat           --trt --linear-exp PROCESSES 4 LABELS longrun CONFIGURATIONS Debug DebugOptimized DEPENDS_ON_TARGETS PoiseuilleChannel )

    waLBerla_execute_test( NO_MODULE_LABEL NAME PoiseuilleChannelTestParallelPlates0 COMMAND $<TARGET_FILE:PoiseuilleChannel> ${CMAKE_CURRENT_SOURCE_DIR}/TestParallelPlates0.dat --trt --linear-exp LABELS longrun                         CONFIGURATIONS Release RelWithDbgInfo DEPENDS_ON_TARGETS PoiseuilleChannel )
    waLBerla_execute_test( NO_MODULE_LABEL NAME PoiseuilleChannelTestParallelPlates2 COMMAND $<TARGET_FILE:PoiseuilleChannel> ${CMAKE_CURRENT_SOURCE_DIR}/TestParallelPlates2.dat --trt --linear-exp LABELS longrun verylongrun PROCESSES 4 CONFIGURATIONS Release RelWithDbgInfo DEPENDS_ON_TARGETS PoiseuilleChannel )
    waLBerla_execute_test( NO_MODULE_LABEL NAME PoiseuilleChannelTestPipe0           COMMAND $<TARGET_FILE:PoiseuilleChannel> ${CMAKE_CURRENT_SOURCE_DIR}/TestPipe0.dat --trt --linear-exp           LABELS longrun                         CONFIGURATIONS Release RelWithDbgInfo DEPENDS_ON_TARGETS PoiseuilleChannel )
    waLBerla_execute_test( NO_MODULE_LABEL NAME PoiseuilleChannelTestPipe2           COMMAND $<TARGET_FILE:PoiseuilleChannel> ${CMAKE_CURRENT_SOURCE_DIR}/TestPipe2.dat --trt --linear-exp           LABELS longrun verylongrun PROCESSES 4 CONFIGURATIONS Release RelWithDbgInfo DEPENDS_ON_TARGETS PoiseuilleChannel )

endif()



