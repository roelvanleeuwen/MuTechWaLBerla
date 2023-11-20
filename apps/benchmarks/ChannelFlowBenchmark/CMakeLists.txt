waLBerla_link_files_to_builddir(*.prm)
waLBerla_link_files_to_builddir(*.py)
waLBerla_link_files_to_builddir(*.obj)

if( WALBERLA_BUILD_WITH_CODEGEN )
    waLBerla_generate_target_from_python(NAME ChannelFlowBenchmarkCodeGen
            FILE ChannelFlowBenchmark_codegen.py
            OUT_FILES initialize_fluid_field.cpp initialize_fluid_field.h
            fluid_lb_step.cpp fluid_lb_step.h
            BC_fluid_NoSlip.cpp BC_fluid_NoSlip.h
            PackInfo_hydro.cpp PackInfo_hydro.h
            GenDefines.h)

    waLBerla_add_executable(NAME ChannelFlowBenchmark
            FILES ChannelFlowBenchmark.cpp InitializerFunctions.cpp SQLProperties.cpp ChannelFlowBenchmark_codegen.py
            DEPENDS blockforest core field postprocessing python_coupling lbm geometry sqlite timeloop ChannelFlowBenchmarkCodeGen)
endif()
