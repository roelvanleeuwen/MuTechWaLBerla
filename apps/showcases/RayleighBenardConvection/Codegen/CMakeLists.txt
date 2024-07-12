waLBerla_link_files_to_builddir(*.prm)
waLBerla_link_files_to_builddir(*.py)
waLBerla_link_files_to_builddir(*.obj)

if( WALBERLA_BUILD_WITH_CODEGEN )
    waLBerla_generate_target_from_python(NAME RayleighBenardConvectionCodeGen
            FILE RayleighBenardConvection_codegen.py
            OUT_FILES initialize_zhang_thermal_field.cpp initialize_zhang_thermal_field.h
            initialize_fluid_field.cpp initialize_fluid_field.h
            #thermal_lb_step.cpp thermal_lb_step.h
            zhang_thermal_lb_step.cpp zhang_thermal_lb_step.h
            fluid_lb_step.cpp fluid_lb_step.h
            BC_thermal_Thot.cpp BC_thermal_Thot.h
            BC_thermal_Tcold.cpp BC_thermal_Tcold.h
            BC_fluid_NoSlip.cpp BC_fluid_NoSlip.h
            PackInfo_thermal.cpp PackInfo_thermal.h
            PackInfo_hydro.cpp PackInfo_hydro.h
            GenDefines.h)

    waLBerla_add_executable(NAME RayleighBenardConvectionCodegen
            FILES RayleighBenardConvectionCodegen.cpp InitializerFunctions.cpp RayleighBenardConvection_codegen.py
            DEPENDS blockforest core field postprocessing python_coupling lbm geometry timeloop RayleighBenardConvectionCodeGen)
endif()
