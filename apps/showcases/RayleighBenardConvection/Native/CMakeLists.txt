waLBerla_link_files_to_builddir(*.prm)
waLBerla_link_files_to_builddir(*.py)
waLBerla_link_files_to_builddir(*.obj)

if( WALBERLA_BUILD_WITH_CODEGEN )
    walberla_generate_target_from_python( NAME      RayleighBenardConvectionLatticeModelGeneration
            FILE      RayleighBenardConvectionLatticeModelGeneration.py
            OUT_FILES RayleighBenardConvectionLatticeModel_fluid.cpp RayleighBenardConvectionLatticeModel_fluid.h
            RayleighBenardConvectionLatticeModel_thermal.cpp RayleighBenardConvectionLatticeModel_thermal.h )

    waLBerla_add_executable(NAME    RayleighBenardConvectionNative
            FILES   RayleighBenardConvectionNative.cpp #InitializerFunctions.cpp #SQLProperties.cpp
            DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing python_coupling#sqlite
            timeloop vtk RayleighBenardConvectionLatticeModelGeneration)
endif()
