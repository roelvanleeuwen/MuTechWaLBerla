waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )

if ( WALBERLA_BUILD_WITH_OPENMESH AND WALBERLA_BUILD_WITH_CODEGEN )

waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )
waLBerla_link_files_to_builddir( *.mtl )

if (WALBERLA_BUILD_WITH_CUDA)
    waLBerla_generate_target_from_python(NAME LagoonCodeGenSparse
            FILE LagoonCodegenSparse.py
            OUT_FILES Lagoon_LbSweep.cu Lagoon_LbSweep.h
            ListLBMList.cpp ListLBMList.h
            Lagoon_MacroSetter.cpp Lagoon_MacroSetter.h
            Lagoon_UBB.cu Lagoon_UBB.h
            Lagoon_Pressure.cu Lagoon_Pressure.h
            Lagoon_PackInfo.cu Lagoon_PackInfo.h
            ListLBMInfoHeader.h)

    waLBerla_add_executable ( NAME LagoonSparse
            FILES LagoonSparse.cpp
            DEPENDS blockforest boundary core cuda field lbm mesh stencil timeloop vtk LagoonCodeGenSparse )
else()
    waLBerla_generate_target_from_python(NAME LagoonCodeGenSparse
            FILE LagoonCodegenSparse.py
            OUT_FILES Lagoon_LbSweep.cpp Lagoon_LbSweep.h
            ListLBMList.cpp ListLBMList.h
            Lagoon_MacroSetter.cpp Lagoon_MacroSetter.h
            Lagoon_UBB.cpp Lagoon_UBB.h
            Lagoon_Pressure.cpp Lagoon_Pressure.h
            Lagoon_PackInfo.cpp Lagoon_PackInfo.h
            ListLBMInfoHeader.h)

    waLBerla_add_executable ( NAME LagoonSparse
                              FILES LagoonSparse.cpp
                              DEPENDS blockforest boundary core field lbm mesh stencil timeloop vtk LagoonCodeGenSparse )

endif()
                          
endif()
