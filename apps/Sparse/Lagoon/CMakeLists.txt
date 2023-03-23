waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )

if ( WALBERLA_BUILD_WITH_OPENMESH AND WALBERLA_BUILD_WITH_CODEGEN )

waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )
waLBerla_link_files_to_builddir( *.mtl )

if (WALBERLA_BUILD_WITH_CUDA)
    waLBerla_generate_target_from_python(NAME LagoonCodeGen
            FILE LagoonCodegen.py
            OUT_FILES Lagoon_LbSweep.cu Lagoon_LbSweep.h
            Lagoon_MacroSetter.cpp Lagoon_MacroSetter.h
            Lagoon_UBB.cu Lagoon_UBB.h
            Lagoon_NoSlip.cu Lagoon_NoSlip.h
            Lagoon_Outflow.cu Lagoon_Outflow.h
            Lagoon_PackInfoEven.cu Lagoon_PackInfoEven.h
            Lagoon_PackInfoOdd.cu Lagoon_PackInfoOdd.h
            Lagoon_InfoHeader.h)

    waLBerla_add_executable ( NAME Lagoon
            FILES Lagoon.cpp
            DEPENDS blockforest boundary core cuda field lbm mesh stencil timeloop vtk LagoonCodeGen )
else()
    waLBerla_generate_target_from_python(NAME LagoonCodeGen
            FILE LagoonCodegen.py
            OUT_FILES Lagoon_LbSweep.cpp Lagoon_LbSweep.h
            Lagoon_MacroSetter.cpp Lagoon_MacroSetter.h
            Lagoon_UBB.cpp Lagoon_UBB.h
            Lagoon_NoSlip.cpp Lagoon_NoSlip.h
            Lagoon_Outflow.cpp Lagoon_Outflow.h
            Lagoon_PackInfoEven.cpp Lagoon_PackInfoEven.h
            Lagoon_PackInfoOdd.cpp Lagoon_PackInfoOdd.h
            Lagoon_InfoHeader.h)

    waLBerla_add_executable ( NAME Lagoon
                              FILES Lagoon.cpp
                              DEPENDS blockforest boundary core field lbm mesh stencil timeloop vtk LagoonCodeGen )

endif()
                          
endif()
