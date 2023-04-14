waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )

if ( WALBERLA_BUILD_WITH_OPENMESH AND WALBERLA_BUILD_WITH_CODEGEN )

    waLBerla_link_files_to_builddir( *.obj )
    waLBerla_link_files_to_builddir( *.stl )
    waLBerla_link_files_to_builddir( *.mtl )

    if( WALBERLA_BUILD_WITH_CUDA)
        waLBerla_generate_target_from_python(NAME LagoonSparseCodeGen
                FILE LagoonSparse.py
                OUT_FILES LBSweep.cu LBSweep.h
                ListLBMList.cpp ListLBMList.h
                MacroSetter.cpp MacroSetter.h
                UBB.cu UBB.h
                Pressure.cu Pressure.h
                NoSlip.cu NoSlip.h
                PackInfoEven.cu PackInfoEven.h
                PackInfoOdd.cu PackInfoOdd.h
                ListLBMInfoHeader.h)

        waLBerla_add_executable ( NAME LagoonSparse
                FILES LagoonSparse.cpp
                DEPENDS blockforest boundary core cuda field lbm mesh stencil timeloop vtk LagoonSparseCodeGen)
    else()
        waLBerla_generate_target_from_python(NAME LagoonSparseCodeGen
                FILE LagoonSparse.py
                OUT_FILES LBSweep.cpp LBSweep.h
                ListLBMList.cpp ListLBMList.h
                MacroSetter.cpp MacroSetter.h
                UBB.cpp UBB.h
                Pressure.cpp Pressure.h
                NoSlip.cpp NoSlip.h
                PackInfoEven.cpp PackInfoEven.h
                PackInfoOdd.cpp PackInfoOdd.h
                ListLBMInfoHeader.h)

        waLBerla_add_executable ( NAME LagoonSparse
                FILES LagoonSparse.cpp
                DEPENDS blockforest boundary core field lbm mesh stencil timeloop vtk LagoonSparseCodeGen)
    endif()
endif()
