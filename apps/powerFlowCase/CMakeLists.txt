if ( WALBERLA_BUILD_WITH_OPENMESH )

    waLBerla_link_files_to_builddir( *.prm )
    waLBerla_link_files_to_builddir( *.stl )
    waLBerla_link_files_to_builddir( *.md )

    waLBerla_add_executable ( NAME powerFlowCase
                            FILES powerFlowCase.cpp
                            DEPENDS blockforest core field geometry timeloop mesh boundary lbm vtk )
    message(STATUS "Building powerFlowCase")

    # waLBerla_add_executable ( NAME powerFlowCase_previous
    #                         FILES powerFlowCase_previous.cpp
    #                         DEPENDS blockforest core field geometry timeloop mesh boundary lbm vtk )
    # message(STATUS "Building powerFlowCase_previous")
endif()

                        