if ( WALBERLA_BUILD_WITH_OPENMESH )

    waLBerla_link_files_to_builddir( *.prm )
    waLBerla_link_files_to_builddir( *.stl )
    waLBerla_link_files_to_builddir( airfoil/* )
    waLBerla_link_files_to_builddir( *.md )

    waLBerla_add_executable ( NAME powerFlowCase
                            FILES PowerFlowCase.cpp
                            DEPENDS blockforest core field geometry timeloop gui )
endif()

                        