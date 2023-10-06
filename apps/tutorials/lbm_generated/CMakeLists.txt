waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )

waLBerla_add_executable( NAME 01_GeneratedLbmBasics
        FILES 01_GeneratedLbmBasics.cpp
        DEPENDS blockforest core field geometry timeloop lbm_generated vtk )
