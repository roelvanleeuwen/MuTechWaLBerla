
waLBerla_link_files_to_builddir( "*.dat" )                 
waLBerla_link_files_to_builddir( "*.prm" )

waLBerla_add_executable( NAME RayleighBenardConvection
                         FILES RayleighBenardConvection.cpp
                         DEPENDS blockforest core field lbm geometry timeloop )