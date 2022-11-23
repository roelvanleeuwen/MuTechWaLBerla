waLBerla_link_files_to_builddir( "*.prm" )

waLBerla_add_executable ( NAME BedGenerationPipingErosion FILES BedGenerationPipingErosion.cpp
        DEPENDS blockforest core mesa_pd vtk )
