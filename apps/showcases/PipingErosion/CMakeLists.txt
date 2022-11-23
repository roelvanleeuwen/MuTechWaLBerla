waLBerla_link_files_to_builddir( "*.prm" )

waLBerla_add_executable ( NAME BedGenerationPipingErosion FILES BedGenerationPipingErosion.cpp
        DEPENDS blockforest core mesa_pd vtk )

waLBerla_add_executable ( NAME PipingErosion FILES PipingErosion.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling mesa_pd timeloop vtk )
