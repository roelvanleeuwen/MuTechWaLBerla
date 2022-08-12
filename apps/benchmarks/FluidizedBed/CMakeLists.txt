waLBerla_link_files_to_builddir( "*.prm" )

waLBerla_add_executable ( NAME FluidizedBed_PSM FILES FluidizedBed.cpp
                          DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling mesa_pd timeloop vtk )

if( WALBERLA_BUILD_WITH_CUDA AND WALBERLA_BUILD_WITH_CODEGEN AND CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60 )
    waLBerla_add_executable ( NAME FluidizedBed_PSM_GPU FILES FluidizedBedGPU.cpp
            DEPENDS blockforest boundary core cuda domain_decomposition field lbm lbm_mesapd_coupling mesa_pd timeloop vtk PSMCodegenPython )
endif()
