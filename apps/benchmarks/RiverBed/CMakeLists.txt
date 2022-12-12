waLBerla_link_files_to_builddir( "*.prm" )

if( WALBERLA_BUILD_WITH_CUDA AND WALBERLA_BUILD_WITH_CODEGEN AND CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60 )
    waLBerla_add_executable ( NAME RiverBed_PSM_GPU FILES RiverBedGPU.cpp
            DEPENDS blockforest boundary core cuda domain_decomposition field lbm lbm_mesapd_coupling mesa_pd timeloop vtk PSMCodegenPython )
endif()
