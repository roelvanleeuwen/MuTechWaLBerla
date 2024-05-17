waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )
waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )



if(WALBERLA_BUILD_WITH_GPU_SUPPORT)
    walberla_add_executable ( NAME VoxelizationTest
            FILES VoxelizationTest.cpp
            DEPENDS blockforest gpu core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk )
else()
    walberla_add_executable ( NAME VoxelizationTest
            FILES VoxelizationTest.cpp
            DEPENDS blockforest core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk )
endif()