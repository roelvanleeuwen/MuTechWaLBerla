waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )
waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )


walberla_generate_target_from_python( NAME VoxelizationTestCodeGen
        FILE VoxelizationTest.py
        OUT_FILES VoxelizationTestStorageSpecification.h VoxelizationTestStorageSpecification.${CODEGEN_FILE_SUFFIX}
        VoxelizationTestSweepCollection.h VoxelizationTestSweepCollection.${CODEGEN_FILE_SUFFIX}
        VoxelizationTestBoundaryCollection.h
        NoSlip.${CODEGEN_FILE_SUFFIX} NoSlip.h
        UBB.${CODEGEN_FILE_SUFFIX} UBB.h
        FixedDensity.${CODEGEN_FILE_SUFFIX} FixedDensity.h
        ExtrapolationOutflow.${CODEGEN_FILE_SUFFIX} ExtrapolationOutflow.h
        VoxelizationTest_InfoHeader.h
        )


if(WALBERLA_BUILD_WITH_GPU_SUPPORT)
    walberla_add_executable ( NAME VoxelizationTest
            FILES VoxelizationTest.cpp ../ObjectRotatorGPUOpenLB.h ../ObjectRotatorGPUOpenLB.cu
            DEPENDS blockforest gpu core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk VoxelizationTestCodeGen )
else()
    walberla_add_executable ( NAME VoxelizationTest
            FILES VoxelizationTest.cpp ../ObjectRotatorOpenLB.h
            DEPENDS blockforest core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk VoxelizationTestCodeGen )
endif()