waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )
waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )


walberla_generate_target_from_python( NAME PSM_Benchmark_TestCodeGen
        FILE PSM_Benchmark_Test.py
        OUT_FILES PSM_Benchmark_TestStorageSpecification.h PSM_Benchmark_TestStorageSpecification.${CODEGEN_FILE_SUFFIX}
        PSM_Benchmark_TestSweepCollection.h PSM_Benchmark_TestSweepCollection.${CODEGEN_FILE_SUFFIX}
        PSM_Benchmark_TestBoundaryCollection.h
        NoSlip.${CODEGEN_FILE_SUFFIX} NoSlip.h
        PSM_Benchmark_Test_InfoHeader.h
        PSM_Conditional_Sweep.${CODEGEN_FILE_SUFFIX} PSM_Conditional_Sweep.h
        )


if(WALBERLA_BUILD_WITH_GPU_SUPPORT)
    walberla_add_executable ( NAME PSM_Benchmark_Test
            FILES PSM_Benchmark_Test.cpp ../PSM_Complex_Geometry/MovingGeometry.h ../PSM_Complex_Geometry/MovingGeometry.impl.gpu.h
            DEPENDS blockforest gpu core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk PSM_Benchmark_TestCodeGen )
else()
    walberla_add_executable ( NAME PSM_Benchmark_Test
            FILES PSM_Benchmark_Test.cpp ../PSM_Complex_Geometry/MovingGeometry.h
            DEPENDS blockforest core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk PSM_Benchmark_TestCodeGen )
endif()