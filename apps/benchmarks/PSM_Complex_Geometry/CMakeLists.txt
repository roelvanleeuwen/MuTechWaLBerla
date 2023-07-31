waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )
waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )


walberla_generate_target_from_python( NAME PSM_ComplexGeometryCodeGen
        FILE PSM_ComplexGeometry.py
        OUT_FILES PSMStorageSpecification.h PSMStorageSpecification.${CODEGEN_FILE_SUFFIX}
        PSMSweepCollection.h PSMSweepCollection.${CODEGEN_FILE_SUFFIX}
        PSMBoundaryCollection.h
        #PSMSweep.${CODEGEN_FILE_SUFFIX} PSMSweep.h
        NoSlip.${CODEGEN_FILE_SUFFIX} NoSlip.h
        UBB.${CODEGEN_FILE_SUFFIX} UBB.h
        FixedDensity.${CODEGEN_FILE_SUFFIX} FixedDensity.h
        ExtrapolationOutflow.${CODEGEN_FILE_SUFFIX} ExtrapolationOutflow.h
        PSM_InfoHeader.h
        )


if(WALBERLA_BUILD_WITH_GPU_SUPPORT)
    walberla_add_executable ( NAME PSM_ComplexGeometry
            FILES PSM_ComplexGeometry.cpp ObjectRotator.h
            DEPENDS blockforest gpu core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk PSM_ComplexGeometryCodeGen )
else()
    walberla_add_executable ( NAME PSM_ComplexGeometry
            FILES PSM_ComplexGeometry.cpp ObjectRotator.h
            DEPENDS blockforest core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk PSM_ComplexGeometryCodeGen )
endif()