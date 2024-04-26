waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )
waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )


walberla_generate_target_from_python( NAME CRORCodeGen
        FILE CROR.py
        OUT_FILES CRORStorageSpecification.h CRORStorageSpecification.${CODEGEN_FILE_SUFFIX}
        CRORSweepCollection.h CRORSweepCollection.${CODEGEN_FILE_SUFFIX}
        CRORBoundaryCollection.h
        NoSlip.${CODEGEN_FILE_SUFFIX} NoSlip.h
        UBB.${CODEGEN_FILE_SUFFIX} UBB.h
        FixedDensity.${CODEGEN_FILE_SUFFIX} FixedDensity.h
        ExtrapolationOutflow.${CODEGEN_FILE_SUFFIX} ExtrapolationOutflow.h
        CROR_InfoHeader.h
        )


if(WALBERLA_BUILD_WITH_GPU_SUPPORT)
    walberla_add_executable ( NAME CROR
            FILES CROR.cpp ../MovingGeometry.h ../MovingGeometry.impl.gpu.h
            DEPENDS blockforest gpu core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk CRORCodeGen )
else()
    walberla_add_executable ( NAME CROR
            FILES CROR.cpp ../MovingGeometry.h
            DEPENDS blockforest core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk CRORCodeGen )
endif()