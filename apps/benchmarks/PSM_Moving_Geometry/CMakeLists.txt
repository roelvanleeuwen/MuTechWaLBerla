waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )
waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )


walberla_generate_target_from_python( NAME PSM_Moving_GeometryCodeGen
        FILE PSM_Moving_Geometry.py
        OUT_FILES PSM_Moving_GeometryStorageSpecification.h PSM_Moving_GeometryStorageSpecification.${CODEGEN_FILE_SUFFIX}
        PSM_Moving_GeometrySweepCollection.h PSM_Moving_GeometrySweepCollection.${CODEGEN_FILE_SUFFIX}
        PSM_Moving_GeometryBoundaryCollection.h
        NoSlip.${CODEGEN_FILE_SUFFIX} NoSlip.h
        UBB_top.${CODEGEN_FILE_SUFFIX} UBB_top.h
        UBB_bot.${CODEGEN_FILE_SUFFIX} UBB_bot.h
        PSM_Moving_Geometry_InfoHeader.h
        PSM_Conditional_Sweep.${CODEGEN_FILE_SUFFIX} PSM_Conditional_Sweep.h
        )


if(WALBERLA_BUILD_WITH_GPU_SUPPORT)
    walberla_add_executable ( NAME PSM_Moving_Geometry
            FILES PSM_Moving_Geometry.cpp
            DEPENDS blockforest gpu core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk PSM_Moving_GeometryCodeGen )
    set_source_files_properties(PSM_Moving_Geometry.cpp PROPERTIES LANGUAGE CUDA)

else()
    walberla_add_executable ( NAME PSM_Moving_Geometry
            FILES PSM_Moving_Geometry.cpp
            DEPENDS blockforest core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk PSM_Moving_GeometryCodeGen )
endif()