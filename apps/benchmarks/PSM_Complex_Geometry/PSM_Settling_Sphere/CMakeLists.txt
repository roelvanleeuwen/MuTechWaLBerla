waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )
waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )


walberla_generate_target_from_python( NAME PSM_Settling_SphereCodeGen
        FILE PSM_Settling_Sphere.py
        OUT_FILES PSM_Settling_SphereStorageSpecification.h PSM_Settling_SphereStorageSpecification.${CODEGEN_FILE_SUFFIX}
        PSM_Settling_SphereSweepCollection.h PSM_Settling_SphereSweepCollection.${CODEGEN_FILE_SUFFIX}
        PSM_Settling_SphereBoundaryCollection.h
        NoSlip.${CODEGEN_FILE_SUFFIX} NoSlip.h
        PSM_Settling_Sphere_InfoHeader.h
        PSM_Conditional_Sweep.${CODEGEN_FILE_SUFFIX} PSM_Conditional_Sweep.h
        )


if(WALBERLA_BUILD_WITH_GPU_SUPPORT)
    walberla_add_executable ( NAME PSM_Settling_Sphere
            FILES PSM_Settling_Sphere.cpp
            DEPENDS blockforest gpu core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk PSM_Settling_SphereCodeGen )

else()
    walberla_add_executable ( NAME PSM_Settling_Sphere
            FILES PSM_Settling_Sphere.cpp
            DEPENDS blockforest core domain_decomposition field geometry mesh mesh_common timeloop lbm stencil vtk PSM_Settling_SphereCodeGen )
endif()