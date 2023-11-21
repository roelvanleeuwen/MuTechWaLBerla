waLBerla_link_files_to_builddir( *.py )
waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )
waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )
waLBerla_link_files_to_builddir( *.mtl )

waLBerla_generate_target_from_python(NAME FlowAroundSphereCodeGen
        FILE FlowAroundSphere.py
        OUT_FILES FlowAroundSphereSweepCollection.h FlowAroundSphereSweepCollection.${CODEGEN_FILE_SUFFIX}
        FlowAroundSphereStorageSpecification.h FlowAroundSphereStorageSpecification.${CODEGEN_FILE_SUFFIX}
        NoSlipBouzidi.h NoSlipBouzidi.${CODEGEN_FILE_SUFFIX}
        UBB.h UBB.${CODEGEN_FILE_SUFFIX}
        Outflow.h Outflow.${CODEGEN_FILE_SUFFIX}
        FlowAroundSphereBoundaryCollection.h
        FlowAroundSphereInfoHeader.h)

if (WALBERLA_BUILD_WITH_CUDA OR WALBERLA_BUILD_WITH_HIP)
    waLBerla_add_executable ( NAME FlowAroundSphere
            FILES FlowAroundSphere.cpp
            DEPENDS blockforest boundary core field gpu lbm_generated mesh stencil timeloop vtk FlowAroundSphereCodeGen )
else()
    waLBerla_add_executable ( NAME FlowAroundSphere
            FILES FlowAroundSphere.cpp
            DEPENDS blockforest boundary core field lbm_generated mesh stencil timeloop vtk FlowAroundSphereCodeGen )

endif(WALBERLA_BUILD_WITH_CUDA OR WALBERLA_BUILD_WITH_HIP)
