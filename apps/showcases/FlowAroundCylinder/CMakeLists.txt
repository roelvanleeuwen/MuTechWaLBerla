waLBerla_link_files_to_builddir( *.py )
waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )
waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )
waLBerla_link_files_to_builddir( *.mtl )

waLBerla_generate_target_from_python(NAME FlowAroundCylinderCodeGen
        FILE FlowAroundCylinder.py
        OUT_FILES FlowAroundCylinderSweepCollection.h FlowAroundCylinderSweepCollection.${CODEGEN_FILE_SUFFIX}
        FlowAroundCylinderStorageSpecification.h FlowAroundCylinderStorageSpecification.${CODEGEN_FILE_SUFFIX}
        FreeSlip.h FreeSlip.${CODEGEN_FILE_SUFFIX}
        NoSlip.h NoSlip.${CODEGEN_FILE_SUFFIX}
        Obstacle.h Obstacle.${CODEGEN_FILE_SUFFIX}
        UBB.h UBB.${CODEGEN_FILE_SUFFIX}
        Outflow.h Outflow.${CODEGEN_FILE_SUFFIX}
        FlowAroundCylinderBoundaryCollection.h
        FlowAroundCylinderInfoHeader.h
        FlowAroundCylinderStaticDefines.h)

if (WALBERLA_BUILD_WITH_CUDA OR WALBERLA_BUILD_WITH_HIP)
    waLBerla_add_executable ( NAME FlowAroundCylinder
            FILES FlowAroundCylinder.cpp Cylinder.cpp Evaluation.cpp
            DEPENDS blockforest boundary core field gpu lbm_generated stencil timeloop vtk FlowAroundCylinderCodeGen )
else()
    waLBerla_add_executable ( NAME FlowAroundCylinder
            FILES FlowAroundCylinder.cpp Cylinder.cpp Evaluation.cpp
            DEPENDS blockforest boundary core field lbm_generated stencil timeloop vtk FlowAroundCylinderCodeGen )

endif(WALBERLA_BUILD_WITH_CUDA OR WALBERLA_BUILD_WITH_HIP)
