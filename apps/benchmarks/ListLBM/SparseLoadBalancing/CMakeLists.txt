waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )
waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )
waLBerla_link_files_to_builddir( *.mtl )
waLBerla_link_files_to_builddir( input*.py )


if ( WALBERLA_BUILD_WITH_OPENMESH AND WALBERLA_BUILD_WITH_CODEGEN )

    if( WALBERLA_BUILD_WITH_GPU_SUPPORT)
        waLBerla_generate_target_from_python(NAME SparseLoadBalancingCodeGen
                FILE SparseLoadBalancing.py
                OUT_FILES SparseLBSweep.${CODEGEN_FILE_SUFFIX} SparseLBSweep.h
                ListLBMList.cpp ListLBMList.h
                SparseMacroSetter.cpp SparseMacroSetter.h
                SparseUBB.${CODEGEN_FILE_SUFFIX} SparseUBB.h
                SparsePressure.${CODEGEN_FILE_SUFFIX} SparsePressure.h
                SparseNoSlip.${CODEGEN_FILE_SUFFIX} SparseNoSlip.h
                SparsePackInfoEven.${CODEGEN_FILE_SUFFIX} SparsePackInfoEven.h
                SparsePackInfoOdd.${CODEGEN_FILE_SUFFIX} SparsePackInfoOdd.h
                SparseLBMInfoHeader.h
                DenseLBSweep.${CODEGEN_FILE_SUFFIX} DenseLBSweep.h
                DenseMacroSetter.cpp DenseMacroSetter.h
                DenseMacroGetter.cpp DenseMacroGetter.h
                DenseUBB.${CODEGEN_FILE_SUFFIX} DenseUBB.h
                DensePressure.${CODEGEN_FILE_SUFFIX} DensePressure.h
                DenseNoSlip.${CODEGEN_FILE_SUFFIX} DenseNoSlip.h
                DensePackInfoEven.${CODEGEN_FILE_SUFFIX} DensePackInfoEven.h
                DensePackInfoOdd.${CODEGEN_FILE_SUFFIX} DensePackInfoOdd.h
                DenseLBMInfoHeader.h
                HybridPackInfoEven.${CODEGEN_FILE_SUFFIX} HybridPackInfoEven.h
                HybridPackInfoOdd.${CODEGEN_FILE_SUFFIX} HybridPackInfoOdd.h)

        waLBerla_add_executable ( NAME SparseLoadBalancing
                FILES SparseLoadBalancing.cpp
                DEPENDS blockforest boundary core gpu field lbm mesh stencil timeloop vtk python_coupling SparseLoadBalancingCodeGen)
    else()
        waLBerla_generate_target_from_python(NAME SparseLoadBalancingCodeGen
                FILE SparseLoadBalancing.py
                OUT_FILES SparseLBSweep.cpp SparseLBSweep.h
                ListLBMList.cpp ListLBMList.h
                SparseMacroSetter.cpp SparseMacroSetter.h
                SparseUBB.cpp SparseUBB.h
                SparsePressure.cpp SparsePressure.h
                SparseNoSlip.cpp SparseNoSlip.h
                SparsePackInfoEven.cpp SparsePackInfoEven.h
                SparsePackInfoOdd.cpp SparsePackInfoOdd.h
                SparseLBMInfoHeader.h
                DenseLBSweep.cpp DenseLBSweep.h
                DenseMacroSetter.cpp DenseMacroSetter.h
                DenseMacroGetter.cpp DenseMacroGetter.h
                DenseUBB.cpp DenseUBB.h
                DensePressure.cpp DensePressure.h
                DenseNoSlip.cpp DenseNoSlip.h
                DensePackInfoEven.cpp DensePackInfoEven.h
                DensePackInfoOdd.cpp DensePackInfoOdd.h
                DenseLBMInfoHeader.h
                HybridPackInfoEven.cpp HybridPackInfoEven.h
                HybridPackInfoOdd.cpp HybridPackInfoOdd.h)

        waLBerla_add_executable ( NAME SparseLoadBalancing
                FILES SparseLoadBalancing.cpp
                DEPENDS blockforest boundary core field lbm mesh stencil timeloop vtk python_coupling SparseLoadBalancingCodeGen)
    endif()
endif()
