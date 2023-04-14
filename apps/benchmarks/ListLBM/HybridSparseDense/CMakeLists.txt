waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )

if ( WALBERLA_BUILD_WITH_OPENMESH AND WALBERLA_BUILD_WITH_CODEGEN )

    waLBerla_link_files_to_builddir( *.obj )
    waLBerla_link_files_to_builddir( *.stl )
    waLBerla_link_files_to_builddir( *.mtl )

    if( WALBERLA_BUILD_WITH_CUDA)
        waLBerla_generate_target_from_python(NAME HybridSparseDenseCodeGen
                FILE HybridSparseDense.py
                OUT_FILES SparseLBSweep.cu SparseLBSweep.h
                ListLBMList.cpp ListLBMList.h
                SparseMacroSetter.cpp SparseMacroSetter.h
                SparseUBB.cu SparseUBB.h
                SparsePressure.cu SparsePressure.h
                SparseNoSlip.cu SparseNoSlip.h
                SparsePackInfoEven.cu SparsePackInfoEven.h
                SparsePackInfoOdd.cu SparsePackInfoOdd.h
                SparseLBMInfoHeader.h
                DenseLBSweep.cu DenseLBSweep.h
                DenseMacroSetter.cpp DenseMacroSetter.h
                DenseMacroGetter.cpp DenseMacroGetter.h
                DenseUBB.cu DenseUBB.h
                DensePressure.cu DensePressure.h
                DenseNoSlip.cu DenseNoSlip.h
                DensePackInfoEven.cu DensePackInfoEven.h
                DensePackInfoOdd.cu DensePackInfoOdd.h
                DenseLBMInfoHeader.h)

        waLBerla_add_executable ( NAME HybridSparseDense
                FILES HybridSparseDense.cpp InitSpherePacking.h InitSpherePacking.cpp SetupHybridCommunication.h
                DEPENDS blockforest boundary core cuda field lbm mesh stencil timeloop vtk HybridSparseDenseCodeGen)
    else()
        waLBerla_generate_target_from_python(NAME HybridSparseDenseCodeGen
                FILE HybridSparseDense.py
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
                DenseLBMInfoHeader.h)

        waLBerla_add_executable ( NAME HybridSparseDense
                FILES HybridSparseDense.cpp InitSpherePacking.h InitSpherePacking.cpp SetupHybridCommunication.h
                DEPENDS blockforest boundary core field lbm mesh stencil timeloop vtk HybridSparseDenseCodeGen)
    endif()
endif()
