waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.png )
waLBerla_link_files_to_builddir( *.obj )
waLBerla_link_files_to_builddir( *.stl )
waLBerla_link_files_to_builddir( *.mtl )
waLBerla_link_files_to_builddir( input.py )



waLBerla_generate_target_from_python(NAME TestCommCodeGen
        FILE TestComm.py
        OUT_FILES  UniformGridGPU_LbKernel.cu UniformGridGPU_LbKernel.h
        UniformGridGPU_PackInfoEven.cu UniformGridGPU_PackInfoEven.h
        UniformGridGPU_PackInfoOdd.cu UniformGridGPU_PackInfoOdd.h
        UniformGridGPU_NoSlip.cu UniformGridGPU_NoSlip.h
        UniformGridGPU_UBB.cu UniformGridGPU_UBB.h
        UniformGridGPU_MacroSetter.cu UniformGridGPU_MacroSetter.h
        UniformGridGPU_StreamOnlyKernel.cu UniformGridGPU_StreamOnlyKernel.h
        UniformGridGPU_InfoHeader.h
        )

waLBerla_add_executable(NAME TestComm
        FILES TestComm.cpp
        DEPENDS blockforest boundary core cuda domain_decomposition field geometry python_coupling timeloop vtk TestCommCodeGen)
