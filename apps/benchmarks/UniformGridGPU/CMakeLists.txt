
waLBerla_link_files_to_builddir( "*.prm" )
waLBerla_link_files_to_builddir( "*.py" )
waLBerla_link_files_to_builddir( "simulation_setup" )


foreach (config srt trt mrt smagorinsky entropic smagorinsky_noopt entropic_kbc_n4
      entropic_kbc_n4_noopt mrt_noopt mrt_full mrt_full_noopt
      cumulant cumulant_d3q27
      srt_d3q27 mrt_d3q27 mrt_d3q27_noopt smagorinsky_d3q27 smagorinsky_d3q27_noopt mrt_full_d3q27 mrt_full_d3q27_noopt)

    waLBerla_generate_target_from_python(NAME UniformGridGPUGenerated_${config}
          FILE UniformGridGPU.py
          CODEGEN_CFG ${config}
          OUT_FILES UniformGridGPU_LatticeModel.cpp UniformGridGPU_LatticeModel.h
          UniformGridGPU_LbKernel.cu UniformGridGPU_LbKernel.h
          UniformGridGPU_NoSlip.cu UniformGridGPU_NoSlip.h
          UniformGridGPU_UBB.cu UniformGridGPU_UBB.h
          UniformGridGPU_PackInfo.cu UniformGridGPU_PackInfo.h
          UniformGridGPU_MacroSetter.cpp UniformGridGPU_MacroSetter.h
          UniformGridGPU_MacroGetter.cpp UniformGridGPU_MacroGetter.h
          UniformGridGPU_Defines.h
          )


    waLBerla_add_executable(NAME UniformGridBenchmarkGPU_${config}
          FILES UniformGridGPU.cpp
          DEPENDS blockforest boundary core cuda domain_decomposition field geometry timeloop vtk gui UniformGridGPUGenerated_${config})
    set_target_properties( UniformGridBenchmarkGPU_${config} PROPERTIES CXX_VISIBILITY_PRESET hidden)
endforeach ()

foreach(streaming_pattern aa) # choose from {pull, push, aa, esotwist}
    foreach(stencil d3q27) # choose from {d3q19 d3q27}
        foreach (collision_setup srt trt mrt cumulant cumulant-overrelax cumulant-nosimp) # choose from {srt trt mrt cumulant entropic smagorinsky}
            set(config ${stencil}_${streaming_pattern}_${collision_setup})
            waLBerla_generate_target_from_python(NAME UniformGridGPUGenerated_InPlace_${config}
                    FILE UniformGridGPU_InPlace.py
                    CODEGEN_CFG ${config}
                    OUT_FILES   UniformGridGPU_InPlace_LbKernel.cu UniformGridGPU_InPlace_LbKernel.h
                    UniformGridGPU_InPlace_PackInfoEven.cu UniformGridGPU_InPlace_PackInfoEven.h
                    UniformGridGPU_InPlace_PackInfoOdd.cu UniformGridGPU_InPlace_PackInfoOdd.h
                    UniformGridGPU_InPlace_NoSlip.cu UniformGridGPU_InPlace_NoSlip.h
                    UniformGridGPU_InPlace_UBB.cu UniformGridGPU_InPlace_UBB.h
                    UniformGridGPU_InPlace_MacroSetter.cu UniformGridGPU_InPlace_MacroSetter.h
                    UniformGridGPU_InPlace_Defines.h
                    )


            waLBerla_add_executable(NAME UniformGridBenchmarkGPU_InPlace_${config}
                    FILES UniformGridGPU_InPlace.cpp
                    DEPENDS blockforest boundary core cuda domain_decomposition field geometry timeloop vtk UniformGridGPUGenerated_InPlace_${config})
            set_target_properties( UniformGridBenchmarkGPU_InPlace_${config} PROPERTIES CXX_VISIBILITY_PRESET hidden)
        endforeach ()
    endforeach()
endforeach()