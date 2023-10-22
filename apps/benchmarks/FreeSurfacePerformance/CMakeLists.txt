waLBerla_link_files_to_builddir( *.prm )

if( WALBERLA_BUILD_WITH_CODEGEN )
   walberla_generate_target_from_python( NAME      GravityWaveLatticeModelGenerationPB
                                         FILE      GravityWaveLatticeModelGeneration.py
                                         OUT_FILES GravityWaveLatticeModel.cpp GravityWaveLatticeModel.h )

   waLBerla_add_executable(NAME    GravityWaveCodegenPBenchmark
                           FILES   GravityWaveCodegen.cpp SQLProperties.cpp
                           DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing sqlite
                                   timeloop vtk GravityWaveLatticeModelGenerationPB)

   waLBerla_add_executable(NAME    GravityWaveCodegenBenchmark
           FILES   GravityWaveCodegenBenchmark.cpp SQLProperties.cpp
           DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing sqlite
           timeloop vtk GravityWaveLatticeModelGenerationPB)

endif()
