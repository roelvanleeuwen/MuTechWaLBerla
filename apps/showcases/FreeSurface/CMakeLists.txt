waLBerla_link_files_to_builddir( *.prm )

waLBerla_add_executable(NAME    BubblyPoiseuille
                        FILES   BubblyPoiseuille.cpp
                        DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing timeloop vtk)

waLBerla_add_executable(NAME    CapillaryWave
                        FILES   CapillaryWave.cpp
                        DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing timeloop vtk)

waLBerla_add_executable(NAME    DamBreakCylindrical
                        FILES   DamBreakCylindrical.cpp
                        DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing timeloop vtk)

waLBerla_add_executable(NAME    DamBreakRectangular
                        FILES   DamBreakRectangular.cpp
                        DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing timeloop vtk)

waLBerla_add_executable(NAME    DropImpact
                        FILES   DropImpact.cpp
                        DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing timeloop vtk)

waLBerla_add_executable(NAME    DropWetting
                        FILES   DropWetting.cpp
                        DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing timeloop vtk)

waLBerla_add_executable(NAME    GravityWave
                        FILES   GravityWave.cpp
                        DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing timeloop vtk)

if( WALBERLA_BUILD_WITH_CODEGEN )
   walberla_generate_target_from_python( NAME      GravityWaveLatticeModelGeneration
                                         FILE      GravityWaveLatticeModelGeneration.py
                                         OUT_FILES GravityWaveLatticeModel.cpp GravityWaveLatticeModel.h )

   waLBerla_add_executable(NAME    GravityWaveCodegen
                           FILES   GravityWaveCodegen.cpp SQLProperties.cpp
                           DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing sqlite
                                   timeloop vtk GravityWaveLatticeModelGeneration)

   walberla_generate_target_from_python( NAME      SchwipSchwapLatticeModelGeneration
                           FILE SchwipSchwapLatticeModelGeneration.py
                           OUT_FILES SchwipSchwapLatticeModel.cpp SchwipSchwapLatticeModel.h SchwipSchwapLatticeModelThermal.cpp SchwipSchwapLatticeModelThermal.h )

   waLBerla_generate_target_from_python(NAME SchwipSchwap_codegen
           FILE SchwipSchwap_codegen.py
           OUT_FILES initialize_thermal_field.cpp initialize_thermal_field.h
           thermal_lb_step.cpp thermal_lb_step.h
           BC_thermal_Thot.cpp BC_thermal_Thot.h
           BC_thermal_Tcold.cpp BC_thermal_Tcold.h
           PackInfo_thermal.cpp PackInfo_thermal.h
           GenDefines.h)

   waLBerla_add_executable(NAME    SchwipSchwap
                           FILES SchwipSchwap.cpp SQLProperties.cpp
                           DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing sqlite
                                   timeloop vtk SchwipSchwapLatticeModelGeneration SchwipSchwap_codegen)
endif()

waLBerla_add_executable(NAME    MovingDrop
                        FILES   MovingDrop.cpp
                        DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing timeloop vtk)

waLBerla_add_executable(NAME    RisingBubble
                        FILES   RisingBubble.cpp
                        DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing timeloop vtk)

waLBerla_add_executable(NAME    TaylorBubble
                        FILES   TaylorBubble.cpp
                        DEPENDS blockforest boundary core domain_decomposition field lbm postprocessing timeloop vtk)