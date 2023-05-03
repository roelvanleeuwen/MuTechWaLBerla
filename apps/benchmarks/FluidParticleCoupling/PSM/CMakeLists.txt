waLBerla_link_files_to_builddir("*.dat")
waLBerla_link_files_to_builddir("*.prm")

waLBerla_add_executable(NAME LubricationForceEvaluationPSM FILES LubricationForceEvaluationPSM.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling mesa_pd
        postprocessing timeloop vtk FluidParticleCouplingGeneratedLBM)


foreach (collision_setup srt trt)
    foreach (solid_collision 1 2 3)
        foreach (weighting 1 2)
            foreach (split 0 1)
                set(config ${collision_setup}_sc${solid_collision}_w${weighting}_s${split})
                waLBerla_add_executable(NAME DragForceOnSpherePSM_${config} FILES DragForceSpherePSM.cpp
                        DEPENDS blockforest core cuda field lbm lbm_mesapd_coupling mesa_pd timeloop PSMCodegenPython_${collision_setup}_sc${solid_collision}_s${split})
                target_compile_definitions(DragForceOnSpherePSM_${config} PRIVATE Weighting=${weighting})

                waLBerla_add_executable(NAME ForcesOnSphereNearPlanePSM_${config} FILES ForcesOnSphereNearPlanePSM.cpp
                        DEPENDS blockforest core cuda field lbm lbm_mesapd_coupling mesa_pd vtk PSMCodegenPython_${collision_setup}_sc${solid_collision}_s${split})
                target_compile_definitions(ForcesOnSphereNearPlanePSM_${config} PRIVATE Weighting=${weighting})

                waLBerla_add_executable(NAME SettlingSphereInBoxPSM_${config} FILES SettlingSphereInBoxPSM.cpp
                        DEPENDS blockforest core cuda field lbm lbm_mesapd_coupling mesa_pd timeloop vtk PSMCodegenPython_${collision_setup}_sc${solid_collision}_s${split})
                target_compile_definitions(SettlingSphereInBoxPSM_${config} PRIVATE Weighting=${weighting})

                waLBerla_add_executable(NAME SphereWallCollisionPSM_${config} FILES SphereWallCollisionPSM.cpp
                        DEPENDS blockforest core cuda field lbm lbm_mesapd_coupling mesa_pd timeloop vtk PSMCodegenPython_${collision_setup}_sc${solid_collision}_s${split})
                target_compile_definitions(SphereWallCollisionPSM_${config} PRIVATE Weighting=${weighting})
            endforeach ()
        endforeach ()
    endforeach ()
endforeach ()
