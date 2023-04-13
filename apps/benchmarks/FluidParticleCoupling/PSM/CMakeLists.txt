waLBerla_add_executable(NAME LubricationForceEvaluationPSM FILES LubricationForceEvaluationPSM.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling mesa_pd
        postprocessing timeloop vtk FluidParticleCouplingGeneratedLBM)

waLBerla_add_executable(NAME SphereWallCollisionPSM FILES SphereWallCollisionPSM.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling
        mesa_pd postprocessing timeloop vtk FluidParticleCouplingGeneratedLBM)

waLBerla_add_executable(NAME SettlingSphereInBoxPSM FILES SettlingSphereInBoxPSM.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling
        mesa_pd postprocessing timeloop vtk FluidParticleCouplingGeneratedLBM)

foreach (collision_setup srt trt)
    foreach (solid_collision 1 2 3)
        foreach (weighting 1 2)
            foreach (split 0 1)
                set(config ${collision_setup}_sc${solid_collision}_w${weighting}_s${split})
                waLBerla_add_executable(NAME DragForceOnSpherePSM_${config} FILES DragForceSpherePSM.cpp
                        DEPENDS blockforest core cuda field lbm lbm_mesapd_coupling mesa_pd timeloop PSMCodegenPython_${collision_setup}_sc${solid_collision}_s${split})
            endforeach ()
        endforeach ()
    endforeach ()
endforeach ()
