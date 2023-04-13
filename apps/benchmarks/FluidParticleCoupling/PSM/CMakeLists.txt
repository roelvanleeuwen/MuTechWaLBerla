waLBerla_add_executable(NAME LubricationForceEvaluationPSM FILES LubricationForceEvaluationPSM.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling mesa_pd
        postprocessing timeloop vtk FluidParticleCouplingGeneratedLBM)

waLBerla_add_executable(NAME SphereWallCollisionPSM FILES SphereWallCollisionPSM.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling
        mesa_pd postprocessing timeloop vtk FluidParticleCouplingGeneratedLBM)

waLBerla_add_executable(NAME SettlingSphereInBoxPSM FILES SettlingSphereInBoxPSM.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling
        mesa_pd postprocessing timeloop vtk FluidParticleCouplingGeneratedLBM)

waLBerla_add_executable(NAME DragForceOnSpherePSM FILES DragForceSpherePSM.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling mesa_pd
        postprocessing timeloop vtk FluidParticleCouplingGeneratedLBMWithForce)
