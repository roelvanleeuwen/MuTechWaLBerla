waLBerla_add_executable(NAME LubricationForceEvaluationPSM FILES LubricationForceEvaluationPSM.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling mesa_pd
        postprocessing timeloop vtk FluidParticleCouplingGeneratedLBM)
