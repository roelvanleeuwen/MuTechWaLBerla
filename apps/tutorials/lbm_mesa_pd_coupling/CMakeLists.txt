waLBerla_add_executable(NAME 01_LBM_PSM_CPU
        FILES 01_LBM_PSM_CPU.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling timeloop mesa_pd)
