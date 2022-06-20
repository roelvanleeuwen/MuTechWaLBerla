import sympy as sp
import pystencils as ps

from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel

from lbmpy.creationfunctions import create_lb_update_rule

from pystencils_walberla import CodeGeneration, generate_sweep, generate_pack_info_from_kernel

#   =====================
#      Code Generation
#   =====================

with CodeGeneration() as ctx:
    #   ========================
    #      General Parameters
    #   ========================
    data_type = "float64" if ctx.double_accuracy else "float32"
    stencil = LBStencil(Stencil.D3Q19)
    omega = sp.Symbol('omega')
    layout = 'fzyx'

    #   PDF Fields
    pdfs, pdfs_tmp = ps.fields(f'pdfs({stencil.Q}), pdfs_tmp({stencil.Q}): {data_type}[3D]', layout=layout)

    #   Velocity Output Field
    velocity = ps.fields(f"velocity({stencil.D}): {data_type}[3D]", layout=layout)
    output = {'velocity': velocity}

    # LBM Optimisation
    lbm_opt = LBMOptimisation(cse_global=True,
                              symbolic_field=pdfs,
                              symbolic_temporary_field=pdfs_tmp,
                              field_layout=layout)

    #   ==================
    #      Method Setup
    #   ==================

    lbm_config = LBMConfig(stencil=stencil,
                           method=Method.SRT,
                           relaxation_rate=omega,
                           force_model=ForceModel.LUO,
                           #output=output
                           )

    lbm_update_rule = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)
    lbm_method = lbm_update_rule.method

    #   ========================
    #      PDF Initialization
    #   ========================

    initial_rho = sp.Symbol('rho_0')

    if ctx.cuda:
        target = ps.Target.GPU
    else:
        target = ps.Target.CPU

    #   LBM Sweep
    generate_sweep(ctx, "SRTSweep", lbm_update_rule, field_swaps=[(pdfs, pdfs_tmp)], target=target)

    #   Pack Info
    generate_pack_info_from_kernel(ctx, "SRTPackInfo", lbm_update_rule, target=target)
