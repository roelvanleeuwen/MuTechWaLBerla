import sympy as sp
import pystencils as ps

from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel

from lbmpy.creationfunctions import create_lb_update_rule, create_lb_method

from pystencils_walberla import (
    CodeGeneration,
    generate_sweep,
    generate_pack_info_from_kernel,
)

#   =====================
#      Code Generation
#   =====================

with CodeGeneration() as ctx:
    #   ========================
    #      General Parameters
    #   ========================
    data_type = "float64" if ctx.double_accuracy else "float32"
    stencil = LBStencil(Stencil.D3Q19)
    omega = sp.Symbol("omega")
    layout = "fzyx"

    #   PDF Fields
    pdfs, pdfs_tmp = ps.fields(
        f"pdfs({stencil.Q}), pdfs_tmp({stencil.Q}): {data_type}[3D]", layout=layout
    )

    #   Fraction Field
    B = ps.fields(f"b({1}): {data_type}[3D]", layout=layout)

    # LBM Optimisation
    lbm_opt = LBMOptimisation(
        cse_global=True,
        symbolic_field=pdfs,
        symbolic_temporary_field=pdfs_tmp,
        field_layout=layout,
    )

    #   ==================
    #      Method Setup
    #   ==================

    lbm_config = LBMConfig(
        stencil=stencil,
        method=Method.SRT,
        relaxation_rate=omega,
        force=sp.symbols("F_:3"),
        force_model=ForceModel.LUO,
        compressible=False,
    )

    method = create_lb_method(lbm_config=lbm_config)

    equilibrium = method.get_equilibrium_terms()
    psm = []
    for eq, f in zip(equilibrium, method.pre_collision_pdf_symbols):
        psm.append((1 - B.center) * sp.Symbol("omega") * (f - eq))

    Fq = method.force_model(method)
    Fq_psm = []
    for fq in Fq:
        Fq_psm.append(fq * (1 - B.center))

    # TODO: check formulas (- and +)
    collision_psm = []
    for f, c, fo in zip(method.pre_collision_pdf_symbols, psm, Fq_psm):
        collision_psm.append(f - c - fo)

    gjd = []
    for d, c in zip(method.post_collision_pdf_symbols, collision_psm):
        gjd.append(ps.Assignment(d, c))

    cqc = method.conserved_quantity_computation.equilibrium_input_equations_from_pdfs(
        method.pre_collision_pdf_symbols, False
    )

    collision_total = cqc.all_assignments + collision_psm

    up = ps.AssignmentCollection(gjd, subexpressions=cqc.all_assignments)
    up.method = method

    lbm_update_rule = create_lb_update_rule(
        collision_rule=up, lbm_config=lbm_config, lbm_optimisation=lbm_opt
    )

    if ctx.cuda:
        target = ps.Target.GPU
    else:
        target = ps.Target.CPU

    #   LBM Sweep
    generate_sweep(
        ctx, "SRTSweep", lbm_update_rule, field_swaps=[(pdfs, pdfs_tmp)], target=target
    )

    #   Pack Info
    generate_pack_info_from_kernel(ctx, "SRTPackInfo", lbm_update_rule, target=target)
