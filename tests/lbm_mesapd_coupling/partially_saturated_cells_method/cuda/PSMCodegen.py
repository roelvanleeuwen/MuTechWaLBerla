import sympy as sp
import pystencils as ps

from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel

from lbmpy.creationfunctions import create_lb_update_rule, create_lb_method

from pystencils_walberla import (
    CodeGeneration,
    generate_sweep,
    generate_pack_info_from_kernel,
)

# Based on the following paper: https://doi.org/10.1016/j.compfluid.2017.05.033

with CodeGeneration() as ctx:
    # =====================
    # Code generation for the modified SRT sweep
    # =====================
    data_type = "float64" if ctx.double_accuracy else "float32"
    stencil = LBStencil(Stencil.D3Q19)
    omega = sp.Symbol("omega")
    layout = "fzyx"

    pdfs, pdfs_tmp = ps.fields(
        f"pdfs({stencil.Q}), pdfs_tmp({stencil.Q}): {data_type}[3D]", layout=layout
    )

    # Solid fraction field
    B = ps.fields(f"b({1}): {data_type}[3D]", layout=layout)

    psm_opt = LBMOptimisation(
        cse_global=True,
        symbolic_field=pdfs,
        symbolic_temporary_field=pdfs_tmp,
        field_layout=layout,
    )

    srt_psm_config = LBMConfig(
        stencil=stencil,
        method=Method.SRT,
        relaxation_rate=omega,
        force=sp.symbols("F_:3"),
        force_model=ForceModel.LUO,
        compressible=False,
    )

    method = create_lb_method(lbm_config=srt_psm_config)

    # SRT collision operator with (1 - solid_fraction) as prefactor
    equilibrium = method.get_equilibrium_terms()
    srt_collision_op_psm = []
    for eq, f in zip(equilibrium, method.pre_collision_pdf_symbols):
        srt_collision_op_psm.append((1.0 - B.center) * sp.Symbol("omega") * (f - eq))

    # Given forcing operator with (1 - solid_fraction) as prefactor
    fq = method.force_model(method)
    fq_psm = []
    for f in fq:
        fq_psm.append((1 - B.center) * f)

    # Assemble right-hand side of collision assignments
    collision_rhs = []
    for f, c, fo in zip(method.pre_collision_pdf_symbols, srt_collision_op_psm, fq_psm):
        collision_rhs.append(f - c + fo)

    # Assemble collision assignments
    collision_assignments = []
    for d, c in zip(method.post_collision_pdf_symbols, collision_rhs):
        collision_assignments.append(ps.Assignment(d, c))

    # Define quantities to compute the equilibrium as functions of the pdfs
    cqc = method.conserved_quantity_computation.equilibrium_input_equations_from_pdfs(
        method.pre_collision_pdf_symbols, False
    )

    up = ps.AssignmentCollection(
        collision_assignments, subexpressions=cqc.all_assignments
    )
    up.method = method

    lbm_update_rule = create_lb_update_rule(
        collision_rule=up, lbm_config=srt_psm_config, lbm_optimisation=psm_opt
    )

    if ctx.cuda:
        target = ps.Target.GPU
    else:
        target = ps.Target.CPU

    # Generate files
    generate_sweep(
        ctx, "SRTSweep", lbm_update_rule, field_swaps=[(pdfs, pdfs_tmp)], target=target
    )

    generate_pack_info_from_kernel(ctx, "SRTPackInfo", lbm_update_rule, target=target)


# =====================
# Code generation for the solid collision kernel
# =====================

solid_psm_config = LBMConfig(
    stencil=stencil, method=Method.SRT, compressible=False, kernel_type="collide_only"
)

method = create_lb_method(lbm_config=solid_psm_config)

# Assemble equilibrium for the particle velocity using the fluid equilibrium
equilibriumFluid = method.get_equilibrium_terms()
equilibriumSolid = []
for eq in equilibriumFluid:
    equilibriumSolid.append(
        eq.subs(
            [
                (sp.Symbol("u_0"), sp.Symbol("u_solid_0")),
                (sp.Symbol("u_1"), sp.Symbol("u_solid_1")),
                (sp.Symbol("u_2"), sp.Symbol("u_solid_2")),
            ]
        )
    )

# Assemble right-hand side of collision assignments
# TODO: use f_inv und equilibriumSolid_inv
collision_rhs = []
for eqFluid, eqSolid, f in zip(
    equilibriumFluid, equilibriumSolid, method.pre_collision_pdf_symbols
):
    collision_rhs.append((f - eqFluid) - (f - eqSolid))

# Assemble collision assignments
collision_assignments = []
for d, c in zip(method.post_collision_pdf_symbols, collision_rhs):
    collision_assignments.append(ps.Assignment(d, c))

# Define quantities to compute the fluid equilibrium as functions of the pdfs.
# The velocity for the solid equilibrium is a function parameter.
cqc = method.conserved_quantity_computation.equilibrium_input_equations_from_pdfs(
    method.pre_collision_pdf_symbols, False
)

up = ps.AssignmentCollection(collision_assignments, subexpressions=cqc.all_assignments)
up.method = method

lbm_update_rule = create_lb_update_rule(
    collision_rule=up, lbm_config=solid_psm_config, lbm_optimisation=psm_opt
)

if ctx.cuda:
    target = ps.Target.GPU
else:
    target = ps.Target.CPU

ast = ps.create_kernel(lbm_update_rule)

f = open("SolidKernel.cuh", "w+")
f.write(ps.get_code_str(ast))
