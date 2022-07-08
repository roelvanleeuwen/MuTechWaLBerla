import sympy as sp
import pystencils as ps

from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel

from lbmpy.creationfunctions import create_lb_update_rule, create_lb_method
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter

from pystencils_walberla import (
    CodeGeneration,
    generate_sweep,
    generate_pack_info_from_kernel,
)

# Based on the following paper: https://doi.org/10.1016/j.compfluid.2017.05.033

with CodeGeneration() as ctx:
    data_type = data_type = "float64" if ctx.double_accuracy else "float32"
    stencil = LBStencil(Stencil.D3Q19)
    omega = sp.Symbol("omega")
    init_density = 1.0
    init_velocity = (0, 0, 0)
    layout = "fzyx"
    MaxParticlesPerCell = 2

    pdfs, pdfs_tmp = ps.fields(
        f"pdfs({stencil.Q}), pdfs_tmp({stencil.Q}): {data_type}[3D]", layout=layout
    )

    # Output
    velocity = ps.fields(f"velocity({stencil.D}): {data_type}[3D]", layout=layout)
    output = {"velocity": velocity}

    particle_velocities, particle_forces, Bs = ps.fields(
        f"particle_velocities({MaxParticlesPerCell * stencil.D}), particle_forces({MaxParticlesPerCell * stencil.D}), Bs({MaxParticlesPerCell}): {data_type}[3D]",
        layout=layout,
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
        output=output,
    )

    # =====================
    # Code generation for the modified SRT sweep
    # =====================

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

    # =====================
    # Code generation for the solid collision kernel
    # =====================

    forces_rhs = [0] * MaxParticlesPerCell * stencil.D
    for p in range(MaxParticlesPerCell):
        equilibriumFluid = method.get_equilibrium_terms()
        equilibriumSolid = []
        for eq in equilibriumFluid:
            eq_sol = eq
            for i in range(stencil.D):
                eq_sol = eq_sol.subs(
                    sp.Symbol("u_" + str(i)),
                    particle_velocities.center(p * stencil.D + i),
                )
            equilibriumSolid.append(eq_sol)

        # Assemble right-hand side of collision assignments
        # TODO: add more solid collision operators
        # Add solid collision part to collision right-hand side and forces right-hand side
        for i, (eqFluid, eqSolid, f, offset) in enumerate(
            zip(
                equilibriumFluid,
                equilibriumSolid,
                method.pre_collision_pdf_symbols,
                stencil,
            )
        ):
            inverse_direction_index = stencil.stencil_entries.index(
                stencil.inverse_stencil_entries[i]
            )
            solid_collision = Bs.center(p) * (
                (
                    method.pre_collision_pdf_symbols[inverse_direction_index]
                    - equilibriumFluid[inverse_direction_index]
                )
                - (f - eqSolid)
            )
            collision_rhs[i] += solid_collision
            for j in range(stencil.D):
                forces_rhs[p * stencil.D + j] -= solid_collision * int(offset[j])

    # =====================
    # Assemble update rule
    # =====================

    # Assemble collision assignments
    collision_assignments = []
    for d, c in zip(method.post_collision_pdf_symbols, collision_rhs):
        collision_assignments.append(ps.Assignment(d, c))

    # Add force calculations to collision assignments
    for p in range(MaxParticlesPerCell):
        for i in range(stencil.D):
            collision_assignments.append(
                ps.Assignment(
                    particle_forces.center(p * stencil.D + i),
                    forces_rhs[p * stencil.D + i],
                )
            )

    # Define quantities to compute the equilibrium as functions of the pdfs
    cqc = method.conserved_quantity_computation.equilibrium_input_equations_from_pdfs(
        method.pre_collision_pdf_symbols, False
    )

    up = ps.AssignmentCollection(
        collision_assignments, subexpressions=cqc.all_assignments
    )
    output_eqs = method.conserved_quantity_computation.output_equations_from_pdfs(
        method.pre_collision_pdf_symbols, srt_psm_config.output
    )
    up = up.new_merged(output_eqs)
    up.method = method

    lbm_update_rule = create_lb_update_rule(
        collision_rule=up, lbm_config=srt_psm_config, lbm_optimisation=psm_opt
    )

    # TODO: what is delta_rho and why do we have to set density to 2.0?
    pdfs_setter = macroscopic_values_setter(
        method, init_density, init_velocity, pdfs.center_vector
    )

    if ctx.cuda:
        target = ps.Target.GPU
    else:
        target = ps.Target.CPU

    # Generate files
    generate_sweep(
        ctx, "PSMSweep", lbm_update_rule, field_swaps=[(pdfs, pdfs_tmp)], target=target
    )

    generate_pack_info_from_kernel(ctx, "PSMPackInfo", lbm_update_rule, target=target)

    generate_sweep(ctx, "InitialPDFsSetter", pdfs_setter, target=target)
