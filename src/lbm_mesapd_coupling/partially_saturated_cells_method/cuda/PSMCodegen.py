import copy
import sympy as sp
import pystencils as ps
from sympy.core.add import Add
from sympy.core.mul import Mul

from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel

from lbmpy.boundaries import NoSlip, UBB, FixedDensity
from lbmpy.creationfunctions import create_lb_update_rule, create_lb_method
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
from pystencils.astnodes import Conditional, SympyAssignment, Block
from pystencils.node_collection import NodeCollection

from pystencils_walberla import (
    CodeGeneration,
    generate_sweep,
    generate_pack_info_from_kernel,
)

from lbmpy_walberla import generate_boundary

# Based on the following paper: https://doi.org/10.1016/j.compfluid.2017.05.033

with CodeGeneration() as ctx:
    data_type = "float64" if ctx.double_accuracy else "float32"
    stencil = LBStencil(Stencil.D3Q19)
    omega = sp.Symbol("omega")
    init_density = sp.Symbol("init_density")
    init_velocity = sp.symbols("init_velocity_:3")
    layout = "fzyx"
    MaxParticlesPerCell = 2

    pdfs, pdfs_tmp = ps.fields(
        f"pdfs({stencil.Q}), pdfs_tmp({stencil.Q}): {data_type}[3D]", layout=layout
    )

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
    # Code generation for the solid parts
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

    # Create assignment collection for the complete update rule
    lbm_update_rule = create_lb_update_rule(
        collision_rule=up, lbm_config=srt_psm_config, lbm_optimisation=psm_opt
    )

    # =====================
    # Add conditionals for the solid parts
    # =====================

    # Transform the assignment collection into a node collection to be able to add conditionals
    node_collection = NodeCollection.from_assignment_collection(lbm_update_rule)

    conditionals = []
    for p in range(MaxParticlesPerCell):
        # One conditional for every potentially overlapping particle
        conditional_assignments = []

        # Move force computations to conditional
        for i in range(stencil.D):
            for node in node_collection.all_assignments:
                if type(node) == SympyAssignment and node.lhs == particle_forces.center(
                    p * stencil.D + i
                ):
                    conditional_assignments.append(node)
                    node_collection.all_assignments.remove(node)

        # Move solid collisions to conditional
        for node in node_collection.all_assignments:
            if type(node) == SympyAssignment and type(node.rhs) == Add:
                rhs = node.rhs.args
                # Maximum one solid collision for each potentially overlapping particle per assignment
                solid_collision = next(
                    (
                        summand
                        for summand in rhs
                        if type(summand) == Mul and Bs.center(p) in summand.args
                    ),
                    None,
                )
                if solid_collision is not None:
                    conditional_assignments.append(
                        SympyAssignment(
                            copy.deepcopy(node.lhs),
                            Add(solid_collision, copy.deepcopy(node.lhs)),
                        )
                    )
                    node.rhs = Add(
                        *[
                            summand
                            for summand in rhs
                            if not (
                                type(summand) == Mul and Bs.center(p) in summand.args
                            )
                        ]
                    )

        conditional = Conditional(Bs.center(p) > 0.0, Block(conditional_assignments))

        conditionals.append(conditional)
        # Append conditional at the end of previous conditional
        # because n+1 particles can only overlap if at least n particles overlap
        if p > 0:
            conditionals[-2].true_block.append(conditional)

    # Add first conditional to node collection, the other conditionals are nested inside the first one
    node_collection.all_assignments.append(conditionals[0])

    pdfs_setter = macroscopic_values_setter(
        method, init_density, init_velocity, pdfs.center_vector
    )

    if ctx.cuda:
        target = ps.Target.GPU
    else:
        target = ps.Target.CPU

    # Generate files
    generate_sweep(
        ctx,
        "PSMSweep",
        node_collection,
        field_swaps=[(pdfs, pdfs_tmp)],
        target=target,
    )

    generate_sweep(
        ctx,
        "PSMSweepSplit",
        node_collection,
        field_swaps=[(pdfs, pdfs_tmp)],
        target=target,
        inner_outer_split=True,
    )

    generate_pack_info_from_kernel(ctx, "PSMPackInfo", lbm_update_rule, target=target)

    generate_sweep(ctx, "InitialPDFsSetter", pdfs_setter, target=target)

    # TODO: check if boundary condition is correct for settling sphere
    generate_boundary(
        ctx,
        "PSM_NoSlip",
        NoSlip(),
        method,
        field_name=pdfs.name,
        streaming_pattern="pull",
        target=target,
    )

    bc_velocity = sp.symbols("bc_velocity_:3")
    generate_boundary(
        ctx,
        "PSM_UBB",
        UBB(bc_velocity),
        method,
        field_name=pdfs.name,
        streaming_pattern="pull",
        target=target,
    )

    bc_density = sp.Symbol("bc_density")
    generate_boundary(
        ctx,
        "PSM_Density",
        FixedDensity(bc_density),
        method,
        field_name=pdfs.name,
        streaming_pattern="pull",
        target=target,
    )