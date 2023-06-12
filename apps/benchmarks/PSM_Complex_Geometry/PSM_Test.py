import copy
import sympy as sp
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.codegen.ast import Assignment

import pystencils as ps

from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel

from lbmpy.creationfunctions import create_lb_update_rule, create_lb_method
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter, macroscopic_values_getter
from lbmpy.boundaries import NoSlip, UBB, FixedDensity

from pystencils.node_collection import NodeCollection
from pystencils.astnodes import Conditional, SympyAssignment, Block

from pystencils_walberla import CodeGeneration, generate_sweep, generate_pack_info_from_kernel
from lbmpy_walberla import generate_boundary

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
    density = ps.fields(f"density({1}): {data_type}[3D]", layout=layout)

    # LBM Optimisation
    lbm_opt = LBMOptimisation(cse_global=True,
                              symbolic_field=pdfs,
                              symbolic_temporary_field=pdfs_tmp,
                              field_layout=layout)

    #   ==================
    #      Method Setup
    #   ==================

    psm_config = LBMConfig(
        stencil=stencil,
        method=Method.SRT,
        relaxation_rate=omega,
        force=sp.symbols("F_:3"),
        force_model=ForceModel.LUO,
        compressible=False,
    )

    lbm_update_rule = create_lb_update_rule(lbm_config=psm_config, lbm_optimisation=lbm_opt)
    lbm_method = lbm_update_rule.method

    #   ========================
    #      PDF Initialization
    #   ========================

    initial_rho = sp.Symbol('rho_0')

    pdfs_setter = macroscopic_values_setter(lbm_method,
                                            initial_rho,
                                            velocity.center_vector,
                                            pdfs.center_vector)
    pdfs_getter = macroscopic_values_getter(
        lbm_method,
        density=density,
        velocity=velocity.center_vector,
        pdfs=pdfs.center_vector)
    generate_sweep(ctx, "MacroSetter", pdfs_setter)
    generate_sweep(ctx, "MacroGetter", pdfs_getter)

    target = ps.Target.GPU if ctx.gpu else ps.Target.CPU

    #   LBM Sweep
    generate_sweep(ctx, "LBMSweep", lbm_update_rule, field_swaps=[(pdfs, pdfs_tmp)], target=target)

    #   Pack Info
    generate_pack_info_from_kernel(ctx, "PackInfo", lbm_update_rule, target=target)


    #   NoSlip Boundary
    generate_boundary(ctx, "NoSlip", NoSlip(), lbm_method, target=target)
    generate_boundary(ctx, "UBB", UBB((0.01,0,0)), lbm_method, target=target)
    generate_boundary(ctx, "FixedDensity", FixedDensity(1), lbm_method, target=target)





    # PSM Sweep

    split = False
    MaxParticlesPerCell = 1
    pdfs_inter = stencil.Q
    SC = 1

    particle_velocities, particle_forces, Bs = ps.fields(
        f"particle_v({MaxParticlesPerCell * stencil.D}), particle_f({MaxParticlesPerCell * stencil.D}), Bs({MaxParticlesPerCell}): {data_type}[3D]",
        layout=layout,
    )
    B = ps.fields(f"b({1}): {data_type}[3D]", layout=layout)


    method = create_lb_method(lbm_config=psm_config)

    # TODO: think about better way to obtain collision operator than to rebuild it
    # Collision operator with (1 - solid_fraction) as prefactor
    equilibrium = method.get_equilibrium_terms()
    collision_op_psm = []
    if psm_config.method == Method.SRT:
        for eq, f in zip(equilibrium, method.pre_collision_pdf_symbols):
            collision_op_psm.append(
                (1.0 - B.center) * psm_config.relaxation_rate * (f - eq)
            )
    # TODO: set magic number
    elif psm_config.method == Method.TRT:
        for i, (eq, f) in enumerate(zip(equilibrium, method.pre_collision_pdf_symbols)):
            inverse_direction_index = stencil.stencil_entries.index(
                stencil.inverse_stencil_entries[i]
            )
            collision_op_psm.append(
                (1.0 - B.center)
                * (
                        psm_config.relaxation_rates[0]
                        * (
                                (f + method.pre_collision_pdf_symbols[inverse_direction_index])
                                / 2
                                - (eq + equilibrium[inverse_direction_index]) / 2
                        )
                        + psm_config.relaxation_rates[1]
                        * (
                                (f - method.pre_collision_pdf_symbols[inverse_direction_index])
                                / 2
                                - (eq - equilibrium[inverse_direction_index]) / 2
                        )
                )
            )
    else:
        raise ValueError("Only SRT and TRT are supported.")

    # Given forcing operator with (1 - solid_fraction) as prefactor
    fq = method.force_model(method)
    fq_psm = []
    for f in fq:
        fq_psm.append((1 - B.center) * f)

    # Assemble right-hand side of collision assignments
    collision_rhs = []
    for f, c, fo in zip(method.pre_collision_pdf_symbols, collision_op_psm, fq_psm):
        collision_rhs.append(f - c + fo)

    # =====================
    # Code generation for the solid parts
    # =====================

    forces_rhs = [0] * MaxParticlesPerCell * stencil.D
    solid_collisions = [0] * stencil.Q
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
            if split:
                eq_sol = eq_sol.subs(
                    sp.Symbol("delta_rho"),
                    sp.Symbol("delta_rho_inter"),
                )
            equilibriumSolid.append(eq_sol)
        if split:
            equilibriumFluidTmp = equilibriumFluid
            equilibriumFluid = []
            for eq in equilibriumFluidTmp:
                for i in range(stencil.D):
                    eq = eq.subs(
                        sp.Symbol("u_" + str(i)),
                        sp.Symbol("u_" + str(i) + "_inter"),
                    )
                eq = eq.subs(
                    sp.Symbol("delta_rho"),
                    sp.Symbol("delta_rho_inter"),
                )
                equilibriumFluid.append(eq)

        # Assemble right-hand side of collision assignments
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
            if SC == 1:
                solid_collision = Bs.center(p) * (
                        (
                                method.pre_collision_pdf_symbols[inverse_direction_index]
                                - equilibriumFluid[inverse_direction_index]
                        )
                        - (f - eqSolid)
                )
            elif SC == 2:
                solid_collision = Bs.center(p) * (
                        (eqSolid - f) + (1 - omega) * (f - eqFluid)
                )
            elif SC == 3:
                solid_collision = Bs.center(p) * (
                        (
                                method.pre_collision_pdf_symbols[inverse_direction_index]
                                - equilibriumSolid[inverse_direction_index]
                        )
                        - (f - eqSolid)
                )
            else:
                raise ValueError("Only SC=1, SC=2 and SC=3 are supported.")
            solid_collisions[i] += solid_collision
            for j in range(stencil.D):
                forces_rhs[p * stencil.D + j] -= solid_collision * int(offset[j])

    # =====================
    # Assemble update rule
    # =====================

    # Assemble collision assignments
    collision_assignments = []
    if not split:
        for d, c, sc in zip(
                method.post_collision_pdf_symbols, collision_rhs, solid_collisions
        ):
            collision_assignments.append(ps.Assignment(d, c + sc))
    else:
        # First, add only fluid collision
        for d, c in zip(pdfs_inter, collision_rhs):
            collision_assignments.append(ps.Assignment(d, c))

        # Second, define quantities to compute the equilibrium as functions of the pdfs_inter
        cqc_inter = (
            method.conserved_quantity_computation.equilibrium_input_equations_from_pdfs(
                pdfs_inter, False
            )
        )
        for cq in cqc_inter.all_assignments:
            for i in range(stencil.D):
                cq = cq.subs(
                    sp.Symbol("u_" + str(i)), sp.Symbol("u_" + str(i) + "_inter")
                )
                cq = cq.subs(
                    sp.Symbol("vel" + str(i) + "Term"),
                    sp.Symbol("vel" + str(i) + "Term_inter"),
                )
            cq = cq.subs(sp.Symbol("delta_rho"), sp.Symbol("delta_rho_inter"))
            cq = cq.subs(sp.Symbol("rho"), sp.Symbol("rho_inter"))
            collision_assignments.append(cq)

        # Third, add solid collision using quantities from pdfs_inter
        for d, d_inter, sc in zip(
                method.post_collision_pdf_symbols, pdfs_inter, solid_collisions
        ):
            collision_assignments.append(ps.Assignment(d, d_inter + sc))

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
        method.pre_collision_pdf_symbols, psm_config.output
    )
    up = up.new_merged(output_eqs)
    up.method = method

    # Create assignment collection for the complete update rule
    lbm_update_rule = create_lb_update_rule(
        collision_rule=up, lbm_config=psm_config, lbm_optimisation=lbm_opt
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



    generate_sweep(
        ctx,
        "PSMSweep",
        node_collection,
        field_swaps=[(pdfs, pdfs_tmp)],
        target=target,
    )