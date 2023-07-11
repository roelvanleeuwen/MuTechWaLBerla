import copy
import sympy as sp
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.codegen.ast import Assignment

import pystencils as ps

from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel

from lbmpy.creationfunctions import create_lb_update_rule, create_lb_method, create_lb_collision_rule
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter, macroscopic_values_getter
from lbmpy.boundaries import NoSlip, UBB, FixedDensity, ExtrapolationOutflow

from pystencils.node_collection import NodeCollection
from pystencils.astnodes import Conditional, SympyAssignment, Block

from pystencils_walberla import CodeGeneration, generate_sweep, generate_info_header
from lbmpy_walberla import generate_lbm_package, lbm_boundary_generator


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
    if ctx.optimize_for_localhost:
        cpu_vec = {"nontemporal": False, "assume_aligned": True}
    else:
        cpu_vec = None

    #   PDF Fields
    pdfs, pdfs_tmp = ps.fields(f'pdfs({stencil.Q}), pdfs_tmp({stencil.Q}): {data_type}[3D]', layout=layout)

    #   Velocity Output Field
    velocity = ps.fields(f"velocity({stencil.D}): {data_type}[3D]", layout=layout)
    density = ps.fields(f"density({1}): {data_type}[3D]", layout=layout)
    macroscopic_fields = {'density': density, 'velocity': velocity}

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
    collision_rule = create_lb_collision_rule(lbm_config=psm_config, lbm_optimisation=lbm_opt)

    no_slip = lbm_boundary_generator(class_name='NoSlip', flag_uid='NoSlip',
                                     boundary_object=NoSlip())
    inlet_vel=sp.symbols("inlet_vel"),
    ubb = lbm_boundary_generator(class_name='UBB', flag_uid='UBB', boundary_object=UBB([0.05, 0, 0], data_type=data_type))
    fixedDensity = lbm_boundary_generator(class_name='FixedDensity', flag_uid='FixedDensity', boundary_object=FixedDensity(1.0))
    extrapolOutflow = lbm_boundary_generator(class_name='ExtrapolationOutflow', flag_uid='ExtrapolationOutflow', boundary_object=ExtrapolationOutflow((1,0,0), lbm_method))

#   ========================
    #      PDF Initialization
    #   ========================

    target = ps.Target.GPU if ctx.gpu else ps.Target.CPU

    generate_lbm_package(ctx, name="PSM",
                         collision_rule=collision_rule,
                         lbm_config=psm_config, lbm_optimisation=lbm_opt,
                         nonuniform=False, boundaries=[no_slip, ubb, fixedDensity, extrapolOutflow],
                         macroscopic_fields=macroscopic_fields,
                         cpu_vectorize_info=cpu_vec, target=target)

    #   ========================
    #      PSM SWEEP
    #   ========================

    split = False
    MaxParticlesPerCell = 1
    pdfs_inter = stencil.Q
    SC = 1

    particle_velocities = ps.fields(f"particle_v({stencil.D}): {data_type}[3D]", layout=layout,)

    data_type_B = "float32"
    B = ps.fields(f"B({1}): {data_type_B}[3D]", layout=layout,)


    method = create_lb_method(lbm_config=psm_config)
    collision_rule = create_lb_collision_rule(
        lbm_config=psm_config, lbm_optimisation=lbm_opt
    )

    collision_rhs = []
    for assignment in collision_rule.main_assignments:
        rhsSum = 0
        for arg in assignment.rhs.args:
            if arg not in method.pre_collision_pdf_symbols:
                rhsSum += (1 - B.center) * arg
            else:
                rhsSum += arg
        collision_rhs.append(rhsSum)

    # =====================
    # Code generation for the solid parts
    # =====================

    solid_collisions = [0] * stencil.Q

    equilibriumFluid = method.get_equilibrium_terms()
    equilibriumSolid = []
    for eq in equilibriumFluid:
        eq_sol = eq
        for i in range(stencil.D):
            eq_sol = eq_sol.subs(
                sp.Symbol("u_" + str(i)),
                particle_velocities.center(i),
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
        inverse_direction_index = stencil.stencil_entries.index(stencil.inverse_stencil_entries[i])
        if SC == 1:
            solid_collision = B.center * (
                    (
                            method.pre_collision_pdf_symbols[inverse_direction_index]
                            - equilibriumFluid[inverse_direction_index]
                    )
                    - (f - eqSolid)
            )
        elif SC == 2:
            solid_collision = B.center * (
                    (eqSolid - f) + (1 - omega) * (f - eqFluid)
            )
        elif SC == 3:
            solid_collision = B.center * (
                    (
                            method.pre_collision_pdf_symbols[inverse_direction_index]
                            - equilibriumSolid[inverse_direction_index]
                    )
                    - (f - eqSolid)
            )
        else:
            raise ValueError("Only SC=1, SC=2 and SC=3 are supported.")
        solid_collisions[i] += solid_collision


    # =====================
    # Assemble update rule
    # =====================

    # Assemble collision assignments
    collision_assignments = []
    if not split:
        for d, c, sc in zip( method.post_collision_pdf_symbols, collision_rhs, solid_collisions):
            collision_assignments.append(ps.Assignment(d, c + sc))


    # Define quantities to compute the equilibrium as functions of the pdfs
    # TODO: maybe incorporate some of these assignments into the subexpressions
    # cqc = method.conserved_quantity_computation.equilibrium_input_equations_from_pdfs(
    #    method.pre_collision_pdf_symbols, False
    # )

    up = ps.AssignmentCollection(
        collision_assignments, subexpressions=collision_rule.subexpressions
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

    # One conditional for every potentially overlapping particle
    conditional_assignments = []

    # Move solid collisions to conditional
    for node in node_collection.all_assignments:
        if type(node) == SympyAssignment and type(node.rhs) == Add:
            rhs = node.rhs.args
            # Maximum one solid collision for each potentially overlapping particle per assignment
            solid_collision = next(
                (
                    summand
                    for summand in rhs
                    if type(summand) == Mul and B.center in summand.args
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
                                type(summand) == Mul and B.center in summand.args
                        )
                    ]
                )

    conditional = Conditional(B.center > 0.0, Block(conditional_assignments))

    conditionals.append(conditional)


    # Add first conditional to node collection, the other conditionals are nested inside the first one
    node_collection.all_assignments.append(conditionals[0])

    # Generate files
    generate_sweep(ctx, "PSMSweep", node_collection, field_swaps=[(pdfs, pdfs_tmp)], target=target)

    field_typedefs = {'VectorField_T': velocity,
                      'ScalarField_T': density}

    generate_info_header(ctx, 'PSM_InfoHeader',
                         field_typedefs=field_typedefs)