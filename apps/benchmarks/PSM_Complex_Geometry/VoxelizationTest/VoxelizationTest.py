import copy
import sympy as sp
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.codegen.ast import Assignment

import pystencils as ps

from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel
from lbmpy.partially_saturated_cells import PSMConfig

from lbmpy.creationfunctions import create_lb_update_rule, create_lb_method, create_lb_collision_rule
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter, macroscopic_values_getter
from lbmpy.boundaries import NoSlip, UBB, FixedDensity, SimpleExtrapolationOutflow

from pystencils.node_collection import NodeCollection
from pystencils.astnodes import Conditional, SympyAssignment, Block
from pystencils import TypedSymbol

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

    fraction_field = ps.fields(f"frac_field({1}): {data_type}[3D]", layout=layout,)
    object_velocity_field = ps.fields(f"obj_vel({stencil.D}): {data_type}[3D]", layout=layout,)

    psm_config = PSMConfig(
        fraction_field=fraction_field,
        object_velocity_field=object_velocity_field,
        SC=1,
    )

    lbm_config = LBMConfig(
        stencil=stencil,
        method=Method.CUMULANT,
        relaxation_rate=omega,
        compressible=True,
        zero_centered=True,
        psm_config=psm_config,
    )

    lbm_opt = LBMOptimisation(cse_global=True,
                              symbolic_field=pdfs,
                              symbolic_temporary_field=pdfs_tmp,
                              field_layout=layout,
                              simplification=True)


    no_slip = lbm_boundary_generator(class_name='NoSlip', flag_uid='NoSlip', boundary_object=NoSlip())
    inlet_vel=sp.symbols("inlet_vel"),
    ubb = lbm_boundary_generator(class_name='UBB', flag_uid='UBB', boundary_object=UBB([0.02, 0, 0], data_type=data_type))
    fixedDensity = lbm_boundary_generator(class_name='FixedDensity', flag_uid='FixedDensity', boundary_object=FixedDensity(1.0))
    extrapolOutflow = lbm_boundary_generator(class_name='ExtrapolationOutflow', flag_uid='ExtrapolationOutflow', boundary_object=SimpleExtrapolationOutflow((1, 0, 0), stencil))


    target = ps.Target.GPU if ctx.gpu else ps.Target.CPU

    collision_rule = create_lb_collision_rule(
        lbm_config=lbm_config,
        lbm_optimisation=lbm_opt
    )

    generate_lbm_package(ctx, name="VoxelizationTest",
                         collision_rule=collision_rule,
                         lbm_config=lbm_config, lbm_optimisation=lbm_opt,
                         nonuniform=True, boundaries=[no_slip, ubb, fixedDensity, extrapolOutflow],
                         macroscopic_fields=macroscopic_fields,
                         cpu_vectorize_info=cpu_vec, target=target)


    field_typedefs = {'VectorField_T': velocity,
                      'ScalarField_T': density}

    generate_info_header(ctx, 'VoxelizationTest_InfoHeader',
                         field_typedefs=field_typedefs)