import copy
import sympy as sp
import numpy as np
from dataclasses import replace

import pystencils as ps

from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel
from lbmpy.partially_saturated_cells import PSMConfig

from lbmpy.creationfunctions import create_lb_collision_rule, create_psm_update_rule
from lbmpy.boundaries import NoSlip, UBB

from pystencils import TypedSymbol

from pystencils_walberla import CodeGeneration, generate_sweep, generate_info_header
from lbmpy_walberla import generate_lbm_package, lbm_boundary_generator

from pystencils.cache import clear_cache
clear_cache()

#   =====================
#      Code Generation
#   =====================

#TODO change values for every resolution
dx = 0.0025
dt = 0.00202857
u_conversion = dt / dx

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

    openmp = True if ctx.openmp else False

    #   PDF Fields
    pdfs, pdfs_tmp = ps.fields(f'pdfs({stencil.Q}), pdfs_tmp({stencil.Q}): {data_type}[3D]', layout=layout)

    #   Velocity Output Field
    velocity = ps.fields(f"velocity({3}): {data_type}[3D]", layout=layout)
    density = ps.fields(f"density({1}): {data_type}[3D]", layout=layout)
    macroscopic_fields = {'density': density, 'velocity': velocity}

    fraction_field = ps.fields(f"frac_field({1}): {data_type}[3D]", layout=layout,)
    object_velocity_field = ps.fields(f"obj_vel({3}): {data_type}[3D]", layout=layout,)
    force_field = ps.fields(f"force({3}): {data_type}[3D]", layout=layout,)

    psm_config = PSMConfig(
        fraction_field=fraction_field,
        object_velocity_field=object_velocity_field,
        SC=1,
        particle_force_field=force_field
    )


    lbm_config = LBMConfig(
        stencil=stencil,
        method=Method.SRT,
        relaxation_rate=omega,
        compressible=False,
        zero_centered=True,
        psm_config=psm_config,
    )

    lbm_config_no_psm = replace(lbm_config, psm_config=None)

    lbm_opt = LBMOptimisation(cse_global=True,
                              symbolic_field=pdfs,
                              symbolic_temporary_field=pdfs_tmp,
                              field_layout=layout,
                              simplification=True)


    no_slip = lbm_boundary_generator(class_name='NoSlip', flag_uid='NoSlip', boundary_object=NoSlip())

    ubb_top_vel = (TypedSymbol("ubb_top_vel_x", data_type), 0, 0)
    ubb_top = lbm_boundary_generator(class_name='UBB_top', flag_uid='UBB_top', boundary_object=UBB(ubb_top_vel, data_type=data_type))

    ubb_bot_vel = (TypedSymbol("ubb_bot_vel_x", data_type), 0, 0)
    ubb_bot = lbm_boundary_generator(class_name='UBB_bot', flag_uid='UBB_bot', boundary_object=UBB(ubb_bot_vel, data_type=data_type))

    #fixedDensity = lbm_boundary_generator(class_name='FixedDensity', flag_uid='FixedDensity', boundary_object=FixedDensity(1.0))
    #extrapolOutflow = lbm_boundary_generator(class_name='ExtrapolationOutflow', flag_uid='ExtrapolationOutflow', boundary_object=SimpleExtrapolationOutflow((1, 0, 0), stencil))


    target = ps.Target.GPU if ctx.gpu else ps.Target.CPU

    collision_rule = create_lb_collision_rule(
        lbm_config=lbm_config,
        lbm_optimisation=lbm_opt
    )


    generate_lbm_package(ctx, name="PSM_Moving_Geometry",
                         collision_rule=collision_rule,
                         lbm_config=lbm_config, lbm_optimisation=lbm_opt,
                         nonuniform=False, boundaries=[no_slip, ubb_top, ubb_bot],
                         macroscopic_fields=macroscopic_fields,
                         cpu_openmp=openmp, cpu_vectorize_info=cpu_vec, target=target)


    field_typedefs = {'VectorField_T': velocity,
                      'ScalarField_T': density}

    psm_condition_update_rule = create_psm_update_rule(lbm_config, lbm_opt)

    generate_sweep(ctx, "PSM_Conditional_Sweep", psm_condition_update_rule, field_swaps=[(pdfs, pdfs_tmp)],
                   cpu_openmp=openmp, cpu_vectorize_info=cpu_vec, target=target)

    generate_info_header(ctx, 'PSM_Moving_Geometry_InfoHeader',
                         field_typedefs=field_typedefs)


