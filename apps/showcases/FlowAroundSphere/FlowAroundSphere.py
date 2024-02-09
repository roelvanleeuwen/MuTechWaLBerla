import sympy as sp
import numpy as np
from pystencils import TypedSymbol, Target

from pystencils.field import fields
from pystencils.simp.subexpression_insertion import insert_constants, insert_aliases

from lbmpy import Stencil, LBStencil, Method, LBMConfig, LBMOptimisation
from lbmpy.boundaries.boundaryconditions import ExtrapolationOutflow, UBB, QuadraticBounceBack, FreeSlip
from lbmpy.creationfunctions import create_lb_collision_rule

from pystencils_walberla import CodeGeneration, generate_info_header
from lbmpy_walberla import generate_lbm_package, lbm_boundary_generator

info_header = """
#pragma once
const char * infoStencil = "{stencil}";
const char * infoStreamingPattern = "{streaming_pattern}";
const char * infoCollisionOperator = "{collision_operator}";
"""

omega = sp.symbols("omega")
inlet_velocity = sp.symbols("u_x")

with CodeGeneration() as ctx:
    dtype = 'float64'
    pdf_dtype = 'float32'

    stencil = LBStencil(Stencil.D3Q27)
    q = stencil.Q
    dim = stencil.D

    streaming_pattern = 'aa'

    pdfs, pdfs_tmp = fields(f"pdfs({stencil.Q}), pdfs_tmp({stencil.Q}): {pdf_dtype}[3D]", layout='fzyx')
    velocity_field, density_field = fields(f"velocity({dim}), density(1) : {dtype}[{dim}D]", layout='fzyx')
    omega_field = fields(f"omega(1) : {dtype}[{dim}D]", layout='fzyx')

    macroscopic_fields = {'density': density_field, 'velocity': velocity_field}

    lbm_config = LBMConfig(
        method=Method.CUMULANT,
        stencil=stencil,
        relaxation_rate=omega,
        compressible=True,
        galilean_correction=False,
        fourth_order_correction=1e-4,
        field_name='pdfs',
        streaming_pattern=streaming_pattern,
    )

    lbm_opt = LBMOptimisation(cse_global=False, cse_pdfs=False,
                              symbolic_field=pdfs, symbolic_temporary_field=pdfs_tmp)

    collision_rule = create_lb_collision_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)
    collision_rule = insert_constants(collision_rule)
    collision_rule = insert_aliases(collision_rule)
    lb_method = collision_rule.method

    if ctx.gpu:
        target = Target.GPU
        openmp = False
        cpu_vec = None
        vp = [('int64_t', 'cudaBlockSize0'),
              ('int64_t', 'cudaBlockSize1'),
              ('int64_t', 'cudaBlockSize2')]

        sweep_block_size = (TypedSymbol("cudaBlockSize0", np.int64),
                            TypedSymbol("cudaBlockSize1", np.int64),
                            TypedSymbol("cudaBlockSize2", np.int64))
        sweep_params = {'block_size': sweep_block_size}

    else:
        if ctx.optimize_for_localhost:
            cpu_vec = {"instruction_set": None}
        else:
            cpu_vec = None

        openmp = True if ctx.openmp else False

        target = Target.CPU
        sweep_params = {}
        vp = ()

    freeslip = lbm_boundary_generator("FreeSlip", flag_uid="FreeSlip", boundary_object=FreeSlip(stencil), field_data_type=pdf_dtype)
    no_slip_interpolated = lbm_boundary_generator(class_name='NoSlip', flag_uid='NoSlip',
                                                  boundary_object=QuadraticBounceBack(omega), field_data_type=pdf_dtype)
    ubb = lbm_boundary_generator(class_name='UBB', flag_uid='UBB',
                                 boundary_object=UBB((inlet_velocity, 0.0, 0.0), data_type=dtype),
                                 field_data_type=pdf_dtype)

    outflow = lbm_boundary_generator(class_name='Outflow', flag_uid='Outflow',
                                     boundary_object=ExtrapolationOutflow(stencil[4], lb_method),
                                     field_data_type=pdf_dtype)

    generate_lbm_package(ctx, name="FlowAroundSphere", collision_rule=collision_rule,
                         lbm_config=lbm_config, lbm_optimisation=lbm_opt,
                         nonuniform=True, boundaries=[freeslip, no_slip_interpolated, ubb, outflow],
                         macroscopic_fields=macroscopic_fields, gpu_indexing_params=sweep_params,
                         target=target, data_type=dtype, pdfs_data_type=pdf_dtype,
                         cpu_vectorize_info=cpu_vec)

    field_typedefs = {'VelocityField_T': velocity_field,
                      'ScalarField_T': density_field}

    # Info header containing correct template definitions for stencil and field
    generate_info_header(ctx, 'FlowAroundSphereInfoHeader',
                         field_typedefs=field_typedefs)

    infoHeaderParams = {
        'stencil': stencil.name.lower(),
        'streaming_pattern': streaming_pattern,
        'collision_operator': lbm_config.method.name.lower(),
    }

    ctx.write_file("FlowAroundSphereStaticDefines.h", info_header.format(**infoHeaderParams))
