from dataclasses import replace
import sympy as sp
import numpy as np
from pystencils import TypedSymbol, Target

from pystencils.field import fields

from lbmpy import Stencil, LBStencil, Method, LBMConfig, LBMOptimisation
from lbmpy.boundaries.boundaryconditions import ExtrapolationOutflow, UBB, NoSlip, FixedDensity
from lbmpy.creationfunctions import create_lb_collision_rule
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
from lbmpy.advanced_streaming.utility import get_timesteps, is_inplace

from pystencils_walberla import CodeGeneration, generate_sweep, generate_info_header
from lbmpy_walberla import RefinementScaling, generate_boundary, generate_lattice_model, generate_lb_pack_info
from lbmpy_walberla import generate_alternating_lbm_sweep, generate_alternating_lbm_boundary
from lbmpy_walberla.additional_data_handler import UBBAdditionalDataHandler, OutflowAdditionalDataHandler

with CodeGeneration() as ctx:
    dtype = 'float64' if ctx.double_accuracy else 'float32'

    streaming_pattern = 'esotwist'
    timesteps = get_timesteps(streaming_pattern)

    stencil = LBStencil(Stencil.D3Q19)
    q = stencil.Q
    dim = stencil.D

    omega = sp.symbols("omega")

    pdfs, pdfs_tmp = fields(f"pdfs({q}), pdfs_tmp({q}): {dtype}[{dim}D]", layout='fzyx')
    velocity_field, density_field = fields(f"velocity({dim}), density(1) : {dtype}[{dim}D]", layout='fzyx')

    output = {
        'density': density_field,
        'velocity': velocity_field
    }

    lbm_config = LBMConfig(
        method=Method.SRT,
        stencil=stencil,
        relaxation_rate=omega,
        galilean_correction=(q == 27),
        field_name='pdfs',
        streaming_pattern=streaming_pattern,
        output=output
    )

    lbm_opt = LBMOptimisation(
        symbolic_field=pdfs,
        cse_global=False, cse_pdfs=False,
    )

    if not is_inplace(streaming_pattern):
        lbm_opt = replace(lbm_opt, symbolic_temporary_field=pdfs_tmp)
        field_swaps = [(pdfs, pdfs_tmp)]
    else:
        field_swaps = []

    collision_rule = create_lb_collision_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

    scaling = RefinementScaling()
    scaling.add_standard_relaxation_rate_scaling(omega)

    lb_method = collision_rule.method
    init_velocity = (0.0,0.0,0.0)
    setter_assignments = macroscopic_values_setter(lb_method, velocity=init_velocity,
                                                   pdfs=pdfs, density=1.0,
                                                   streaming_pattern=streaming_pattern,
                                                   previous_timestep=timesteps[0])

    stencil_typedefs = {'Stencil_T': stencil,
                        'CommunicationStencil_T': stencil}
    field_typedefs = {'PdfField_T': pdfs,
                      'VelocityField_T': velocity_field,
                      'ScalarField_T': density_field}

    if ctx.cuda:
        target = Target.GPU
        openmp = False
        cpu_vec = None
        vp = [('int32_t', 'cudaBlockSize0'),
              ('int32_t', 'cudaBlockSize1'),
              ('int32_t', 'cudaBlockSize2')]

        sweep_block_size = (TypedSymbol("cudaBlockSize0", np.int32),
                            TypedSymbol("cudaBlockSize1", np.int32),
                            TypedSymbol("cudaBlockSize2", np.int32))
        sweep_params = {'block_size': sweep_block_size}

    else:
        if ctx.optimize_for_localhost:
            cpu_vec = {"nontemporal": True, "assume_aligned": True}
        else:
            cpu_vec = None

        openmp = True if ctx.openmp else False

        target = Target.CPU
        sweep_params = {}
        vp = ()

    generate_alternating_lbm_sweep(ctx, 'Lagoon_LbSweep',
                                   collision_rule, lbm_config, target=target, lbm_optimisation=lbm_opt,
                                   inner_outer_split=True,
                                   varying_parameters=vp, gpu_indexing_params=sweep_params, field_swaps=field_swaps,
                                   cpu_openmp=openmp, cpu_vectorize_info=cpu_vec)

    generate_sweep(ctx, 'Lagoon_MacroSetter', setter_assignments, target=Target.CPU, cpu_openmp=openmp)

    outflow_normal = (1, 0, 0)

    outflow = ExtrapolationOutflow(outflow_normal, lb_method, streaming_pattern=streaming_pattern, data_type=dtype)
    outflow_data_handler = OutflowAdditionalDataHandler(stencil, outflow, target=target)

    generate_alternating_lbm_boundary(ctx, 'Lagoon_UBB', UBB((0.05, 0.0, 0.0), data_type=dtype), lb_method,
                                      target=target, streaming_pattern=streaming_pattern, cpu_openmp=openmp)

    generate_alternating_lbm_boundary(ctx, 'Lagoon_NoSlip', NoSlip(), lb_method,
                                      target=target, streaming_pattern=streaming_pattern, cpu_openmp=openmp)

    generate_alternating_lbm_boundary(ctx, 'Lagoon_Outflow', outflow, lb_method,
                                      target=target, streaming_pattern=streaming_pattern,
                                      additional_data_handler=outflow_data_handler, cpu_openmp=openmp)

    # communication setting OpenMP here is usually bad
    generate_lb_pack_info(ctx, 'Lagoon_PackInfo', stencil, pdfs,
                          streaming_pattern=streaming_pattern, always_generate_separate_classes=True, target=target)

    # Info header containing correct template definitions for stencil and field
    generate_info_header(ctx, 'Lagoon_InfoHeader',
                         stencil_typedefs=stencil_typedefs, field_typedefs=field_typedefs)
