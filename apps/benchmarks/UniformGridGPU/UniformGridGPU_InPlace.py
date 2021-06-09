import sympy as sp
import numpy as np
import pystencils as ps
from lbmpy.creationfunctions import create_lb_collision_rule
from lbmpy.advanced_streaming import Timestep
from pystencils_walberla import CodeGeneration, generate_sweep
from pystencils.data_types import TypedSymbol
from pystencils.fast_approximation import insert_fast_sqrts, insert_fast_divisions
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
from lbmpy.stencils import get_stencil

from lbmpy_walberla import generate_alternating_lbm_sweep, generate_lb_pack_info

omega = sp.symbols("omega")
omega_free = sp.Symbol("omega_free")
compile_time_block_size = False

if compile_time_block_size:
    sweep_block_size = (128, 1, 1)
else:
    sweep_block_size = (TypedSymbol("cudaBlockSize0", np.int32),
                        TypedSymbol("cudaBlockSize1", np.int32),
                        1)

gpu_indexing_params = {'block_size': sweep_block_size}

options_dict = {
    'srt': {
        'method': 'srt',
        'relaxation_rate': omega,
        'compressible': False,
    },
    'trt': {
        'method': 'trt',
        'relaxation_rate': omega,
    },
    'mrt': {
        'method': 'mrt',
        'relaxation_rates': [omega, 1.3, 1.4, omega, 1.2, 1.1],
    },
    'entropic': {
        'method': 'mrt',
        'compressible': True,
        'relaxation_rates': [omega, omega, omega_free, omega_free, omega_free],
        'entropic': True,
    },
    'smagorinsky': {
        'method': 'srt',
        'smagorinsky': True,
        'relaxation_rate': omega,
    }
}

info_header = """
#include "stencil/D3Q{q}.h"\nusing Stencil_T = walberla::stencil::D3Q{q};
const char * infoStencil = "{stencil}";
const char * infoConfigName = "{configName}";
const bool infoCseGlobal = {cse_global};
const bool infoCsePdfs = {cse_pdfs};
"""

with CodeGeneration() as ctx:
    config_tokens = ctx.config.split('_')
    optimize = True
    if config_tokens[-1] == 'noopt':
        optimize = False
        config_tokens = config_tokens[:-1]

    streaming_pattern = config_tokens[-1]
    if streaming_pattern not in ['aa', 'esotwist']:
        raise ValueError(f"{streaming_pattern} is no valid in-place streaming pattern.")

    stencil_str = config_tokens[-2]
    stencil = get_stencil(stencil_str)

    if len(stencil[0]) != 3:
        raise ValueError("This app only works with 3D stencils")

    config_key = '_'.join(config_tokens[:-2])
    options = options_dict.get(config_key, options_dict['srt'])

    q = len(stencil)
    pdfs, velocity_field = ps.fields(f"pdfs({q}), velocity(3) : double[3D]", layout='fzyx')

    common_options = {
        'stencil': stencil,
        'field_name': pdfs.name,
        'output': {
            'velocity': velocity_field
        },
        'optimization': {
            'target': 'gpu',
            'cse_global': True,
            'cse_pdfs': False,
            'symbolic_field': pdfs,
            'field_layout': 'fzyx',
            'gpu_indexing_params': gpu_indexing_params,
        }
    }

    options.update(common_options)

    vp = [
        ('int32_t', 'cudaBlockSize0'),
        ('int32_t', 'cudaBlockSize1')
    ]

    # LB Sweep
    collision_rule = create_lb_collision_rule(**options)

    if optimize:
        collision_rule = insert_fast_divisions(collision_rule)
        collision_rule = insert_fast_sqrts(collision_rule)

    lb_method = collision_rule.method

    generate_alternating_lbm_sweep(ctx, 'UniformGridGPU_InPlace_LbKernel', collision_rule, streaming_pattern,
                                   optimization=options['optimization'],
                                   inner_outer_split=True,
                                   varying_parameters=vp)

    # getter & setter
    setter_assignments = macroscopic_values_setter(lb_method, density=1.0, velocity=velocity_field.center_vector,
                                                   pdfs=pdfs,
                                                   streaming_pattern=streaming_pattern,
                                                   previous_timestep=Timestep.EVEN)
    generate_sweep(ctx, 'UniformGridGPU_InPlace_MacroSetter', setter_assignments, target='gpu')

    # communication
    generate_lb_pack_info(ctx, 'UniformGridGPU_InPlace_PackInfo', stencil, pdfs,
                          streaming_pattern=streaming_pattern, target='gpu')

    infoHeaderParams = {
        'stencil': stencil_str,
        'q': q,
        'configName': ctx.config,
        'cse_global': int(options['optimization']['cse_global']),
        'cse_pdfs': int(options['optimization']['cse_pdfs']),
    }
    ctx.write_file("UniformGridGPU_InPlace_Defines.h", info_header.format(**infoHeaderParams))
