from lbmpy.advanced_streaming.utility import get_timesteps, Timestep
from pystencils.field import fields
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
from lbmpy.stencils import get_stencil
from lbmpy.creationfunctions import create_lb_collision_rule, create_lb_method, create_lb_update_rule
from lbmpy.boundaries import NoSlip, UBB, ExtrapolationOutflow
from lbmpy_walberla.additional_data_handler import UBBAdditionalDataHandler, OutflowAdditionalDataHandler
from pystencils_walberla import CodeGeneration, generate_sweep
from lbmpy_walberla import RefinementScaling, generate_boundary, generate_lb_pack_info

import sympy as sp

stencil = get_stencil("D3Q27")
q = len(stencil)
dim = len(stencil[0])
streaming_pattern = 'esotwist'
timesteps = get_timesteps(streaming_pattern)

pdfs, velocity_field, density_field = fields(f"pdfs({q}), velocity({dim}), density(1) : double[{dim}D]", layout='fzyx')
omega = sp.Symbol("omega")
u_max = sp.Symbol("u_max")

output = {
    'density': density_field,
    'velocity': velocity_field
}

options = {'method': 'cumulant',
           'stencil': stencil,
           'relaxation_rate': omega,
           'galilean_correction': True,
           'field_name': 'pdfs',
           'streaming_pattern': streaming_pattern,
           'output': output,
           'optimization': {'symbolic_field': pdfs,
                            'cse_global': False,
                            'cse_pdfs': False}}

method = create_lb_method(**options)

# getter & setter
setter_assignments = macroscopic_values_setter(method, velocity=velocity_field.center_vector,
                                               pdfs=pdfs, density=1,
                                               streaming_pattern=streaming_pattern,
                                               previous_timestep=timesteps[0])

# opt = {'instruction_set': 'sse', 'assume_aligned': True, 'nontemporal': False, 'assume_inner_stride_one': True}

collision_rule = create_lb_collision_rule(lb_method=method, **options)
update_rule_even = create_lb_update_rule(collision_rule=collision_rule, timestep=Timestep.EVEN, **options)
update_rule_odd = create_lb_update_rule(collision_rule=collision_rule, timestep=Timestep.ODD, **options)

info_header = f"""
using namespace walberla;
#include "stencil/D{dim}Q{q}.h"
using Stencil_T = walberla::stencil::D{dim}Q{q};
using PdfField_T = GhostLayerField<real_t, {q}>;
using VelocityField_T = GhostLayerField<real_t, {dim}>;
using ScalarField_T = GhostLayerField<real_t, 1>;
    """

stencil = method.stencil

with CodeGeneration() as ctx:
    if ctx.cuda:
        target = 'gpu'
    else:
        target = 'cpu'

    # sweeps
    generate_sweep(ctx, 'FlowAroundSphereCodeGen_EvenSweep', update_rule_even, target=target)
    generate_sweep(ctx, 'FlowAroundSphereCodeGen_OddSweep', update_rule_odd, target=target)
    generate_sweep(ctx, 'FlowAroundSphereCodeGen_MacroSetter', setter_assignments, target=target)

    # boundaries
    ubb = UBB(lambda *args: None, dim=dim)
    ubb_data_handler = UBBAdditionalDataHandler(stencil, ubb)
    outflow = ExtrapolationOutflow(stencil[4], method, streaming_pattern=streaming_pattern)
    outflow_data_handler = OutflowAdditionalDataHandler(stencil, outflow, target=target)

    generate_boundary(ctx, 'FlowAroundSphereCodeGen_UBB', ubb, method,
                      target=target, streaming_pattern=streaming_pattern, always_generate_separate_classes=True,
                      additional_data_handler=ubb_data_handler)

    generate_boundary(ctx, 'FlowAroundSphereCodeGen_NoSlip', NoSlip(), method, target=target,
                      streaming_pattern=streaming_pattern, always_generate_separate_classes=True)

    generate_boundary(ctx, 'FlowAroundSphereCodeGen_Outflow', outflow, method, target=target,
                      streaming_pattern=streaming_pattern, always_generate_separate_classes=True,
                      additional_data_handler=outflow_data_handler)

    # communication
    generate_lb_pack_info(ctx, 'FlowAroundSphereCodeGen_PackInfo', stencil, pdfs,
                          streaming_pattern=streaming_pattern, always_generate_separate_classes=True, target=target)

    # Info header containing correct template definitions for stencil and field
    ctx.write_file("FlowAroundSphereCodeGen_InfoHeader.h", info_header)
