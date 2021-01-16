from pystencils.field import fields
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter, macroscopic_values_getter
from lbmpy.stencils import get_stencil
from lbmpy.creationfunctions import create_lb_method, create_lb_update_rule
from lbmpy.boundaries import NoSlip, UBB, ExtrapolationOutflow
from pystencils_walberla import CodeGeneration, generate_sweep, generate_pack_info_from_kernel
from lbmpy_walberla import RefinementScaling, generate_boundary, generate_lattice_model

import sympy as sp


stencil = get_stencil("D3Q27")
q = len(stencil)
dim = len(stencil[0])

pdfs, velocity_field, density_field = fields(f"pdfs({q}), velocity({dim}), density(1) : double[{dim}D]", layout='fzyx')
omega = sp.Symbol("omega")
u_max = sp.Symbol("u_max")

options = {'method': 'cumulant',
           'stencil': stencil,
           'relaxation_rate': omega,
           'galilean_correction': True,
           'field_name': 'pdfs',
           'temporary_field_name': 'pdfs_tmp',
           'optimization': {'symbolic_field': pdfs,
                            'cse_global': False,
                            'cse_pdfs': False}}

method = create_lb_method(**options)

# getter & setter
setter_assignments = macroscopic_values_setter(method, velocity=velocity_field.center_vector,
                                               pdfs=pdfs.center_vector, density=1)
getter_assignments = macroscopic_values_getter(method, velocity=velocity_field.center_vector,
                                               pdfs=pdfs.center_vector, density=density_field)

# opt = {'instruction_set': 'sse', 'assume_aligned': True, 'nontemporal': False, 'assume_inner_stride_one': True}

update_rule = create_lb_update_rule(lb_method=method, **options)

info_header = f"""
using namespace walberla;
#include "stencil/D{dim}Q{q}.h"
using Stencil_T = walberla::stencil::D{dim}Q{q};
using PdfField_T = GhostLayerField<real_t, {q}>;
using VelocityField_T = GhostLayerField<real_t, {dim}>;
using ScalarField_T = GhostLayerField<real_t, 1>;
    """

stencil = method.stencil

shift = {}
for i, stencil_dir in enumerate(stencil):
    if all(n == 0 or n == -s for s, n in zip(stencil_dir, stencil[4])):
        shift[pdfs.center(i)] = pdfs[stencil[4]](i)

with CodeGeneration() as ctx:
    target = 'cpu'

    # sweeps
    generate_sweep(ctx, 'ChannelFlowCodeGen_Sweep', update_rule, field_swaps=[('pdfs', 'pdfs_tmp')], target=target)
    generate_sweep(ctx, 'ChannelFlowCodeGen_MacroSetter', setter_assignments, target=target)
    generate_sweep(ctx, 'ChannelFlowCodeGen_MacroGetter', getter_assignments, target=target)

    # boundaries
    generate_boundary(ctx, 'ChannelFlowCodeGen_UBB', UBB(lambda *args: None, dim=dim), method, target=target)
    generate_boundary(ctx, 'ChannelFlowCodeGen_NoSlip', NoSlip(), method, target=target)
    outflow = ExtrapolationOutflow(stencil[4], method)
    generate_boundary(ctx, 'ChannelFlowCodeGen_Outflow', outflow, method, target=target)

    # communication
    generate_pack_info_from_kernel(ctx, 'ChannelFlowCodeGen_PackInfo', update_rule, target=target)
    generate_pack_info_from_kernel(ctx, 'ChannelFlowCodeGen_PackInfo_Outflow', update_rule,
                                   shifting_dict=shift, target=target)

    # Info header containing correct template definitions for stencil and field
    ctx.write_file("ChannelFlowCodeGen_InfoHeader.h", info_header)
