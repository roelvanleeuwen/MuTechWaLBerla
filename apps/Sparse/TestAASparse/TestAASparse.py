from dataclasses import replace
import sympy as sp
import numpy as np
from pystencils import Field, FieldType, TypedSymbol, Target

from pystencils.field import fields

from lbmpy import Stencil, LBStencil, Method, LBMConfig, LBMOptimisation
from lbmpy.boundaries.boundaryconditions import FixedDensity, UBB, NoSlip
from lbmpy.creationfunctions import create_lb_method, create_lb_collision_rule
from lbmpy.sparse import create_macroscopic_value_setter_sparse, create_lb_update_rule_sparse
from lbmpy.advanced_streaming import is_inplace

from pystencils_walberla import CodeGeneration, generate_sweep, generate_info_header
from lbmpy_walberla.sparse import generate_list_class, generate_sparse_sweep, generate_sparse_boundary, generate_sparse_pack_info, generate_alternating_sparse_lbm_sweep, generate_alternating_sparse_boundary, generate_alternating_sparse_pack_info
from lbmpy_walberla import RefinementScaling

with CodeGeneration() as ctx:

    dtype = 'float64' if ctx.double_accuracy else 'float32'

    stencil = LBStencil(Stencil.D3Q19)
    q = stencil.Q
    dim = stencil.D

    omega = sp.symbols("omega")
    inlet_velocity = sp.symbols("u_x")

    pdfs, pdfs_tmp = fields(f"pdf_field({q}), pdf_field_tmp({q}): {dtype}[1D]", layout='fzyx')
    pdfs.field_type = FieldType.CUSTOM
    pdfs_tmp.field_type = FieldType.CUSTOM
    index_list = Field.create_generic("idx", spatial_dimensions=1, index_dimensions=1, dtype=np.uint32)

    omega_field = Field.create_generic("omega_field", spatial_dimensions=1, index_dimensions=1, dtype=dtype)


    lbm_config = LBMConfig(
        method=Method.SRT,
        stencil=stencil,
        relaxation_rate=omega,
        galilean_correction=(q == 27),
        streaming_pattern='aa'
    )

    lbm_opt = LBMOptimisation(
        cse_global=False,
        cse_pdfs=False,
    )

    if not is_inplace(lbm_config.streaming_pattern):
        field_swaps=[(pdfs, pdfs_tmp)]
    else:
        field_swaps=[]

    scaling = RefinementScaling()
    scaling.add_standard_relaxation_rate_scaling(omega)

    if ctx.cuda:
        target = Target.GPU
        vp = [('int32_t', 'cudaBlockSize0'),
              ('int32_t', 'cudaBlockSize1'),
              ('int32_t', 'cudaBlockSize2')]

        sweep_block_size = (TypedSymbol("cudaBlockSize0", np.int32),
                            TypedSymbol("cudaBlockSize1", np.int32),
                            TypedSymbol("cudaBlockSize2", np.int32))

        sweep_params = {'block_size': sweep_block_size}

    else:
        target = Target.CPU
        sweep_params = {}
        vp = ()



    method = create_lb_method(lbm_config=lbm_config)
    collision_rule = create_lb_collision_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

    generated_list_class_name = "ListLBMList"
    stencil_typedefs = {'Stencil_T': stencil}
    list_template = f"using List_T = walberla::lbmpy::{generated_list_class_name};"

    generate_list_class(ctx, generated_list_class_name, index_list, pdfs, stencil, target=target)

    sparse_setter_eqs = create_macroscopic_value_setter_sparse(method, pdfs, 1.0, (0.0, 0.0, 0.0))
    generate_sparse_sweep(ctx, 'MacroSetter', sparse_setter_eqs, stencil=stencil, target=Target.CPU)





    generate_alternating_sparse_lbm_sweep(ctx, 'LBSweep', collision_rule, lbm_config, pdfs, index_list, dst=pdfs_tmp, field_swaps=field_swaps, target=target, inner_outer_split=True, varying_parameters=vp, gpu_indexing_params=sweep_params)

    ubb = UBB((inlet_velocity, 0, 0))
    generate_alternating_sparse_boundary(ctx, 'UBB', ubb, method, streaming_pattern=lbm_config.streaming_pattern, target=target)
    fixed_density = FixedDensity(sp.Symbol("density"))
    generate_alternating_sparse_boundary(ctx, 'Pressure', fixed_density, method, streaming_pattern=lbm_config.streaming_pattern, target=target)
    generate_alternating_sparse_boundary(ctx, 'NoSlip', NoSlip(), method, streaming_pattern=lbm_config.streaming_pattern, target=target)

    generate_alternating_sparse_pack_info(ctx, 'PackInfo', stencil, lbm_config.streaming_pattern, target=target)

    generate_info_header(ctx, 'ListLBMInfoHeader', stencil_typedefs=stencil_typedefs, additional_code=list_template)
