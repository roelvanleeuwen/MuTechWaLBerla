import sympy as sp
import numpy as np
from dataclasses import replace

from pystencils import Field, FieldType, TypedSymbol, Target
from pystencils.field import fields

from lbmpy import Stencil, LBStencil, Method, LBMConfig, LBMOptimisation
from lbmpy.boundaries.boundaryconditions import FixedDensity, UBB, NoSlip
from lbmpy.creationfunctions import create_lb_method, create_lb_collision_rule
from lbmpy.sparse import create_macroscopic_value_setter_sparse, create_lb_update_rule_sparse
from lbmpy.advanced_streaming import is_inplace, Timestep
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter, macroscopic_values_getter

from pystencils_walberla import CodeGeneration, generate_sweep, generate_info_header
from lbmpy_walberla.sparse import generate_list_class, generate_sparse_sweep, generate_sparse_boundary, generate_sparse_pack_info, generate_alternating_sparse_lbm_sweep, generate_alternating_sparse_boundary, generate_alternating_sparse_pack_info
from lbmpy_walberla import generate_alternating_lbm_sweep, generate_lb_pack_info, generate_alternating_lbm_boundary

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

    lbm_config = LBMConfig(
        method=Method.SRT,
        stencil=stencil,
        relaxation_rate=omega,
        streaming_pattern='pull'
    )

    lbm_opt = LBMOptimisation(
        cse_global=False,
        cse_pdfs=False,
    )

    if not is_inplace(lbm_config.streaming_pattern):
        field_swaps=[(pdfs, pdfs_tmp)]
    else:
        field_swaps=[]

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
    inner_outer_split = True


    ########################################## Sparse kernels ###################################################
    generated_list_class_name = "ListLBMList"
    stencil_typedefs = {'Stencil_T': stencil}
    list_template = f"using List_T = walberla::lbmpy::{generated_list_class_name};"
    generate_list_class(ctx, generated_list_class_name, index_list, pdfs, stencil, target=target)

    sparse_setter_eqs = create_macroscopic_value_setter_sparse(method, pdfs, 1.0, (0.0, 0.0, 0.0))
    generate_sparse_sweep(ctx, 'SparseMacroSetter', sparse_setter_eqs, stencil=stencil, target=Target.CPU)

    generate_alternating_sparse_lbm_sweep(ctx, 'SparseLBSweep', collision_rule, lbm_config, pdfs, index_list, dst=pdfs_tmp, field_swaps=field_swaps, target=target, inner_outer_split=inner_outer_split, varying_parameters=vp, gpu_indexing_params=sweep_params)

    ubb = UBB((inlet_velocity, 0, 0))
    generate_alternating_sparse_boundary(ctx, 'SparseUBB', ubb, method, streaming_pattern=lbm_config.streaming_pattern, target=target)
    fixed_density = FixedDensity(sp.Symbol("density"))
    generate_alternating_sparse_boundary(ctx, 'SparsePressure', fixed_density, method, streaming_pattern=lbm_config.streaming_pattern, target=target)
    generate_alternating_sparse_boundary(ctx, 'SparseNoSlip', NoSlip(), method, streaming_pattern=lbm_config.streaming_pattern, target=target)

    generate_alternating_sparse_pack_info(ctx, 'SparsePackInfo', stencil, lbm_config.streaming_pattern, target=target)

    generate_info_header(ctx, 'SparseLBMInfoHeader', stencil_typedefs=stencil_typedefs, additional_code=list_template)



    ########################################## Dense kernels ###################################################

    pdfs, pdfs_tmp, = fields(f"pdfs({q}), pdfs_tmp({q}): double[3D]",  layout='fzyx')
    velocity_field, density_field = fields(f"velocity({stencil.D}), density(1): double[3D]", layout='fzyx')


    setter_assignments = macroscopic_values_setter(method, density=1.0, velocity=(0.0, 0.0, 0.0), pdfs=pdfs, streaming_pattern=lbm_config.streaming_pattern)
    generate_sweep(ctx, 'DenseMacroSetter', setter_assignments, target=Target.CPU)

    getter_assignments = macroscopic_values_getter(method, density=density_field, velocity=velocity_field.center_vector,
                                                   pdfs=pdfs,
                                                   streaming_pattern=lbm_config.streaming_pattern,
                                                   previous_timestep=Timestep.BOTH)
    generate_sweep(ctx, 'DenseMacroGetter', getter_assignments, target=Target.CPU)



    if not is_inplace(lbm_config.streaming_pattern):
        field_swaps=[(pdfs, pdfs_tmp)]
    else:
        field_swaps=[]


    lbm_opt = replace(lbm_opt, symbolic_field=pdfs, symbolic_temporary_field=pdfs_tmp)

    generate_alternating_lbm_sweep(ctx, 'DenseLBSweep', collision_rule, lbm_config=lbm_config,
                                   lbm_optimisation=lbm_opt, target=target,
                                   varying_parameters=vp, gpu_indexing_params=sweep_params,
                                   inner_outer_split=inner_outer_split, field_swaps=field_swaps)

    generate_alternating_lbm_boundary(ctx, 'DenseUBB', ubb, method, field_name=pdfs.name, streaming_pattern=lbm_config.streaming_pattern, target=target)

    generate_alternating_lbm_boundary(ctx, 'DensePressure', fixed_density, method, field_name=pdfs.name, streaming_pattern=lbm_config.streaming_pattern, target=target)

    generate_alternating_lbm_boundary(ctx, 'DenseNoSlip', NoSlip(), method, field_name=pdfs.name, streaming_pattern=lbm_config.streaming_pattern, target=target)

    generate_lb_pack_info(ctx, 'DensePackInfo', stencil, pdfs, streaming_pattern=lbm_config.streaming_pattern, target=target, always_generate_separate_classes=True)

    field_typedefs = {'PdfField_T': pdfs, 'VelocityField_T': velocity_field, 'ScalarField_T': density_field}
    generate_info_header(ctx, 'DenseLBMInfoHeader', stencil_typedefs=stencil_typedefs, field_typedefs=field_typedefs)
