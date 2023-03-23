from dataclasses import replace
import sympy as sp
import numpy as np
from pystencils import Field, FieldType, TypedSymbol, Target

from pystencils.field import fields

from lbmpy import Stencil, LBStencil, Method, LBMConfig, LBMOptimisation
from lbmpy.boundaries.boundaryconditions import FixedDensity, UBB
from lbmpy.creationfunctions import create_lb_method, create_lb_collision_rule
from lbmpy.sparse import create_macroscopic_value_setter_sparse, create_lb_update_rule_sparse

from pystencils_walberla import CodeGeneration, generate_sweep, generate_info_header
from lbmpy_walberla.sparse import generate_sparse_sweep, generate_sparse_boundary, generate_list_class, generate_sparse_pack_info

with CodeGeneration() as ctx:
    dtype = 'float64' if ctx.double_accuracy else 'float32'

    stencil = LBStencil(Stencil.D3Q27)
    q = stencil.Q
    dim = stencil.D

    omega = sp.symbols("omega")
    inlet_velocity = sp.symbols("u_x")

    pdfs, pdfs_tmp = fields(f"pdf_field({q}), pdf_field_tmp({q}): {dtype}[1D]", layout='fzyx')
    pdfs.field_type = FieldType.CUSTOM
    pdfs_tmp.field_type = FieldType.CUSTOM

    index_field = Field.create_generic("idx", spatial_dimensions=1, index_dimensions=1, dtype=np.uint32)
    cell_index_field = Field.create_generic("cell_index_field", spatial_dimensions=1, index_dimensions=1,
                                            dtype=np.uint32)

    omega_field = Field.create_generic("omega_field", spatial_dimensions=1, index_dimensions=1, dtype=dtype)


    lbm_config = LBMConfig(
        method=Method.CUMULANT,
        stencil=stencil,
        relaxation_rate=omega_field.center,
        compressible=True,
        galilean_correction=(q == 27),
        streaming_pattern='pull'
    )



    lbm_optimisation = LBMOptimisation(cse_global=False, cse_pdfs=False)

    method = create_lb_method(lbm_config=lbm_config)
    collision_rule = create_lb_collision_rule(lbm_config=lbm_config, lbm_optimisation=lbm_optimisation)

    sparse_setter_eqs = create_macroscopic_value_setter_sparse(method, pdfs, 1, (0.0, 0.0, 0.0))


    sparse_update_rule = create_lb_update_rule_sparse(collision_rule, pdfs, pdfs_tmp,
                                                      index_field, cell_index_field,
                                                      kernel_type='stream_pull_collide')

    generated_list_class_name = "ListLBMList"
    stencil_typedefs = {'Stencil_T': stencil}
    list_template = f"using List_T = walberla::lbmpy::{generated_list_class_name};"

    if ctx.cuda:
        target = Target.GPU
    else:
        target = Target.CPU

    generate_sparse_sweep(ctx, 'Lagoon_MacroSetter', sparse_setter_eqs, stencil=stencil, target=Target.CPU,
                          cpu_vectorize_info={'instruction_set': None})

    generate_sparse_sweep(ctx, 'Lagoon_LbSweep', sparse_update_rule, stencil=stencil,
                          field_swaps=[(pdfs, pdfs_tmp)], inner_outer_split=True,
                          target=target, cpu_vectorize_info={'instruction_set': None})

    ubb = UBB((inlet_velocity, 0, 0))
    fixed_density = FixedDensity(sp.Symbol("density"))
    generate_sparse_boundary(ctx, 'Lagoon_UBB', ubb, method, target=target)
    generate_sparse_boundary(ctx, 'Lagoon_Pressure', fixed_density, method, target=target)

    generate_sparse_pack_info(ctx, 'Lagoon_PackInfo',  stencil, target=target)

    generate_list_class(ctx, generated_list_class_name, index_field, pdfs, stencil, target=target)

    infoHeaderParams = {
        'stencil': lbm_config.stencil.name,
        'collision_setup': lbm_config.method.name,
        'cse_global': int(lbm_optimisation.cse_global),
        'cse_pdfs': int(lbm_optimisation.cse_pdfs),
    }

    generate_info_header(ctx, 'ListLBMInfoHeader', stencil_typedefs=stencil_typedefs,
                         additional_code=list_template)
