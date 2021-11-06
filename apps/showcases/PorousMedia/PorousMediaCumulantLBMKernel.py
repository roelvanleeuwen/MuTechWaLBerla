import sympy as sp
import pystencils as ps

from lbmpy import Stencil, LBStencil
from lbmpy.creationfunctions import create_lb_collision_rule, create_lb_update_rule

from lbmpy_walberla import RefinementScaling, generate_lattice_model
from pystencils_walberla import CodeGeneration, generate_sweep, generate_pack_info_from_kernel


with CodeGeneration() as ctx:
    data_type = "float64" if ctx.double_accuracy else "float32"
    stencil = LBStencil(Stencil.D3Q19, ordering="walberla")
    omega = sp.Symbol('omega')
    layout = 'fzyx'
    pdf_field = ps.fields(f"pdfs(19): {data_type}[3D]", layout=layout)

    optimizations = {'cse_global': True,
                     'cse_pdfs': False,
                     'split': True,
                     'symbolic_field': pdf_field,
                     'field_layout': layout,
                     'pre_simplification': True
                     }

    params = {'stencil': stencil,
              'method': 'cumulant',
              'relaxation_rate': omega,
              'equilibrium_order': 4,
              'maxwellian_moments': True,
              'compressible': True,
              }

    collision_rule = create_lb_collision_rule(optimization=optimizations, **params)
    update_rule = create_lb_update_rule(collision_rule=collision_rule, optimization=optimizations)

    refinement_scaling = RefinementScaling()
    refinement_scaling.add_standard_relaxation_rate_scaling(omega)

    generate_lattice_model(ctx, "LbCodeGen_LatticeModel", collision_rule, refinement_scaling=refinement_scaling,
                           field_layout=layout)
