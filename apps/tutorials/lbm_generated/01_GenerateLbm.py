import sympy as sp

from pystencils import fields

from lbmpy import Stencil, LBStencil, Method, LBMConfig, LBMOptimisation, create_lb_collision_rule
from lbmpy.boundaries import NoSlip, FreeSlip, UBB, ExtrapolationOutflow

from pystencils_walberla import CodeGeneration
from lbmpy_walberla import generate_lbm_package, lbm_boundary_generator

stencil = LBStencil(Stencil.D3Q19)
streaming_pattern = 'pull'
omega = sp.Symbol('omega')

#   Create symbolic fields
pdfs, pdfs_tmp = fields(f"pdfs({stencil.Q}), pdfs_tmp({stencil.Q}): double[3D]", layout='fzyx')
density_field, velocity_field = fields(f"density, velocity(3) : double[3D]", layout='fzyx')
macroscopic_fields = {'density': density_field, 'velocity': velocity_field}


#   Define parameters of the lattice Boltzmann method
lbm_config = LBMConfig(stencil=stencil,
                       streaming_pattern=streaming_pattern,
                       method=Method.CENTRAL_MOMENT,
                       relaxation_rate=omega,
                       compressible=True)

lbm_opt = LBMOptimisation()

#   Create the collision rule in algebraic form
collision_rule = create_lb_collision_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

lb_method = collision_rule.method

#   Set up boundary condition objects

noslip = lbm_boundary_generator(class_name='NoSlip', flag_uid='NoSlip', boundary_object=NoSlip())
freeslip = lbm_boundary_generator(class_name='FreeSlip', flag_uid='FreeSlip', boundary_object=FreeSlip(stencil))

ubb_velocity = sp.symbols(f'boundary_velocity_:{stencil.D}')
ubb = lbm_boundary_generator(class_name='UBB', flag_uid='UBB', boundary_object=UBB(ubb_velocity))

outflow_east_obj = ExtrapolationOutflow(normal_direction=(1, 0, 0), lb_method=lb_method)
outflow = lbm_boundary_generator(class_name=f'Outflow{stencil.name}', flag_uid='Outflow',
                                 boundary_object=outflow_east_obj)

with CodeGeneration() as ctx:
    #   This call generates:
    #     1. The Lattice Storage Specification
    #     2. The LBM Stream/Collide sweep
    #     3. Code for Population field initialization, and density/velocity output
    #     4. Boundary condition implementations
    generate_lbm_package(ctx,
                         name="GeneratedLbm",
                         collision_rule=collision_rule,
                         lbm_config=lbm_config,
                         lbm_optimisation=lbm_opt,
                         nonuniform=True,
                         boundaries=[noslip, freeslip, ubb, outflow],
                         macroscopic_fields=macroscopic_fields)
