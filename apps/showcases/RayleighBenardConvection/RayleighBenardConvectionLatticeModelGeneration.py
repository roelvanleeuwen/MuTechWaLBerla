import sympy as sp

from lbmpy.creationfunctions import LBMConfig, LBMOptimisation, create_lb_collision_rule, create_lb_update_rule
from lbmpy.enums import ForceModel, Method, Stencil
from lbmpy.stencils import LBStencil

from pystencils_walberla import CodeGeneration, generate_pack_info_from_kernel
from lbmpy_walberla import generate_lattice_model

# general parameters
generatedMethod = "SRT"  # "TRT", "SRT"
stencilStr = "D3Q19"
stencil = LBStencil(Stencil.D3Q19 if stencilStr == "D3Q19" else Stencil.D3Q27)
force = sp.symbols('force_:3')
layout = 'fzyx'

if generatedMethod == "Cumulants":
    omega = sp.Symbol('omega')
    # method definition
    lbm_config = LBMConfig(stencil=stencil,
                           method=Method.CUMULANT,
                           relaxation_rate=omega,
                           compressible=True, # Cumulants are always compressible ( at least to my knowledge )
                           force=force,
                           force_model=ForceModel.CUMULANT,
                           zero_centered=False,
                           streaming_pattern='pull',
                           galilean_correction=True if stencil == LBStencil(Stencil.D3Q27) else False)  # free surface implementation only works with pull pattern
elif generatedMethod == "TRT":
    omega_e = sp.Symbol('omega_e')
    omega_o = sp.Symbol('omega_o')
    # method definition
    lbm_config = LBMConfig(stencil=stencil,
                           method=Method.TRT,
                           smagorinsky=False,
                           relaxation_rates=[omega_e, omega_o],
                           compressible=False,
                           force=force,
                           force_model=ForceModel.GUO,
                           zero_centered=False,
                           streaming_pattern='pull')  # free surface implementation only works with pull pattern
elif generatedMethod == "SRT":
    omega = sp.Symbol('omega')
    # method definition
    lbm_config = LBMConfig(stencil=stencil,
                           method=Method.SRT,
                           smagorinsky=False,
                           relaxation_rate=omega,
                           compressible=False,
                           force=force,
                           force_model=ForceModel.GUO,
                           zero_centered=False,
                           streaming_pattern='pull')  # free surface implementation only works with pull pattern

# optimizations to be used by the code generator
lbm_opt = LBMOptimisation(cse_global=True,
                          field_layout=layout)

collision_rule = create_lb_collision_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)
update_rule = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

with CodeGeneration() as ctx:
    generate_lattice_model(ctx, "RBCLatticeModel", collision_rule, field_layout=layout)
    generate_pack_info_from_kernel(ctx, "RBCPackInfo", update_rule)

