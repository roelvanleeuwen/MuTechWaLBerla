import sympy as sp
import pystencils as ps
from lbmpy.creationfunctions import LBMConfig, LBMOptimisation, create_lb_collision_rule
from lbmpy.enums import ForceModel, Method, Stencil
from lbmpy.stencils import LBStencil

from pystencils_walberla import CodeGeneration
from lbmpy_walberla import generate_lattice_model


with CodeGeneration() as ctx:
    # general parameters
    layout = 'fzyx'
    data_type = "float64" if ctx.double_accuracy else "float32"

    stencil_fluid = LBStencil(Stencil.D3Q19)
    omega_fluid = sp.Symbol('omega_fluid')
    force_field = ps.fields(f"force(3): {data_type}[3D]", layout='fzyx')

    velocity_field = ps.fields(
        f"vel_field({stencil_fluid.D}): [{stencil_fluid.D}D]", layout=layout
    )

    # method definition
    lbm_config = LBMConfig(stencil=stencil_fluid,
                           method=Method.SRT,
                           relaxation_rate=omega_fluid,
                           compressible=True,
                           force=force_field,
                           force_model=ForceModel.GUO,
                           zero_centered=False,
                           output={"velocity": velocity_field},
                           streaming_pattern='pull')  # free surface implementation only works with pull pattern

    # optimizations to be used by the code generator
    lbm_opt = LBMOptimisation(cse_global=True,
                              field_layout=layout)

    collision_rule_fluid = create_lb_collision_rule(lbm_config=lbm_config,
                                              lbm_optimisation=lbm_opt)

    generate_lattice_model(ctx, "ViscousLDCLatticeModel", collision_rule_fluid, field_layout=layout)

    # -------------------------- THERMAL ---------------------------------------
    stencil_thermal = LBStencil(Stencil.D3Q7)
    omega_thermal = sp.Symbol('omega_thermal')

    velocity_field = ps.fields(f"vel_field({stencil_fluid.D}): [{stencil_fluid.D}D]", layout=layout)
    density_field = ps.fields(f"density_field: [{stencil_fluid.D}D]", layout=layout)
    temperature_field = ps.fields(f"temperature_field: [{stencil_thermal.D}D]", layout=layout)

    # method definition
    lbm_config = LBMConfig(stencil=stencil_thermal,
                           method=Method.SRT,
                           relaxation_rate=omega_thermal,
                           compressible=True,
                           velocity_input=velocity_field,
                           equilibrium_order=1,
                           output={"density": temperature_field},
                           zero_centered=False,
                           streaming_pattern='pull')  # free surface implementation only works with pull pattern

    # optimizations to be used by the code generator
    lbm_opt = LBMOptimisation(field_layout=layout)

    collision_rule_thermal = create_lb_collision_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

    generate_lattice_model(ctx, "ViscousLDCLatticeModelThermal", collision_rule_thermal, field_layout=layout)
