import sympy as sp
import pystencils as ps
from lbmpy.creationfunctions import LBMConfig, LBMOptimisation, create_lb_collision_rule
from lbmpy.enums import ForceModel, Method, Stencil
from lbmpy.stencils import LBStencil

from pystencils_walberla import CodeGeneration
from lbmpy_walberla import generate_lattice_model


with CodeGeneration() as ctx:
    stencil_thermal = LBStencil(Stencil.D3Q7)
    stencil_fluid = LBStencil(Stencil.D3Q19)

    target = ps.Target.CPU
    layout = "fzyx"
    zero_centered = False
    force_model = ForceModel.SIMPLE

    omega_fluid = sp.Symbol("omega_fluid")
    omega_thermal = sp.Symbol("omega_thermal")
    gravity_LBM = sp.Symbol("gravity")  #! gravity? gravitational acceleration? ...? what exactly do I use / need here?
    rho = sp.Symbol("rho")

    # optimizations to be used by the code generator
    lbm_opt = LBMOptimisation(cse_global=True, field_layout=layout)

    velocity_field = ps.fields(
        f"vel_field({stencil_fluid.D}): [{stencil_fluid.D}D]", layout=layout
    )
    temperature_field = ps.fields(f"temperature_field: [{stencil_thermal.D}D]", layout=layout)

    force = sp.Matrix([0, gravity_LBM * temperature_field(0) * rho, 0])

    #------------------------------ Fluid LBM ------------------------------
    # Fluid LBM
    lbm_config_fluid = LBMConfig(
        stencil=stencil_fluid,
        method=Method.SRT,
        relaxation_rate=omega_fluid,
        compressible=False,
        zero_centered=zero_centered,
        output={"velocity": velocity_field},
        force=force,
        force_model=ForceModel.SIMPLE,
    )  # Simple force model (automatisch w_i, c_i, 3)

    #> currently optimizations turned off!
    collision_rule_fluid = create_lb_collision_rule(lbm_config=lbm_config_fluid)#,
                                                    #lbm_optimisation=lbm_opt)

    generate_lattice_model(ctx, "RayleighBenardConvectionLatticeModel_fluid", collision_rule_fluid, field_layout=layout)

    #------------------------------ Thermal LBM ------------------------------
    # Thermal LBM
    lbm_config_thermal = LBMConfig(
        stencil=stencil_thermal,
        method=Method.SRT,
        relaxation_rate=omega_thermal,
        compressible=True,
        zero_centered=False,  #? think about that -> if "defualt" delta_rho not known
        velocity_input=velocity_field,
        equilibrium_order=1,
        entropic=False,
        output={"density": temperature_field},
    )
    #> currently optimizations turned off!
    collision_rule_thermal = create_lb_collision_rule(lbm_config=lbm_config_thermal)#,
                                                      #lbm_optimisation=lbm_opt)

    generate_lattice_model(ctx, "RayleighBenardConvectionLatticeModel_thermal", collision_rule_thermal, field_layout=layout)

print("\t >>> finished code generation successfully <<<")
