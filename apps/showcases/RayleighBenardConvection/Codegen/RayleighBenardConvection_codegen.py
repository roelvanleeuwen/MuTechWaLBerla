#!/usr/bin/python3

from pystencils import fields, TypedSymbol
from pystencils.enums import Target
from pystencils.simp import sympy_cse
from pystencils import AssignmentCollection

from lbmpy.boundaries import NoSlip, UBB, FixedDensity
from lbmpy.creationfunctions import (
    create_lb_method,
    create_lb_update_rule,
    LBMConfig,
    LBMOptimisation,
)
from lbmpy.enums import Stencil, Method, ForceModel
from lbmpy.macroscopic_value_kernels import pdf_initialization_assignments
from lbmpy.stencils import LBStencil
from lbmpy.maxwellian_equilibrium import get_weights
import pystencils as ps

from pystencils_walberla import (
    CodeGeneration,
    generate_sweep,
    generate_pack_info_from_kernel,
    generate_info_header,
)
from lbmpy_walberla import generate_boundary#, generate_lb_pack_info

import sympy as sp
import numpy as np

#todo: aktuell nur D2Q9
stencil_thermal = LBStencil(Stencil.D3Q7)
stencil_fluid = LBStencil(Stencil.D3Q19)

target = ps.Target.CPU
layout = "fzyx"
zero_centered = False
force_model = ForceModel.SIMPLE

omega_fluid = sp.Symbol("omega_fluid")
omega_thermal = sp.Symbol("omega_thermal")
gravity_LBM = sp.Symbol("gravity")  #! gravity? gravitational acceleration? ...? what exactly do I use / need here?
Thot = sp.Symbol("Thot")
Tcold = sp.Symbol("Tcold")
rho = sp.Symbol("rho")


pdf_fluid = fields(
    f"lb_fluid_field({stencil_fluid.Q}): [{stencil_fluid.D}D]", layout=layout
)
pdf_fluid_tmp = fields(
    f"lb_fluid_field_tmp({stencil_fluid.Q}): [{stencil_fluid.D}D]", layout=layout
)

#pdf_thermal = fields(
#    f"lb_thermal_field({stencil_thermal.Q}): [{stencil_thermal.D}D]", layout=layout
#)
#pdf_thermal_tmp = fields(
#    f"lb_thermal_field_tmp({stencil_thermal.Q}): [{stencil_thermal.D}D]", layout=layout
#)

velocity_field = fields(
    f"vel_field({stencil_fluid.D}): [{stencil_fluid.D}D]", layout=layout
)
#density_field = fields(f"density_field: [{stencil_fluid.D}D]", layout=layout)
#temperature_field = fields(f"temperature_field: [{stencil_thermal.D}D]", layout=layout)

#force = sp.Matrix([0, gravity_LBM * temperature_field(0) * rho, 0])

#> zhang
pdf_zhang_thermal = fields(
    f"zhang_lb_thermal_field({stencil_thermal.Q}): [{stencil_thermal.D}D]", layout=layout
)
pdf_zhang_thermal_tmp = fields(
    f"zhang_lb_thermal_field_tmp({stencil_thermal.Q}): [{stencil_thermal.D}D]", layout=layout
)
zhang_energy_density_field = fields(f"zhang_energy_density_field: [{stencil_thermal.D}D]", layout=layout)
force = sp.Matrix([0, gravity_LBM * zhang_energy_density_field(0) * rho, 0])
#>

# Fluid LBM
lbm_config_fluid = LBMConfig(
    stencil=stencil_fluid,
    method=Method.SRT,
    relaxation_rate=omega_fluid,
    compressible=False,
    zero_centered=zero_centered,
    output={"velocity": velocity_field},
    force=force,  # density has to be deleted here probably if I work with the symbol rho
    force_model=ForceModel.SIMPLE,
)  # Simple force model (automatisch w_i, c_i, 3)
fluid_step = create_lb_update_rule(
    lbm_config=lbm_config_fluid,
    lbm_optimisation=LBMOptimisation(
        symbolic_field=pdf_fluid, symbolic_temporary_field=pdf_fluid_tmp
    ),
)
method_fluid = fluid_step.method

setter_eqs_fluid = pdf_initialization_assignments(
    method_fluid, 1.0, velocity_field.center_vector, pdf_fluid.center_vector
)

# Thermal LBM
lbm_config_thermal = LBMConfig(
    stencil=stencil_thermal,
    method=Method.SRT,
    relaxation_rate=omega_thermal,
    compressible=True,
    zero_centered=False,
    velocity_input=velocity_field,
    equilibrium_order=1,
    entropic=False,
    output={"density": zhang_energy_density_field},
)
thermal_step = create_lb_update_rule(
    lbm_config=lbm_config_thermal,
    lbm_optimisation=LBMOptimisation(
        symbolic_field=pdf_zhang_thermal, symbolic_temporary_field=pdf_zhang_thermal_tmp
    ),
)
method_thermal = thermal_step.method

#setter_eqs_thermal = pdf_initialization_assignments(
#    method_thermal,
#    temperature_field.center,
#    velocity_field.center_vector,
#    pdf_thermal.center_vector,
#)

#> Zhang Thermal LBM
setter_zhang_eqs_thermal = pdf_initialization_assignments(
    method_thermal,
    zhang_energy_density_field.center,
    velocity_field.center_vector,
    pdf_zhang_thermal.center_vector,
)

#hi = dh.add_array('hi', values_per_cell=len(stencil_thermal))
#dh.fill('hi', 0.0, ghost_layers=True)
#hi_tmp = dh.add_array('hi_tmp', values_per_cell=len(stencil_thermal))
#dh.fill('hi_tmp', 0.0, ghost_layers=True)
cs2 = 1/3
# todo: aktuell einfach zum schnellen testen k so gewählt, dass selbes omega für thermo rauskommt wie für fluid
k = (1/omega_fluid - 1/2) * cs2
#energyDensity = dh.add_array('energyDensity', values_per_cell=1)
#dh.fill('energyDensity', 0.0, ghost_layers=True)
@ps.kernel
def assignmentCollectionTest(s):
    u = sp.symbols(f"u_:{stencil_thermal.D}")
    phi_r = sp.symbols(f"phi_r_:{stencil_thermal.D}")
    phi = sp.symbols(f"phi_:{stencil_thermal.Q}")

    s.tau_h @= k / cs2 + 1/2

    s.rho @= sum(pdf_fluid.center(i) for i in range(stencil_thermal.Q))
    for d in range(stencil_thermal.D):
        u[d] @= sum(stencil_thermal[i][d] * pdf_fluid.center(i) for i in range(stencil_thermal.Q)) / s.rho

    #s.h @= sum(pdf_fluid.center(i) * pdf_zhang_thermal.center(i) for i in range(stencil_thermal.Q)) / s.rho
    s.h @= sum(pdf_fluid[-np.array(stencil_thermal[i])][i] * pdf_zhang_thermal[-np.array(stencil_thermal[i])][i] for i in range(stencil_thermal.Q)) / s.rho
    zhang_energy_density_field[0,0,0] @= s.h
    for d in range(stencil_thermal.D):
        #phi_r[d] @= sum(stencil_thermal[j][d] * pdf_fluid.center(j) * (pdf_zhang_thermal.center(j) - s.h) for j in range(stencil_thermal.Q))
        phi_r[d] @= sum(stencil_thermal[j][d] * pdf_fluid[-np.array(stencil_thermal[j])][j] * (pdf_zhang_thermal[-np.array(stencil_thermal[j])][j] - s.h) for j in range(stencil_thermal.Q))
    for i in range(stencil_thermal.Q):
        phi[i] @= sum(((stencil_thermal[i][d] - u[d]) / (s.rho * cs2) * phi_r[d]) for d in range(stencil_thermal.D))
    for i in range(stencil_thermal.Q):
        pdf_zhang_thermal_tmp[0,0,0][i] @= s.h + (1 - 1 / s.tau_h) * phi[i]


# todo: hardcoded seperation of main assignments and subexpressions
zhang_ac = ps.AssignmentCollection(main_assignments=assignmentCollectionTest[-stencil_thermal.Q:], subexpressions=assignmentCollectionTest[:-stencil_thermal.Q])
strategy = ps.simp.SimplificationStrategy()
strategy.add(ps.simp.apply_to_all_assignments(sp.expand))
strategy.add(ps.simp.sympy_cse)
zhang_ac_cse = strategy(zhang_ac)
zhang_thermal_collision_kernel_function = ps.create_kernel(zhang_ac_cse).compile()
#>

cpu_vec = {'assume_inner_stride_one': True, 'nontemporal': True}

# Code Generation
with CodeGeneration() as ctx:
    # Initializations
    #generate_sweep(ctx, "initialize_thermal_field", setter_eqs_thermal, target=target)
    generate_sweep(ctx, "initialize_zhang_thermal_field", setter_zhang_eqs_thermal, target=target)
    generate_sweep(ctx, "initialize_fluid_field", setter_eqs_fluid, target=target)

    # lattice Boltzmann steps
    #generate_sweep(ctx, 'thermal_lb_step', thermal_step,
    #               field_swaps=[(pdf_thermal, pdf_thermal_tmp)],
    #               target=Target.CPU)
        #> zhang temperature LBM step
    generate_sweep(ctx, 'zhang_thermal_lb_step', zhang_ac_cse,
                   field_swaps=[(pdf_zhang_thermal, pdf_zhang_thermal_tmp)], target=Target.CPU)
    generate_sweep(ctx, 'fluid_lb_step', fluid_step,
                   field_swaps=[(pdf_fluid, pdf_fluid_tmp)],
                   target=Target.CPU)

    # Boundary conditions
    generate_boundary(ctx, 'BC_thermal_Thot', FixedDensity(Thot), method_thermal,
                      target=Target.CPU)  # bottom wall
    generate_boundary(ctx, 'BC_thermal_Tcold', FixedDensity(Tcold), method_thermal,
                      target=Target.CPU)  # top wall
    generate_boundary(ctx, 'BC_fluid_NoSlip', NoSlip(), method_fluid, target=Target.CPU)  # top & bottom wall

    # Communication
    #generate_lb_pack_info(ctx, 'PackInfo_thermal', stencil_thermal, pdf_thermal, streaming_pattern='pull', target=Target.CPU)
    #generate_lb_pack_info(ctx, 'PackInfo_hydro', stencil_fluid, pdf_fluid, streaming_pattern='pull', target=Target.CPU)
    generate_pack_info_from_kernel(ctx, 'PackInfo_thermal', thermal_step, target=Target.CPU)
    generate_pack_info_from_kernel(ctx, 'PackInfo_hydro', fluid_step, target=Target.CPU)

    stencil_typedefs = {
        "Stencil_thermal_T": stencil_thermal,
        "Stencil_fluid_T": stencil_fluid,
    }
    field_typedefs = {
        "PdfField_thermal_T": pdf_zhang_thermal,
        "PdfField_fluid_T": pdf_fluid,
        "VelocityField_T": velocity_field,
        "ZhangEnergyDensityField_T": zhang_energy_density_field,
    }
    additional_code = f"""
        const char * StencilNameThermal = "{stencil_thermal.name}";
        const char * StencilNamefluid = "{stencil_fluid.name}";
        """

    generate_info_header(
        ctx,
        "GenDefines",
        stencil_typedefs=stencil_typedefs,
        field_typedefs=field_typedefs,
        additional_code=additional_code,
    )

print("finished code generation successfully")