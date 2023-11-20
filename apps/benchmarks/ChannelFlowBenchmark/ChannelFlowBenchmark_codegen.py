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

stencil_fluid = LBStencil(Stencil.D3Q19)

target = ps.Target.CPU
layout = "fzyx"
zero_centered = True

omega_fluid = sp.Symbol("omega_fluid")
force_in_x = sp.Symbol("force_in_x")
rho = sp.Symbol("rho")


pdf_fluid = fields(
    f"lb_fluid_field({stencil_fluid.Q}): [{stencil_fluid.D}D]", layout=layout
)
pdf_fluid_tmp = fields(
    f"lb_fluid_field_tmp({stencil_fluid.Q}): [{stencil_fluid.D}D]", layout=layout
)

velocity_field = fields(
    f"vel_field({stencil_fluid.D}): [{stencil_fluid.D}D]", layout=layout
)
density_field = fields(f"density_field: [{stencil_fluid.D}D]", layout=layout)

force = sp.Matrix([force_in_x, 0, 0])

# Fluid LBM
lbm_config_fluid = LBMConfig(
    stencil=stencil_fluid,
    method=Method.SRT,
    relaxation_rate=omega_fluid,
    compressible=False,
    zero_centered=zero_centered,
    output={"velocity": velocity_field},
    force=force,  # density has to be deleted here probably if I work with the symbol rho
    force_model=ForceModel.GUO,
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

cpu_vec = {'assume_inner_stride_one': True, 'nontemporal': True}

# Code Generation
with CodeGeneration() as ctx:
    # Initializations
    generate_sweep(ctx, "initialize_fluid_field", setter_eqs_fluid, target=target)

    # lattice Boltzmann steps
    generate_sweep(ctx, 'fluid_lb_step', fluid_step,
                   field_swaps=[(pdf_fluid, pdf_fluid_tmp)],
                   target=Target.CPU)

    # Boundary conditions
    #> periodic BCs missing for thermal and fluid part in x-direction (necessary?)
    generate_boundary(ctx, 'BC_fluid_NoSlip', NoSlip(), method_fluid, target=Target.CPU)  # top & bottom wall

    # Communication
    generate_pack_info_from_kernel(ctx, 'PackInfo_hydro', fluid_step, target=Target.CPU)

    stencil_typedefs = {
        "Stencil_fluid_T": stencil_fluid,
    }
    field_typedefs = {
        "PdfField_fluid_T": pdf_fluid,
        "VelocityField_T": velocity_field,
    }
    additional_code = f"""
        const char * StencilNamefluid = "{stencil_fluid.name}";
        """

    generate_info_header(
        ctx,
        "GenDefines",
        stencil_typedefs=stencil_typedefs,
        field_typedefs=field_typedefs,
        additional_code=additional_code,
    )
