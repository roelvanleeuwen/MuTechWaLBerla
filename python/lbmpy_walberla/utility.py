from pystencils import CreateKernelConfig, fields
from pystencils.typing import create_type

from lbmpy.advanced_streaming import Timestep
from lbmpy.stencils import LBStencil

from pystencils_walberla.compat import get_default_dtype

def timestep_suffix(timestep: Timestep):
    """ get the suffix as string for a timestep

    :param timestep: instance of class lbmpy.advanced_streaming.Timestep
    :return: either "even", "odd" or an empty string
    """
    return ("_" + str(timestep)) if timestep != Timestep.BOTH else ''


def create_pdf_field(config: CreateKernelConfig, name: str, stencil: LBStencil, field_layout: str = 'fzyx'):
    default_dtype = get_default_dtype(config) 
    data_type = default_dtype.numpy_dtype
    return fields(f'{name}({stencil.Q}) :{data_type}[{stencil.D}D]', layout=field_layout)

