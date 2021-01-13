import pystencils_walberla.boundary
from lbmpy.boundaries.boundaryhandling import create_lattice_boltzmann_boundary_kernel
from lbmpy.boundaries import ExtrapolationOutflow, UBB
from lbmpy_walberla.additional_data_handler import UBBAdditionalDataHandler, OutflowAdditionalDataHandler


def generate_boundary(generation_context,
                      class_name,
                      boundary_object,
                      lb_method,
                      field_name='pdfs',
                      **create_kernel_params):
    def boundary_creation_function(field, index_field, stencil, boundary_functor, target='cpu', openmp=True, **kwargs):
        return create_lattice_boltzmann_boundary_kernel(field,
                                                        index_field,
                                                        lb_method,
                                                        boundary_functor,
                                                        target=target,
                                                        **kwargs)

    additonal_data_handler = None
    stencil_info = None
    if isinstance(boundary_object, ExtrapolationOutflow):
        stencil_info = []
        for i, d in enumerate(lb_method.stencil):
            if d == boundary_object.normal_direction:
                direction = d if boundary_object.dim == 3 else d + (0,)
                stencil_info.append((i, direction, ", ".join([str(e) for e in direction])))
        additonal_data_handler = OutflowAdditionalDataHandler(boundary_object, field_name)

    elif isinstance(boundary_object, UBB) and boundary_object.velocity_is_callable:
        additonal_data_handler = UBBAdditionalDataHandler(boundary_object)

    pystencils_walberla.boundary.generate_boundary(generation_context,
                                                   class_name,
                                                   boundary_object,
                                                   field_name=field_name,
                                                   neighbor_stencil=lb_method.stencil,
                                                   index_shape=[len(lb_method.stencil)],
                                                   stencil_info=stencil_info,
                                                   kernel_creation_function=boundary_creation_function,
                                                   namespace='lbm',
                                                   additonal_data_handler=additonal_data_handler,
                                                   **create_kernel_params)
