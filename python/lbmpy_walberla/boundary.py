import pystencils_walberla.boundary
from lbmpy.boundaries.boundaryhandling import create_lattice_boltzmann_boundary_kernel
from lbmpy.advanced_streaming import AccessPdfValues, numeric_offsets, numeric_index
from lbmpy.boundaries import ExtrapolationOutflow, UBB


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

    stencil_info = None
    if isinstance(boundary_object, ExtrapolationOutflow):
        normal_direction = boundary_object.normal_direction
        dim = boundary_object.dim
        pdf_acc = AccessPdfValues(boundary_object.stencil, streaming_pattern=boundary_object.streaming_pattern,
                                  timestep=boundary_object.zeroth_timestep, streaming_dir='out')
        stencil_info = []
        for i, d in enumerate(lb_method.stencil):
            if d == normal_direction:
                direction = d if dim == 3 else d + (0,)
                stencil_info.append((i, direction, ", ".join([str(e) for e in direction])))

        init_list = []
        for key, value in get_init_dict(boundary_object.stencil, normal_direction, pdf_acc).items():
            init_list.append(f"element.{key} = pdfs->get({value});")

        additional_context = {'OutflowBoundary': True,
                              'InitStructOutflowBoundary': "\n".join(init_list)}
    elif isinstance(boundary_object, UBB) and boundary_object.velocity_is_callable:
        additional_context = {'UBBCallback': boundary_object.velocity_is_callable}
    else:
        additional_context = {}

    pystencils_walberla.boundary.generate_boundary(generation_context,
                                                   class_name,
                                                   boundary_object,
                                                   field_name=field_name,
                                                   neighbor_stencil=lb_method.stencil,
                                                   index_shape=[len(lb_method.stencil)],
                                                   stencil_info=stencil_info,
                                                   kernel_creation_function=boundary_creation_function,
                                                   namespace='lbm',
                                                   additonal_context_items=additional_context,
                                                   **create_kernel_params)


def get_init_dict(stencil, normal_direction, pdf_accessor):
    """The Extrapolation Outflow boundary needs additional data. This function provides a list of all values
    which have to be initialised"""
    result = {}
    position = ["it.x()", "it.y()", "it.z()"]
    for j, stencil_dir in enumerate(stencil):
        pos = []
        if all(n == 0 or n == -s for s, n in zip(stencil_dir, normal_direction)):
            offsets = numeric_offsets(pdf_accessor.accs[j])
            for p, o in zip(position, offsets):
                pos.append(p + " + cell_idx_c(" + str(o) + ")")
            pos.append(str(numeric_index(pdf_accessor.accs[j])[0]))
            result[f'pdf_{j}'] = ', '.join(pos)
            result[f'pdf_nd_{j}'] = ', '.join(pos)

    return result
