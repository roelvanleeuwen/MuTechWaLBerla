import pystencils_walberla.boundary
from lbmpy.boundaries.boundaryhandling import create_lattice_boltzmann_boundary_kernel
from lbmpy.advanced_streaming import AccessPdfValues, numeric_offsets, numeric_index
from lbmpy.boundaries import ExtrapolationOutflow, UBB

index_vector_init_template = """
if ( isFlagSet( it.neighbor({offset}), boundaryFlag ))
{{
    {init_element}
    {init_additional_data}
    indexVectorAll.push_back( element );
    if( inner.contains( it.x(), it.y(), it.z() ))
        indexVectorInner.push_back( element );
    else
        indexVectorOuter.push_back( element );
}}
"""


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

    stencil_info = [(i, d, ", ".join([str(e) for e in d])) for i, d in enumerate(lb_method.stencil)]
    inv_dirs = []
    for direction in lb_method.stencil:
        inverse_dir = tuple([-i for i in direction])
        inv_dirs.append(lb_method.stencil.index(inverse_dir))
    if isinstance(boundary_object, ExtrapolationOutflow):
        index_vector_initialisation = generate_index_vector_initialisation(stencil_info, len(lb_method.stencil[0]),
                                                                           boundary_object, "IndexInfo", inv_dirs)
        additional_context = {'index_vector_initialisation': index_vector_initialisation,
                              'outflow_boundary': True}
    elif isinstance(boundary_object, UBB):
        additional_context = {'UBBCallback': True}
    else:
        additional_context = {}

    pystencils_walberla.boundary.generate_boundary(generation_context,
                                                   class_name,
                                                   boundary_object,
                                                   field_name=field_name,
                                                   neighbor_stencil=lb_method.stencil,
                                                   index_shape=[len(lb_method.stencil)],
                                                   kernel_creation_function=boundary_creation_function,
                                                   namespace='lbm',
                                                   additonal_context_items=additional_context,
                                                   **create_kernel_params)


def generate_index_vector_initialisation(stencil_info, dim, boundary_object, struct_name, inverse_directions):
    """Generates code to initialise an index vector for boundary treatment. In case of the Outflow boundary
       the Index vector needs additional data to store PDF values of a previous timestep.
    Args:
        stencil_info:       containing direction index, direction vector and an offset as string
        dim:                number of dimesions for the simulation
        boundary_object:    lbmpy boundary object
        struct_name:        name of the struct which forms the elements of the index vector
        inverse_directions: inverse of the direction vector of the stencil
    """
    code_lines = []
    inner_or_boundary = boundary_object.inner_or_boundary

    normal_direction = None
    pdf_acc = None
    if isinstance(boundary_object, ExtrapolationOutflow):
        normal_direction = boundary_object.normal_direction
        pdf_acc = AccessPdfValues(boundary_object.stencil, streaming_pattern=boundary_object.streaming_pattern,
                                  timestep=boundary_object.zeroth_timestep, streaming_dir='out')

    for dirIdx, dirVec, offset in stencil_info:
        init_list = []
        offset_for_dimension = offset + ", 0" if dim == 3 else offset

        if inner_or_boundary:
            init_element = f"auto element = {struct_name}( it.x(), it.y(), " \
                           + (f"it.z(), " if dim == 3 else "") + f"{dirIdx} );"
        else:
            init_element = f"auto element = {struct_name}( it.x() + cell_idx_c({dirVec[0]}), " \
                           f"it.y() + cell_idx_c({dirVec[1]}), " \
                           + (f"it.z() + cell_idx_c({dirVec[2]}), " if dim == 3 else "") \
                           + f"{inverse_directions[dirIdx]} );"

        if normal_direction and normal_direction == dirVec:
            for key, value in get_init_dict(boundary_object.stencil, normal_direction, pdf_acc).items():
                init_list.append(f"element.{key} = pdfs->get({value});")

            code_lines.append(index_vector_init_template.format(offset=offset_for_dimension,
                                                                init_element=init_element,
                                                                init_additional_data="\n    ".join(init_list)))
        elif normal_direction and normal_direction != dirVec:
            continue

        else:
            code_lines.append(index_vector_init_template.format(offset=offset_for_dimension,
                                                                init_element=init_element,
                                                                init_additional_data="\n    ".join(init_list)))

    return "\n".join(code_lines)


def get_init_dict(stencil, normal_direction, pdf_accessor):
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
