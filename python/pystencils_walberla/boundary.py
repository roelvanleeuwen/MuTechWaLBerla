from typing import OrderedDict
import numpy as np
from jinja2 import Environment, PackageLoader, StrictUndefined
from lbmpy.advanced_streaming import AccessPdfValues, numeric_offsets, numeric_index
from lbmpy.boundaries import ExtrapolationOutflow
from pystencils import Field, FieldType
from pystencils.boundaries.boundaryhandling import create_boundary_kernel
from pystencils.boundaries.createindexlist import (
    boundary_index_array_coordinate_names, direction_member_name,
    numpy_data_type_for_boundary_object)
from pystencils.data_types import TypedSymbol, create_type
from pystencils_walberla.codegen import KernelInfo, default_create_kernel_parameters
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env

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
                      field_name,
                      neighbor_stencil,
                      index_shape,
                      field_type=FieldType.GENERIC,
                      kernel_creation_function=None,
                      target='cpu',
                      namespace='pystencils',
                      **create_kernel_params):
    struct_name = "IndexInfo"
    boundary_object.name = class_name
    dim = len(neighbor_stencil[0])

    create_kernel_params = default_create_kernel_parameters(generation_context, create_kernel_params)
    create_kernel_params["target"] = target
    del create_kernel_params["cpu_vectorize_info"]

    if not create_kernel_params["data_type"]:
        create_kernel_params["data_type"] = 'double' if generation_context.double_accuracy else 'float32'
    index_struct_dtype = numpy_data_type_for_boundary_object(boundary_object, dim)

    field = Field.create_generic(field_name, dim,
                                 np.float64 if generation_context.double_accuracy else np.float32,
                                 index_dimensions=len(index_shape), layout='fzyx', index_shape=index_shape,
                                 field_type=field_type)

    index_field = Field('indexVector', FieldType.INDEXED, index_struct_dtype, layout=[0],
                        shape=(TypedSymbol("indexVectorSize", create_type(np.int64)), 1), strides=(1, 1))

    if not kernel_creation_function:
        kernel_creation_function = create_boundary_kernel

    kernels = kernel_creation_function(field, index_field, neighbor_stencil, boundary_object, **create_kernel_params)
    if isinstance(kernels, dict):
        sweep_to_kernel_info_dict = OrderedDict()
        dummy_kernel_info = None
        for sweep_class, sweep_kernel in kernels.items():
            sweep_kernel.function_name = "boundary_" + boundary_object.name + '_' + sweep_class
            sweep_kernel.assumed_inner_stride_one = False
            kernel_info = KernelInfo(sweep_kernel)
            sweep_to_kernel_info_dict[sweep_class] = kernel_info
            if dummy_kernel_info is None:
                dummy_kernel_info = kernel_info
            # elif not dummy_kernel_info.has_same_interface(kernel_info):
            #     raise ValueError("Multiple boundary sweeps must have the same kernel interface!")
        multi_sweep = True
    else:
        multi_sweep = False
        kernel = kernels
        kernel.function_name = "boundary_" + boundary_object.name
        kernel.assumed_inner_stride_one = False
        kernel_info = KernelInfo(kernel)
        sweep_to_kernel_info_dict = { '' : kernel_info }
        dummy_kernel_info = kernel_info

    # waLBerla is a 3D framework. Therefore, a zero for the z index has to be added if we work in 2D
    if dim == 2:
        stencil = ()
        for d in neighbor_stencil:
            d = d + (0,)
            stencil = stencil + (d,)
    else:
        stencil = neighbor_stencil

    stencil_info = [(i, d, ", ".join([str(e) for e in d])) for i, d in enumerate(stencil)]
    inv_dirs = []
    for direction in stencil:
        inverse_dir = tuple([-i for i in direction])
        inv_dirs.append(stencil.index(inverse_dir))

    index_vector_initialisation = generate_index_vector_initialisation(stencil_info, dim, boundary_object,
                                                                       struct_name, inv_dirs)

    context = {
        'class_name': boundary_object.name,
        'sweep_classes': sweep_to_kernel_info_dict,
        'multi_sweep': multi_sweep,
        'dummy_kernel_info' : dummy_kernel_info,
        'StructName': struct_name,
        'StructDeclaration': struct_from_numpy_dtype(struct_name, index_struct_dtype),
        'stencil_info': stencil_info,
        'inverse_directions': inv_dirs,
        'dim': dim,
        'target': target,
        'namespace': namespace,
        'index_vector_initialisation': index_vector_initialisation,
        'outflow_boundary': isinstance(boundary_object, ExtrapolationOutflow)
    }

    env = Environment(loader=PackageLoader('pystencils_walberla'), undefined=StrictUndefined)
    add_pystencils_filters_to_jinja_env(env)

    header = env.get_template('Boundary.tmpl.h').render(**context)
    source = env.get_template('Boundary.tmpl.cpp').render(**context)

    source_extension = "cpp" if target == "cpu" else "cu"
    generation_context.write_file("{}.h".format(class_name), header)
    generation_context.write_file("{}.{}".format(class_name, source_extension), source)


def generate_staggered_boundary(generation_context, class_name, boundary_object,
                                dim, neighbor_stencil, index_shape, target='cpu', **kwargs):
    assert dim == len(neighbor_stencil[0])
    generate_boundary(generation_context, class_name, boundary_object, 'field', neighbor_stencil, index_shape,
                      FieldType.STAGGERED, target=target, **kwargs)


def generate_staggered_flux_boundary(generation_context, class_name, boundary_object,
                                     dim, neighbor_stencil, index_shape, target='cpu', **kwargs):
    assert dim == len(neighbor_stencil[0])
    generate_boundary(generation_context, class_name, boundary_object, 'flux', neighbor_stencil, index_shape,
                      FieldType.STAGGERED_FLUX, target=target, **kwargs)


def struct_from_numpy_dtype(struct_name, numpy_dtype):
    result = f"struct {struct_name} {{ \n"

    equality_compare = []
    constructor_params = []
    constructor_initializer_list = []
    for name, (sub_type, offset) in numpy_dtype.fields.items():
        pystencils_type = create_type(sub_type)
        result += f"    {pystencils_type} {name};\n"
        if name in boundary_index_array_coordinate_names or name == direction_member_name:
            constructor_params.append(f"{pystencils_type} {name}_")
            constructor_initializer_list.append(f"{name}({name}_)")
        else:
            constructor_initializer_list.append(f"{name}()")
        if pystencils_type.is_float():
            equality_compare.append(f"floatIsEqual({name}, o.{name})")
        else:
            equality_compare.append(f"{name} == o.{name}")

    result += "    %s(%s) : %s {}\n" % \
              (struct_name, ", ".join(constructor_params), ", ".join(constructor_initializer_list))
    result += "    bool operator==(const %s & o) const {\n        return %s;\n    }\n" % \
              (struct_name, " && ".join(equality_compare))
    result += "};\n"
    return result


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
