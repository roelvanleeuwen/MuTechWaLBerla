from jinja2 import Environment, PackageLoader, StrictUndefined

from pystencils import Target
from pystencils.stencil import inverse_direction
from pystencils.typing import get_base_type

from pystencils_walberla import config_from_context
from lbmpy_walberla.sparse.jinja_filters import add_sparse_jinja_env


__all__ = ['generate_list_class']


def generate_list_class(generation_context, class_name, index_field, pdfs, stencil,
                        layout='fzyx', target=Target.CPU, data_type=None, cpu_openmp=None, cpu_vectorize_info=None,
                        **create_kernel_params):
    """Generates a waLBerla sweep from a pystencils representation.

    The constructor of the C++ sweep class expects all kernel parameters (fields and parameters) in alphabetical order.
    Fields have to passed using BlockDataID's pointing to walberla fields

    Args:
        generation_context: build system context filled with information from waLBerla's CMake. The context for example
                            defines where to write generated files, if OpenMP is available or which SIMD instruction
                            set should be used. See waLBerla examples on how to get a context.
        class_name: name of the generated sweep class
        assignments: list of assignments defining the stencil update rule or a :class:`KernelFunction`
        namespace: the generated class is accessible as walberla::<namespace>::<class_name>
        field_swaps: sequence of field pairs (field, temporary_field). The generated sweep only gets the first field
                     as argument, creating a temporary field internally which is swapped with the first field after
                     each iteration.
        varying_parameters: Depending on the configuration, the generated kernels may receive different arguments for
                            different setups. To not have to adapt the C++ application when then parameter change,
                            the varying_parameters sequence can contain parameter names, which are always expected by
                            the C++ class constructor even if the kernel does not need them.
        inner_outer_split: if True generate a sweep that supports separate iteration over inner and outer regions
                           to allow for communication hiding.
        target: An pystencils Target to define cpu or gpu code generation. See pystencils.Target
        data_type: default datatype for the kernel creation. Default is double
        cpu_openmp: if loops should use openMP or not.
        cpu_vectorize_info: dictionary containing necessary information for the usage of a SIMD instruction set.
        **create_kernel_params: remaining keyword arguments are passed to `pystencils.create_kernel`
    """
    config = config_from_context(generation_context, target=target, data_type=data_type, cpu_openmp=cpu_openmp,
                                 cpu_vectorize_info=cpu_vectorize_info, **create_kernel_params)

    target = config.target

    assert pdfs.index_shape[0] == stencil.Q

    if stencil.D == 2:
        walberla_stencil = ()
        for d in stencil:
            d = d + (0,)
            walberla_stencil = walberla_stencil + (d,)
    else:
        walberla_stencil = stencil

    cx, cy, cz = list(), list(), list()
    for direction in walberla_stencil:
        cx.append(str(direction[0]))
        cy.append(str(direction[1]))
        cz.append(str(direction[2]))

    direction_vectors = {'cx': ", ".join(cx), 'cy': ", ".join(cy), 'cz': ", ".join(cz)}

    stencil_info = [(i, d, ", ".join([str(e) for e in d])) for i, d in enumerate(walberla_stencil)]

    inv_dirs = []
    for direction in walberla_stencil:
        inv_dirs.append(walberla_stencil.index(inverse_direction(direction)))

    inv_dirs_vector = ", ".join([str(inv_dir) for inv_dir in inv_dirs])

    # TODO: implement this
    # if 'instruction_set' in config.cpu_vectorize_info:
    #     alignment = config.cpu_vectorize_info.instruction_set.width
    # else:
    alignment = 0

    env = Environment(loader=PackageLoader('lbmpy_walberla'), undefined=StrictUndefined)
    add_sparse_jinja_env(env)

    jinja_context = {
        'class_name': class_name,
        'stencil': stencil_info,
        'inv_dir': inv_dirs,
        'Q': stencil.Q,
        'direction_vectors': direction_vectors,
        'inv_dirs_vector': inv_dirs_vector,
        'index_type': get_base_type(index_field.dtype).c_name,
        'alignment': alignment,
        'target': target.name.lower(),
    }
    header = env.get_template("List.tmpl.h").render(**jinja_context)
    source = env.get_template("List.tmpl.cpp").render(**jinja_context)

    source_extension = "cpp"
    generation_context.write_file(f"{class_name}.h", header)
    generation_context.write_file(f"{class_name}.{source_extension}", source)
