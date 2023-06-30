from jinja2 import Environment, PackageLoader, StrictUndefined

import pystencils as ps
from pystencils import Assignment, create_kernel, Field, FieldType
from pystencils.typing import create_type
from pystencils.stencil import inverse_direction
from pystencils.boundaries.createindexlist import numpy_data_type_for_boundary_object
from pystencils.simp import add_subexpressions_for_field_reads

from lbmpy.boundaries.boundaryhandling import create_lattice_boltzmann_boundary_kernel
from lbmpy.advanced_streaming import Timestep, BetweenTimestepsIndexing

from pystencils_walberla import config_from_context
from pystencils_walberla.kernel_selection import (
    KernelFamily, AbstractKernelSelectionNode, KernelCallNode, HighLevelInterfaceSpec)
from lbmpy_walberla.alternating_sweeps import EvenIntegerCondition, OddIntegerCondition, TimestepTrackerMapping
from pystencils_walberla.additional_data_handler import AdditionalDataHandler
from lbmpy_walberla.sparse.jinja_filters import add_sparse_jinja_env
from lbmpy.advanced_streaming import is_inplace, inverse_dir_index

from pystencils import Target, TypedSymbol

import numpy as np
import sympy as sp


def generate_alternating_sparse_boundary(generation_context,
                                         class_name,
                                         boundary_object,
                                         lb_method,
                                         streaming_pattern='pull',
                                         field_name='pdfs',
                                         namespace='lbmpy',
                                         target=Target.CPU,
                                         data_type=None,
                                         cpu_openmp=None,
                                         generate_functor=True,
                                         **create_kernel_params):
    if boundary_object.additional_data:
        raise NotImplemented("Additional data is not supported for sparse boundaries at the moment")

    struct_name = "IndexInfo"
    boundary_object.name = class_name
    stencil = lb_method.stencil
    dim = stencil.D
    index_shape = [len(lb_method.stencil)]
    field_type = FieldType.GENERIC

    config = config_from_context(generation_context, target=target, data_type=data_type, cpu_openmp=cpu_openmp,
                                 **create_kernel_params)
    create_kernel_params = config.__dict__
    del create_kernel_params['target']
    del create_kernel_params['index_fields']

    field_data_type = np.float64 if config.data_type == "float64" else np.float32

    index_struct_dtype = np.dtype([("in", np.int64), ("out", np.int64), ("dir", np.int64)]
                                  + [(i[0], i[1].numpy_dtype) for i in boundary_object.additional_data], align=True)

    field = Field.create_generic(field_name, dim,
                                 field_data_type,
                                 index_dimensions=len(index_shape), layout='fzyx', index_shape=index_shape,
                                 field_type=field_type)

    pdf_field_sparse = ps.fields(f"pdf_field({stencil.Q}): [1D]")
    pdf_field_sparse.field_type = FieldType.CUSTOM
    index_list = ps.fields(f"idx({stencil.Q}): uint32[1D]")
    index_list.field_type = FieldType.CUSTOM


    index_field = Field('indexVector', FieldType.INDEXED, index_struct_dtype, layout=[0],
                        shape=(TypedSymbol("indexVectorSize", create_type(np.int64)), 1), strides=(1, 1))

    def make_even_or_odd_boundary_kernel(prev_timestep):

        indexing = BetweenTimestepsIndexing(field, lb_method.stencil,
                                            prev_timestep, streaming_pattern, np.int64, np.int64)

        f_out, f_in = indexing.proxy_fields
        dir_symbol = indexing.dir_symbol
        inv_dir = indexing.inverse_dir_symbol

        boundary_assignments = boundary_object(f_out, f_in, dir_symbol, inv_dir, lb_method, index_field)
        boundary_assignments = substitute_proxies_sparse(boundary_assignments, pdf_field_sparse,
                                                         f_out, f_in, index_field, index_list, prev_timestep, stencil)

        elements = [Assignment(dir_symbol, index_field[0]('dir'))]
        elements += boundary_assignments.all_assignments

        kernel = create_kernel(elements, index_fields=[index_field], coordinate_names=("in", "out"), target=target)
        if prev_timestep == Timestep.EVEN:
            kernel.function_name = 'even'
        else:
            kernel.function_name = 'odd'

        for node in boundary_object.get_additional_code_nodes(lb_method)[::-1]:
            kernel.body.insert_front(node)

        kernel = KernelCallNode(kernel)
        return kernel

    if is_inplace(streaming_pattern):
        kernel_even = make_even_or_odd_boundary_kernel(Timestep.EVEN)
        kernel_odd = make_even_or_odd_boundary_kernel(Timestep.ODD)
    else:
        kernel_even = make_even_or_odd_boundary_kernel(Timestep.BOTH)
        kernel_odd = kernel_even

    tree = EvenIntegerCondition('timestep', kernel_even, kernel_odd, np.uint8)
    interface_mappings = [TimestepTrackerMapping(tree.parameter_symbol)]

    kernel_family = KernelFamily(tree, class_name)

    interface_spec = HighLevelInterfaceSpec(kernel_family.kernel_selection_parameters, interface_mappings)

    headers = [f'#include "stencil/D{stencil.D}Q{stencil.Q}.h"',
               f'using Stencil_T = walberla::stencil::D{stencil.D}Q{stencil.Q};',
               f'#include "lbm/inplace_streaming/TimestepTracker.h"']

    additional_data_handler = AdditionalDataHandler(stencil=stencil)
    walberla_stencil = additional_data_handler._walberla_stencil

    stencil_arrays = []
    for i, name in enumerate(["x", "y", "z"]):
        offset = [d[i] for d in walberla_stencil]
        stencil_arrays.append(array_pattern("int8_t", f"c{name}", offset))

    inv_dirs = [walberla_stencil.index(inverse_direction(d)) for d in walberla_stencil]
    stencil_arrays.append(array_pattern("uint8_t", f"inv_dir", inv_dirs))

    context = {
        'kernel': kernel_family,
        'class_name': boundary_object.name,
        'generate_functor': generate_functor,
        'interface_spec': interface_spec,
        'StructName': struct_name,
        'StructDeclaration': struct_from_numpy_dtype(struct_name, index_struct_dtype),
        'dim': dim,
        'target': target.name.lower(),
        'namespace': namespace,
        'inner_or_boundary': boundary_object.inner_or_boundary,
        'single_link': boundary_object.single_link,
        'additional_data_handler': additional_data_handler,
        'additional_headers': headers,
        'stencil_arrays': stencil_arrays,
        'Q': stencil.Q
    }

    env = Environment(loader=PackageLoader('lbmpy_walberla'), undefined=StrictUndefined)
    add_sparse_jinja_env(env)

    header = env.get_template('BoundarySparse.tmpl.h').render(**context)
    source = env.get_template('BoundarySparse.tmpl.cpp').render(**context)

    source_extension = "cpp" if target == Target.CPU else "cu"
    generation_context.write_file(f"{class_name}.h", header)
    generation_context.write_file(f"{class_name}.{source_extension}", source)




def generate_sparse_boundary(generation_context,
                             class_name,
                             boundary_object,
                             lb_method,
                             field_name='pdfs',
                             namespace='lbmpy',
                             target=Target.CPU,
                             data_type=None,
                             cpu_openmp=None,
                             generate_functor=True,
                             **create_kernel_params):
    if boundary_object.additional_data:
        raise NotImplemented("Additional data is not supported for sparse boundaries at the moment")

    struct_name = "IndexInfo"
    boundary_object.name = class_name
    stencil = lb_method.stencil
    dim = stencil.D
    index_shape = [len(lb_method.stencil)]
    field_type = FieldType.GENERIC

    prev_timestep = Timestep.BOTH
    streaming_pattern = 'pull'

    config = config_from_context(generation_context, target=target, data_type=data_type, cpu_openmp=cpu_openmp,
                                 **create_kernel_params)
    create_kernel_params = config.__dict__
    del create_kernel_params['target']
    del create_kernel_params['index_fields']

    field_data_type = np.float64 if config.data_type == "float64" else np.float32

    index_struct_dtype = np.dtype([("in", np.int64), ("out", np.int64), ("dir", np.int64)]
                                  + [(i[0], i[1].numpy_dtype) for i in boundary_object.additional_data], align=True)

    field = Field.create_generic(field_name, dim,
                                 field_data_type,
                                 index_dimensions=len(index_shape), layout='fzyx', index_shape=index_shape,
                                 field_type=field_type)

    pdf_field_sparse = ps.fields(f"pdf_field({stencil.Q}): [1D]")
    pdf_field_sparse.field_type = FieldType.CUSTOM
    index_list = ps.fields(f"idx({stencil.Q}): uint32[1D]")
    index_list.field_type = FieldType.CUSTOM

    index_field = Field('indexVector', FieldType.INDEXED, index_struct_dtype, layout=[0],
                        shape=(TypedSymbol("indexVectorSize", create_type(np.int64)), 1), strides=(1, 1))

    indexing = BetweenTimestepsIndexing(field, lb_method.stencil,
                                        prev_timestep, streaming_pattern, np.int64, np.int64)

    f_out, f_in = indexing.proxy_fields
    dir_symbol = indexing.dir_symbol
    inv_dir = indexing.inverse_dir_symbol

    boundary_assignments = boundary_object(f_out, f_in, dir_symbol, inv_dir, lb_method, index_field)

    boundary_assignments = substitute_proxies_sparse(boundary_assignments, pdf_field_sparse,
                                                     f_out, f_in, index_field, index_list, prev_timestep, stencil)

    #   Code Elements inside the loop
    elements = [Assignment(dir_symbol, index_field[0]('dir'))]
    elements += boundary_assignments

    kernel = create_kernel(elements, index_fields=[index_field], coordinate_names=("in", "out"), target=target)

    #   Code Elements ahead of the loop
    index_arrs_node = indexing.create_code_node()
    for node in boundary_object.get_additional_code_nodes(lb_method)[::-1]:
        kernel.body.insert_front(node)
    kernel.body.insert_front(index_arrs_node)

    selection_tree = KernelCallNode(kernel)
    kernel_family = KernelFamily(selection_tree, class_name)

    interface_mappings = set()
    interface_spec = HighLevelInterfaceSpec(kernel_family.kernel_selection_parameters, interface_mappings)

    headers = [f'#include "stencil/D{stencil.D}Q{stencil.Q}.h"',
               f'using Stencil_T = walberla::stencil::D{stencil.D}Q{stencil.Q};']

    additional_data_handler = AdditionalDataHandler(stencil=stencil)
    walberla_stencil = additional_data_handler._walberla_stencil

    stencil_arrays = []
    for i, name in enumerate(["x", "y", "z"]):
        offset = [d[i] for d in walberla_stencil]
        stencil_arrays.append(array_pattern("int8_t", f"c{name}", offset))

    inv_dirs = [walberla_stencil.index(inverse_direction(d)) for d in walberla_stencil]
    stencil_arrays.append(array_pattern("uint8_t", f"inv_dir", inv_dirs))

    context = {
        'kernel': kernel_family,
        'class_name': boundary_object.name,
        'generate_functor': generate_functor,
        'interface_spec': interface_spec,
        'StructName': struct_name,
        'StructDeclaration': struct_from_numpy_dtype(struct_name, index_struct_dtype),
        'dim': dim,
        'target': target.name.lower(),
        'namespace': namespace,
        'inner_or_boundary': boundary_object.inner_or_boundary,
        'single_link': boundary_object.single_link,
        'additional_data_handler': additional_data_handler,
        'additional_headers': headers,
        'stencil_arrays': stencil_arrays,
        'Q': stencil.Q
    }

    env = Environment(loader=PackageLoader('lbmpy_walberla'), undefined=StrictUndefined)
    add_sparse_jinja_env(env)

    header = env.get_template('BoundarySparse.tmpl.h').render(**context)
    source = env.get_template('BoundarySparse.tmpl.cpp').render(**context)

    source_extension = "cpp" if target == Target.CPU else "cu"
    generation_context.write_file(f"{class_name}.h", header)
    generation_context.write_file(f"{class_name}.{source_extension}", source)


def substitute_proxies_sparse(assignments, pdf_field_sparse, f_out, f_in, index_field, index_list, timestep, stencil):
    if isinstance(assignments, ps.Assignment):
        assignments = [assignments]

    if not isinstance(assignments, ps.AssignmentCollection):
        assignments = ps.AssignmentCollection(assignments)

    accessor_subs = dict()
    if timestep == Timestep.BOTH:
        for fa in assignments.atoms(ps.Field.Access):
            if fa.field == f_out:
                accessor_subs[fa] = pdf_field_sparse.absolute_access((index_field[0]('out'),), fa.index)
            elif fa.field == f_in:
                accessor_subs[fa] = pdf_field_sparse.absolute_access((index_field[0]('in'),), ())
            else:
                continue

    elif timestep == Timestep.EVEN:
        for fa in assignments.atoms(ps.Field.Access):
            idx = fa.index[0]
            if isinstance(idx, sp.Indexed) and idx.base == sp.IndexedBase('invdir'):
                idx = idx.indices[0]

            if isinstance(sp.sympify(idx), sp.Integer):
                idx = inverse_dir_index(stencil, idx)
                accessor_subs[fa] = pdf_field_sparse.absolute_access((index_field[0]('out'),), (idx,))

            else:
                if fa.field == f_out:
                    accessor_subs[fa] = pdf_field_sparse.absolute_access((index_field[0]('out'),), (idx,))
                elif fa.field == f_in:
                    accessor_subs[fa] = pdf_field_sparse.absolute_access((index_field[0]('in'),), ())
                else:
                    continue

    else: #Timestep.ODD

        for fa in assignments.atoms(ps.Field.Access):

            idx = fa.index[0]
            if isinstance(idx, sp.Indexed) and idx.base == sp.IndexedBase('invdir'):
                idx = idx.indices[0]

            if isinstance(sp.sympify(idx), sp.Integer):
                pull_index = index_list.absolute_access((index_field[0]('out'),), ( idx,))
                accessor_subs[fa] = pdf_field_sparse.absolute_access((pull_index,), ())
            else:
                if fa.field == f_out:
                    accessor_subs[fa] = pdf_field_sparse.absolute_access((index_field[0]('in'),), ())
                elif fa.field == f_in:
                    accessor_subs[fa] = pdf_field_sparse.absolute_access((index_field[0]('out'),), (idx,))
                else:
                    continue

    result = assignments.new_with_substitutions(accessor_subs)
    return result


def struct_from_numpy_dtype(struct_name, numpy_dtype):
    result = f"struct {struct_name} {{ \n"

    equality_compare = []
    constructor_params = []
    constructor_initializer_list = []
    for name, (sub_type, offset) in numpy_dtype.fields.items():
        pystencils_type = create_type(sub_type)
        result += f"    {pystencils_type} {name};\n"
        if name in ["in", "out", "dir"]:
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


def array_pattern(dtype, name, content):
    return f"const {str(dtype)} {name} [] = {{ {','.join(str(c) for c in content)} }};"
