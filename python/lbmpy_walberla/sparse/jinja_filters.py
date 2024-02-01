# For backward compatibility with version < 3.0.0
try:
    from jinja2 import pass_context as jinja2_context_decorator
except ImportError:
    from jinja2 import contextfilter as jinja2_context_decorator

from pystencils import Target, Backend
from pystencils.backends.cbackend import generate_c
from pystencils.typing import TypedSymbol, get_base_type
from pystencils.field import FieldType
from pystencils.sympyextensions import prod

temporary_fieldMemberTemplate = """
private: std::set< {type} *, lbm::SwapableCompare< {type} * > > cache_{original_field_name}_;"""

temporary_fieldTemplate = """
{{
    // Getting temporary field {tmp_field_name}
    auto it = cache_{original_field_name}_.find( {original_field_name} );
    if( it != cache_{original_field_name}_.end() )
    {{
        {tmp_field_name} = *it;
    }}
    else
    {{
        {tmp_field_name} = {original_field_name}->cloneUninitialized();
        cache_{original_field_name}_.insert({tmp_field_name});
    }}
}}
"""

temporary_constructor = """
~{class_name}() {{  {contents} }}
"""

delete_loop = """
    for(auto p: cache_{original_field_name}_) {{
        delete p;
    }}
"""

list_fields = {'pdf_field', 'idx', 'pdf_field_tmp', 'cell_index_field', 'omega_field'}


def make_field_type(dtype, is_gpu):
    if is_gpu:
        return f"cuda::GPUField<{dtype}>"
    else:
        return f"lbm::SparseField<{dtype}>"


def get_field_fsize(field):
    """Determines the size of the index coordinate. Since walberla fields only support one index dimension,
    pystencils fields with multiple index dimensions are linearized to a single index dimension.
    """
    assert field.has_fixed_index_shape, \
        f"All Fields have to be created with fixed index coordinate shape using index_shape=(q,) {str(field.name)}"

    if field.index_dimensions == 0:
        return 1
    else:
        return prod(field.index_shape)


def get_field_stride(param, inner_or_outer):
    field = param.fields[0]
    type_str = get_base_type(param.symbol.dtype).c_name
    if inner_or_outer == "inner" and field.name in {'idx', 'cell_index_field'}:
        stride_names = ['fStride()', 'sizeIDXInner()']
    elif inner_or_outer == 'outer' and field.name in {'idx', 'cell_index_field'}:
        stride_names = ['fStride()', 'sizeIDXOuter()']
    else:
        stride_names = ['fStride()', 'xStride()']

    stride_names = [f"{type_str}(list->{e})" for e in stride_names]
    strides = stride_names[:field.spatial_dimensions]
    if field.index_dimensions > 0:
        additional_strides = [1]
        for shape in reversed(field.index_shape[1:]):
            additional_strides.append(additional_strides[-1] * shape)
        assert len(additional_strides) == field.index_dimensions
        f_stride_name = stride_names[-1]
        strides.extend([f"{type_str}({e} * {f_stride_name})" for e in reversed(additional_strides)])
    return strides[param.symbol.coordinate]


def generate_declaration(kernel_info, target=Target.CPU):
    """Generates the declaration of the kernel function"""
    ast = kernel_info.ast
    result = generate_c(ast, signature_only=True, dialect=Backend.CUDA if target == Target.GPU else Backend.C) + ";"
    result = "namespace internal_%s {\n%s\n}" % (ast.function_name, result,)
    return result


def generate_definition(kernel_info, target=Target.CPU):
    """Generates the definition (i.e. implementation) of the kernel function"""
    ast = kernel_info.ast
    result = generate_c(ast, dialect=Backend.CUDA if target == Target.GPU else Backend.C)
    result = "namespace internal_%s {\nstatic %s\n}" % (ast.function_name, result)
    return result


def generate_declarations(kernel_family, target=Target.CPU):
    declarations = []
    for ast in kernel_family.all_asts:
        code = generate_c(ast, signature_only=True, dialect=Backend.CUDA if target == Target.GPU else Backend.C) + ";"
        code = "namespace internal_%s {\n%s\n}\n" % (ast.function_name, code,)
        declarations.append(code)
    return "\n".join(declarations)


def generate_definitions(kernel_family, target=Target.CPU):
    definitions = []
    for ast in kernel_family.all_asts:
        code = generate_c(ast, dialect=Backend.CUDA if target == Target.GPU else Backend.C)
        code = "namespace internal_%s {\nstatic %s\n}\n" % (ast.function_name, code)
        definitions.append(code)
    return "\n".join(definitions)


def field_extraction_code(field, is_temporary, declaration_only=False,
                          no_declaration=False, is_gpu=False, update_member=False):
    """Returns code string for getting a field pointer.

    This can happen in two ways: either the field is extracted from a walberla block, or a temporary field to swap is
    created.

    Args:
        field: the field for which the code should be created
        is_temporary: new_filtered field from block (False) or create a temporary copy of an existing field (True)
        declaration_only: only create declaration instead of the full code
        no_declaration: create the extraction code, and assume that declarations are elsewhere
        is_gpu: if the field is a GhostLayerField or a GpuField
        update_member: specify if function is used inside a constructor; add _ to members
    """
    # Determine size of f coordinate which is a template parameter
    field_name = field.name
    dtype = get_base_type(field.dtype)
    field_type = make_field_type(dtype, is_gpu)

    if not is_temporary:
        dtype = get_base_type(field.dtype)
        field_type = make_field_type(dtype, is_gpu)
        if declaration_only:
            return f"{field_type} * {field_name}_;"
        else:
            prefix = "" if no_declaration else "auto "
            if update_member:
                return f"{prefix}{field_name}_ = block->getData< {field_type} >({field_name}ID);"
            else:
                if "tmp" in field_name:
                    return f"{prefix}{field_name} = list->gettmpPDFbegining();"
                elif "pdf" in field_name:
                    return f"{prefix}{field_name} = list->getPDFbegining();"
                elif "idx" in field_name:
                    return f"{prefix}{field_name} = list->getidxbeginning();"
                elif "cell_index_field" in field_name:
                    return f"{prefix}{field_name} = list->getCellIndexList();"
                else:
                    return f"{prefix}{field_name} = block->getData< {field_type} >({field_name}ID);"
    else:
        assert field_name.endswith('_tmp')
        original_field_name = field_name[:-len('_tmp')]
        if declaration_only:
            return f"{field_type} * {field_name}_;"
        # else:
        #     declaration = f"{field_type} * {field_name};"
        #     tmp_field_str = temporary_fieldTemplate.format(original_field_name=original_field_name,
        #                                                    tmp_field_name=field_name, type=field_type)
        #     return tmp_field_str if no_declaration else declaration + tmp_field_str


@jinja2_context_decorator
def generate_block_data_to_field_extraction(ctx, kernel_info, parameters_to_ignore=(), parameters=None,
                                            declarations_only=False, no_declarations=False, update_member=False):
    """Generates code that extracts all required fields of a kernel from a walberla block storage."""
    if parameters is not None:
        assert parameters_to_ignore == ()
        field_parameters = []
        for param in kernel_info.parameters:
            if param.is_field_pointer and param.field_name in parameters:
                field_parameters.append(param.fields[0])
    else:
        field_parameters = []
        for param in kernel_info.parameters:
            if param.is_field_pointer and param.field_name not in parameters_to_ignore:
                field_parameters.append(param.fields[0])

    normal_fields = {f for f in field_parameters if f.name not in kernel_info.temporary_fields}
    temporary_fields = {f for f in field_parameters if f.name in kernel_info.temporary_fields}

    args = {
        'declaration_only': declarations_only,
        'no_declaration': no_declarations,
        'is_gpu': ctx['target'] == 'gpu',
    }

    result = "\n".join(
        field_extraction_code(field=field, is_temporary=False, update_member=update_member, **args) for field in
        normal_fields) + "\n"
    # result += "\n".join(
    #     field_extraction_code(field=field, is_temporary=True, update_member=update_member, **args) for field in
    #     temporary_fields)
    return result


def generate_refs_for_kernel_parameters(kernel_info, prefix, parameters_to_ignore=(), ignore_fields=False):
    symbols = {p.field_name for p in kernel_info.parameters if p.is_field_pointer and not ignore_fields}
    symbols.update(p.symbol.name for p in kernel_info.parameters if not p.is_field_parameter)
    symbols.difference_update(parameters_to_ignore)
    return "\n".join("auto & %s = %s%s_;" % (s, prefix, s) for s in symbols)


@jinja2_context_decorator
def generate_call(ctx, kernel, stream='0', spatial_shape_symbols=(), inner_or_outer=None):
    """Generates the function call to a pystencils kernel

    Args:
        ctx: code generation context
        kernel: pystencils kernel
        stream: optional name of cuda stream variable
        spatial_shape_symbols: relevant only for gpu kernels - to determine CUDA block and grid sizes the iteration
                               region (i.e. field shape) has to be known. This can normally be inferred by the kernel
                               parameters - however in special cases like boundary conditions a manual specification
                               may be necessary.
        inner_or_outer: if True call for inner region is specified, if False call for outer region is specified
    """
    ast_params = kernel.parameters
    vec_info = ctx.get('cpu_vectorize_info', None)
    instruction_set = kernel.get_ast_attr('instruction_set')
    if vec_info:
        assume_inner_stride_one = vec_info['assume_inner_stride_one']
        assume_aligned = vec_info['assume_aligned']
        nontemporal = vec_info['nontemporal']
    else:
        assume_inner_stride_one = nontemporal = False
        assume_aligned = False

    cpu_openmp = ctx.get('cpu_openmp', False)

    kernel_call_lines = []

    def get_start_coordinates(field_object):
        return [0] * field_object.spatial_dimensions

    def get_end_coordinates(field_object):
        shape_names = ['xSize()'][:field_object.spatial_dimensions]
        return [f"cell_idx_c({field_object.name}->{e})" for e in shape_names]

    for param in ast_params:
        if param.is_field_parameter and FieldType.is_indexed(param.fields[0]):
            continue

        if param.is_field_pointer:
            if ctx['target'] == 'cpu':
                if "tmp" in param.field_name:
                    kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->gettmpPDFbegining();")
                elif "pdf" in param.field_name:
                    kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getPDFbegining();")
                elif "idx" in param.field_name:
                    if inner_or_outer == "inner":
                        kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getidxInnerbeginning();")
                    elif inner_or_outer == 'outer':
                        kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getidxOuterbeginning();")
                    else:
                        kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getidxbeginning();")
                elif "cell_index_field" in param.field_name:
                    if inner_or_outer == "inner":
                        kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getCellIndexListInner();")
                    elif inner_or_outer == 'outer':
                        kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getCellIndexListOuter();")
                    else:
                        kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getCellIndexList();")
                elif "omega" in param.field_name:
                    kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getomegasbegining();")

            elif ctx['target'] == 'gpu':
                if "tmp" in param.field_name:
                    kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getGPUtmpPDFbegining();")
                elif "pdf" in param.field_name:
                    kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getGPUPDFbegining();")
                elif "idx" in param.field_name:
                    if inner_or_outer == "inner":
                        kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getGPUidxInnerbeginning();")
                    elif inner_or_outer == 'outer':
                        kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getGPUidxOuterbeginning();")
                    else:
                        kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getGPUidxbeginning();")
                elif "cell_index_field" in param.field_name:
                    if inner_or_outer == "inner":
                        kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getGPUCellIndexListInner();")
                    elif inner_or_outer == 'outer':
                        kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getGPUCellIndexListOuter();")
                    else:
                        kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getGPUCellIndexList();")
                elif "omega" in param.field_name:
                    kernel_call_lines.append(f"{param.symbol.dtype} {param.symbol.name} = list->getGPUomegasbegining();")
            else:
                raise NotImplementedError(f"Only CPU or GPU is supported as target, not {ctx['target']}")

        elif param.is_field_stride:
            casted_stride = get_field_stride(param, inner_or_outer)
            type_str = get_base_type(param.symbol.dtype).c_name
            kernel_call_lines.append(f"const {type_str} {param.symbol.name} = {casted_stride};")
        elif param.is_field_shape:
            type_str = get_base_type(param.symbol.dtype).c_name
            if inner_or_outer == "inner":
                kernel_call_lines.append(f"const {type_str} {param.symbol.name} = list->numFluidCellsInner();")
            elif inner_or_outer == 'outer':
                kernel_call_lines.append(f"const {type_str} {param.symbol.name} = list->numFluidCellsOuter();")
            else:
                kernel_call_lines.append(f"const {type_str} {param.symbol.name} = list->numFluidCells();")
            kernel_call_lines.append(f"if ( {param.symbol.name} == 0) return;")


    kernel_call_lines.append(kernel.generate_kernel_invocation_code(stream=stream,
                                                                    spatial_shape_symbols=spatial_shape_symbols))

    return "\n".join(kernel_call_lines)


def generate_swaps(kernel_info):
    """Generates code to swap main fields with temporary fields"""
    swaps = ""
    for src, dst in kernel_info.field_swaps:
        swaps += f"list->swapTmpPdfs();"
    return swaps


def generate_constructor_initializer_list(kernel_info, parameters_to_ignore=None):
    if parameters_to_ignore is None:
        parameters_to_ignore = []

    parameters_to_ignore += kernel_info.temporary_fields
    parameters_to_ignore += list_fields

    parameter_initializer_list = ["listID(listID_)"]
    for param in kernel_info.parameters:
        if param.is_field_pointer and param.field_name not in parameters_to_ignore:
            parameter_initializer_list.append(f"{param.field_name}ID({param.field_name}ID_)")
        elif not param.is_field_parameter and param.symbol.name not in parameters_to_ignore:
            parameter_initializer_list.append(f"{param.symbol.name}_({param.symbol.name})")
    return ", ".join(parameter_initializer_list)


def generate_constructor_parameters(kernel_info, parameters_to_ignore=None):
    if parameters_to_ignore is None:
        parameters_to_ignore = []

    varying_parameters = []
    if hasattr(kernel_info, 'varying_parameters'):
        varying_parameters = kernel_info.varying_parameters
    varying_parameter_names = tuple(e[1] for e in varying_parameters)
    parameters_to_ignore += kernel_info.temporary_fields + varying_parameter_names
    parameters_to_ignore += list_fields

    parameter_list = ['BlockDataID listID_']
    for param in kernel_info.parameters:
        if param.is_field_pointer and param.field_name not in parameters_to_ignore:
            parameter_list.append(f"BlockDataID {param.field_name}ID_")
        elif not param.is_field_parameter and param.symbol.name not in parameters_to_ignore:
            parameter_list.append(f"{param.symbol.dtype} {param.symbol.name}")
    varying_parameters = ["%s %s" % e for e in varying_parameters]
    return ", ".join(parameter_list + varying_parameters)


def generate_constructor_call_arguments(kernel_info, parameters_to_ignore=None):
    if parameters_to_ignore is None:
        parameters_to_ignore = []

    varying_parameters = []
    if hasattr(kernel_info, 'varying_parameters'):
        varying_parameters = kernel_info.varying_parameters
    varying_parameter_names = tuple(e[1] for e in varying_parameters)
    parameters_to_ignore += kernel_info.temporary_fields + varying_parameter_names

    parameter_list = []
    for param in kernel_info.parameters:
        if param.is_field_pointer and param.field_name not in parameters_to_ignore:
            parameter_list.append(f"{param.field_name}ID")
        elif not param.is_field_parameter and param.symbol.name not in parameters_to_ignore:
            parameter_list.append(f'{param.symbol.name}_')
    varying_parameters = [f"{e}_" for e in varying_parameter_names]
    return ", ".join(parameter_list + varying_parameters)


@jinja2_context_decorator
def generate_members(ctx, kernel_info, parameters_to_ignore=(), only_fields=False):
    fields = {f.name: f for f in kernel_info.fields_accessed}

    params_to_skip = tuple(parameters_to_ignore) + tuple(kernel_info.temporary_fields) + tuple(list_fields)
    params_to_skip += tuple(e[1] for e in kernel_info.varying_parameters)
    is_gpu = ctx['target'] == 'gpu'

    result = [f"BlockDataID listID;"]
    for param in kernel_info.parameters:
        if only_fields and not param.is_field_parameter:
            continue
        if param.is_field_pointer and param.field_name not in params_to_skip:
            result.append(f"BlockDataID {param.field_name}ID;")
        elif not param.is_field_parameter and param.symbol.name not in params_to_skip:
            result.append(f"{param.symbol.dtype} {param.symbol.name}_;")

    # for field_name in kernel_info.temporary_fields:
    #     f = fields[field_name]
    #     if field_name in parameters_to_ignore:
    #         continue
    #     assert field_name.endswith('_tmp')
    #     original_field_name = field_name[:-len('_tmp')]
    #     f_size = get_field_fsize(f)
    #     field_type = make_field_type(get_base_type(f.dtype), is_gpu)
    #     result.append(temporary_fieldMemberTemplate.format(type=field_type, original_field_name=original_field_name))

    if hasattr(kernel_info, 'varying_parameters'):
        result.extend(["%s %s_;" % e for e in kernel_info.varying_parameters])

    return "\n".join(result)


def generate_destructor(kernel_info, class_name):
    return ""


@jinja2_context_decorator
def nested_class_method_definition_prefix(ctx, nested_class_name):
    outer_class = ctx['class_name']
    if len(nested_class_name) == 0:
        return outer_class
    else:
        return f"{outer_class}::{nested_class_name}"


def generate_list_of_expressions(expressions, prepend=''):
    if len(expressions) == 0:
        return ''
    return prepend + ", ".join(expressions)


def type_identifier_list(nested_arg_list):
    """
    Filters a nested list of strings and TypedSymbols and returns a comma-separated string.
    Strings are passed through as they are, but TypedSymbols are formatted as C-style
    'type identifier' strings, e.g. 'uint32_t ghost_layers'.
    """
    result = []

    def recursive_flatten(arg_list):
        for s in arg_list:
            if isinstance(s, str):
                result.append(s)
            elif isinstance(s, TypedSymbol):
                result.append(f"{s.dtype} {s.name}")
            else:
                recursive_flatten(s)

    recursive_flatten(nested_arg_list)
    return ", ".join(result)


def identifier_list(nested_arg_list):
    """
    Filters a nested list of strings and TypedSymbols and returns a comma-separated string.
    Strings are passed through as they are, but TypedSymbols are replaced by their name.
    """
    result = []

    def recursive_flatten(arg_list):
        for s in arg_list:
            if isinstance(s, str):
                result.append(s)
            elif isinstance(s, TypedSymbol):
                result.append(s.name)
            else:
                recursive_flatten(s)

    recursive_flatten(nested_arg_list)
    return ", ".join(result)


def add_sparse_jinja_env(jinja_env):
    jinja_env.filters['generate_definition'] = generate_definition
    jinja_env.filters['generate_declaration'] = generate_declaration
    jinja_env.filters['generate_definitions'] = generate_definitions
    jinja_env.filters['generate_declarations'] = generate_declarations
    jinja_env.filters['generate_members'] = generate_members
    jinja_env.filters['generate_constructor_parameters'] = generate_constructor_parameters
    jinja_env.filters['generate_constructor_initializer_list'] = generate_constructor_initializer_list
    jinja_env.filters['generate_constructor_call_arguments'] = generate_constructor_call_arguments
    jinja_env.filters['generate_call'] = generate_call
    jinja_env.filters['generate_block_data_to_field_extraction'] = generate_block_data_to_field_extraction
    jinja_env.filters['generate_swaps'] = generate_swaps
    jinja_env.filters['generate_refs_for_kernel_parameters'] = generate_refs_for_kernel_parameters
    jinja_env.filters['generate_destructor'] = generate_destructor
    jinja_env.filters['nested_class_method_definition_prefix'] = nested_class_method_definition_prefix
    jinja_env.filters['type_identifier_list'] = type_identifier_list
    jinja_env.filters['identifier_list'] = identifier_list
    jinja_env.filters['list_of_expressions'] = generate_list_of_expressions
