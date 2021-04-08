from typing import Sequence
from collections import OrderedDict
from functools import reduce
from jinja2.filters import do_indent
from pystencils import TypedSymbol
from pystencils.backends.cbackend import get_headers
from pystencils.backends.cuda_backend import CudaSympyPrinter
from pystencils.kernelparameters import SHAPE_DTYPE

# ---------------------------------- Selection Tree --------------------------------------------------------------------


class AbstractKernelSelectionNode:
    def __init__(self, branch_true, branch_false):
        self.branch_true = branch_true
        self.branch_false = branch_false

    @property
    def selection_parameters(self):
        return self.branch_true.symbols | self.branch_false.symbols

    @property
    def condition_text(self):
        raise NotImplementedError('condition_text must be provided by subclass.')

    @property
    def all_kernel_calls(self):
        return self.branch_true.all_kernel_calls + self.branch_false.all_kernel_calls

    def get_selection_parameter_list(self):
        all_params = self.selection_parameters
        all_names = set(p.name for p in all_params)
        if len(all_names) < len(all_params):
            raise ValueError('There existed selection parameters of same name, but different type.')
        return sorted(all_params)

    def get_code(self, **kwargs):
        true_branch_code = self.branch_true.get_code(**kwargs)
        false_branch_code = self.branch_false.get_code(**kwargs)

        true_branch_code = do_indent(true_branch_code, width=4, indentfirst=True)
        false_branch_code = do_indent(false_branch_code, width=4, indentfirst=True)

        code = f"if({self.condition_text}) {{\n"
        code += true_branch_code
        code += "\n} else {\n"
        code += false_branch_code
        code += "\n}"
        return code


class KernelCallNode(AbstractKernelSelectionNode):
    def __init__(self, ast):
        self.ast = ast
        self.parameters = ast.get_parameters()  # cache parameters here
        super(KernelCallNode, self).__init__(None, None)

    @property
    def selection_parameters(self):
        return set()

    @property
    def condition_text(self):
        raise Exception('There is no condition on a leaf node.')

    @property
    def all_kernel_calls(self):
        return [self]

    def get_code(self, **kwargs):
        ast = self.ast
        ast_params = self.parameters
        is_cpu = self.ast.target == 'cpu'
        call_parameters = ", ".join([p.symbol.name for p in ast_params])

        if not is_cpu:
            stream = kwargs.get('stream', '0')
            spatial_shape_symbols = kwargs.get('spatial_shape_symbols', ())

            if not spatial_shape_symbols:
                spatial_shape_symbols = [p.symbol for p in ast_params if p.is_field_shape]
                spatial_shape_symbols.sort(key=lambda e: e.coordinate)
            else:
                spatial_shape_symbols = [TypedSymbol(s, SHAPE_DTYPE) for s in spatial_shape_symbols]

            assert spatial_shape_symbols, "No shape parameters in kernel function arguments.\n"\
                "Please only use kernels for generic field sizes!"

            indexing_dict = ast.indexing.call_parameters(spatial_shape_symbols)
            sp_printer_c = CudaSympyPrinter()
            kernel_call_lines = [
                "dim3 _block(int(%s), int(%s), int(%s));" % tuple(sp_printer_c.doprint(e)
                                                                  for e in indexing_dict['block']),
                "dim3 _grid(int(%s), int(%s), int(%s));" % tuple(sp_printer_c.doprint(e)
                                                                 for e in indexing_dict['grid']),
                "internal_%s::%s<<<_grid, _block, 0, %s>>>(%s);" % (ast.function_name, ast.function_name,
                                                                    stream, call_parameters),
            ]

            return "\n".join(kernel_call_lines)
        else:
            return f"internal_{ast.function_name}::{ast.function_name}({call_parameters});"


class SimpleBooleanCondition(AbstractKernelSelectionNode):
    def __init__(self,
                 parameter_name: str,
                 branch_true: AbstractKernelSelectionNode,
                 branch_false: AbstractKernelSelectionNode):
        self.parameter_symbol = TypedSymbol(parameter_name, bool)
        super(SimpleBooleanCondition, self).__init__(branch_true, branch_false)

    @property
    def selection_parameters(self):
        return { self.parameter_symbol }

    @property
    def condition_text(self):
        return self.parameter_symbol.name


# ---------------------------------- Kernel Info -----------------------------------------------------------------------


class KernelFamily:
    def __init__(self, kernel_selection_tree: AbstractKernelSelectionNode,
                 class_name: str,
                 temporary_fields=(), field_swaps=(), varying_parameters=(),
                 assumed_inner_stride_one=False):
        self.kernel_selection_tree = kernel_selection_tree
        self.kernel_selection_parameters = kernel_selection_tree.get_selection_parameter_list()
        self.temporary_fields = tuple(temporary_fields)
        self.field_swaps = tuple(field_swaps)
        self.varying_parameters = tuple(varying_parameters)
        self.assumed_inner_stride_one = assumed_inner_stride_one

        all_kernel_calls = self.kernel_selection_tree.all_kernel_calls
        all_param_lists = [k.parameters for k in all_kernel_calls]
        asts_list = [k.ast for k in all_kernel_calls]
        representative_ast = asts_list[0]

        #   Eliminate duplicates
        self.all_asts = set(asts_list)

        #   Check function names for uniqueness and reformat them
        #   using the class name
        function_names = [ast.function_name.lower() for ast in self.all_asts]
        unique_names = set(function_names)
        if len(unique_names) < len(function_names):
            raise ValueError('Function names of kernel family members must be unique!')

        prefix = class_name.lower()
        for ast in self.all_asts:
            ast.function_name = prefix + '_' + ast.function_name

        all_fields = [k.ast.fields_accessed for k in all_kernel_calls]

        #   Collect function parameters and accessed fields
        self.parameters = merge_lists_of_symbols(all_param_lists)
        self.fields_accessed = reduce(lambda x, y: x | y, all_fields)

        #   Collect Ghost Layers and target
        self.ghost_layers = representative_ast.ghost_layers
        self.target = representative_ast.target

        for ast in self.all_asts:
            if ast.ghost_layers != self.ghost_layers:
                raise ValueError(
                    f'Inconsistency in kernel family: Member ghost_layers was different in {ast}!')

            if ast.target != self.target:
                raise ValueError(
                    f'Inconsistency in kernel family: Member target was different in {ast}!')

    def get_headers(self):
        all_headers = [get_headers(ast) for ast in self.all_asts]
        return reduce(merge_sorted_lists, all_headers)

    def generate_kernel_invocation_code(self, **kwargs):
        return self.kernel_selection_tree.get_code(**kwargs)


# --------------------------- High-Level Sweep Interface Specification ------------------------------------------------


class AbstractInterfaceArgumentMapping:
    def __init__(self, high_level_args: Sequence[TypedSymbol], low_level_arg: TypedSymbol):
        self.high_level_args = high_level_args
        self.low_level_arg = low_level_arg

    @property
    def mapping_code(self):
        raise NotImplementedError()

    @property
    def headers(self):
        return set()


class IdentityMapping(AbstractInterfaceArgumentMapping):

    def __init__(self, low_level_arg: TypedSymbol):
        self.high_level_args = (low_level_arg,)
        self.low_level_arg = low_level_arg

    @property
    def mapping_code(self):
        return self.low_level_arg.name


def _create_interface_mapping_dict(low_level_args: Sequence[TypedSymbol],
                                   mappings: Sequence[AbstractInterfaceArgumentMapping]):
    mapping_dict = OrderedDict()
    for m in mappings:
        larg = m.low_level_arg
        if larg not in low_level_args:
            raise ValueError(f'Low-level argument {larg} did not exist.')
        if larg.name in mapping_dict:
            raise ValueError(f'More than one mapping was given for low-level argument {larg}')
        mapping_dict[larg.name] = m

    for arg in low_level_args:
        mapping_dict.setdefault(arg.name, IdentityMapping(arg))

    return mapping_dict


class HighLevelInterfaceSpec:
    def __init__(self, low_level_args: Sequence[TypedSymbol],
                 mappings: Sequence[AbstractInterfaceArgumentMapping]):
        self.low_level_args = low_level_args
        mapping_dict = _create_interface_mapping_dict(low_level_args, mappings)
        self.mappings = mapping_dict.values()
        high_level_args = OrderedDict()
        self.mapping_codes = []
        self.headers = set()
        for larg in low_level_args:
            m = mapping_dict[larg.name]
            self.mapping_codes.append(m.mapping_code)
            self.headers |= m.headers
            for harg in m.high_level_args:
                if high_level_args.setdefault(harg.name, harg) != harg:
                    raise ValueError(f'High-Level Argument {harg} was given multiple times with different types.')

        self.high_level_args = list(high_level_args.values())


# ---------------------------------- Helpers --------------------------------------------------------------------------


def merge_sorted_lists(lx, ly, sort_key=lambda x: x, identity_check_key=None):
    if identity_check_key is None:
        identity_check_key = sort_key
    nx = len(lx)
    ny = len(ly)

    def recursive_merge(lx, ly, ix, iy):
        if ix == nx:
            return ly[iy:]
        if iy == ny:
            return lx[ix:]
        x = lx[ix]
        y = ly[iy]
        skx = sort_key(x)
        sky = sort_key(y)
        if skx == sky:
            if identity_check_key(x) == identity_check_key(y):
                return [x] + recursive_merge(lx, ly, ix + 1, iy + 1)
            else:
                raise ValueError(f'Elements <{x}> and <{y}> with equal sort key where not identical!')
        elif skx < sky:
            return [x] + recursive_merge(lx, ly, ix + 1, iy)
        else:
            return [y] + recursive_merge(lx, ly, ix, iy + 1)
    return recursive_merge(lx, ly, 0, 0)


def merge_lists_of_symbols(lists):
    def merger(lx, ly):
        return merge_sorted_lists(lx, ly, sort_key=lambda x: x.symbol.name, identity_check_key=lambda x: x.symbol)
    return reduce(merger, lists)
