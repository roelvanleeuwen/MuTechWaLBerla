from functools import reduce
from jinja2.filters import do_indent
from pystencils import TypedSymbol

# ---------------------------------- Selection Tree --------------------------------------------------------------------------


class AbstractKernelSelectionNode:
    def __init__(self, branch_true, branch_false):
        self.branch_true = branch_true
        self.branch_false = branch_false

    @property
    def symbols(self):
        return self.branch_true.symbols + self.branch_false.symbols

    @property
    def condition_text(self):
        raise NotImplementedError('condition_text must be provided by subclass.')

    @property
    def all_kernel_calls(self):
        return self.branch_true.all_kernel_calls + self.branch_false.all_kernel_calls

    def get_code(self):
        true_branch_code = self.branch_true.get_code()
        false_branch_code = self.branch_false.get_code()

        true_branch_code = do_indent(true_branch_code, width=4, indentfirst=True)
        false_branch_code = do_indent(false_branch_code, width=4, indentfirst=True)

        code = f"if({self.condition_text}) {{\n"
        code += true_branch_code
        code += "\n} else {\n"
        code += false_branch_code
        code += "\n}"
        return code


class KernelCallNode(AbstractKernelSelectionNode):
    def __init__(self, ast, target='cpu', stream='0'):
        self.target = target
        self.ast = ast
        self.parameters = ast.get_parameters()  # cache parameters here
        self.stream = stream
        super(KernelCallNode, self).__init__(None, None)

    @property
    def condition_text(self):
        raise Exception('There is no condition on a leaf node.')

    @property
    def all_kernel_calls(self):
        return [self]

    def get_code(self):
        ast = self.ast
        ast_params = self.parameters
        is_cpu = self.target == 'cpu'
        call_parameters = ", ".join([p.symbol.name for p in ast_params])

        if not is_cpu:
            return f"internal_{ast.function_name}::{ast.function_name}<<<_grid, _block, 0, {self.stream}>>>({call_parameters});"
        else:
            return f"internal_{ast.function_name}::{ast.function_name}({call_parameters});"


class BooleanParameterSelectionNode(AbstractKernelSelectionNode):
    def __init__(self,
                 parameter_name: str,
                 branch_true: AbstractKernelSelectionNode,
                 branch_false: AbstractKernelSelectionNode):
        self.symbol = TypedSymbol(parameter_name, bool)
        super(BooleanParameterSelectionNode, self).__init__(branch_true, branch_false)

    @property
    def symbols(self):
        return [self.symbol]

    @property
    def condition_text(self):
        return self.symbol.name


# ---------------------------------- Kernel Info --------------------------------------------------------------------------


class KernelInfo:
    def __init__(self, ast, temporary_fields=(), field_swaps=(), varying_parameters=()):
        self.ast = ast
        self.temporary_fields = tuple(temporary_fields)
        self.field_swaps = tuple(field_swaps)
        self.varying_parameters = tuple(varying_parameters)
        self.parameters = ast.get_parameters()  # cache parameters here


class KernelFamily:
    def __init__(self, kernel_selection_tree: AbstractKernelSelectionNode, temporary_fields=(), field_swaps=(), varying_parameters=()):
        self.kernel_selection_tree = kernel_selection_tree
        self.temporary_fields = tuple(temporary_fields)
        self.field_swaps = tuple(field_swaps)
        self.varying_parameters = tuple(varying_parameters)
        all_kernel_calls = self.kernel_selection_tree.all_kernel_calls
        all_param_lists = [k.parameters for k in all_kernel_calls]
        def merger(lx, ly): return merge_sorted_lists(
            lx, ly, sort_key=lambda x: x.symbol.name, identity_check_key=lambda x: x.symbol)
        self.parameters = reduce(merger, all_param_lists)


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
