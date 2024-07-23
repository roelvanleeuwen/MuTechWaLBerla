from functools import reduce

from pystencils import Target, TypedSymbol

from pystencils_walberla.utility import merge_sorted_lists
from pystencils_walberla.compat import backend_printer, SHAPE_DTYPE, KernelFunction, IS_PYSTENCILS_2


# TODO KernelInfo and KernelFamily should have same interface
class KernelInfo:
    def __init__(self, ast: KernelFunction, temporary_fields=(), field_swaps=(), varying_parameters=()):
        self.ast = ast
        self.temporary_fields = tuple(temporary_fields)
        self.field_swaps = tuple(field_swaps)
        self.varying_parameters = tuple(varying_parameters)
        self.parameters = ast.get_parameters()  # cache parameters here

        if ast.target == Target.GPU and IS_PYSTENCILS_2:
            #   TODO
            raise NotImplementedError("Generating GPU kernels is not yet supported with pystencils 2.0")

    @property
    def fields_accessed(self):
        return self.ast.fields_accessed

    def get_ast_attr(self, name, default=None):
        """Returns the value of an attribute of the AST managed by this KernelInfo.
        For compatibility with KernelFamily."""
        try:
            return self.ast.__getattribute__(name)
        except AttributeError:
            return self.ast.metadata.get(name, default)

    def get_headers(self):
        all_headers = [list(self.ast.required_headers)]
        return reduce(merge_sorted_lists, all_headers)

    def generate_kernel_invocation_code(self, **kwargs):
        ast = self.ast
        ast_params = self.parameters
        fnc_name = ast.function_name
        is_cpu = self.ast.target == Target.CPU
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
            sp_printer_c = backend_printer()
            block = tuple(sp_printer_c.doprint(e) for e in indexing_dict['block'])
            grid = tuple(sp_printer_c.doprint(e) for e in indexing_dict['grid'])

            kernel_call_lines = [
                f"dim3 _block(uint32_c({block[0]}), uint32_c({block[1]}), uint32_c({block[2]}));",
                f"dim3 _grid(uint32_c({grid[0]}), uint32_c({grid[1]}), uint32_c({grid[2]}));",
                f"internal_{fnc_name}::{fnc_name}<<<_grid, _block, 0, {stream}>>>({call_parameters});"
            ]

            return "\n".join(kernel_call_lines)
        else:
            return f"internal_{fnc_name}::{fnc_name}({call_parameters});"
