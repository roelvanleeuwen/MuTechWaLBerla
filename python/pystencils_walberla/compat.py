from pystencils import __version__ as ps_version

#   Determine if we're running pystencils 1.x or 2.x
version_tokes = ps_version.split(".")

PS_VERSION = int(version_tokes[0])

IS_PYSTENCILS_2 = PS_VERSION == 2

if IS_PYSTENCILS_2:
    #   pystencils 2.x

    from typing import Any
    from enum import Enum, auto

    from pystencils import DEFAULTS, Target, create_type
    from pystencils.types import PsType, PsDereferencableType, PsCustomType
    from pystencils import KernelFunction
    from pystencils.backend.emission import emit_code, CAstPrinter

    def get_base_type(dtype: PsType):
        while isinstance(dtype, PsDereferencableType):
            dtype = dtype.base_type
        return dtype

    BasicType = PsType

    SHAPE_DTYPE = DEFAULTS.index_dtype

    def custom_type(typename: str):
        return PsCustomType(typename)
    
    def get_default_dtype(config):
        return create_type(config.default_dtype) 

    class Backend(Enum):
        C = auto()
        CUDA = auto()

    def generate_c(
        kfunc: KernelFunction,
        signature_only: bool = False,
        dialect: Any = None,
        custom_backend=None,
        with_globals=False,
    ) -> str:

        assert not with_globals
        assert custom_backend is None

        if signature_only:
            return CAstPrinter().print_signature(kfunc)
        else:
            return emit_code(kfunc)

    def backend_printer(**kwargs):
        return CAstPrinter()

    def get_headers(kfunc: KernelFunction) -> set[str]:
        return kfunc.required_headers

    def target_string(target: Target) -> str:
        if target.is_cpu():
            return "cpu"
        elif target.is_gpu():
            return "cpu"
        else:
            raise Exception("Invalid target.")

    def get_supported_instruction_sets():
        return ()

else:
    #   pystencils 1.x

    from pystencils import Target
    from pystencils.typing import get_base_type, BasicType
    from pystencils.typing.typed_sympy import SHAPE_DTYPE
    from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets
    from pystencils.enums import Backend
    from pystencils.backends.cbackend import generate_c, get_headers, CustomSympyPrinter, KernelFunction
    
    def custom_type(typename: str):
        return typename
    
    def get_default_dtype(config):
        return config.data_type.default_factory()

    def backend_printer(**kwargs):
        return CustomSympyPrinter()

    def target_string(target: Target) -> str:
        return target.name.lower()
