import numpy as np
import sympy as sp
from itertools import product

from jinja2 import Environment, PackageLoader, StrictUndefined

from pystencils import Assignment, create_kernel, Field, FieldType, fields
from pystencils.cpu import create_kernel as create_cpu_kernel, add_openmp
from pystencils.stencil import offset_to_direction_string
from pystencils import TypedSymbol
from pystencils.stencil import inverse_direction
from pystencils.bit_masks import flag_cond

from pystencils_walberla.codegen import default_create_kernel_parameters
from pystencils_walberla.kernel_selection import KernelFamily, KernelCallNode, SwitchNode
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env

from lbmpy.stencils import get_stencil
from lbmpy.advanced_streaming import get_accessor, is_inplace, get_timesteps, Timestep
from lbmpy.advanced_streaming.communication import _extend_dir

from lbmpy_walberla.alternating_sweeps import EvenIntegerCondition


def generate_packing_kernels(generation_context, class_name: str, stencil, streaming_pattern,
                             namespace='lbm', nonuniform=False, **create_kernel_params):
    if 'cpu_vectorize_info' in create_kernel_params:
        vec_params = create_kernel_params['cpu_vectorize_info']
        if 'instruction_set' in vec_params and vec_params['instruction_set'] is not None:
            raise NotImplementedError("Vectorisation of packing kernels is not implemented.")

    if isinstance(stencil, str):
        stencil = get_stencil(stencil)

    create_kernel_params = default_create_kernel_parameters(generation_context, create_kernel_params)
    target = create_kernel_params.get('target', 'cpu')
    if target == 'gpu':
        raise NotImplementedError("Packing kernels for GPU are not yet implemented")
    create_kernel_params['cpu_vectorize_info']['instruction_set'] = None

    cg = PackingKernelsCodegen(stencil, streaming_pattern, class_name, create_kernel_params)

    kernels = cg.create_uniform_kernel_families()

    if nonuniform:
        kernels = cg.create_nonuniform_kernel_families(kernels_dict=kernels)

    values_per_cell = len(stencil)
    dimension = len(stencil[0])

    jinja_context = {
        'class_name': class_name,
        'namespace': namespace,
        'kernels': kernels,
        'inplace': is_inplace(streaming_pattern),
        'direction_sizes': cg.get_direction_sizes(),
        'stencil_size': values_per_cell,
        'dimension': dimension,
        'src_field': cg.src_field,
        'dst_field': cg.dst_field
    }

    if nonuniform:
        jinja_context['mask_field'] = cg.mask_field

    template_name = "NonuniformPackingKernels" if nonuniform else "PackingKernels"

    env = Environment(loader=PackageLoader('lbmpy_walberla'), undefined=StrictUndefined)
    add_pystencils_filters_to_jinja_env(env)
    header = env.get_template(f"{template_name}.tmpl.h").render(**jinja_context)
    source = env.get_template(f"{template_name}.tmpl.cpp").render(**jinja_context)

    source_extension = "cpp" if target == "cpu" else "cu"
    generation_context.write_file("{}.h".format(class_name), header)
    generation_context.write_file("{}.{}".format(class_name, source_extension), source)


#   ------------------------------ INTERNAL ----------------------------------------------------------------------------


class PackingKernelsCodegen:

    def __init__(self, stencil, streaming_pattern, class_name, create_kernel_params):
        self.stencil = stencil
        self.dim = len(stencil[0])
        self.values_per_cell = len(stencil)
        self.full_stencil = get_stencil('D3Q27') if self.dim == 3 else get_stencil('D2Q9')
        self.streaming_pattern = streaming_pattern
        self.inplace = is_inplace(streaming_pattern)
        self.class_name = class_name
        self.create_kernel_params = create_kernel_params

        self.src_field, self.dst_field = fields(
            f'pdfs_src({self.values_per_cell}), pdfs_dst({self.values_per_cell}) : [{self.dim}D]',
            dtype=self.create_kernel_params['data_type'])
        self.accessors = [get_accessor(streaming_pattern, t) for t in get_timesteps(streaming_pattern)]
        self.mask_field = fields(f'mask : uint32 [{self.dim}D]')

    def create_uniform_kernel_families(self, kernels_dict=None):
        kernels = dict() if kernels_dict is None else kernels_dict

        kernels['packAll'] = self.get_pack_all_kernel_family()
        kernels['unpackAll'] = self.get_unpack_all_kernel_family()
        kernels['localCopyAll'] = self.get_local_copy_all_kernel_family()

        kernels['packDirection'] = self.get_pack_direction_kernel_family()
        kernels['unpackDirection'] = self.get_unpack_direction_kernel_family()
        kernels['localCopyDirection'] = self.get_local_copy_direction_kernel_family()
        return kernels

    def create_nonuniform_kernel_families(self, kernels_dict=None):
        kernels = dict() if kernels_dict is None else kernels_dict
        kernels['unpackRedistribute'] = self.get_unpack_redistribute_kernel_family()
        kernels['packPartialCoalescence'] = self.get_pack_partial_coalescence_kernel_family()
        kernels['unpackCoalescence'] = self.get_unpack_coalescence_kernel_family()

        return kernels

    # --------------------------- Pack / Unpack / LocalCopy All --------------------------------------------------------

    def get_pack_all_ast(self, timestep):
        buffer = self._buffer(self.values_per_cell)
        src, _ = self._stream_out_accs(timestep)
        assignments = [Assignment(buffer(i), src[i]) for i in range(self.values_per_cell)]
        ast = create_kernel(assignments, **self.create_kernel_params, ghost_layers=0)
        ast.function_name = 'pack_ALL' + self._timestep_suffix(timestep)
        return ast

    def get_pack_all_kernel_family(self):
        if not self.inplace:
            tree = KernelCallNode(self.get_pack_all_ast(Timestep.BOTH))
        else:
            even_call = KernelCallNode(self.get_pack_all_ast(Timestep.EVEN))
            odd_call = KernelCallNode(self.get_pack_all_ast(Timestep.ODD))
            tree = EvenIntegerCondition('timestep', even_call, odd_call, parameter_dtype=np.uint8)
        return KernelFamily(tree, self.class_name)

    def get_unpack_all_ast(self, timestep):
        buffer = self._buffer(self.values_per_cell)
        _, dst = self._stream_out_accs(timestep)
        assignments = [Assignment(dst[i], buffer(i)) for i in range(self.values_per_cell)]
        ast = create_kernel(assignments, **self.create_kernel_params, ghost_layers=0)
        ast.function_name = 'unpack_ALL' + self._timestep_suffix(timestep)
        return ast

    def get_unpack_all_kernel_family(self):
        if not self.inplace:
            tree = KernelCallNode(self.get_unpack_all_ast(Timestep.BOTH))
        else:
            even_call = KernelCallNode(self.get_unpack_all_ast(Timestep.EVEN))
            odd_call = KernelCallNode(self.get_unpack_all_ast(Timestep.ODD))
            tree = EvenIntegerCondition('timestep', even_call, odd_call, parameter_dtype=np.uint8)
        return KernelFamily(tree, self.class_name)

    def get_local_copy_all_ast(self, timestep):
        src, dst = self._stream_out_accs(timestep)
        assignments = [Assignment(dst[i], src[i]) for i in range(self.values_per_cell)]
        ast = create_kernel(assignments, **self.create_kernel_params, ghost_layers=0)
        ast.function_name = 'localCopy_ALL' + self._timestep_suffix(timestep)
        return ast

    def get_local_copy_all_kernel_family(self):
        if not self.inplace:
            tree = KernelCallNode(self.get_local_copy_all_ast(Timestep.BOTH))
        else:
            even_call = KernelCallNode(self.get_local_copy_all_ast(Timestep.EVEN))
            odd_call = KernelCallNode(self.get_local_copy_all_ast(Timestep.ODD))
            tree = EvenIntegerCondition('timestep', even_call, odd_call, parameter_dtype=np.uint8)
        return KernelFamily(tree, self.class_name)

    # --------------------------- Pack / Unpack / LocalCopy Direction --------------------------------------------------

    def get_pack_direction_ast(self, comm_dir, timestep):
        assert not all(d == 0 for d in comm_dir)
        dir_string = offset_to_direction_string(comm_dir)
        streaming_dirs = self.get_streaming_dirs(comm_dir)
        buffer = self._buffer(len(streaming_dirs))
        src, _ = self._stream_out_accs(timestep)
        assignments = []
        dir_indices = sorted(self.stencil.index(d) for d in streaming_dirs)
        if len(dir_indices) == 0:
            return None
        for i, d in enumerate(dir_indices):
            assignments.append(Assignment(buffer(i), src[d]))
        ast = create_kernel(assignments, **self.create_kernel_params, ghost_layers=0)
        ast.function_name = f'pack_{dir_string}' + self._timestep_suffix(timestep)
        return ast

    def get_pack_direction_kernel_family(self):
        return self._construct_directionwise_kernel_family(self.get_pack_direction_ast)

    def get_unpack_direction_ast(self, comm_dir, timestep):
        assert not all(d == 0 for d in comm_dir)
        dir_string = offset_to_direction_string(comm_dir)
        streaming_dirs = self.get_streaming_dirs(inverse_direction(comm_dir))
        buffer = self._buffer(len(streaming_dirs))
        _, dst = self._stream_out_accs(timestep)
        assignments = []
        dir_indices = sorted(self.stencil.index(d) for d in streaming_dirs)
        if len(dir_indices) == 0:
            return None
        for i, d in enumerate(dir_indices):
            assignments.append(Assignment(dst[d], buffer(i)))
        ast = create_kernel(assignments, **self.create_kernel_params, ghost_layers=0)
        ast.function_name = f'unpack_{dir_string}' + self._timestep_suffix(timestep)
        return ast

    def get_unpack_direction_kernel_family(self):
        return self._construct_directionwise_kernel_family(self.get_unpack_direction_ast)

    def get_local_copy_direction_ast(self, comm_dir, timestep):
        assert not all(d == 0 for d in comm_dir)
        dir_string = offset_to_direction_string(comm_dir)
        streaming_dirs = self.get_streaming_dirs(comm_dir)
        src, dst = self._stream_out_accs(timestep)
        assignments = []
        dir_indices = sorted(self.stencil.index(d) for d in streaming_dirs)
        if len(dir_indices) == 0:
            return None
        for d in dir_indices:
            assignments.append(Assignment(dst[d], src[d]))
        ast = create_kernel(assignments, **self.create_kernel_params, ghost_layers=0)
        ast.function_name = f'localCopy_{dir_string}' + self._timestep_suffix(timestep)
        return ast

    def get_local_copy_direction_kernel_family(self):
        return self._construct_directionwise_kernel_family(self.get_local_copy_direction_ast)

    # --------------------------- Pack / Unpack / LocalCopy Coarse to Fine ---------------------------------------------

    def get_unpack_redistribute_ast(self, comm_dir, timestep):
        assert not all(d == 0 for d in comm_dir)
        dir_string = offset_to_direction_string(comm_dir)
        streaming_dirs = self.get_streaming_dirs(inverse_direction(comm_dir))
        dir_indices = sorted(self.stencil.index(d) for d in streaming_dirs)
        buffer = self._buffer(self.values_per_cell)
        _, dst = self._stream_out_accs(timestep)
        orthos = self.orthogonal_principals(comm_dir)
        sub_dirs = self.contained_principals(comm_dir)
        orthogonal_combinations = self.linear_combinations(orthos)
        subdir_combinations = self.linear_combinations_nozero(sub_dirs)
        second_gl_dirs = [o + s for o, s in product(orthogonal_combinations, subdir_combinations)]
        negative_dir_correction = np.array([(1 if d == -1 else 0) for d in comm_dir])
        assignments = []
        for offset in orthogonal_combinations:
            o = offset + negative_dir_correction
            for d in range(self.values_per_cell):
                field_acc = dst[d].get_shifted(*o)
                assignments.append(Assignment(field_acc, buffer(d)))

        for offset in second_gl_dirs:
            o = offset + negative_dir_correction
            for d in dir_indices:
                field_acc = dst[d].get_shifted(*o)
                assignments.append(Assignment(field_acc, buffer(d)))

        data_type = self.create_kernel_params['data_type']
        function_name = f'unpackRedistribute_{dir_string}' + self._timestep_suffix(timestep)
        iteration_slice = tuple(slice(None, None, 2) for k in range(self.dim))
        ast = create_cpu_kernel(assignments, function_name=function_name, iteration_slice=iteration_slice,
                                type_info=data_type, ghost_layers=0, allow_double_writes=True)
        if self.create_kernel_params['cpu_openmp']:
            add_openmp(ast, num_threads=self.create_kernel_params['cpu_openmp'])
        return ast

    def get_unpack_redistribute_kernel_family(self):
        return self._construct_directionwise_kernel_family(self.get_unpack_redistribute_ast)

    def get_local_copy_redistribute_ast(self, comm_dir, timestep):
        #   TODO
        raise NotImplementedError()

    def get_local_copy_redistribute_kernel_family(self):
        #   TODO
        raise NotImplementedError()

    # --------------------------- Pack / Unpack / LocalCopy Fine to Coarse ---------------------------------------------

    def get_pack_partial_coalescence_ast(self, comm_dir, timestep):
        assert not all(d == 0 for d in comm_dir)
        dir_string = offset_to_direction_string(comm_dir)
        streaming_dirs = self.get_streaming_dirs(comm_dir)
        dir_indices = sorted(self.stencil.index(d) for d in streaming_dirs)
        buffer = self._buffer(self.values_per_cell)
        src, _ = self._stream_in_accs(timestep.next())
        mask = self.mask_field

        offsets = list(product(*((0,1) for _ in range(self.dim))))
        assignments = []
        for i, d in enumerate(dir_indices):
            acc = 0
            for o in offsets:
                acc += flag_cond(d, mask[o], src[d].get_shifted(*o))
            assignments.append(Assignment(buffer(i), acc))

        iteration_slice = tuple(slice(None, None, 2) for k in range(self.dim))
        ast = create_kernel(assignments, ghost_layers=0, iteration_slice=iteration_slice, **self.create_kernel_params)
        ast.function_name = f'packPartialCoalescence_{dir_string}' + self._timestep_suffix(timestep)
        return ast

    def get_pack_partial_coalescence_kernel_family(self):
        return self._construct_directionwise_kernel_family(self.get_pack_partial_coalescence_ast)

    def get_unpack_coalescence_ast(self, comm_dir, timestep):
        assert not all(d == 0 for d in comm_dir)
        dir_string = offset_to_direction_string(comm_dir)
        streaming_dirs = self.get_streaming_dirs(inverse_direction(comm_dir))
        dir_indices = sorted(self.stencil.index(d) for d in streaming_dirs)
        buffer = self._buffer(self.values_per_cell)
        _, dst = self._stream_in_accs(timestep.next())

        coalescence_factor = sp.Rational(1, self.dim)

        assignments = []
        for i, d in enumerate(dir_indices):
            assignments.append(Assignment(dst[d], dst[d] + coalescence_factor * buffer(i)))

        ast = create_kernel(assignments, **self.create_kernel_params, ghost_layers=0)
        ast.function_name = f'unpackCoalescence_{dir_string}' + self._timestep_suffix(timestep)
        return ast

    def get_unpack_coalescence_kernel_family(self):
        return self._construct_directionwise_kernel_family(self.get_unpack_coalescence_ast)

    #   TODO
    def get_local_copy_partial_coalescence_ast(self, comm_dir, timestep):
        raise NotImplementedError()

    def get_local_copy_partial_coalescence_kernel_family(self):
        raise NotImplementedError()

    # ------------------------------------------ Utility ---------------------------------------------------------------

    def get_streaming_dirs(self, comm_dir):
        if all(d == 0 for d in comm_dir):
            return set()
        else:
            return set(_extend_dir(comm_dir)) & set(self.stencil)

    def get_direction_sizes(self):
        return [len(self.get_streaming_dirs(d)) for d in self.full_stencil]

    def principal(self, i):
        e_i = np.zeros(self.dim, dtype=int)
        e_i[i] = 1
        return e_i

    def principals(self):
        """Returns the principal directions for the given dimension"""
        return tuple(self.principal(i) for i in range(self.dim))

    def orthogonal_principals(self, comm_dir):
        """Returns the positive principal directions orthogonal to the comm_dir"""
        return tuple(p for i, p in enumerate(self.principals()) if comm_dir[i] == 0)

    def contained_principals(self, comm_dir):
        """Returns the (positive or negative) principal directions contained in comm_dir"""
        vecs = []
        for i, d in enumerate(comm_dir):
            if d != 0:
                vecs.append(d * self.principal(i))
        return vecs

    def linear_combinations(self, vectors):
        if not vectors:
            return [np.zeros(self.dim, dtype=int)]
        else:
            rest = self.linear_combinations(vectors[1:])
            return rest + [vectors[0] + r for r in rest]

    def linear_combinations_nozero(self, vectors):
        if len(vectors) == 1:
            return [vectors[0]]
        else:
            rest = self.linear_combinations_nozero(vectors[1:])
            return rest + [vectors[0]] + [vectors[0] + r for r in rest]

    # --------------------------- Private Members ----------------------------------------------------------------------

    def _construct_directionwise_kernel_family(self, create_ast_callback):
        subtrees = []
        directionSymbol = TypedSymbol('dir', dtype='stencil::Direction')
        for t in get_timesteps(self.streaming_pattern):
            cases_dict = dict()
            for comm_dir in self.full_stencil:
                if all(d == 0 for d in comm_dir):
                    continue
                dir_string = offset_to_direction_string(comm_dir)
                ast = create_ast_callback(comm_dir, t)
                if ast is None:
                    continue
                kernel_call = KernelCallNode(ast)
                cases_dict[f"stencil::{dir_string}"] = kernel_call
            subtrees.append(SwitchNode(directionSymbol, cases_dict))

        if not self.inplace:
            tree = subtrees[0]
        else:
            tree = EvenIntegerCondition('timestep', subtrees[Timestep.EVEN.idx], subtrees[Timestep.ODD.idx])
        return KernelFamily(tree, self.class_name)

    def _stream_out_accs(self, timestep):
        accessor = self.accessors[timestep.idx]
        src_stream_out_accs = accessor.write(self.src_field, self.stencil)
        dst_stream_out_accs = accessor.write(self.dst_field, self.stencil)
        return src_stream_out_accs, dst_stream_out_accs

    def _stream_in_accs(self, timestep):
        accessor = self.accessors[timestep.idx]
        src_stream_in_accs = accessor.read(self.src_field, self.stencil)
        dst_stream_in_accs = accessor.read(self.dst_field, self.stencil)
        return src_stream_in_accs, dst_stream_in_accs

    def _timestep_suffix(self, timestep):
        return ("_" + str(timestep)) if timestep != Timestep.BOTH else ''

    def _buffer(self, size):
        return Field.create_generic('buffer', spatial_dimensions=1, field_type=FieldType.BUFFER,
                                    dtype=self.create_kernel_params['data_type'],
                                    index_shape=(size,))
