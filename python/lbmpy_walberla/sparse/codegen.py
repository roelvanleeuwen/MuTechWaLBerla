from collections import OrderedDict, defaultdict
from dataclasses import replace
import numpy as np

from jinja2 import Environment, PackageLoader, StrictUndefined

from lbmpy import LBStencil, Stencil
from lbmpy.advanced_streaming import Timestep, is_inplace, get_accessor
from lbmpy.advanced_streaming.communication import _extend_dir
from lbmpy.sparse import create_lb_update_rule_sparse

from pystencils import Assignment, Target,  Field, create_kernel, FieldType
from pystencils.astnodes import KernelFunction
from pystencils.stencil import inverse_direction, offset_to_direction_string
from pystencils.backends.cbackend import get_headers

from lbmpy_walberla.sparse.jinja_filters import add_sparse_jinja_env
from lbmpy_walberla.alternating_sweeps import EvenIntegerCondition, OddIntegerCondition, TimestepTrackerMapping

from pystencils_walberla.codegen import generate_selective_sweep, config_from_context, KernelInfo, comm_directions, generate_pack_info
from pystencils_walberla.kernel_selection import AbstractInterfaceArgumentMapping, AbstractConditionNode, KernelCallNode, KernelFamily, HighLevelInterfaceSpec
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env




__all__ = ['generate_sparse_sweep', 'generate_sparse_pack_info']


def generate_sparse_sweep(generation_context, class_name, assignments, stencil,
                          namespace='lbmpy', field_swaps=(), varying_parameters=(),
                          inner_outer_split=False,
                          target=Target.CPU, data_type=None, cpu_openmp=None, cpu_vectorize_info=None,
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

    def to_name(f):
        return f.name if isinstance(f, Field) else f
    config = config_from_context(generation_context, target=target, data_type=data_type, cpu_openmp=cpu_openmp,
                                 cpu_vectorize_info=cpu_vectorize_info, **create_kernel_params)

    target = config.target

    if isinstance(assignments, KernelFunction):
        ast = assignments
    else:
        ast = create_kernel(assignments, config=config)

    ast.function_name = class_name.lower()
    selection_tree = KernelCallNode(ast)
    generate_selective_sparse_sweep(generation_context, class_name, selection_tree, target=target,
                                    namespace=namespace, field_swaps=field_swaps, varying_parameters=varying_parameters, inner_outer_split=inner_outer_split,
                                    cpu_vectorize_info=cpu_vectorize_info, cpu_openmp=cpu_openmp)



def generate_sparse_pack_info(generation_context, class_name: str, stencil,
                              namespace='lbmpy', operator=None, gl_to_inner=False,
                              target=Target.CPU, data_type=None, cpu_openmp=None,
                              **create_kernel_params):
    """Generates a waLBerla GPU PackInfo

    Args:
        generation_context: see documentation of `generate_sweep`
        class_name: name of the generated class
        directions_to_pack_terms: maps tuples of directions to read field accesses, specifying which values have to be
                                  packed for which direction
        namespace: inner namespace of the generated class
        operator: optional operator for, e.g., reduction pack infos
        gl_to_inner: communicates values from ghost layers of sender to interior of receiver
        target: An pystencils Target to define cpu or gpu code generation. See pystencils.Target
        data_type: default datatype for the kernel creation. Default is double
        cpu_openmp: if loops should use openMP or not.
        **create_kernel_params: remaining keyword arguments are passed to `pystencils.create_kernel`
    """
    template_name = "CpuPackInfoSparse.tmpl" if target == Target.CPU else "GpuPackInfoSparse.tmpl"

    jinja_context = {
        'class_name': class_name,
        'namespace': namespace,
        'Q': stencil.Q,
        'timestep': 'Even'

    }

    env = Environment(loader=PackageLoader('lbmpy_walberla'), undefined=StrictUndefined)
    add_sparse_jinja_env(env)

    header = env.get_template(template_name + ".h").render(**jinja_context)
    source = env.get_template(template_name + ".cpp").render(**jinja_context)

    source_extension = "cpp" if target == Target.CPU else "cu"
    generation_context.write_file(f"{class_name}.h", header)
    generation_context.write_file(f"{class_name}.{source_extension}", source)



def generate_alternating_sparse_pack_info(generation_context, class_name: str, stencil, streaming_pattern,
                                          namespace='lbmpy', operator=None, gl_to_inner=False,
                                          target=Target.CPU, data_type=None,
                                          cpu_openmp=None, **create_kernel_params):

    template_name = "CpuPackInfoSparse.tmpl" if target == Target.CPU else "GpuPackInfoSparse.tmpl"

    timesteps = ["Even", "Odd"]
    for timestep in timesteps:
        if streaming_pattern == 'aa':
            real_timestep = timestep
        elif streaming_pattern == 'pull':
            real_timestep = "Even"
        else:
            ValueError("Packinfo for streaming pattern", streaming_pattern, "not implemented")

        class_name_timestep = class_name + timestep
        jinja_context = {
            'class_name': class_name_timestep,
            'namespace': namespace,
            'Q': stencil.Q,
            'timestep': real_timestep
        }

        env = Environment(loader=PackageLoader('lbmpy_walberla'), undefined=StrictUndefined)
        add_sparse_jinja_env(env)

        header = env.get_template(template_name + ".h").render(**jinja_context)
        source = env.get_template(template_name + ".cpp").render(**jinja_context)

        source_extension = "cpp" if target == Target.CPU else "cu"
        generation_context.write_file(f"{class_name_timestep}.h", header)
        generation_context.write_file(f"{class_name_timestep}.{source_extension}", source)



def generate_hybrid_pack_info(generation_context, class_name_prefix: str, stencil, pdf_field,
                              streaming_pattern='pull', lb_collision_rule=None,
                              namespace='lbmpy', operator=None, target=Target.CPU,
                              data_type=None, cpu_openmp=None, gl_to_inner=False,
                              **create_kernel_params):

    timesteps = [Timestep.EVEN, Timestep.ODD]

    common_spec = defaultdict(set)

    if lb_collision_rule is not None:
        assignments = lb_collision_rule.all_assignments
        reads = set()
        for a in assignments:
            if not isinstance(a, Assignment):
                continue
            reads.update(a.rhs.atoms(Field.Access))
        for fa in reads:
            assert all(abs(e) <= 1 for e in fa.offsets)
            if all(offset == 0 for offset in fa.offsets):
                continue
            comm_direction = inverse_direction(fa.offsets)
            for comm_dir in comm_directions(comm_direction):
                common_spec[(comm_dir,)].add(fa.field.center(*fa.index))

    full_stencil = LBStencil(Stencil.D3Q27) if stencil.D == 3 else LBStencil(Stencil.D2Q9)

    for t in timesteps:
        directions_to_pack_terms = common_spec.copy()
        write_accesses = get_accessor(streaming_pattern, t).write(pdf_field, stencil)
        for comm_dir in full_stencil:
            if all(d == 0 for d in comm_dir):
                continue

            for streaming_dir in set(_extend_dir(comm_dir)) & set(stencil):
                d = stencil.index(streaming_dir)
                fa = write_accesses[d]
                directions_to_pack_terms[(comm_dir,)].add(fa)
        if t == Timestep.EVEN:
            class_name_suffix = 'Even'
        elif t == Timestep.ODD:
            class_name_suffix = 'Odd'
        else:
            class_name_suffix = ''

        class_name = class_name_prefix + class_name_suffix

        if streaming_pattern == 'aa':
            real_timestep = class_name_suffix
        elif streaming_pattern == 'pull':
            real_timestep = "Even"
        else:
            ValueError("Packinfo for streaming pattern", streaming_pattern, "not implemented")

        if cpu_openmp:
            raise ValueError("The packing kernels are already called inside an OpenMP parallel region. Thus "
                             "additionally parallelising each kernel is not supported.")
        items = [(e[0], sorted(e[1], key=lambda x: int(x.index[0]))) for e in directions_to_pack_terms.items()]
        items = sorted(items, key=lambda e: e[0])
        directions_to_pack_terms = OrderedDict(items)

        config = config_from_context(generation_context, target=target, data_type=data_type, cpu_openmp=cpu_openmp,
                                     **create_kernel_params)

        config_zero_gl = config_from_context(generation_context, target=target, data_type=data_type, cpu_openmp=cpu_openmp,
                                             ghost_layers=0, **create_kernel_params)

        # Vectorisation of the pack info is not implemented.
        config = replace(config, cpu_vectorize_info=None)
        config_zero_gl = replace(config_zero_gl, cpu_vectorize_info=None)

        config = replace(config, allow_double_writes=True)
        config_zero_gl = replace(config_zero_gl, allow_double_writes=True)

        template_name = "CpuPackInfoHybrid.tmpl" if config.target == Target.CPU else 'GpuPackInfoHybrid.tmpl'

        fields_accessed = set()
        for terms in directions_to_pack_terms.values():
            for term in terms:
                assert isinstance(term, Field.Access)  # and all(e == 0 for e in term.offsets)
                fields_accessed.add(term)

        field_names = {fa.field.name for fa in fields_accessed}

        data_types = {fa.field.dtype for fa in fields_accessed}
        if len(data_types) == 0:
            raise ValueError("No fields to pack!")
        if len(data_types) != 1:
            err_detail = "\n".join(f" - {f.name} [{f.dtype}]" for f in fields_accessed)
            raise NotImplementedError("Fields of different data types are used - this is not supported.\n" + err_detail)
        dtype = data_types.pop()

        pack_kernels = OrderedDict()
        unpack_kernels = OrderedDict()
        all_accesses = set()
        elements_per_cell = OrderedDict()
        for direction_set, terms in directions_to_pack_terms.items():
            for d in direction_set:
                if not all(abs(i) <= 1 for i in d):
                    raise NotImplementedError("Only first neighborhood supported")

            buffer = Field.create_generic('buffer', spatial_dimensions=1, field_type=FieldType.BUFFER,
                                          dtype=dtype.numpy_dtype, index_shape=(len(terms),))

            direction_strings = tuple(offset_to_direction_string(d) for d in direction_set)
            all_accesses.update(terms)

            pack_assignments = [Assignment(buffer(i), term) for i, term in enumerate(terms)]
            pack_ast = create_kernel(pack_assignments, config=config_zero_gl)
            pack_ast.function_name = 'pack_{}'.format("_".join(direction_strings))
            if operator is None:
                unpack_assignments = [Assignment(term, buffer(i)) for i, term in enumerate(terms)]
            else:
                unpack_assignments = [Assignment(term, operator(term, buffer(i))) for i, term in enumerate(terms)]
            unpack_ast = create_kernel(unpack_assignments, config=config_zero_gl)
            unpack_ast.function_name = 'unpack_{}'.format("_".join(direction_strings))

            pack_kernels[direction_strings] = KernelInfo(pack_ast)
            unpack_kernels[direction_strings] = KernelInfo(unpack_ast)
            elements_per_cell[direction_strings] = len(terms)
        fused_kernel = create_kernel([Assignment(buffer.center, t) for t in all_accesses], config=config)

        jinja_context = {
            'class_name': class_name,
            'pack_kernels': pack_kernels,
            'unpack_kernels': unpack_kernels,
            'fused_kernel': KernelInfo(fused_kernel),
            'elements_per_cell': elements_per_cell,
            'headers': get_headers(fused_kernel),
            'target': config.target.name.lower(),
            'dtype': dtype,
            'field_name': field_names.pop(),
            'namespace': namespace,
            'gl_to_inner': gl_to_inner,
            'timestep': real_timestep
        }
        env = Environment(loader=PackageLoader('pystencils_walberla'), undefined=StrictUndefined)
        add_pystencils_filters_to_jinja_env(env)
        header = env.get_template(template_name + ".h").render(**jinja_context)
        source = env.get_template(template_name + ".cpp").render(**jinja_context)

        source_extension = "cpp" if config.target == Target.CPU else "cu"
        generation_context.write_file(f"{class_name}.h", header)
        generation_context.write_file(f"{class_name}.{source_extension}", source)





def generate_alternating_sparse_lbm_sweep(generation_context, class_name, collision_rule,
                                   lbm_config, pdfs, index_field, dst=None, lbm_optimisation=None,
                                   namespace='lbmpy', field_swaps=(), varying_parameters=(),
                                   inner_outer_split=False, ghost_layers_to_include=0,
                                   target=Target.CPU, data_type=None,
                                   cpu_openmp=None, cpu_vectorize_info=None, max_threads=None,
                                   **kernel_parameters):
    """Generates an Alternating lattice Boltzmann sweep class. This is in particular meant for
    in-place streaming patterns, but can of course also be used with two-fields patterns (why make it
    simple if you can make it complicated?). From a collision rule, two kernel implementations are
    generated; one for even, and one for odd timesteps. At run time, the correct one is selected
    according to a time step counter. This counter can be managed by the `walberla::lbm::TimestepTracker`
    class.

    Args:
        generation_context: See documentation of `pystencils_walberla.generate_sweep`
        class_name: Name of the generated class
        collision_rule: LbCollisionRule as returned by `lbmpy.create_lb_collision_rule`.
        lbm_config: configuration of the LB method. See lbmpy.LBMConfig
        lbm_optimisation: configuration of the optimisations of the LB method. See lbmpy.LBMOptimisation
        namespace: see documentation of `generate_sweep`
        field_swaps: see documentation of `generate_sweep`
        varying_parameters: see documentation of `generate_sweep`
        inner_outer_split: see documentation of `generate_sweep`
        ghost_layers_to_include: see documentation of `generate_sweep`
        target: An pystencils Target to define cpu or gpu code generation. See pystencils.Target
        data_type: default datatype for the kernel creation. Default is double
        cpu_openmp: if loops should use openMP or not.
        cpu_vectorize_info: dictionary containing necessary information for the usage of a SIMD instruction set.
        max_threads: only relevant for GPU kernels. Will be argument of `__launch_bounds__`.
        kernel_parameters: other parameters passed to the creation of a pystencils.CreateKernelConfig
    """
    config = config_from_context(generation_context, target=target, data_type=data_type, cpu_openmp=cpu_openmp,
                                 cpu_vectorize_info=cpu_vectorize_info, **kernel_parameters)

    # Add the lbm collision rule to the config
    lbm_config = replace(lbm_config, collision_rule=collision_rule)
    even_lbm_config = replace(lbm_config, timestep=Timestep.EVEN)

    sparse_update_rule_even = create_lb_update_rule_sparse(collision_rule, even_lbm_config, pdfs, index_field, dst, inner_outer_split)
    ast_even = create_kernel(sparse_update_rule_even, config=config)
    ast_even.function_name = 'even'
    kernel_even = KernelCallNode(ast_even)

    if is_inplace(lbm_config.streaming_pattern):
        odd_lbm_config = replace(lbm_config, timestep=Timestep.ODD)
        sparse_update_rule_odd = create_lb_update_rule_sparse(collision_rule, odd_lbm_config, pdfs, index_field, dst, inner_outer_split)

        config = replace(config, allow_double_writes=True)
        ast_odd = create_kernel(sparse_update_rule_odd, config=config)
        ast_odd.function_name = 'odd'
        kernel_odd = KernelCallNode(ast_odd)
    else:
        kernel_odd = kernel_even

    tree = EvenIntegerCondition('timestep', kernel_even, kernel_odd, np.uint8)
    interface_mappings = [TimestepTrackerMapping(tree.parameter_symbol)]


    generate_selective_sparse_sweep(generation_context, class_name, tree, interface_mappings, target,
                                    namespace, field_swaps, varying_parameters, inner_outer_split, ghost_layers_to_include,
                                    cpu_vectorize_info, cpu_openmp, max_threads)





def generate_selective_sparse_sweep(generation_context, class_name, selection_tree, interface_mappings=(), target=None,
                             namespace='lbmpy', field_swaps=(), varying_parameters=(),
                             inner_outer_split=False, ghost_layers_to_include=0,
                             cpu_vectorize_info=None, cpu_openmp=False, max_threads=None):

    def to_name(f):
        return f.name if isinstance(f, Field) else f

    field_swaps = tuple((to_name(e[0]), to_name(e[1])) for e in field_swaps)
    temporary_fields = tuple(e[1] for e in field_swaps)

    kernel_family = KernelFamily(selection_tree, class_name,
                                 temporary_fields, field_swaps, varying_parameters)
    if target is None:
        target = kernel_family.get_ast_attr('target')
    elif target != kernel_family.get_ast_attr('target'):
        raise ValueError('Mismatch between target parameter and AST targets.')

    #if not (generation_context.cuda or generation_context.hip) and target == Target.GPU:
    if not generation_context.cuda and target == Target.GPU:
        return

    representative_field = {p.field_name for p in kernel_family.parameters if p.is_field_parameter}
    representative_field = sorted(representative_field)[0]

    env = Environment(loader=PackageLoader('lbmpy_walberla'), undefined=StrictUndefined)
    add_sparse_jinja_env(env)

    interface_spec = HighLevelInterfaceSpec(kernel_family.kernel_selection_parameters, interface_mappings)

    jinja_context = {
        'kernel': kernel_family,
        'namespace': namespace,
        'class_name': class_name,
        'target': target.name.lower(),
        'field': representative_field,
        'ghost_layers_to_include': ghost_layers_to_include,
        'inner_outer_split': inner_outer_split,
        'interface_spec': interface_spec,
        'generate_functor': True,
        'cpu_vectorize_info': cpu_vectorize_info,
        'cpu_openmp': cpu_openmp,
        'max_threads': max_threads
    }
    header = env.get_template("SweepSparse.tmpl.h").render(**jinja_context)
    source = env.get_template("SweepSparse.tmpl.cpp").render(**jinja_context)

    source_extension = "cpp" if target == Target.CPU else "cu"
    generation_context.write_file(f"{class_name}.h", header)
    generation_context.write_file(f"{class_name}.{source_extension}", source)