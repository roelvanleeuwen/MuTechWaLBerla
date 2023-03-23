from typing import Dict, Sequence, Tuple
from collections import OrderedDict
from dataclasses import replace
import numpy as np

from jinja2 import Environment, PackageLoader, StrictUndefined

from pystencils import Target
from pystencils import Field, create_kernel
from pystencils.astnodes import KernelFunction
from pystencils.typing import collate_types

from pystencils_walberla.codegen import KernelInfo
from lbmpy_walberla.sparse.jinja_filters import add_sparse_jinja_env
from lbmpy.sparse import create_lb_update_rule_sparse

from pystencils_walberla.codegen import generate_selective_sweep, config_from_context
from pystencils_walberla.kernel_selection import AbstractInterfaceArgumentMapping, AbstractConditionNode, KernelCallNode, KernelFamily, HighLevelInterfaceSpec
from lbmpy.creationfunctions import create_lb_ast
from lbmpy.advanced_streaming import Timestep, is_inplace

from lbmpy_walberla.alternating_sweeps import EvenIntegerCondition, OddIntegerCondition, TimestepTrackerMapping



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
    template_name = "CPUPackInfoSparse.tmpl" if target == Target.CPU else "GPUPackInfoSparse.tmpl"

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
    template_name = "CPUPackInfoSparse.tmpl" if target == Target.CPU else "GPUPackInfoSparse.tmpl"


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