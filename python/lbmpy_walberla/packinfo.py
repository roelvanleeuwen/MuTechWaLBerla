from collections import defaultdict
from lbmpy.advanced_streaming.utility import Timestep, get_accessor, get_timesteps
from lbmpy.advanced_streaming.communication import _extend_dir
from pystencils.stencil import inverse_direction
from pystencils_walberla.codegen import comm_directions, generate_pack_info
from pystencils import Assignment, Field


def generate_pack_infos_for_lbm_kernel(generation_context,
                                       class_name_prefix: str,
                                       lb_collision_rule,
                                       pdf_field,
                                       streaming_pattern='pull',
                                       include_non_pdf_fields=True,
                                       always_generate_seperate_classes=False,
                                       **create_kernel_params):
    """Generates waLBerla MPI PackInfos for an LBM kernel, based on a given method
    and streaming pattern. For in-place streaming patterns, two PackInfos are generated;
    one for the even and another for the odd time steps.

    Args:
        generation_context: see documentation of `generate_sweep`
        class_name_prefix: Prefix of the desired class name which will be addended with
                           'Even' or 'Odd' for in-place kernels
        lb_collision_rule: The collision rule defining the lattice boltzmann kernel, 
                           as returned by `create_lb_collision_rule`.
        streaming_pattern: The employed streaming pattern.
        **create_kernel_params: remaining keyword arguments are passed to `pystencils.create_kernel`
    """
    timesteps = [Timestep.EVEN, Timestep.ODD] \
                if always_generate_seperate_classes \
                else get_timesteps(streaming_pattern)

    common_spec = defaultdict(set)
    assignments = lb_collision_rule.all_assignments
    stencil = lb_collision_rule.method.stencil

    if include_non_pdf_fields:
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

    for t in timesteps:
        spec = common_spec.copy()
        write_accesses = get_accessor(streaming_pattern, t).write(pdf_field, stencil)
        for comm_dir in stencil:
            if all(d == 0 for d in comm_dir):
                continue
                
            for streaming_dir in set(_extend_dir(comm_dir)) & set(stencil):
                d = stencil.index(streaming_dir)
                fa = write_accesses[d]
                spec[(comm_dir,)].add(fa)

        class_name = class_name_prefix + ('' if t == Timestep.BOTH else str(t))
        generate_pack_info(generation_context, class_name, spec, **create_kernel_params)

