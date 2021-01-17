from lbmpy.advanced_streaming import AccessPdfValues, numeric_offsets, numeric_index
from lbmpy.boundaries import ExtrapolationOutflow, UBB

from pystencils_walberla.additional_data_handler import AdditionalDataHandler


class UBBAdditionalDataHandler(AdditionalDataHandler):
    def __init__(self, stencil, dim, boundary_object):
        assert isinstance(boundary_object, UBB)
        self._boundary_object = boundary_object
        super(UBBAdditionalDataHandler, self).__init__(stencil=stencil, dim=dim)

    @property
    def constructor_arguments(self):
        return ", std::function<Vector3<real_t>(const Cell &, const shared_ptr<StructuredBlockForest>&, IBlock&)>& " \
               "velocityCallback "

    @property
    def initialiser_list(self):
        return "elementInitaliser(velocityCallback),"

    @property
    def additional_arguments_for_fill_function(self):
        return "blocks, "

    @property
    def additional_parameters_for_fill_function(self):
        return " const shared_ptr<StructuredBlockForest> &blocks, "

    @property
    def data_initialisation(self):
        init_list = ["Vector3<real_t> InitialisatonAdditionalData = elementInitaliser(Cell(it.x(), it.y(), it.z()), "
                     "blocks, *block);", "element.vel_0 = InitialisatonAdditionalData[0];",
                     "element.vel_1 = InitialisatonAdditionalData[1];"]
        if self._dim == 3:
            init_list.append("element.vel_2 = InitialisatonAdditionalData[2];")

        return "\n".join(init_list)

    @property
    def additional_member_variable(self):
        return "std::function<Vector3<real_t>(const Cell &, const shared_ptr<StructuredBlockForest>&, IBlock&)> " \
               "elementInitaliser; "


class OutflowAdditionalDataHandler(AdditionalDataHandler):
    def __init__(self, stencil, dim, boundary_object, field_name, build_for_gpu=False):
        assert isinstance(boundary_object, ExtrapolationOutflow)
        self._boundary_object = boundary_object
        self._stencil = boundary_object.stencil
        self._lb_method = boundary_object.lb_method
        self._normal_direction = boundary_object.normal_direction
        self._field_name = field_name
        self._build_for_gpu = build_for_gpu
        super(OutflowAdditionalDataHandler, self).__init__(stencil=stencil, dim=dim)

    @property
    def additional_field_data(self):
        suffix = "CPU" if self._build_for_gpu else ""
        if self._build_for_gpu:
            gpu_field = f"{self._field_name}GPU"
            cpu_field = f"{self._field_name}"
            xs = f"{gpu_field}->xSize()"
            ys = f"{gpu_field}->ySize()"
            zs = f"{gpu_field}->zSize()"
            gl = f"{gpu_field}->nrOfGhostLayers()"
            layout = f"{gpu_field}->layout()"
            result = [f"auto {gpu_field} = block->getData< cuda::GPUField< real_t > >({self._field_name}ID);",
                      f"    auto {cpu_field} = new GhostLayerField<real_t ,{len(self._stencil)}>"
                      f"( {xs}, {ys}, {zs}, {gl}, {layout} );",
                      f"    cuda::fieldCpy( *{cpu_field}, *{gpu_field} );"]
            return "\n".join(result)
        else:
            return f"auto {self._field_name} = block->getData< field::GhostLayerField<double, " \
                   f"{len(self._stencil)}> >({self._field_name}ID{suffix}); "

    @property
    def data_initialisation(self):
        pdf_acc = AccessPdfValues(self._boundary_object.stencil,
                                  streaming_pattern=self._boundary_object.streaming_pattern,
                                  timestep=self._boundary_object.zeroth_timestep,
                                  streaming_dir='out')

        init_list = []
        for key, value in self.get_init_dict(pdf_acc).items():
            init_list.append(f"element.{key} = {self._field_name}->get({value});")

        return "\n".join(init_list)

    @property
    def stencil_info(self):
        stencil_info = []
        for i, d in enumerate(self._stencil):
            if d == self._normal_direction:
                direction = d if self._dim == 3 else d + (0,)
                stencil_info.append((i, direction, ", ".join([str(e) for e in direction])))
        return stencil_info

    def get_init_dict(self, pdf_accessor):
        """The Extrapolation Outflow boundary needs additional data. This function provides a list of all values
        which have to be initialised"""
        result = {}
        position = ["it.x()", "it.y()", "it.z()"]
        for j, stencil_dir in enumerate(self._stencil):
            pos = []
            if all(n == 0 or n == -s for s, n in zip(stencil_dir, self._normal_direction)):
                offsets = numeric_offsets(pdf_accessor.accs[j])
                for p, o in zip(position, offsets):
                    pos.append(p + " + cell_idx_c(" + str(o) + ")")
                pos.append(str(numeric_index(pdf_accessor.accs[j])[0]))
                result[f'pdf_{j}'] = ', '.join(pos)
                result[f'pdf_nd_{j}'] = ', '.join(pos)

        return result
