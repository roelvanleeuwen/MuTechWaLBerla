class AdditionalDataHandler:
    """Base class that defines how to handle boundary conditions holding additional data."""

    def __init__(self, dim=3):
        self._dim = dim

    @property
    def constructor_arguments(self):
        return ""

    @property
    def initialiser_list(self):
        return ""

    @property
    def additional_arguments_for_fill_function(self):
        return ""

    @property
    def additional_parameters_for_fill_function(self):
        return ""

    @property
    def additional_field_data(self):
        return ""

    @property
    def data_initialisation(self):
        return ""

    @property
    def additional_member_variable(self):
        return ""
