# -*- coding: utf-8 -*-

from mesa_pd.accessor import create_access
from mesa_pd.utility import generate_file


class CohesionInitialization:
    def __init__(self):
        self.context = {'interface': []}
        self.context['interface'].append(create_access("uid", "walberla::id_t", access="g"))
        self.context['interface'].append(
            create_access("contactHistory", "std::map<walberla::id_t, walberla::mesa_pd::Vec3>", access="gs"))

    def generate(self, module):
        ctx = {'module': module, **self.context}

        generate_file(module['module_path'], 'kernel/CohesionInitialization.templ.h', ctx)