#!/usr/bin/env python3
"""
This is a waLBerla parameter file that tests (almost) all parameter combinations for GPU communication.
Build waLBerla with -DWALBERLA_BUILD_WITH_PYTHON=1  then run e.g.
 ./UniformGridGPU_d3q27_aa_srt simulation_setup/benchmark_configs.py

Look at the end of the file to select the benchmark to run
"""

import os
import waLBerla as wlb
#from waLBerla.tools.config import block_decomposition
#from waLBerla.tools.sqlitedb import sequenceValuesToScalars, checkAndUpdateSchema, storeSingle
import sys
import sqlite3
from math import prod

# Number of time steps run for a workload of 128^3 per GPU
# if double as many cells are on the GPU, half as many time steps are run etc.
# increase this to get more reliable measurements
TIME_STEPS_FOR_128_BLOCK = 1000
DB_FILE = os.environ.get('DB_FILE', "gpu_benchmark.sqlite3")

BASE_CONFIG = {
    'DomainSetup': {
        'cellsPerBlock': (256, 128, 128),
        'periodic': (1, 1, 1),
    },
    'Parameters': {
        'omega': 1.8,
        'cudaEnabledMPI': False,
        'warmupSteps': 5,
        'outerIterations': 3,
    }
}


class Scenario:
    def __init__(self, cells_per_block=(256, 128, 128), periodic=(1, 1, 1), cuda_blocks=(256, 1, 1),
                 timesteps=None, time_step_strategy="normal", omega=1.8, cuda_enabled_mpi=False,
                 inner_outer_split=(1, 1, 1), warmup_steps=5, outer_iterations=3, init_shear_flow=False,
                 additional_info=None):

        self.blocks = (2, 2, 1)#block_decomposition(wlb.mpi.numProcesses())

        self.cells_per_block = cells_per_block
        self.periodic = periodic

        self.time_step_strategy = time_step_strategy
        self.omega = omega
        self.timesteps = timesteps
        self.cuda_enabled_mpi = cuda_enabled_mpi
        self.inner_outer_split = inner_outer_split
        self.init_shear_flow = init_shear_flow
        self.warmup_steps = warmup_steps
        self.outer_iterations = outer_iterations
        self.cuda_blocks = cuda_blocks

        self.vtk_write_frequency = 0

        self.config_dict = self.config(print_dict=False)
        self.additional_info = additional_info

    @wlb.member_callback
    def config(self, print_dict=True):
        from pprint import pformat
        config_dict = {
            'DomainSetup': {
                'blocks': self.blocks,
                'cellsPerBlock': self.cells_per_block,
                'periodic': self.periodic,
            },
            'Parameters': {
                'omega': self.omega,
                'cudaEnabledMPI': self.cuda_enabled_mpi,
                'warmupSteps': self.warmup_steps,
                'outerIterations': self.outer_iterations,
                'timeStepStrategy': self.time_step_strategy,
                'timesteps': self.timesteps,
                'initShearFlow': self.init_shear_flow,
                'gpuBlockSize': self.cuda_blocks,
                'innerOuterSplit': self.inner_outer_split,
                'vtkWriteFrequency': self.vtk_write_frequency
            }
        }
        if print_dict:
            wlb.log_info_on_root("Scenario:\n" + pformat(config_dict))
            if self.additional_info:
                wlb.log_info_on_root("Additional Info:\n" + pformat(self.additional_info))
        return config_dict


def emptyChannel():
    scenarios = wlb.ScenarioManager()
    scenario = Scenario( cells_per_block=(10, 10, 10), time_step_strategy="noOverlap", cuda_enabled_mpi=True, timesteps=100)
    scenarios.add(scenario)



emptyChannel()