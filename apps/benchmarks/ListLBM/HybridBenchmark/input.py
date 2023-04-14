import os
from math import prod
import sys
import sqlite3

import numpy as np

import waLBerla as wlb
from waLBerla.tools.config import block_decomposition
from waLBerla.tools.sqlitedb import sequenceValuesToScalars, checkAndUpdateSchema, storeSingle

DB_FILE = os.environ.get('DB_FILE', "ListLBMBenchmark.sqlite3")

class Scenario:
    def __init__(self, cells_per_block=(64, 64, 10),
                 timesteps=1000, time_step_strategy="noOverlap", omega=1.4, cuda_enabled_mpi=True,
                 inner_outer_split=(1, 1, 1), vtk_write_frequency=0,
                 porosity=0.5, porositySwitch=0.0):

        self.timesteps = timesteps
        self.vtkWriteFrequency = vtk_write_frequency

        self.cells_per_block = cells_per_block
        self.porositySwitch = porositySwitch
        self.porosity = porosity

        self.inner_outer_split = inner_outer_split
        self.time_step_strategy = time_step_strategy
        self.cuda_enabled_mpi = cuda_enabled_mpi

        self.omega = omega

        self.config_dict = self.config()

    @wlb.member_callback
    def config(self):
        return {
            'DomainSetup': {
                'cellsPerBlock': self.cells_per_block,
                'weakScaling': True
        },
            'Parameters': {
                'timesteps': self.timesteps,
                'omega': self.omega,
                'timeStepStrategy': self.time_step_strategy,
                'innerOuterSplit': self.inner_outer_split,
                'cudaEnabledMPI': self.cuda_enabled_mpi,
                'vtkWriteFrequency': self.vtkWriteFrequency,
                'porositySwitch': self.porositySwitch,
                'porosity': self.porosity,
                'runBoundaries': True,
                'remainingTimeLoggerFrequency': 2
        },
            'Boundaries': {
                'Border': [
                    {'direction': 'N', 'walldistance': -1, 'flag': 'NoSlip'},
                    {'direction': 'S', 'walldistance': -1, 'flag': 'NoSlip'},
                    {'direction': 'W', 'walldistance': -1, 'flag': 'NoSlip'},
                    {'direction': 'E', 'walldistance': -1, 'flag': 'NoSlip'},
                    {'direction': 'T', 'walldistance': -1, 'flag': 'NoSlip'},
                    {'direction': 'B', 'walldistance': -1, 'flag': 'NoSlip'},
                ]
            }
        }


def porosity_benchmark():
    wlb.log_info_on_root("Running different porosities")
    scenarios = wlb.ScenarioManager()
    porosities = [0.1 * i for i in range(10+1)]
    for porosity in porosities:
        scenario = Scenario(porosity=porosity)
        scenarios.add(scenario)


porosity_benchmark()
