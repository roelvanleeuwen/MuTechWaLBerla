import numpy as np
import waLBerla as wlb
import math


class Scenario:
    def __init__(self, benchmark=True,
                 cells=(32, 32, 32),
                 blockDecomposition="3D",
                 timesteps=1000,
                 run_on_cluster=False):
        self.periodic = (1, 0, 1)
        self.cells = cells
        self.timesteps = timesteps
        self.vtk_write_frequency = 200
        self.scenario = 1

        self.omega_fluid = 1.8
        self.force_in_x = 2e-6

        #self.weakScaling = weak_scaling # True
        #self.scalingType = scaling_type  #"fluid", "thermal", "rbc"
        #self.benchmarkingIterations = 5

        self.benchmark = benchmark
        self.warmupSteps = 2
        self.blockDecomposition = blockDecomposition
        self.barrier_after_sweep = False  # ANPASSEN!

        if run_on_cluster:
            self.dbPath = "/home/hpc/b144dc/b144dc11/repos/walberla-origin/build/apps/benchmarks/ChannelFlowBenchmark/"
            tmp = ""
            if self.barrier_after_sweep:
                tmp = "_wMPIBarrier"
            self.dbFilename = f"channelFlow_{blockDecomposition}{tmp}_{cells[0]}_t{timesteps}.sqlite"
        else:
            self.dbPath = "./"  # ANPASSEN!
            self.dbFilename = "channelFlow.sqlite"

    @wlb.member_callback
    def config(self):
        return {
            'DomainSetup': {
                #'blocks': self.blocks,
                #'domainSize': self.domain_size,
                #'cellsPerBlock': self.cells,
                'cellsPerBlock': self.cells,
                'periodic': self.periodic,
                'dx': 1.0,
            },
            'Parameters': {
                'timesteps': self.timesteps,
                'vtkWriteFrequency': self.vtk_write_frequency,
                'remainingTimeLoggerFrequency': 10.0,
                'scenario': self.scenario,
            },
            'PhysicalParameters': {
                'omegaFluid': self.omega_fluid,
                'forceInX': self.force_in_x,
            },
            'BenchmarkParameters': {
                #'weakScaling': self.weakScaling,
                #'scalingType': self.scalingType,  #"fluid", "thermal", "rbc"
                #'benchmarkingIterations': self.benchmarkingIterations,
                'benchmark': self.benchmark,
                'warmupSteps': self.warmupSteps,
                'blockDecomposition': self.blockDecomposition,  #"2D", "3D"
                'barrier_after_sweep': self.barrier_after_sweep,
            },
            'EvaluationParameters': {
                'dbPath': self.dbPath,
                'dbFilename': self.dbFilename,
            },
            'Boundaries_Hydro': {
                'Border': [
                    {'direction': 'N', 'walldistance': -1, 'flag': 'BC_fluid_NoSlip'},
                    {'direction': 'S', 'walldistance': -1, 'flag': 'BC_fluid_NoSlip'},
                ]
            },
        }


def runSimulation():
    scenarios = wlb.ScenarioManager()
    scenarios.add(Scenario())


def weakScalingBenchmark():
    scenarios = wlb.ScenarioManager()
    run_on_cluster = False

    if run_on_cluster:
        block_sizes = [(i, i, i) for i in (32, 64)]
        timesteps = [1000, 10000]
        blockDecompositions = ["3D", "2D"]

        for block_size in block_sizes:
            for timestep in timesteps:
                for blockDecomposition in blockDecompositions:
                    scenario = Scenario(benchmark=True,
                                        blockDecomposition=blockDecomposition,
                                        cells=block_size,
                                        timesteps=timestep,
                                        run_on_cluster=run_on_cluster)
                    scenarios.add(scenario)
    else:
        block_size = (32,32,32)
        timesteps = 1000
        blockDecomposition = "2D"
        scenario = Scenario(benchmark=True,
                            blockDecomposition=blockDecomposition,
                            cells=block_size,
                            timesteps=timesteps,
                            run_on_cluster=run_on_cluster)
        scenarios.add(scenario)


#runSimulation()
weakScalingBenchmark()
