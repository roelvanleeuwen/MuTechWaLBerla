import numpy as np
import waLBerla as wlb
import math


class Scenario:
    def __init__(self, benchmark=True, scaling_type=None, cells=(32, 32, 1), timesteps=100):
        #> Domain Parameters
        #self.domain_size = (32, 32, 1)
        self.blocks = (4, 1, 1)
        self.periodic = (1, 0, 1)
        domain_size = (cells[0] * self.blocks[0], cells[1] * self.blocks[1], cells[2] * self.blocks[2])
        #self.cells = (self.domain_size[0] // self.blocks[0], self.domain_size[1] // self.blocks[1], self.domain_size[2] // self.blocks[2])
        self.cells = cells
        #print(f"self.cells = {self.cells}")
        #> Standard Parameters
        self.timesteps = timesteps
        self.vtk_write_frequency = 200
        self.scenario = 1
        #> Physical Parameters
        #! Prandtl
        self.Prandtl = 1
        #! Rayleigh
        self.Rayleigh = 50000
        #! omega_fluid
        self.omega_fluid = 1.95
        #! omega_thermal
        self.length_x_SI = 2
        self.length_conversion = self.length_x_SI / domain_size[0]
        self.viscosity_LBM = 1./3 * (1. / self.omega_fluid - 1./2)
        self.time_conversion = self.viscosity_LBM * self.length_conversion * self.length_conversion / \
                               np.sqrt(self.Prandtl / self.Rayleigh)
        self.thermal_diffusivity_LBM = np.sqrt(1. / (self.Prandtl * self.Rayleigh)) * self.time_conversion / \
                                       (self.length_conversion * self.length_conversion)
        self.omega_thermal = 1. / (3 * self.thermal_diffusivity_LBM + 1./2)
        #print(f"omega_fluid = {self.omega_fluid} | omega_thermal = {self.omega_thermal}")
        #! temperature_hot
        self.temperature_hot = 50
        #! temperature_cold
        self.temperature_cold = - self.temperature_hot
        #print(f"temperature_hot = {self.temperature_hot} | temperature_cold = {self.temperature_cold}")
        #! gravity_LBM
        self.gravity_SI = 9.81
        self.gravity_conversion = self.length_conversion / (self.time_conversion * self.time_conversion)  #! look
        self.gravity_LBM = self.gravity_SI / self.gravity_conversion
        #> Initialization Parameters
        self.delta_temperature = abs(self.temperature_hot) + abs(self.temperature_cold)
        self.init_amplitude = self.delta_temperature / 200 * domain_size[0]
        self.init_temperature_range = 2 * self.delta_temperature / domain_size[0]
        #print(f"init_amplitude = {self.init_amplitude}")
        #print(f"init_temperature_range = {self.init_temperature_range}")

        #> Benchmark Parameters
        self.benchmark = benchmark
        #self.weakScaling = weak_scaling # True
        self.scalingType = scaling_type  #"fluid", "thermal", "rbc"
        self.benchmarkingIterations = 1
        self.warmupSteps = 10
        self.dbPath = './'
        self.dbFilename = 'rbc_ws_benchmark.sqlite'

        #print(f"gravity = {self.gravity_LBM}")

        #?self.viscosity_SI = 1.516e-5  #? look
        #?self.thermal_diffusivity_SI = 2.074e-5  #? look

        #?self.alpha = 3.43e-3  #? look

    @wlb.member_callback
    def config(self):
        return {
            'DomainSetup': {
                #'blocks': self.blocks,
                #'domainSize': self.domain_size,
                #'cellsPerBlock': self.cells,
                'cellsPerBlock': self.cells,
                'blocks': self.blocks,
                'periodic': self.periodic,
                'dx': 1.0,
            },
            'Parameters': {
                'timesteps': self.timesteps,
                'vtkWriteFrequency': self.vtk_write_frequency,
                'remainingTimeLoggerFrequency': 10.0,
                'scenario': self.scenario,
            },
            'PhysicalParameters': {  #todo check gravity!
                'omegaFluid': self.omega_fluid,
                #'omegaThermal': self.omega_thermal,
                'temperatureHot': self.temperature_hot,
                'temperatureCold': self.temperature_cold,
                'gravitationalAcceleration': self.gravity_LBM,  #? is this the right gravity I use here?
                                                                #? What do I have to use here?
                                                                #?    - gravitaional acceleration
                                                                #?    - gravitaional density
                                                                #?    - ...
            },
            'InitializationParameters': {
                'initAmplitude': self.init_amplitude,
                'initTemperatureRange': self.init_temperature_range,
            },
            'BenchmarkParameters': {
                'benchmark': self.benchmark,
                #'weakScaling': self.weakScaling,
                'scalingType': self.scalingType,  #"fluid", "thermal", "rbc"
                #'benchmarkingIterations': self.benchmarkingIterations,
                'warmupSteps': self.warmupSteps,
                'dbPath': self.dbPath,
                'dbFilename': self.dbFilename,
            },
            'Boundaries_Hydro': {
                'Border': [
                    {'direction': 'N', 'walldistance': -1, 'flag': 'BC_fluid_NoSlip'},
                    {'direction': 'S', 'walldistance': -1, 'flag': 'BC_fluid_NoSlip'},
                ]
            },
            'Boundaries_Thermal': {
                'Border': [
                    {'direction': 'N', 'walldistance': -1, 'flag': 'BC_thermal_Tcold'},
                    {'direction': 'S', 'walldistance': -1, 'flag': 'BC_thermal_Thot'},
                ]
            },
        }


def runSimulation():
    scenarios = wlb.ScenarioManager()
    scenarios.add(Scenario(timesteps=10001))


def weakScalingBenchmark():
    scenarios = wlb.ScenarioManager()
    block_sizes = [(i, i, i) for i in (8, 16, 32, 64, 128)]

    for scaling_type in ['rbc', 'fluid', 'thermal']:
        for block_size in block_sizes:
            print(f"block_size = {block_size}")
            print(f"scaling_type = {scaling_type}")
            scenario = Scenario(benchmark=True, scaling_type=scaling_type, cells=block_size)
            scenarios.add(scenario)
            print("---------------------------")


weakScalingBenchmark()
#runSimulation()
