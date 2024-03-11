import os

import waLBerla as wlb

DB_FILE = os.environ.get('DB_FILE', "ListLBMBenchmark.sqlite3")

class Scenario:
    def __init__(self, cells_per_block=(64, 64, 20), periodic=(False,False,False), dx=1.0,
                 timesteps=1001, time_step_strategy="noOverlap", omega=0.8, gpu_enabled_mpi=False,
                 gpu_block_size=(256, 1, 1), inner_outer_split=(1, 1, 1), vtk_write_frequency=0,
                 inflow_velocity=(0.01, 0, 0), porosity=0.0, porosity_switch=0.8, run_hybrid=True,
                 geometry_setup="randomNoslip", spheres_radius=9, sphere_shift=10, sphere_fill=(1.0, 1.0, 1.0),
                 mesh_file="None", run_boundaries=True, use_cartesian_communicator=False, balance_load=False):

        self.timesteps = timesteps
        self.vtkWriteFrequency = vtk_write_frequency

        self.inflow_velocity = inflow_velocity

        self.cells_per_block = cells_per_block
        self.periodic = periodic
        self.porosity_switch = porosity_switch
        self.porosity = porosity
        self.balance_load = balance_load
        self.use_cartesian_communicator = use_cartesian_communicator
        self.gpu_block_size = gpu_block_size

        self.inner_outer_split = inner_outer_split
        self.time_step_strategy = time_step_strategy
        self.gpu_enabled_mpi = gpu_enabled_mpi
        self.run_boundaries = run_boundaries
        self.run_hybrid = run_hybrid
        self.omega = omega
        self.dx = dx

        self.geometry_setup = geometry_setup

        self.spheres_radius = spheres_radius
        self.sphere_shift = sphere_shift
        self.sphere_fill = sphere_fill

        self.mesh_file = mesh_file

        self.config_dict = self.config()



    @wlb.member_callback
    def config(self):
        return {
            'DomainSetup': {
                'cellsPerBlock': self.cells_per_block,
                'periodic': self.periodic,
                'weakScaling': True,
                'geometrySetup': self.geometry_setup,
                'meshFile': self.mesh_file
        },
            'Parameters': {
                'timesteps': self.timesteps,
                'omega': self.omega,
                'initialVelocity':self.inflow_velocity,
                'timeStepStrategy': self.time_step_strategy,
                'innerOuterSplit': self.inner_outer_split,
                'gpuEnabledMPI': self.gpu_enabled_mpi,
                'gpuBlockSize': self.gpu_block_size,
                'vtkWriteFrequency': self.vtkWriteFrequency,
                'porositySwitch': self.porosity_switch,
                'runHybrid': self.run_hybrid,
                'porosity': self.porosity,
                'balanceLoad': self.balance_load,
                'runBoundaries': self.run_boundaries,
                'remainingTimeLoggerFrequency': 10,
                'useCartesian': self.use_cartesian_communicator,
                'writeDomainDecompositionAndReturn': False,
                'dx': self.dx,

                'SpheresRadius': self.spheres_radius,
                'SphereShift': self.sphere_shift,
                'SphereFillDomainRatio': self.sphere_fill,


        },
            'Boundaries': {
                'Border': [
                    #{'direction': 'N', 'walldistance': -1, 'flag': 'NoSlip'},
                    #{'direction': 'S', 'walldistance': -1, 'flag': 'NoSlip'},
                    #{'direction': 'W', 'walldistance': -1, 'flag': 'UBB'},
                    #{'direction': 'E', 'walldistance': -1, 'flag': 'PressureOutflow'},
                    #{'direction': 'T', 'walldistance': -1, 'flag': 'NoSlip'},
                    #{'direction': 'B', 'walldistance': -1, 'flag': 'NoSlip'},
                ]
            }
        }


def porosity_benchmark():
    wlb.log_info_on_root("Running different porosities")
    scenarios = wlb.ScenarioManager()
    porosities = [0.02 * i for i in range(100+1)]
    for porosity in porosities:
        scenario = Scenario(porosity=porosity, run_hybrid=False, porosity_switch=1.0, cells_per_block=(256, 256, 256), geometry_setup="randomNoslip", inflow_velocity=(0, 0, 0), run_boundaries=False, time_step_strategy="kernelOnly")
        scenarios.add(scenario)

def randomNoslip():
    scenarios = wlb.ScenarioManager()
    scenario = Scenario(porosity=0.8, porosity_switch=1.0, cells_per_block=(128, 128, 128), geometry_setup="randomNoslip", inflow_velocity=(0,0,0), run_boundaries=False, time_step_strategy="kernelOnly")
    scenarios.add(scenario)

def spheres():
    scenarios = wlb.ScenarioManager()
    spheres_radius = 7
    sphere_shift = 3
    sphere_fill = (1.0, 1.0, 1.0)
    scenario = Scenario(vtk_write_frequency=100, geometry_setup="spheres", spheres_radius=spheres_radius,
                        sphere_shift=sphere_shift, sphere_fill=sphere_fill, porosity_switch=1.0, cells_per_block=(128, 128, 128), timesteps=101)
    scenarios.add(scenario)

def Artery():
    scenarios = wlb.ScenarioManager()
    #mesh_file = "Artery.obj"
    mesh_file = "coronary_colored_medium.obj"
    scenario = Scenario(dx=0.0257, cells_per_block=(128, 128, 128), vtk_write_frequency=0, geometry_setup="artery", mesh_file=mesh_file, timesteps=20, omega=1.7,  porosity_switch=1.0, run_hybrid=False, time_step_strategy="Overlap", run_boundaries=True, balance_load=True)
    scenarios.add(scenario)

def ArterySparseVsDense():
    scenarios = wlb.ScenarioManager()
    mesh_file = "coronary_colored_medium.obj"

    cells_per_block_options = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256), (320, 320, 320), (450, 450, 450), (512, 512, 512)]
    for cells_per_block in cells_per_block_options:
        scenario = Scenario(dx=0.0257, cells_per_block=cells_per_block, geometry_setup="artery", mesh_file=mesh_file, timesteps=1000,  porosity_switch=1.0, run_hybrid=True, time_step_strategy="Overlap", run_boundaries=True, gpu_enabled_mpi=True)
        scenarios.add(scenario)
        #scenario = Scenario(dx=0.0257, cells_per_block=cells_per_block, geometry_setup="artery", mesh_file=mesh_file, timesteps=1000,  porosity_switch=1.0, run_hybrid=True, time_step_strategy="Overlap", run_boundaries=True, gpu_enabled_mpi=True, balance_load=True)
        #scenarios.add(scenario)
        #scenario = Scenario(dx=0.0257, cells_per_block=cells_per_block, geometry_setup="artery", mesh_file=mesh_file, timesteps=1000,  porosity_switch=0.0, run_hybrid=True, time_step_strategy="Overlap", run_boundaries=True, gpu_enabled_mpi=True)
        #scenarios.add(scenario)

def smallArtery():
    scenarios = wlb.ScenarioManager()
    mesh_file = "Artery.obj"
    scenario = Scenario(dx=0.3, cells_per_block=(64, 64, 64), vtk_write_frequency=0, geometry_setup="artery", mesh_file=mesh_file, timesteps=10, omega=1.7,  porosity_switch=1.0, run_hybrid=False, time_step_strategy="noOverlap", run_boundaries=True)
    scenarios.add(scenario)

def particleBed():
    scenarios = wlb.ScenarioManager()
    blocksX = 4
    domainSizeX = 0.1
    dx = domainSizeX / (blocksX * 32)
    scenario = Scenario(geometry_setup="particleBed", vtk_write_frequency=1, timesteps=1, omega=1.5, cells_per_block=(32, 32, 32), porosity_switch=0.8, run_hybrid=True, dx=dx, periodic=(True, True, True), balance_load=True)
    scenarios.add(scenario)


def particleBedSparseVsDense():
    scenarios = wlb.ScenarioManager()
    blocksX = 4
    domainSizeX = 0.1
    cellsPerBlocksVec = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]
    #cellsPerBlocksVec = [(256, 256, 256)]
    dx = domainSizeX / (blocksX * cellsPerBlocksVec[-1][0])
    for cellsPerBlocks in cellsPerBlocksVec:
        scenario = Scenario(geometry_setup="particleBed", vtk_write_frequency=0, timesteps=1001, omega=1.5, cells_per_block=cellsPerBlocks, porosity_switch=0.0, run_hybrid=True, dx=dx, periodic=(True, True, True), time_step_strategy="kernelOnly")
        scenarios.add(scenario)
        scenario = Scenario(geometry_setup="particleBed", vtk_write_frequency=0, timesteps=1001, omega=1.5, cells_per_block=cellsPerBlocks, porosity_switch=1.0, run_hybrid=False, dx=dx, periodic=(True, True, True), time_step_strategy="kernelOnly")
        scenarios.add(scenario)
        scenario = Scenario(geometry_setup="particleBed", vtk_write_frequency=0, timesteps=1001, omega=1.5, cells_per_block=cellsPerBlocks, porosity_switch=0.8, run_hybrid=True, dx=dx, periodic=(True, True, True), time_step_strategy="kernelOnly")
        scenarios.add(scenario)

def particleBedBlockSizes():
    cellsPerBlocksVec = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256), (320, 320, 320), (450, 450, 450)]
    blocksX = 4
    domainSizeX = 0.1
    scenarios = wlb.ScenarioManager()
    for cellsPerBlocks in cellsPerBlocksVec:
        dx = domainSizeX / (blocksX * cellsPerBlocks[0])
        scenario = Scenario(geometry_setup="particleBed", vtk_write_frequency=0, timesteps=1000, omega=1.5, cells_per_block=cellsPerBlocks, porosity_switch=1.0, run_hybrid=True, dx=dx, periodic=(False, True, True), time_step_strategy="kernelOnly")
        scenarios.add(scenario)
        scenario = Scenario(geometry_setup="particleBed", vtk_write_frequency=0, timesteps=1000, omega=1.5, cells_per_block=cellsPerBlocks, porosity_switch=1.0, run_hybrid=True, dx=dx, periodic=(False, True, True), time_step_strategy="noOverlap")
        scenarios.add(scenario)


def emptyChannel():
    scenarios = wlb.ScenarioManager()
    scenario = Scenario(porosity=1.0, periodic=(0,1,1), vtk_write_frequency=0, geometry_setup="randomNoslip", inflow_velocity=(0.01, 0, 0), omega=1.9, porosity_switch=1.1, run_hybrid=True,cells_per_block=(100, 100, 100),
                        time_step_strategy="noOverlap", gpu_enabled_mpi=False)
    scenarios.add(scenario)

def scalingBenchmark():
    cells_per_block = (320, 320, 320)
    gpu_enabled_mpi = True
    scenarios = wlb.ScenarioManager()
    scenario = Scenario(cells_per_block=cells_per_block, geometry_setup="randomNoslip", porosity=1.0, porosity_switch=1.0, run_hybrid=False, time_step_strategy="noOverlap", inner_outer_split=(0, 0, 0), run_boundaries=True, gpu_enabled_mpi=gpu_enabled_mpi)
    scenarios.add(scenario)
    scenario = Scenario(cells_per_block=cells_per_block, geometry_setup="randomNoslip", porosity=1.0, porosity_switch=1.0, run_hybrid=False, time_step_strategy="Overlap", inner_outer_split=(1, 1, 1), run_boundaries=True, gpu_enabled_mpi=gpu_enabled_mpi)
    scenarios.add(scenario)
    scenario = Scenario(cells_per_block=cells_per_block, geometry_setup="randomNoslip", porosity=1.0, porosity_switch=1.0, run_hybrid=False, time_step_strategy="Overlap", inner_outer_split=(32, 1, 1), run_boundaries=True, gpu_enabled_mpi=gpu_enabled_mpi)
    scenarios.add(scenario)


def testGPUComm():
    cells_per_block=(256, 256, 256)
    scenarios = wlb.ScenarioManager()
    scenario = Scenario(cells_per_block=cells_per_block, geometry_setup="randomNoslip", porosity=1.0, porosity_switch=0.0, run_hybrid=True, time_step_strategy="kernelOnly")
    scenarios.add(scenario)
    scenario = Scenario(cells_per_block=cells_per_block, geometry_setup="randomNoslip", porosity=1.0, porosity_switch=0.0, run_hybrid=True, time_step_strategy="communicationOnly")
    scenarios.add(scenario)

def testCartesianComm():
    cells_per_block=(50, 50, 50)
    scenarios = wlb.ScenarioManager()
    scenario = Scenario(cells_per_block=cells_per_block, periodic=(False,True,True), geometry_setup="randomNoslip", porosity=1.0, porosity_switch=0.0, run_hybrid=True,
                        time_step_strategy="noOverlap", inner_outer_split=(0, 0, 0), run_boundaries=True, use_cartesian_communicator=True)
    scenarios.add(scenario)
    scenario = Scenario(cells_per_block=cells_per_block, periodic=(False,True,True), geometry_setup="randomNoslip", porosity=1.0, porosity_switch=0.0, run_hybrid=True,
                        time_step_strategy="noOverlap", inner_outer_split=(0, 0, 0), run_boundaries=True, use_cartesian_communicator=False)
    scenarios.add(scenario)
    #scenario = Scenario(cells_per_block=cells_per_block, periodic=(0,1,1), geometry_setup="randomNoslip", porosity=1.0, porosity_switch=0.0, run_hybrid=True,
    #                    time_step_strategy="Overlap", inner_outer_split=(1, 1, 1), run_boundaries=True, use_cartesian_communicator=False)
    #scenarios.add(scenario)
    #scenario = Scenario(cells_per_block=cells_per_block, periodic=(0,1,1), geometry_setup="randomNoslip", porosity=1.0, porosity_switch=0.0, run_hybrid=True,
    #                    time_step_strategy="Overlap", inner_outer_split=(1, 1, 1), run_boundaries=True, use_cartesian_communicator=True)
    #scenarios.add(scenario)



#randomNoslip()

#porosity_benchmark()

#spheres()
Artery()
#smallArtery()

#ArterySparseVsDense()

#particleBed()
#particleBedSparseVsDense()

#particleBedBlockSizes()
#emptyChannel()
#scalingBenchmark()
#testGPUComm()
#testCartesianComm()
