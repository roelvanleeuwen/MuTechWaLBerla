import os

import waLBerla as wlb

DB_FILE = os.environ.get('DB_FILE', "ListLBMBenchmark.sqlite3")

class Scenario:
    def __init__(self, cells_per_block=(64, 64, 20),
                 timesteps=1000, time_step_strategy="noOverlap", omega=0.8, cuda_enabled_mpi=True,
                 inner_outer_split=(1, 1, 1), vtk_write_frequency=0, inflow_velocity=(0.01,0,0),
                 porosity=0.5, porositySwitch=0.8, run_hybrid = True, geometry_setup="randomNoslip",
                 spheres_radius=9, sphere_shift = 10, sphere_fill = (1.0, 1.0, 1.0), mesh_file="None", run_boundaries=True):

        self.timesteps = timesteps
        self.vtkWriteFrequency = vtk_write_frequency

        self.inflow_velocity = inflow_velocity

        self.cells_per_block = cells_per_block
        self.porositySwitch = porositySwitch
        self.porosity = porosity

        self.inner_outer_split = inner_outer_split
        self.time_step_strategy = time_step_strategy
        self.cuda_enabled_mpi = cuda_enabled_mpi
        self.run_boundaries = run_boundaries
        self.run_hybrid = run_hybrid
        self.omega = omega

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
                'cudaEnabledMPI': self.cuda_enabled_mpi,
                'vtkWriteFrequency': self.vtkWriteFrequency,
                'porositySwitch': self.porositySwitch,
                'runHybrid': self.run_hybrid,
                'porosity': self.porosity,
                'runBoundaries': self.run_boundaries,
                'remainingTimeLoggerFrequency': 10,


                'SpheresRadius': self.spheres_radius,
                'SphereShift': self.sphere_shift,
                'SphereFillDomainRatio':self.sphere_fill,


        },
            'Boundaries': {
                'Border': [
                    {'direction': 'N', 'walldistance': -1, 'flag': 'NoSlip'},
                    {'direction': 'S', 'walldistance': -1, 'flag': 'NoSlip'},
                    {'direction': 'W', 'walldistance': -1, 'flag': 'UBB'},
                    {'direction': 'E', 'walldistance': -1, 'flag': 'PressureOutflow'},
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
        scenario = Scenario(porosity=porosity, geometry_setup="randomNoslip", inflow_velocity=(0,0,0), run_boundaries=False)
        scenarios.add(scenario)

def randomNoslip():
    scenarios = wlb.ScenarioManager()
    scenario = Scenario(porosity=0.9, vtk_write_frequency=50, geometry_setup="randomNoslip", inflow_velocity=(0,0,0))
    scenarios.add(scenario)

def spheres():
    scenarios = wlb.ScenarioManager()
    spheres_radius = 7
    sphere_shift = 3
    sphere_fill = (0.55, 1.0, 1.0)
    scenario = Scenario(vtk_write_frequency=50, geometry_setup="spheres", spheres_radius=spheres_radius,
                        sphere_shift=sphere_shift, sphere_fill=sphere_fill, porositySwitch=0.75, cells_per_block=(20, 20, 20), timesteps=1000)
    scenarios.add(scenario)

def Artery():
    scenarios = wlb.ScenarioManager()
    mesh_file = "Artery.obj"
    scenario = Scenario(vtk_write_frequency=0, geometry_setup="artery", mesh_file=mesh_file, timesteps=1000, omega=1.9, cells_per_block=(50, 50, 50), porositySwitch=0.8, run_hybrid=True, time_step_strategy="noOverlap", run_boundaries=True)
    scenarios.add(scenario)

def particleBed():
    scenarios = wlb.ScenarioManager()
    scenario = Scenario(geometry_setup="particleBed", vtk_write_frequency=100, timesteps=1000, omega=1.9, cells_per_block=(50, 100, 50), porositySwitch=0.8)
    scenarios.add(scenario)

def emptyChannel():
    scenarios = wlb.ScenarioManager()
    scenario = Scenario(porosity=1.0, vtk_write_frequency=0, geometry_setup="randomNoslip", inflow_velocity=(0.01, 0, 0), omega=1.9, porositySwitch=1.1, run_hybrid=True,cells_per_block=(100, 100, 100),
                        time_step_strategy="noOverlap", cuda_enabled_mpi=True)
    scenarios.add(scenario)

def scalingBenchmark():
    cells_per_block=(320, 320, 320)
    scenarios = wlb.ScenarioManager()
    scenario = Scenario(cells_per_block=cells_per_block, geometry_setup="randomNoslip", porosity=0.8, porositySwitch=0.0, run_hybrid=True, time_step_strategy="Overlap", inner_outer_split=(1, 1, 1) ,run_boundaries=True)
    scenarios.add(scenario)
    scenario = Scenario(cells_per_block=cells_per_block, geometry_setup="randomNoslip", porosity=0.8, porositySwitch=0.0, run_hybrid=True, time_step_strategy="noOverlap", inner_outer_split=(0, 0, 0), run_boundaries=True)
    scenarios.add(scenario)
    scenario = Scenario(cells_per_block=cells_per_block, geometry_setup="randomNoslip", porosity=0.8, porositySwitch=0.0, run_hybrid=True, time_step_strategy="Overlap", inner_outer_split=(32, 1, 1) ,run_boundaries=True)
    scenarios.add(scenario)

#randomNoslip()
#spheres()
#Artery()
#particleBed()
#emptyChannel()
scalingBenchmark()
