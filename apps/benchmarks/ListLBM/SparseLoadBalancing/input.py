import os

import waLBerla as wlb

DB_FILE = os.environ.get('DB_FILE', "ListLBMBenchmark.sqlite3")

class Scenario:
    def __init__(self, cells_per_block=(20, 20, 20), periodic=(False,False,False),
                 timesteps=1001, time_step_strategy="noOverlap", omega=0.8, cuda_enabled_mpi=False,
                 inner_outer_split=(1, 1, 1), vtk_write_frequency=0, inflow_velocity=(0.01, 0, 0),
                 porosity=0.5, porosity_switch=0.8, run_hybrid=True, geometry_setup="randomNoslip",
                 spheres_radius=9, sphere_shift=10, sphere_fill=(1.0, 1.0, 1.0), mesh_file="None", run_boundaries=True, 
                 use_cartesian_communicator=False):

        self.timesteps = timesteps
        self.vtkWriteFrequency = vtk_write_frequency

        self.inflow_velocity = inflow_velocity

        self.cells_per_block = cells_per_block
        self.periodic = periodic
        self.porosity_switch = porosity_switch
        self.porosity = porosity
        self.use_cartesian_communicator = use_cartesian_communicator

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
                'cudaEnabledMPI': self.cuda_enabled_mpi,
                'vtkWriteFrequency': self.vtkWriteFrequency,
                'porositySwitch': self.porosity_switch,
                'runHybrid': self.run_hybrid,
                'porosity': self.porosity,
                'runBoundaries': self.run_boundaries,
                'remainingTimeLoggerFrequency': 10,
                'useCartesian': self.use_cartesian_communicator,


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

def randomNoslip():
    scenarios = wlb.ScenarioManager()
    scenario = Scenario(run_hybrid=False, timesteps=1, porosity=0.9, vtk_write_frequency=1, geometry_setup="randomNoslip")
    scenarios.add(scenario)

randomNoslip()
