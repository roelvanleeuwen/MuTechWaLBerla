# Tutorials and Demo Applications using the new-style LBM backend

## Introduction

### The lbm_generated Backend

Recently, a new module and backend for lattice Boltzmann simulations has been added to the main branch of waLBerla. 
The module `lbm_generated` is strongly based on code generation using `lbmpy` and is intended to replace the old `lbm` module.
Any new projects using *waLBerla* for LBM simulations are encouraged to utilize the new LBM backend.

This directory contains a number of simple applications utilizing this backend to demonstrate its usage.
It is still work in progress; do not hesitate to contact me with questions or suggestions
([frederik.hennig@fau.de](mailto:frederik.hennig@fau.de)).

The fundamental idea behind the `generated_lbm` package is that most functional components are automatically generated
from Python scripts utilizing our packages
[pystencils](https://pypi.org/project/pystencils/) and [lbmpy](https://pypi.org/project/lbmpy/).

The basic components being generated are the following:

 - The *Lattice Storage Specification* defines the storage format of the population field. This includes the *Stencil*,
   *streaming pattern*, *zero-centered storage*, and the rules for *ghost-layer exchange*.
 - The *Sweep Collection* encompasses all LBM bulk algorithms. These are the (fused) *streaming* and *collision* steps,
   as well as the *initialization* of the PDF field from an initial flow state, and the output of density and velocity
   from the populations.
 - *Boundary conditions* can be generated on their own, or assembled into a *boundary collection*.

All implementations of the collision step, as well as all boundary conditions implemented in *lbmpy* are thus available
for use within *waLBerla*.

Usually, a user would write a single Python script to generate all of these components.
For getting started, however, the `lbm_generated` package contains a set of pre-generated algorithms that may instead
be used:

 - D3Q19, Incompressible SRT, Pull-Pattern:
   - Storage Specification: `src/lbm_generated/storage_specification/D3Q19StorageSpecification.h`
   - Sweep Collection: `src/lbm_generated/sweep_collection/D3Q19SRT.h`
   - Boundary Collection: `src/lbm_generated/boundary/D3Q19BoundaryCollection.h`
 - D3Q27, Incompressible SRT, Pull-Pattern:
     - Storage Specification: `src/lbm_generated/storage_specification/D3Q27StorageSpecification.h`
     - Sweep Collection: `src/lbm_generated/sweep_collection/D3Q27SRT.h`
     - Boundary Collection: `src/lbm_generated/boundary/D3Q27BoundaryCollection.h`

## Build System Setup

For these tutorials, the *waLBerla* build system has to be generated with the following cache variables set:
- `WALBERLA_BUILD_WITH_PYTHON=ON` (for Python parameter files)
- `WALBERLA_BUILD_WITH_CODEGEN=ON` (for the code generator)

Furthermore, the `Python_ROOT_DIR` cache variable should point to a (ideally, virtual) Python environment where
*pystencils* and *lbmpy* are installed.

It is assumed that the CMake build tree was created in the directory `walberla/build`.    

## Tutorial 01: Basic Usage of generated_lbm

This tutorial demonstrates the usage of the `generated_lbm` backend and the `lbmpy`-based code generator for
building a simulation application using uniform grids.

### Files

 - Code Generation: `01_GenerateLBM.py`
 - Application: `01_GeneratedLbmBasics.cpp`
 - Parameters: `parameters/Box.prm`

### Compilation and Execution

**Compilation:**

```shell
$ cd build
$ cmake --build . --target 01_GeneratedLbmExe
```

**Execution:**

```shell
$ cd build/apps/tutorials/lbm_generated
$ ./01_GeneratedLbmExe parameters/Box.prm 
```

**Output:**

 - VTK: The velocity field is output to `vtk_out/vtk.pvd` for visualization in ParaView.

## Tutorial 02: Generating Grids

This tutorial demonstrates how one might generate a locally refined *waLBerla* simulation grid for more complex
simulations.

While it is possible to generate the grid and run the simulation together the same application, especially for large,
massively parallel runs, this is not recommended. The grid generation would run redundantly on each process, which
not only wastes resources, but has in the past lead to applications running out of memory during the grid generation phase
at large scales (>500 nodes, >20,000 cores). 

### Files

- Application: `02_GridGenerator.cpp`
- Parameters: `parameters/Couette.prm`

### Compilation and Execution

**Compilation:**

```shell
$ cd build
$ cmake --build . --target 02_GridGenerator
```

**Execution:**

```shell
$ cd build/apps/tutorials/lbm_generated
$ ./02_GridGenerator parameters/Couette.prm 
```

**Output:**

The generated grid is stored as a blockforest file in `couette.bfs`.
Its structure is also written in VTK format to `couette.vtk`, for visualization in ParaView.

## Tutorial 03: Simulations on Locally Refined Grids

This demo application shows how to set up and run a simulation on a pre-generated locally refined grid.
It differs from tutorial 01 in the following regards:

 - The block forest, which manages the grid, is not created within the application but imported from the blockforest
   file generated in tutorial 02;
 - The MPI communication is set up using the special nonuniform communication scheme provided by the `lbm_generated`
   module, which at the same time facilitates interpolation
 - The time loop is populated using `LBMMeshRefinement.addRefinementToTimeLoop`, which sets up the recursive time
   stepping scheme of the spatially and temporally refined LBM.

### Files

- Code Generation: `03_NonuniformLbm.py`
- Application: `03_NonuniformGrid.cpp`
- Parameters: `parameters/Couette.prm`

### Compilation and Execution

**Compilation:**

```shell
$ cd build
$ cmake --build . --target 03_NonuniformGrid
```

**Execution:**

```shell
$ cd build/apps/tutorials/lbm_generated
$ ./03_NonuniformGrid parameters/Couette.prm 
```

**Output:**

- VTK: The velocity field is output to `vtk_out/vtk.pvd` for visualization in ParaView.