waLBerla_link_files_to_builddir( parameter_files )

waLBerla_generate_target_from_python(
        NAME 01_GeneratedLbmLib
        FILE 01_GenerateLbm.py
        OUT_FILES
        GeneratedLbmStorageSpecification.cpp GeneratedLbmStorageSpecification.h
        GeneratedLbmSweepCollection.cpp GeneratedLbmSweepCollection.h
        GeneratedLbmBoundaryCollection.h
        NoSlip.h NoSlip.cpp
        FreeSlip.h FreeSlip.cpp
        UBB.h UBB.cpp
        OutflowD3Q19.h OutflowD3Q19.cpp
)

waLBerla_add_executable( NAME 01_GeneratedLbmExe
        FILES 01_GeneratedLbmBasics.cpp
        DEPENDS 01_GeneratedLbmLib blockforest core field geometry timeloop lbm_generated vtk )

waLBerla_add_executable( NAME 02_GridGenerator
        FILES 02_GridGenerator.cpp
        DEPENDS blockforest core geometry vtk python_coupling)

waLBerla_add_executable( NAME 03_NonuniformGrid
        FILES 03_NonuniformGrid.cpp
        DEPENDS 01_GeneratedLbmLib blockforest core field geometry timeloop lbm_generated vtk )
