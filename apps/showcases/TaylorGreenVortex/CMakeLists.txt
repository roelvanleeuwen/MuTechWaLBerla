waLBerla_link_files_to_builddir( "*.prm" )
waLBerla_link_files_to_builddir( "*.py" )

waLBerla_generate_target_from_python(NAME TaylorGreenVortexGenerated
        FILE TaylorGreenVortex.py
        OUT_FILES TaylorGreenVortexStorageSpecification.h TaylorGreenVortexStorageSpecification.cpp
        TaylorGreenVortexSweepCollection.h TaylorGreenVortexSweepCollection.cpp
        NoSlip.h NoSlip.cpp
        UBB.h UBB.cpp
        TaylorGreenVortexBoundaryCollection.h
        TaylorGreenVortexHeader.h)

waLBerla_add_executable ( NAME TaylorGreenVortex
        FILES TaylorGreenVortex.cpp
        DEPENDS TaylorGreenVortexGenerated blockforest core field geometry lbm_generated timeloop )
