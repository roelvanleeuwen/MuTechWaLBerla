waLBerla_link_files_to_builddir( "*.prm" )
waLBerla_link_files_to_builddir( "*.py" )

waLBerla_generate_target_from_python(NAME ChannelGenerated
        FILE Channel.py
        OUT_FILES ChannelStorageSpecification.h ChannelStorageSpecification.cpp
        ChannelSweepCollection.h ChannelSweepCollection.cpp
        NoSlip.h NoSlip.cpp
        UBB.h UBB.cpp
        ChannelBoundaryCollection.h
        ChannelHeader.h)

waLBerla_add_executable ( NAME Channel
        FILES Channel.cpp
        DEPENDS ChannelGenerated blockforest core field geometry lbm_generated timeloop )
