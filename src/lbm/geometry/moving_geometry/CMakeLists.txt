target_sources( lbm
    PRIVATE
    MovingGeometry.h
    PredefinedMovingGeometry.h
)

if(WALBERLA_BUILD_WITH_GPU_SUPPORT)
    target_sources( lbm
            PRIVATE
            MovingGeometry.cu)
else()
    target_sources( lbm
            PRIVATE
            MovingGeometry.cpp)
endif()