add_library(
      lodepng
      lodepng.cpp
      lodepng.h
)

target_include_directories(lodepng PUBLIC SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}")
if (MSVC)
   target_compile_options( lodepng PRIVATE /W )
else ()
   target_compile_options( lodepng PRIVATE -W )
endif ()