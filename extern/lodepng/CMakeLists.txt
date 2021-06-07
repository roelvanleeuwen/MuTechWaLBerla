add_library(
      lodepng
      lodepng.cpp
      lodepng.h
)

target_include_directories(lodepng PUBLIC SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(lodepng PRIVATE -Wno-conversion)