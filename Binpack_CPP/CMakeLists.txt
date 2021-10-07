cmake_minimum_required(VERSION 3.2)
project("Binpack" CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")# -lpthread")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# Get the binpack lib
add_subdirectory(src/Binpack_lib)

# Get the bin packing executables
add_subdirectory(src)

