cmake_minimum_required(VERSION 3.2)
project(executables)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# expes for 2D fixed requirement and higher density
add_executable(main_density2D main_density2D.cpp lower_bounds.hpp algos/algos2D.hpp lower_bounds.cpp algos/algos2D.cpp)
target_link_libraries(main_density2D PRIVATE Binpack_lib)

# expes for TS varying requirement and higher density
add_executable(main_densityTS main_densityTS.cpp lower_bounds.hpp algos/algosTS.hpp lower_bounds.cpp algos/algosTS.cpp)
target_link_libraries(main_densityTS PRIVATE Binpack_lib)


# expes for 2D fixed requirement and large scale
add_executable(main_large2D main_large2D.cpp lower_bounds.hpp algos/algos2D.hpp lower_bounds.cpp algos/algos2D.cpp)
target_link_libraries(main_large2D PRIVATE Binpack_lib)

# expes for TS varying requirement and large scale
add_executable(main_largeTS main_largeTS.cpp lower_bounds.hpp algos/algosTS.hpp lower_bounds.cpp algos/algosTS.cpp)
target_link_libraries(main_largeTS PRIVATE Binpack_lib)

# expes for TS scalability
add_executable(main_scalabilityTS main_scalabilityTS.cpp lower_bounds.hpp algos/algosTS.hpp lower_bounds.cpp algos/algosTS.cpp)
target_link_libraries(main_scalabilityTS PRIVATE Binpack_lib)

# expes for TS density with batches of apps
# add_executable(main_batchTS main_batchTS.cpp lower_bounds.hpp algos/algosTS.hpp lower_bounds.cpp algos/algosTS.cpp)
# target_link_libraries(main_batchTS PRIVATE Binpack_lib)

install(TARGETS main_density2D main_densityTS
    main_large2D main_largeTS main_scalabilityTS
    RUNTIME DESTINATION bin)