cmake_minimum_required(VERSION 3.2)
project("Binpack_lib")

set(HEADER_FILES
	bins.hpp
    application.hpp
    instance.hpp

    csv.h # Copied from https://github.com/ben-strasser/fast-cpp-csv-parser
)

set(SOURCE_FILES
	bins.cpp
	application.cpp
	instance.cpp
)

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
