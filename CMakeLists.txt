cmake_minimum_required(VERSION 3.15)
project(ModularProject)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
  
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add flags for Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
message(STATUS "Applying Debug build flags")

# g3 = full debug info including macros
add_compile_options(-fsplit-stack -g3 -O0 -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
message(STATUS "Applying Release build flags")

# g3 = full debug info including macros
add_compile_options(-O3 -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
endif()

# Library from src/
file(GLOB_RECURSE LIB_SOURCES ${CMAKE_SOURCE_DIR}/lib/*.cpp ${CMAKE_SOURCE_DIR}/lib/*.hpp)

add_library(gradlib INTERFACE)
target_include_directories(gradlib INTERFACE ${CMAKE_SOURCE_DIR}/lib)

# Find all executables in experiment/
file(GLOB EXPERIMENT_SOURCES experiments/*.cpp)

foreach(EXP_SRC ${EXPERIMENT_SOURCES})
    get_filename_component(EXP_NAME ${EXP_SRC} NAME_WE)
    add_executable(${EXP_NAME} ${EXP_SRC})
    target_link_libraries(${EXP_NAME} PRIVATE gradlib)
endforeach()
