cmake_minimum_required(VERSION 3.16)

project(Lina LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LINA_BUILD_TESTS "Enable tests for Lina" OFF)
option(LINA_BUILD_BENCHMARKS "Enable benchmarks for Lina" OFF)

set(LINA_LIBRARY_NAME lina)

add_subdirectory(lina)

if (LINA_BUILD_TESTS)
    message(STATUS "Building tests is enabled.")

    add_subdirectory(test/test_build/)

    include(CTest)
    enable_testing()

    add_subdirectory(test/)
endif ()

if (LINA_BUILD_BENCHMARKS)
    message(STATUS "Building benchmarks is enabled.")

    add_subdirectory(benchmarks/)
endif ()
