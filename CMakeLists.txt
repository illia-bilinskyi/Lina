cmake_minimum_required(VERSION 3.16)

project(Lina LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LINA_BUILD_TESTS "Enable tests for Lina" OFF)

set(LINA_LIBRARY_NAME lina)

add_subdirectory(lina)

if (LINA_BUILD_TESTS)
    message(STATUS "Building tests is enabled.")
    include(CTest)   # Enables testing functionality
    enable_testing() # Required so `ctest` can discover tests

    add_subdirectory(test/)
endif ()
