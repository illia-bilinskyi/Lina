cmake_minimum_required(VERSION 3.16)

project(Lina LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CORE_LIBRARY_CMAKE_TARGET_NAME lina)

add_library(${CORE_LIBRARY_CMAKE_TARGET_NAME} INTERFACE lina.h)
target_include_directories(${CORE_LIBRARY_CMAKE_TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

include(CTest)   # Enables testing functionality
enable_testing() # Required so `ctest` can discover tests

add_subdirectory(test/)
