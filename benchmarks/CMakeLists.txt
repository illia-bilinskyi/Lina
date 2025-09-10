# Fetch external libraries for comparison
include(FetchContent)

# GLM library
FetchContent_Declare(
        glm
        URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
        DOWNLOAD_EXTRACT_TIMESTAMP ON
)

# Eigen library  
FetchContent_Declare(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP ON
)

FetchContent_MakeAvailable(glm eigen)

# Benchmark executable
set(LINA_BENCHMARK_TARGET_NAME benchmark_lina)

add_executable(${LINA_BENCHMARK_TARGET_NAME}
        benchmark_lina.cpp
        benchmark_comparison.cpp
)

target_link_libraries(${LINA_BENCHMARK_TARGET_NAME}
        ${LINA_LIBRARY_NAME}
        glm::glm
        Eigen3::Eigen
)

target_include_directories(${LINA_BENCHMARK_TARGET_NAME} PRIVATE
        ${glm_SOURCE_DIR}
        ${eigen_SOURCE_DIR}
)

set_property(TARGET ${LINA_BENCHMARK_TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

# Add benchmark as a test (optional)
add_test(NAME benchmarks COMMAND ${LINA_BENCHMARK_TARGET_NAME})