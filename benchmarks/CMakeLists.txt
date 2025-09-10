# Benchmark executable
set(LINA_BENCHMARK_TARGET_NAME benchmark_lina)

add_executable(${LINA_BENCHMARK_TARGET_NAME}
        benchmark_lina.cpp
)

target_link_libraries(${LINA_BENCHMARK_TARGET_NAME}
        ${LINA_LIBRARY_NAME}
)

set_property(TARGET ${LINA_BENCHMARK_TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

# Add benchmark as a test (optional)
add_test(NAME benchmarks COMMAND ${LINA_BENCHMARK_TARGET_NAME})