include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP ON
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# Disable CTest targets from GoogleTest
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(gtest_disable_pthreads OFF CACHE BOOL "" FORCE)

# Or more specifically for GoogleTest CTest integration
set(GTEST_CREATE_SHARED_LIBRARY OFF CACHE BOOL "" FORCE)
set(gtest_build_tests OFF CACHE BOOL "" FORCE)
set(gtest_build_samples OFF CACHE BOOL "" FORCE)
set(gmock_build_tests OFF CACHE BOOL "" FORCE)

set(EXCLUDE_FROM_ALL TRUE)
FetchContent_MakeAvailable(googletest)
set(EXCLUDE_FROM_ALL FALSE)

enable_testing()

set(LINA_TEST_TARGET_NAME test_lina)

add_executable(${LINA_TEST_TARGET_NAME}
        test_matrix.cpp
        test_vector.cpp
        test_matrix_operations.cpp
        test_vector_operations.cpp
        test_vector_matrix_operations.cpp
        test_transforms.cpp
        test_camera.cpp
        test_inverse_matrix.cpp
        test_str_operations.cpp
        test_constexpr_math.cpp
)
target_link_libraries(${LINA_TEST_TARGET_NAME}
        ${LINA_LIBRARY_NAME}
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(${LINA_TEST_TARGET_NAME})