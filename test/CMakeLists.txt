include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP ON
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(TEST_LIBRARY_CMAKE_TARGET_NAME test_${CORE_LIBRARY_CMAKE_TARGET_NAME})

add_executable(${TEST_LIBRARY_CMAKE_TARGET_NAME} test_lina.cpp)
target_link_libraries(${TEST_LIBRARY_CMAKE_TARGET_NAME}
        ${CORE_LIBRARY_CMAKE_TARGET_NAME}
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(${TEST_LIBRARY_CMAKE_TARGET_NAME})