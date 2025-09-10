add_executable(test_build_lina
        test_build.cpp
)

target_link_libraries(test_build_lina
        ${LINA_LIBRARY_NAME}
)

set_property(TARGET test_build_lina PROPERTY CXX_STANDARD_REQUIRED ON)

# Add benchmark as a test (optional)
add_test(NAME test_build COMMAND test_build_lina)