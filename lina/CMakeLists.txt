add_library(${LINA_LIBRARY_NAME} INTERFACE lina.h)
add_library(lina::lina ALIAS ${LINA_LIBRARY_NAME})

target_include_directories(${LINA_LIBRARY_NAME} INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)