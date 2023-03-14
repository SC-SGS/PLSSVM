## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################


message(STATUS "Building Python language bindings for PLSSVM.")

## try finding fmt
find_package(pybind11 2.10.3 QUIET)
if (pybind11_FOUND)
    message(STATUS "Found package pybind11.")
else ()
    message(STATUS "Couldn't find package pybind11. Building from source ...")
    set(PLSSVM_pybind11_VERSION v2.10.3)
    # fetch pybind11 library for creating Python bindings
    FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG ${PLSSVM_pybind11_VERSION}
            GIT_SHALLOW TRUE
            QUIET
            )
    FetchContent_MakeAvailable(pybind11)
#    add_dependencies(${PLSSVM_BASE_LIBRARY_NAME} pybind11)
    target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
            $<BUILD_INTERFACE:${pybind11_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
            )
#    # append pybind11 to installed targets if build from source
#    list(APPEND PLSSVM_TARGETS_TO_INSTALL "fmt")
    message(STATUS "Installed pybind11 version ${PLSSVM_pybind11_VERSION}.")
endif ()
#target_link_libraries(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC fmt::fmt)

set(PLSSVM_PYTHON_BINDINGS_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/target_platforms.cpp
        ${CMAKE_CURRENT_LIST_DIR}/backend_types.cpp
        ${CMAKE_CURRENT_LIST_DIR}/file_format_types.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernel_function_types.cpp
        ${CMAKE_CURRENT_LIST_DIR}/parameter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/model.cpp
        ${CMAKE_CURRENT_LIST_DIR}/data_set.cpp
        ${CMAKE_CURRENT_LIST_DIR}/version/version.cpp
        ${CMAKE_CURRENT_LIST_DIR}/exceptions/exceptions.cpp
        ${CMAKE_CURRENT_LIST_DIR}/main.cpp
        )

set(PLSSVM_PYTHON_BINDINGS_LIBRARY_NAME plssvm)
pybind11_add_module(${PLSSVM_PYTHON_BINDINGS_LIBRARY_NAME} ${PLSSVM_PYTHON_BINDINGS_SOURCES})
target_link_libraries(${PLSSVM_PYTHON_BINDINGS_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})
target_compile_definitions(${PLSSVM_PYTHON_BINDINGS_LIBRARY_NAME} PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

find_package(PythonLibs REQUIRED)
target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE ${GPU_SAMPLER_LIBRARY_NAME})

