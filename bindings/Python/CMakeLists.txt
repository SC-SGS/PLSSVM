## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(${PROJECT_SOURCE_DIR}/cmake/utility_macros.cmake)

message(STATUS "Building Python language bindings for PLSSVM.")

find_package(Python COMPONENTS Interpreter Development)

## try finding pybind11
find_package(pybind11 2.11.1 QUIET)
if (pybind11_FOUND)
    message(STATUS "Found package pybind11.")
else ()
    message(STATUS "Couldn't find package pybind11. Building from source ...")
    set(PLSSVM_pybind11_VERSION v2.11.1)
    # fetch pybind11 library for creating Python bindings
    FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG ${PLSSVM_pybind11_VERSION}
            GIT_SHALLOW TRUE
            QUIET
    )
    FetchContent_MakeAvailable(pybind11)
    target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
            $<BUILD_INTERFACE:${pybind11_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    message(STATUS "Installed pybind11 version ${PLSSVM_pybind11_VERSION}.")
endif ()

# set source files that are always used
set(PLSSVM_PYTHON_BINDINGS_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/exceptions/exceptions.cpp

        ${CMAKE_CURRENT_LIST_DIR}/detail/performance_tracker.cpp

        ${CMAKE_CURRENT_LIST_DIR}/version/version.cpp

        ${CMAKE_CURRENT_LIST_DIR}/backend_types.cpp
        ${CMAKE_CURRENT_LIST_DIR}/classification_types.cpp
        ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
        ${CMAKE_CURRENT_LIST_DIR}/data_set.cpp
        ${CMAKE_CURRENT_LIST_DIR}/file_format_types.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernel_function_types.cpp
        ${CMAKE_CURRENT_LIST_DIR}/model.cpp
        ${CMAKE_CURRENT_LIST_DIR}/parameter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/solver_types.cpp
        ${CMAKE_CURRENT_LIST_DIR}/target_platforms.cpp
        ${CMAKE_CURRENT_LIST_DIR}/verbosity_levels.cpp

        ${CMAKE_CURRENT_LIST_DIR}/sklearn.cpp

        ${CMAKE_CURRENT_LIST_DIR}/main.cpp
)
# set backends specific source files if the respective backend is used
if (TARGET ${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME})
    list(APPEND PLSSVM_PYTHON_BINDINGS_SOURCES ${CMAKE_CURRENT_LIST_DIR}/backends/openmp_csvm.cpp)
endif ()
if (TARGET ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME})
    list(APPEND PLSSVM_PYTHON_BINDINGS_SOURCES ${CMAKE_CURRENT_LIST_DIR}/backends/cuda_csvm.cpp)
endif ()
if (TARGET ${PLSSVM_HIP_BACKEND_LIBRARY_NAME})
    list(APPEND PLSSVM_PYTHON_BINDINGS_SOURCES ${CMAKE_CURRENT_LIST_DIR}/backends/hip_csvm.cpp)
endif ()
if (TARGET ${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME})
    list(APPEND PLSSVM_PYTHON_BINDINGS_SOURCES ${CMAKE_CURRENT_LIST_DIR}/backends/opencl_csvm.cpp)
endif ()
if (TARGET ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME})
    list(APPEND PLSSVM_PYTHON_BINDINGS_SOURCES ${CMAKE_CURRENT_LIST_DIR}/backends/sycl.cpp)
endif ()
if (TARGET ${PLSSVM_SYCL_BACKEND_ADAPTIVECPP_LIBRARY_NAME})
    list(APPEND PLSSVM_PYTHON_BINDINGS_SOURCES ${CMAKE_CURRENT_LIST_DIR}/backends/adaptivecpp_csvm.cpp)
endif ()
if (TARGET ${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME})
    list(APPEND PLSSVM_PYTHON_BINDINGS_SOURCES ${CMAKE_CURRENT_LIST_DIR}/backends/dpcpp_csvm.cpp)
endif ()

# create pybind11 module
set(PLSSVM_PYTHON_BINDINGS_LIBRARY_NAME plssvm)
pybind11_add_module(${PLSSVM_PYTHON_BINDINGS_LIBRARY_NAME} ${PLSSVM_PYTHON_BINDINGS_SOURCES})

# set default label type
set(PLSSVM_PYTHON_BINDINGS_POSSIBLE_LABEL_TYPE "bool;char;signed char;unsigned char;short;unsigned short;int;unsigned int;long;unsigned long;long long;unsigned long long;float;double;long double;std::string")
set(PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE "std::string" CACHE STRING "The preferred type of the labels for the Python bindings.")
set_property(CACHE PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE PROPERTY STRINGS ${PLSSVM_PYTHON_BINDINGS_POSSIBLE_LABEL_TYPE})
if (NOT "${PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE}" IN_LIST PLSSVM_PYTHON_BINDINGS_POSSIBLE_LABEL_TYPE)
    message(FATAL_ERROR "The provided label_type \"${PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE}\" is not one of the allowed values: \"${PLSSVM_PYTHON_BINDINGS_POSSIBLE_LABEL_TYPE}\"")
endif ()
message(STATUS "The preferred label_type for the Python bindings is \"${PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE}\".")

# add necessary compile definitions for the default real_type and label_type
target_compile_definitions(${PLSSVM_PYTHON_BINDINGS_LIBRARY_NAME} PRIVATE PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE=${PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE})
if (PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE STREQUAL "std::string")
    target_compile_definitions(${PLSSVM_PYTHON_BINDINGS_LIBRARY_NAME} PRIVATE PLSSVM_PYTHON_BINDINGS_LABEL_TYPE_IS_STRING)
endif ()

# add necessary compile options
target_include_directories(${PLSSVM_PYTHON_BINDINGS_LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(${PLSSVM_PYTHON_BINDINGS_LIBRARY_NAME} PRIVATE ${PLSSVM_ALL_LIBRARY_NAME})
target_compile_definitions(${PLSSVM_PYTHON_BINDINGS_LIBRARY_NAME} PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

# disable clang compiler warning
target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wno-self-assign-overloaded>)

# append pybind11 bindings library to installed targets
append_local_and_parent(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_PYTHON_BINDINGS_LIBRARY_NAME})