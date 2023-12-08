## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "OpenCL:  ")

# check if OpenCL can be enabled
message(CHECK_START "Checking for OpenCL backend")

find_package(OpenCL)

if (NOT OpenCL_FOUND)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_OPENCL_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: OpenCL!")
    endif ()
    return()
endif ()
message(CHECK_PASS "found")


# explicitly set sources
set(PLSSVM_OPENCL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/detail/command_queue.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/context.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/error_code.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/kernel.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
        ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
        ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../gpu_device_ptr.cpp
)

# set target properties
set_local_and_parent(PLSSVM_OPENCL_BACKEND_LIBRARY_NAME plssvm-OpenCL)
add_library(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_OPENCL_SOURCES})
target_include_directories(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC OpenCL)

target_sources(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE
        FILE_SET HEADERS
        BASE_DIRS
            "${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL"
        FILES
            "${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL/cg_explicit/blas.cl"
            "${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL/cg_explicit/kernel_matrix_assembly.cl"
            "${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL/predict_kernel.cl"
)

# link base library against OpenCL library
target_link_libraries(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set OpenCL include path to the *.cl files
target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_OPENCL_KERNEL_SOURCE_DIR="${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL")

# set OpenCL library name (concatenated to the full kernel source string, to make sure that different OpenCL libraries
# are recognized by the sha256 caching algorithm)
target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_OPENCL_LIBRARY="${OpenCL_LIBRARY}")

# set OpenCL target version (to silence compiler warnings)
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC CL_TARGET_OPENCL_VERSION=220)

# set current CMAKE_BUILD_TYPE included in the temp folder where the kernels are compiled to
target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

# needed to link the OpenCL backend to a third-party executable
target_compile_options(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE -fPIE)

# set compile definition that the OpenCL backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_HAS_OPENCL_BACKEND)
target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_OPENCL_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME})

# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)


# generate summary string
set(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_COMPILER " - OpenCL:")
include(${PROJECT_SOURCE_DIR}/cmake/assemble_summary_string.cmake)
assemble_summary_string(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS)
# do not print any special target architecture information
string(REPLACE " (${PLSSVM_CPU_TARGET_ARCHS})" "" PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}")
string(REPLACE " (${PLSSVM_NVIDIA_TARGET_ARCHS})" "" PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}")
string(REPLACE " (${PLSSVM_AMD_TARGET_ARCHS})" "" PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}")
string(REPLACE " (${PLSSVM_INTEL_TARGET_ARCHS})" "" PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}")
set(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_COMPILER}${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}" PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)