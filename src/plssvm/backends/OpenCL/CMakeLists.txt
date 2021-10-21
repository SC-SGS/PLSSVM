## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# check if OpenCL can be enabled
message(CHECK_START "Checking for OpenCL backend")

find_package(OpenCL)

if(NOT OpenCL_FOUND)
    message(CHECK_FAIL "not found")
    if(PLSSVM_ENABLE_OPENCL_BACKEND MATCHES "ON")
        message(FATAL_ERROR "Cannot find requested backend: OpenCL!")
    endif()
    return()
endif()
message(CHECK_PASS "found")


# explicitly set sources
set(PLSSVM_OPENCL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/command_queue.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/error_code.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_csvm.cpp
)

# set target properties
set_local_and_parent(PLSSVM_OPENCL_BACKEND_LIBRARY_NAME svm-OpenCL)
add_library(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} STATIC ${PLSSVM_OPENCL_SOURCES})
target_include_directories(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC OpenCL)

# link base library against OpenCL library
target_link_libraries(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set kernel file directory
#get_target_property(PLSSVM_BASE_LIBRARY_NAME_INCLUDE_DIR ${PLSSVM_BASE_LIBRARY_NAME} INCLUDE_DIRECTORIES)
target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY="${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL/")

# set OpenCL target version
target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC CL_TARGET_OPENCL_VERSION=220)

# set compile definition that the OpenCL backend is available
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
set(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_COMPILER}${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}" PARENT_SCOPE)