## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "OpenCL:  ")

message(CHECK_START "Checking for OpenCL platforms")
########################################################################################################################
##                                         parse target platform information                                          ##
########################################################################################################################
if (DEFINED PLSSVM_OPENCL_TARGET_PLATFORMS)
    set(PLSSVM_OPENCL_TARGET_PLATFORMS ${PLSSVM_OPENCL_TARGET_PLATFORMS} CACHE STRING "The OpenCL target platforms to compile for." FORCE)
elseif (DEFINED ENV{PLSSVM_OPENCL_TARGET_PLATFORMS})
    set(PLSSVM_OPENCL_TARGET_PLATFORMS $ENV{PLSSVM_OPENCL_TARGET_PLATFORMS} CACHE STRING "The OpenCL target platforms to compile for." FORCE)
elseif (DEFINED PLSSVM_TARGET_PLATFORMS)
    set(PLSSVM_OPENCL_TARGET_PLATFORMS ${PLSSVM_TARGET_PLATFORMS} CACHE STRING "The OpenCL target platforms to compile for." FORCE)
elseif (DEFINED ENV{PLSSVM_TARGET_PLATFORMS})
    set(PLSSVM_OPENCL_TARGET_PLATFORMS $ENV{PLSSVM_TARGET_PLATFORMS} CACHE STRING "The OpenCL target platforms to compile for." FORCE)
endif ()

## PLSSVM_TARGET_PLATFORMS must not be empty
if (PLSSVM_OPENCL_TARGET_PLATFORMS STREQUAL "")
    message(FATAL_ERROR "Eather PLSSVM_TARGET_PLATFORMS or PLSSVM_OPENCL_TARGET_PLATFORMS must not be empty to compile with the OpenCL backend!")
endif ()

## parse provided target platforms
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/parse_architecture_info.cmake)
set(PLSSVM_PLATFORM_NAME_LIST "automatic")
foreach (PLSSVM_PLATFORM ${PLSSVM_OPENCL_TARGET_PLATFORMS})

    if (PLSSVM_PLATFORM MATCHES "^cpu")
        # parse provided CPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_CPU_TARGET_ARCHS PLSSVM_NUM_CPU_TARGET_ARCHS)
        if (PLSSVM_NUM_CPU_TARGET_ARCHS GREATER 1)
            message(FATAL_ERROR "Target platform \"cpu\" must at most have one architecture specification!")
        endif ()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_CPU_TARGET)
        list(APPEND PLSSVM_PLATFORM_NAME_LIST "cpu")
    elseif (PLSSVM_PLATFORM MATCHES "^nvidia")
        # parse provided NVIDIA GPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_NVIDIA_TARGET_ARCHS PLSSVM_NUM_NVIDIA_TARGET_ARCHS)
        if (PLSSVM_NUM_NVIDIA_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"nvidia\" must at least have one architecture specification!")
        endif ()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_NVIDIA_TARGET)
        list(APPEND PLSSVM_PLATFORM_NAME_LIST "gpu_nvidia")
    elseif (PLSSVM_PLATFORM MATCHES "^amd")
        # parse provided AMD GPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_AMD_TARGET_ARCHS PLSSVM_NUM_AMD_TARGET_ARCHS)
        if (PLSSVM_NUM_AMD_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"amd\" must at least have one architecture specification!")
        endif ()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_AMD_TARGET)
        list(APPEND PLSSVM_PLATFORM_NAME_LIST "gpu_amd")
    elseif (PLSSVM_PLATFORM MATCHES "^intel")
        # parse provided Intel GPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_INTEL_TARGET_ARCHS PLSSVM_NUM_INTEL_TARGET_ARCHS)
        if (PLSSVM_NUM_INTEL_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"intel\" must at least have one architecture specification!")
        endif ()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_INTEL_TARGET)
        list(APPEND PLSSVM_PLATFORM_NAME_LIST "gpu_intel")
    else ()
        message(FATAL_ERROR "Unrecognized target platform \"${PLSSVM_PLATFORM}\"! Must be one of: cpu nvidia amd intel")
    endif ()
endforeach ()

if ((NOT DEFINED PLSSVM_CPU_TARGET_ARCHS) AND (NOT DEfINED PLSSVM_NVIDIA_TARGET_ARCHS) AND (NOT DEFINED PLSSVM_AMD_TARGET_ARCHS) AND (NOT DEFINED PLSSVM_INTEL_TARGET_ARCHS))
    if (PLSSVM_ENABLE_OPENCL_BACKEND MATCHES "ON")
        message(SEND_ERROR "Found requested OpenCL backend, but no targets were specified!")
    else ()
        message(STATUS "Found OpenCL backend, but no targets were specified!")
    endif ()
    message(CHECK_FAIL "skipped")
    return()
endif ()

get_target_property(BASE_PLSSVM_TARGET_PLATFORMS ${PLSSVM_BASE_LIBRARY_NAME} PLSSVM_TARGET_PLATFORMS)
if (NOT BASE_PLSSVM_TARGET_PLATFORMS)
    set_target_properties(${PLSSVM_BASE_LIBRARY_NAME} PROPERTIES PLSSVM_TARGET_PLATFORMS "${PLSSVM_PLATFORM}")
else ()
    set_target_properties(${PLSSVM_BASE_LIBRARY_NAME} PROPERTIES PLSSVM_TARGET_PLATFORMS "${BASE_PLSSVM_TARGET_PLATFORMS} - OpenCL: ${PLSSVM_OPENCL_TARGET_PLATFORMS}")
endif ()

# generate summary string
set(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_COMPILER " - OpenCL:")
include(${PROJECT_SOURCE_DIR}/cmake/assemble_summary_string.cmake)
assemble_summary_string(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS)
# do not print any special target architecture information
string(REPLACE " (${PLSSVM_CPU_TARGET_ARCHS})" "" PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}")
string(REPLACE " (${PLSSVM_NVIDIA_TARGET_ARCHS})" "" PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}")
string(REPLACE " (${PLSSVM_AMD_TARGET_ARCHS})" "" PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}")
string(REPLACE " (${PLSSVM_INTEL_TARGET_ARCHS})" "" PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}")
set(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_COMPILER}${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}")
set(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING}" PARENT_SCOPE)

message(CHECK_PASS "found " ${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING})

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
    ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
)

# set target properties
set_local_and_parent(PLSSVM_OPENCL_BACKEND_LIBRARY_NAME plssvm-OpenCL)
add_library(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_OPENCL_SOURCES})
target_include_directories(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC OpenCL)

target_sources(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE
               FILE_SET HEADERS
               BASE_DIRS
               "${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL/kernel"
               FILES
               "${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL/kernel/cg_explicit/blas.cl"
               "${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL/kernel/cg_explicit/kernel_matrix_assembly.cl"
               "${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL/kernel/cg_implicit/kernel_matrix_assembly_blas.cl"
               "${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL/kernel/kernel_functions.cl"
               "${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL/kernel/predict_kernel.cl"
               "${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL/kernel/predict_kernel_linear.cl"
)

# link base library against OpenCL library
target_link_libraries(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set OpenCL include path to the *.cl files
set(PLSSVM_OPENCL_KERNEL_SOURCE_DIRECTORY "${PROJECT_SOURCE_DIR}/include/plssvm/backends/OpenCL/kernel")
target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_OPENCL_KERNEL_SOURCE_DIR="${PLSSVM_OPENCL_KERNEL_SOURCE_DIRECTORY}")

# set OpenCL library name (concatenated to the full kernel source string, to make sure that different OpenCL libraries
# are recognized by the sha256 caching algorithm)
target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_OPENCL_LIBRARY="${OpenCL_LIBRARY}")

# option only available if NVIDIA GPUs are requested as target platforms
if (PLSSVM_NUM_NVIDIA_TARGET_ARCHS GREATER 0)
    # add option to completely disable PTX inline assembly
    option(PLSSVM_OPENCL_BACKEND_ENABLE_PTX_INLINE_ASSEMBLY "Enables the usage of PTX inline assembly on NVIDIA GPUs to greatly improve the atomicAdd performance! Only available for sm greater or equal than sm_60." ON)
    if (PLSSVM_OPENCL_BACKEND_ENABLE_PTX_INLINE_ASSEMBLY)
        # test whether the requested sm version is new enough
        list(TRANSFORM PLSSVM_NVIDIA_TARGET_ARCHS REPLACE "sm_" "")
        foreach (sm_value IN LISTS PLSSVM_NVIDIA_TARGET_ARCHS)
            if (${sm_value} LESS 60)
                message(FATAL_ERROR "Requested PTX inline assembly to speed up the FP32/FP64 atomicAdd, but the supplied sm_${sm_value} is too old!")
            endif ()
        endforeach ()

        message(STATUS "Enabled PTX inline assembly on NVIDIA GPUs to greatly improve the atomicAdd performance.")
        target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_OPENCL_BACKEND_USE_PTX_INLINE_ASSEMBLY)
    endif ()
endif ()

# set OpenCL target version (to silence compiler warnings)
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC CL_TARGET_OPENCL_VERSION=220)

# set current CMAKE_BUILD_TYPE included in the temp folder where the kernels are compiled to
target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

# set whether the kernel source should be compiled with fast math enabled or not
if (PLSSVM_ENABLE_FAST_MATH)
    target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_ENABLE_FAST_MATH)
endif ()

# needed to link the OpenCL backend to a third-party executable
target_compile_options(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE -fPIE)

# set compile definition that the OpenCL backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_HAS_OPENCL_BACKEND)
target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_OPENCL_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME})

# mark backend library as install target
append_local_and_parent(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME})



list(POP_BACK CMAKE_MESSAGE_INDENT)