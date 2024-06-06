## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "OpenMP:  ")

message(CHECK_START "Checking for OpenMP platforms")
########################################################################################################################
##                                         parse target platform information                                          ##
########################################################################################################################
if (DEFINED PLSSVM_OPENMP_TARGET_PLATFORMS)
    set(PLSSVM_OPENMP_TARGET_PLATFORMS ${PLSSVM_OPENMP_TARGET_PLATFORMS} CACHE STRING "The OpenMP target platforms to compile for." FORCE)
elseif (DEFINED ENV{PLSSVM_OPENMP_TARGET_PLATFORMS})
    set(PLSSVM_OPENMP_TARGET_PLATFORMS $ENV{PLSSVM_OPENMP_TARGET_PLATFORMS} CACHE STRING "The OpenMP target platforms to compile for." FORCE)
elseif (DEFINED PLSSVM_TARGET_PLATFORMS)
    set(PLSSVM_OPENMP_TARGET_PLATFORMS ${PLSSVM_TARGET_PLATFORMS} CACHE STRING "The OpenMP target platforms to compile for." FORCE)
elseif (DEFINED ENV{PLSSVM_TARGET_PLATFORMS})
    set(PLSSVM_OPENMP_TARGET_PLATFORMS $ENV{PLSSVM_TARGET_PLATFORMS} CACHE STRING "The OpenMP target platforms to compile for." FORCE)
endif ()



## parse provided target platforms
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/parse_architecture_info.cmake)
set(PLSSVM_PLATFORM_NAME_LIST "automatic")
foreach (PLSSVM_PLATFORM ${PLSSVM_OPENMP_TARGET_PLATFORMS})
    if (PLSSVM_PLATFORM MATCHES "^cpu")
        # parse provided CPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_CPU_TARGET_ARCHS PLSSVM_NUM_CPU_TARGET_ARCHS)
        if (PLSSVM_NUM_CPU_TARGET_ARCHS GREATER 1)
            message(FATAL_ERROR "Target platform \"cpu\" must at most have one architecture specification!")
        endif ()
        list(APPEND PLSSVM_PLATFORM_NAME_LIST "cpu")
    elseif (PLSSVM_PLATFORM MATCHES "^nvidia")
        # parse provided NVIDIA GPU architectures
        message(WARNING "The OpenMP backend currently does not support offloading and therefore the target platform \"nvidia\", ignoring it!")
    elseif (PLSSVM_PLATFORM MATCHES "^amd")
        # parse provided AMD GPU architectures
        message(WARNING "The OpenMP backend currently does not support offloading and therefore the target platform \"amd\", ignoring it!")
    elseif (PLSSVM_PLATFORM MATCHES "^intel")
        # parse provided Intel GPU architectures
        message(WARNING "The hip backend currently does not support offloading and therefore the target platform \"intel\", ignoring it!")
    else ()
        message(FATAL_ERROR "Unrecognized target platform \"${PLSSVM_PLATFORM}\"! Must be one of: cpu nvidia amd intel")
    endif ()
endforeach ()

## PLSSVM_TARGET_PLATFORMS must not be empty
if (PLSSVM_OMP_TARGET_PLATFORMS STREQUAL "")
    message(FATAL_ERROR "Eather PLSSVM_TARGET_PLATFORMS or PLSSVM_OMP_TARGET_PLATFORMS must not be empty to compile with the OpenMP backend!")
endif ()

if ((NOT DEFINED PLSSVM_CPU_TARGET_ARCHS) AND ( NOT "cpu" IN_LIST PLSSVM_PLATFORM_NAME_LIST))
    if (PLSSVM_ENABLE_OPENMP_BACKEND MATCHES "ON")
        message(SEND_ERROR "Found requested OpenMP backend, but no \"cpu\" targets were specified!")
    else ()
        message(STATUS "Found OpenMP backend, but no \"cpu\" targets were specified!")
    endif ()
    message(CHECK_FAIL "skipped")
    return()
endif ()
message(CHECK_PASS "found " ${PLSSVM_CPU_TARGET_ARCHS})

# check if OpenMP can be enabled
message(CHECK_START "Checking for OpenMP backend")

find_package(OpenMP)

if (NOT OPENMP_FOUND)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_OPENMP_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: OpenMP!")
    endif ()
    return()
endif ()
message(CHECK_PASS "found ")

# explicitly set sources
set(PLSSVM_OPENMP_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
)

# set target properties
set_local_and_parent(PLSSVM_OPENMP_BACKEND_LIBRARY_NAME plssvm-OpenMP)
add_library(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_OPENMP_SOURCES})
target_compile_definitions(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_CPU_TARGET)

target_link_libraries(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:OpenMP::OpenMP_CXX>)
# special command line options for MSVC:
# -openmp:llvm -> enables unsigned loop indexes in OpenMP parallel for loops
# -openmp:experimental -> enables OpenMP's SIMD instructions
target_compile_options(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:-openmp:llvm -openmp:experimental>)

# additional compilation flags
target_compile_options(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wconversion>)

# work-around since hipcc doesn't define _OPENMP
if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    # hipcc must be clang
    execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --version
            OUTPUT_VARIABLE PLSSVM_CLANG_VERSION_OUTPUT)
    # check if version message contains hipcc specific values
    if ("${PLSSVM_CLANG_VERSION_OUTPUT}" MATCHES ".*HIP.*" AND "${PLSSVM_CLANG_VERSION_OUTPUT}" MATCHES ".*AMD clang.*")
        message(STATUS "Add compiler options \"-x c++\" as workaround for hipcc (https://docs.amd.com/projects/HIP/en/docs-5.2.0/user_guide/faq.html#why-openmp-is-undefined-when-compiling-with-fopenmp).")
        target_compile_options(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} PRIVATE -x c++)
    endif ()
endif ()

# link base library against OpenMP library
target_link_libraries(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set compile definition that the OpenMP backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_OPENMP_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME})

# mark backend library as install target
append_local_and_parent(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME})

# generate summary string
set(PLSSVM_OPENMP_BACKEND_SUMMARY_STRING " - OpenMP: cpu " PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)