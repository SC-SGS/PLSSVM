## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# TODO: only tested on AMD GPUs -> test on NVIDIA GPUs

list(APPEND CMAKE_MESSAGE_INDENT "HIP:  ")

message(CHECK_START "Checking for HIP platforms")
########################################################################################################################
##                                         parse target platform information                                          ##
########################################################################################################################
if (DEFINED PLSSVM_HIP_TARGET_PLATFORMS)
    set(PLSSVM_HIP_TARGET_PLATFORMS ${PLSSVM_HIP_TARGET_PLATFORMS} CACHE STRING "The hip target platforms to compile for." FORCE)
elseif (DEFINED ENV{PLSSVM_HIP_TARGET_PLATFORMS})
    set(PLSSVM_HIP_TARGET_PLATFORMS $ENV{PLSSVM_HIP_TARGET_PLATFORMS} CACHE STRING "The hip target platforms to compile for." FORCE)
elseif (DEFINED PLSSVM_TARGET_PLATFORMS)
    set(PLSSVM_HIP_TARGET_PLATFORMS ${PLSSVM_TARGET_PLATFORMS} CACHE STRING "The hip target platforms to compile for." FORCE)
elseif (DEFINED ENV{PLSSVM_TARGET_PLATFORMS})
    set(PLSSVM_HIP_TARGET_PLATFORMS $ENV{PLSSVM_TARGET_PLATFORMS} CACHE STRING "The hip target platforms to compile for." FORCE)
endif ()

## PLSSVM_TARGET_PLATFORMS must not be empty
if (PLSSVM_HIP_TARGET_PLATFORMS STREQUAL "")
    message(FATAL_ERROR "Eather PLSSVM_TARGET_PLATFORMS or PLSSVM_HIP_TARGET_PLATFORMS must not be empty to compile with the hip backend!")
endif ()

## parse provided target platforms
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/parse_architecture_info.cmake)
foreach (PLSSVM_PLATFORM ${PLSSVM_HIP_TARGET_PLATFORMS})
    if (PLSSVM_PLATFORM MATCHES "^cpu")
        message(WARNING "The hip backend currently does not support the target platform \"cpu\", ignoring it!")
    elseif (PLSSVM_PLATFORM MATCHES "^nvidia")
        # parse provided NVIDIA GPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_NVIDIA_TARGET_ARCHS PLSSVM_NUM_NVIDIA_TARGET_ARCHS)
        if (PLSSVM_NUM_NVIDIA_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"nvidia\" must at least have one architecture specification!")
        endif ()
    elseif (PLSSVM_PLATFORM MATCHES "^amd")
        # parse provided AMD GPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_AMD_TARGET_ARCHS PLSSVM_NUM_AMD_TARGET_ARCHS)
        if (PLSSVM_NUM_AMD_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"amd\" must at least have one architecture specification!")
        endif ()
    elseif (PLSSVM_PLATFORM MATCHES "^intel")
        # parse provided Intel GPU architectures
        message(WARNING "The hip backend currently does not support the target platform \"intel\", ignoring it!")
    else ()
        message(FATAL_ERROR "Unrecognized target platform \"${PLSSVM_PLATFORM}\"! Must be one of: cpu nvidia amd intel")
    endif ()
endforeach ()

if (NOT DEFINED PLSSVM_AMD_TARGET_ARCHS AND NOT DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
    if (PLSSVM_ENABLE_HIP_BACKEND MATCHES "ON")
        message(SEND_ERROR "Found requested HIP backend, but no \"amd\" or \"nvidia\"targets were specified!")
    else ()
        message(STATUS "Found HIP backend, but no \"amd\" or \"nvidia\" targets were specified!")
    endif ()
    message(CHECK_FAIL "skipped")
    return()
endif ()


# generate summary string
if (DEFINED PLSSVM_AMD_TARGET_ARCHS)
    set(PLSSVM_HIP_BACKEND_SUMMARY_STRING " - HIP: amd (${PLSSVM_AMD_TARGET_ARCHS})")
endif ()
if (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
    set(PLSSVM_HIP_BACKEND_SUMMARY_STRING "${PLSSVM_HIP_BACKEND_SUMMARY_STRING} - HIP: nvidia (${PLSSVM_NVIDIA_TARGET_ARCHS})")
endif ()
set(PLSSVM_HIP_BACKEND_SUMMARY_STRING "${PLSSVM_HIP_BACKEND_SUMMARY_STRING}" PARENT_SCOPE)

message(CHECK_PASS "found " ${PLSSVM_HIP_BACKEND_SUMMARY_STRING})

# check if HIP can be enabled
message(CHECK_START "Checking for HIP backend")

include(CheckLanguage)
check_language(HIP)



if (NOT CMAKE_HIP_COMPILER)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_HIP_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: HIP!")
    endif ()
    return()
endif ()
message(CHECK_PASS "found")

enable_language(HIP)
find_package(HIP REQUIRED)

## set necessary flags
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -std=c++17 -Wall -Wextra -Wdouble-promotion -Wshadow -Wcast-qual -Wnull-dereference -Wextra-semi -Wunreachable-code -Wuninitialized -Wmost -Wconversion")
# set fast-math if requested
if (PLSSVM_ENABLE_FAST_MATH)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -ffast-math")
endif ()
set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "${CMAKE_HIP_FLAGS_RELWITHDEBINFO} -O2")
set(CMAKE_HIP_FLAGS_RELEASE "${CMAKE_HIP_FLAGS_RELEASE} -O3")

# explicitly set sources
set(PLSSVM_HIP_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.hip
    ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.hip
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.hip
    ${CMAKE_CURRENT_LIST_DIR}/csvm.hip
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.hip
)

# set target properties
set_local_and_parent(PLSSVM_HIP_BACKEND_LIBRARY_NAME plssvm-HIP)
add_library(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_HIP_SOURCES})

target_compile_definitions(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_AMD_TARGET)

# https://gitlab.kitware.com/cmake/cmake/-/issues/23210
#set_target_properties(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTIES HIP_SEPARABLE_COMPILATION ON)

target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC -Wno-sign-compare)

# set target architecture
set_property(TARGET ${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTY HIP_ARCHITECTURES ${PLSSVM_AMD_TARGET_ARCHS} ${PLSSVM_NVIDIA_TARGET_ARCHS})

# link base library against HIP library
target_link_libraries(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PRIVATE hip::device hip::host)
set_target_properties(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE HIP)
target_link_libraries(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set compile definition that the HIP backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_HAS_HIP_BACKEND)
target_compile_definitions(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_HIP_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_HIP_BACKEND_LIBRARY_NAME})

# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_HIP_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)

set_target_properties(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE HIP)

