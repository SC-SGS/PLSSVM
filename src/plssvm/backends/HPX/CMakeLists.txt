# Authors: Alexander Van Craen, Marcel Breyer, Alexander Strack
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "HPX:  ")


message(CHECK_START "Checking for HPX platforms")
########################################################################################################################
##                                         parse target platform information                                          ##
########################################################################################################################
if (DEFINED PLSSVM_HPX_TARGET_PLATFORMS)
    set(PLSSVM_HPX_TARGET_PLATFORMS ${PLSSVM_HPX_TARGET_PLATFORMS} CACHE STRING "The HPX target platforms to compile for." FORCE)
elseif (DEFINED ENV{PLSSVM_HPX_TARGET_PLATFORMS})
    set(PLSSVM_HPX_TARGET_PLATFORMS $ENV{PLSSVM_HPX_TARGET_PLATFORMS} CACHE STRING "The HPX target platforms to compile for." FORCE)
elseif (DEFINED PLSSVM_TARGET_PLATFORMS)
    set(PLSSVM_HPX_TARGET_PLATFORMS ${PLSSVM_TARGET_PLATFORMS} CACHE STRING "The HPX target platforms to compile for." FORCE)
elseif (DEFINED ENV{PLSSVM_TARGET_PLATFORMS})
    set(PLSSVM_HPX_TARGET_PLATFORMS $ENV{PLSSVM_TARGET_PLATFORMS} CACHE STRING "The HPX target platforms to compile for." FORCE)
endif ()



## parse provided target platforms
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/parse_architecture_info.cmake)
set(PLSSVM_PLATFORM_NAME_LIST "automatic")
foreach (PLSSVM_PLATFORM ${PLSSVM_HPX_TARGET_PLATFORMS})
    if (PLSSVM_PLATFORM MATCHES "^cpu")
        # parse provided CPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_CPU_TARGET_ARCHS PLSSVM_NUM_CPU_TARGET_ARCHS)
        if (PLSSVM_NUM_CPU_TARGET_ARCHS GREATER 1)
            message(FATAL_ERROR "Target platform \"cpu\" must at most have one architecture specification!")
        endif ()
        list(APPEND PLSSVM_PLATFORM_NAME_LIST "cpu")
    elseif (PLSSVM_PLATFORM MATCHES "^nvidia")
        # parse provided NVIDIA GPU architectures
        message(WARNING "The HPX backend currently does not support offloading and therefore the target platform \"nvidia\", ignoring it!")
    elseif (PLSSVM_PLATFORM MATCHES "^amd")
        # parse provided AMD GPU architectures
        message(WARNING "The HPX backend currently does not support offloading and therefore the target platform \"amd\", ignoring it!")
    elseif (PLSSVM_PLATFORM MATCHES "^intel")
        # parse provided Intel GPU architectures
        message(WARNING "The HPX backend currently does not support offloading and therefore the target platform \"intel\", ignoring it!")
    else ()
        message(FATAL_ERROR "Unrecognized target platform \"${PLSSVM_PLATFORM}\"! Must be one of: cpu nvidia amd intel")
    endif ()
endforeach ()

message(CHECK_PASS "found")

# check if HPX can be enabled
message(CHECK_START "Checking for HPX backend")

find_package(HPX 1.9.0)

if (NOT HPX_FOUND)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_HPX_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: HPX!")
    endif ()
    return()
else ()
    if (NOT DEFINED PLSSVM_CPU_TARGET_ARCHS)
        if (PLSSVM_ENABLE_HPX_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested HPX backend, but no \"cpu\" targets were specified!")
        else ()
            message(STATUS "Found HPX backend, but no \"cpu\" targets were specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
endif ()
message(CHECK_PASS "found ")

# explicitly set sources
set(PLSSVM_HPX_SOURCES ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp)

# set target properties
set_local_and_parent(PLSSVM_HPX_BACKEND_LIBRARY_NAME plssvm-HPX)
add_library(${PLSSVM_HPX_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_HPX_SOURCES})
target_link_libraries(${PLSSVM_HPX_BACKEND_LIBRARY_NAME} PUBLIC HPX::hpx)
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_CPU_TARGET)

# additional compilation flags
target_compile_options(${PLSSVM_HPX_BACKEND_LIBRARY_NAME} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,IntelLLVM>:-Wconversion>)

# link base library against HPX library
target_link_libraries(${PLSSVM_HPX_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set compile definition that the HPX backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_HPX_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_HPX_BACKEND_LIBRARY_NAME})

# mark backend library as install target
append_local_and_parent(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_HPX_BACKEND_LIBRARY_NAME})

# generate summary string
set(PLSSVM_HPX_BACKEND_SUMMARY_STRING " - HPX: cpu " PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)
