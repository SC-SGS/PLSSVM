## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "CUDA:  ")

message(CHECK_START "Checking for CUDA platforms")
########################################################################################################################
##                                         parse target platform information                                          ##
########################################################################################################################
if (DEFINED PLSSVM_CUDA_TARGET_PLATFORMS)
    set(PLSSVM_CUDA_TARGET_PLATFORMS ${PLSSVM_CUDA_TARGET_PLATFORMS} CACHE STRING "The CUDA target platforms to compile for." FORCE)
elseif (DEFINED ENV{PLSSVM_CUDA_TARGET_PLATFORMS})
    set(PLSSVM_CUDA_TARGET_PLATFORMS $ENV{PLSSVM_CUDA_TARGET_PLATFORMS} CACHE STRING "The CUDA target platforms to compile for." FORCE)
elseif (DEFINED PLSSVM_TARGET_PLATFORMS)
    set(PLSSVM_CUDA_TARGET_PLATFORMS ${PLSSVM_TARGET_PLATFORMS} CACHE STRING "The CUDA target platforms to compile for." FORCE)
elseif (DEFINED ENV{PLSSVM_TARGET_PLATFORMS})
    set(PLSSVM_CUDA_TARGET_PLATFORMS $ENV{PLSSVM_TARGET_PLATFORMS} CACHE STRING "The CUDA target platforms to compile for." FORCE)
endif ()

## PLSSVM_TARGET_PLATFORMS must not be empty
if (PLSSVM_CUDA_TARGET_PLATFORMS STREQUAL "")
    message(FATAL_ERROR "Eather PLSSVM_TARGET_PLATFORMS or PLSSVM_CUDA_TARGET_PLATFORMS must not be empty to compile with the CUDA backend!")
endif ()

## parse provided target platforms
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/parse_architecture_info.cmake)
foreach (PLSSVM_PLATFORM ${PLSSVM_CUDA_TARGET_PLATFORMS})
    if (PLSSVM_PLATFORM MATCHES "^cpu")
        message(WARNING "The CUDA backend currently does not support the target platform \"cpu\", ignoring it!")
    elseif (PLSSVM_PLATFORM MATCHES "^nvidia")
        # parse provided NVIDIA GPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_NVIDIA_TARGET_ARCHS PLSSVM_NUM_NVIDIA_TARGET_ARCHS)
        if (PLSSVM_NUM_NVIDIA_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"nvidia\" must at least have one architecture specification!")
        endif ()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_NVIDIA_TARGET)
    elseif (PLSSVM_PLATFORM MATCHES "^amd")
        # parse provided AMD GPU architectures
        message(WARNING "The CUDA backend currently does not support the target platform \"amd\", ignoring it!")
    elseif (PLSSVM_PLATFORM MATCHES "^intel")
        # parse provided Intel GPU architectures
        message(WARNING "The CUDA backend currently does not support the target platform \"intel\", ignoring it!")
    else ()
        message(FATAL_ERROR "Unrecognized target platform \"${PLSSVM_PLATFORM}\"! Must be one of: cpu nvidia amd intel")
    endif ()
endforeach ()

if (NOT DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
    if (PLSSVM_ENABLE_CUDA_BACKEND MATCHES "ON")
        message(SEND_ERROR "Found requested CUDA backend, but no \"nvidia\" targets were specified!")
    else ()
        message(STATUS "Found CUDA backend, but no \"nvidia\" targets were specified!")
    endif ()
    message(CHECK_FAIL "skipped")
    return()
endif ()


# generate summary string
set(PLSSVM_CUDA_BACKEND_SUMMARY_STRING " - CUDA: nvidia (${PLSSVM_NVIDIA_TARGET_ARCHS})" PARENT_SCOPE)
# check if CUDA can be enabled
message(CHECK_START "Checking for CUDA backend")

include(CheckLanguage)
check_language(CUDA)



if (NOT CMAKE_CUDA_COMPILER)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_CUDA_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: CUDA!")
    endif ()
    return()
endif ()
message(CHECK_PASS "found")

enable_language(CUDA)
find_package(CUDAToolkit)

# set necessary flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # use nvcc to compile CUDA code
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-fPIC' -Xcudafe --diag_suppress=unsigned_compare_with_zero -lineinfo")
    # build type specific flags
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xptxas=-v -Xptxas -dlcm=cg -Xcudafe --diag_suppress=unrecognized_gcc_pragma")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -O2")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
    # set fast-math if requested
    if (PLSSVM_ENABLE_FAST_MATH)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    endif ()
    # set OpenMP specific flags
    if (OpenMP_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-fopenmp'")
    endif ()
elseif (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    # use clang to compile CUDA code
    # build type specific flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra -Wdouble-promotion -Wshadow -Wcast-qual -Wnull-dereference -Wextra-semi -Wunreachable-code -Wuninitialized -Wmost -Wconversion")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -O2")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")
    # set fast-math if requested
    if (PLSSVM_ENABLE_FAST_MATH)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ffast-math")
    endif ()
    # set OpenMP specific flags
    if (OpenMP_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fopenmp")
    endif ()
endif ()

# explicitly set sources
set(PLSSVM_CUDA_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cu
    ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.cu
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cu
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cu
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
)

# set target properties
set_local_and_parent(PLSSVM_CUDA_BACKEND_LIBRARY_NAME plssvm-CUDA)
add_library(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_CUDA_SOURCES})
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set_target_properties(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()
if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC -fPIC)
    target_link_options(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC -lcudart)
endif ()

target_link_libraries(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC CUDA::cudart)

# prepare architectures for CMake's CUDA_ARCHITECTURES
list(TRANSFORM PLSSVM_NVIDIA_TARGET_ARCHS REPLACE "sm_" "")
# set target architecture
set_property(TARGET ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES ${PLSSVM_NVIDIA_TARGET_ARCHS})

# link base library against CUDA library
target_link_libraries(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})
#get include directories form base
target_include_directories(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC $<TARGET_PROPERTY:${PLSSVM_BASE_LIBRARY_NAME},INTERFACE_INCLUDE_DIRECTORIES>)

# TODO this is a workaround to get the include directories from the base library, fix this in the future!!
get_target_property(PLSSVM_BASE_LIBRARY_INCLUDE_DIRECTORIES ${PLSSVM_BASE_LIBRARY_NAME} INCLUDE_DIRECTORIES)

STRING(REGEX REPLACE "\\$<INSTALL[^>]*>;?" ":" PLSSVM_BASE_LIBRARY_INCLUDE_DIRECTORIES "${PLSSVM_BASE_LIBRARY_INCLUDE_DIRECTORIES}")
STRING(REGEX REPLACE "\\$<BUILD_INTERFACE:" ":" PLSSVM_BASE_LIBRARY_INCLUDE_DIRECTORIES "${PLSSVM_BASE_LIBRARY_INCLUDE_DIRECTORIES}")
STRING(REGEX REPLACE ">;" ":" PLSSVM_BASE_LIBRARY_INCLUDE_DIRECTORIES "${PLSSVM_BASE_LIBRARY_INCLUDE_DIRECTORIES}")
STRING(REGEX REPLACE ";" ":" PLSSVM_BASE_LIBRARY_INCLUDE_DIRECTORIES "${PLSSVM_BASE_LIBRARY_INCLUDE_DIRECTORIES}")
STRING(REGEX REPLACE ":+" ":" PLSSVM_BASE_LIBRARY_INCLUDE_DIRECTORIES "${PLSSVM_BASE_LIBRARY_INCLUDE_DIRECTORIES}")


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I ${PLSSVM_BASE_LIBRARY_INCLUDE_DIRECTORIES}")


# set compile definition that the CUDA backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_HAS_CUDA_BACKEND)
target_compile_definitions(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_CUDA_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME})

# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_CUDA_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)


list(POP_BACK CMAKE_MESSAGE_INDENT)