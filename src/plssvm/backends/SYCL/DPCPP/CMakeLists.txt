# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

message(CHECK_START "Checking for DPC++ as SYCL implementation")

# check if current compiler is Intel's clang llvm fork aka DPC++
set(PLSSVM_SYCL_BACKEND_CHECK_FOR_DPCPP_COMPILER OFF)
if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(STATUS "Found Intel icpx compiler.")
    set(PLSSVM_SYCL_BACKEND_CHECK_FOR_DPCPP_COMPILER ON)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # intel llvm fork must be clang
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} --help OUTPUT_VARIABLE PLSSVM_DPCPP_CLANG_HELP_OUTPUT)
    # check if help message contains DPC++ specific values
    if ("${PLSSVM_DPCPP_CLANG_HELP_OUTPUT}" MATCHES ".*intel.*" AND "${PLSSVM_DPCPP_CLANG_HELP_OUTPUT}" MATCHES ".*-fsycl.*")
        message(STATUS "Found Intel LLVM fork (https://github.com/intel/llvm).")
        set(PLSSVM_SYCL_BACKEND_CHECK_FOR_DPCPP_COMPILER ON)
    endif ()
endif ()

if (PLSSVM_SYCL_BACKEND_CHECK_FOR_DPCPP_COMPILER)
    message(CHECK_PASS "found")
    append_local_and_parent(PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS "dpcpp")

    # set DPC++ specific targets
    set(PLSSVM_SYCL_DPCPP_SOURCES ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.cpp
                                  ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    )

    # set target properties
    set(PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME plssvm-SYCL_DPCPP CACHE INTERNAL "")
    add_library(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} SHARED ${PLSSVM_SYCL_SOURCES} ${PLSSVM_SYCL_DPCPP_SOURCES})
    append_local_and_parent(PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME}")

    # necessary target_*_directories if more than one SYCL implementation could be found
    if (DEFINED ENV{PLSSVM_SYCL_DPCPP_INCLUDE_DIR})
        message(STATUS "PLSSVM_SYCL_DPCPP_INCLUDE_DIR is \"$ENV{PLSSVM_SYCL_DPCPP_INCLUDE_DIR}\"")
        target_include_directories(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC $ENV{PLSSVM_SYCL_DPCPP_INCLUDE_DIR})
    endif ()

    # add target compile definitions for DPC++
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC PLSSVM_HAS_SYCL_BACKEND)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_COMPILER=0)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_COMPILER_NAME="DPC++")

    # set whether the kernels should be compiled with fast math enabled or not
    if (PLSSVM_ENABLE_FAST_MATH)
        target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -ffast-math)
    else ()
        # DPC++ does compile with fast-math by default -> explicitly disable it if not requested otherwise!
        target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -fno-fast-math)
    endif ()

    # enable DPC++ SYCL support
    target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -sycl-std=2020 -fsycl -fno-sycl-id-queries-fit-in-int)
    target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -fsycl)
    # add additional compiler warnings
    target_compile_options(
        ${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME}
        PRIVATE -Wconversion
                -Wall
                -Wextra
                -Wmost
                -Wdouble-promotion
                -fno-common
                -Wshadow
                -Wcast-qual
                -Wextra-semi
                -Wunreachable-code
                -Wuninitialized
                -Wno-ctor-dtor-privacy
    )
    
    # assemble the icpx specific compiler flags and add them to the DPCPP backend target
    include(${PROJECT_SOURCE_DIR}/cmake/assemble_icpx_sycl_target_flags.cmake)
    assemble_icpx_sycl_target_flags(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE)

    # be able to choose between the Level-Zero and OpenCL DPC++ backend for CPUs or Intel GPUs
    option(PLSSVM_SYCL_BACKEND_DPCPP_USE_LEVEL_ZERO "Enable DPC++'s Level-Zero backend in favor of the OpenCL backend." ON)
    if (PLSSVM_SYCL_BACKEND_DPCPP_USE_LEVEL_ZERO)
        message(STATUS "Using DPC++'s Level-Zero backend.")
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE="level-zero")
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE="level-zero")
    else ()
        message(STATUS "Using DPC++'s OpenCL backend.")
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE="opencl")
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE="opencl")
    endif ()

    target_link_libraries(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})
else ()
    message(CHECK_FAIL "not found")
endif ()
