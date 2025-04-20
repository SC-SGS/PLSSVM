# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

message(CHECK_START "searching for AdaptiveCpp")

# try finding Intel TBB
find_package(TBB QUIET)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND TBB_FOUND)
    # check for the AdaptiveCpp compiler
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("--acpp-stdpar" PLSSVM_HAS_ACPP_STDPAR_FLAG)
    if (NOT PLSSVM_HAS_ACPP_STDPAR_FLAG)
        # the stdpar flag couldn't be found!
        if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
            message(
                SEND_ERROR "Found a Clang, but AdaptiveCpp's stdpar compiler flag isn't supported (mostly likely because the used Clang isn't AdaptiveCpp)!"
            )
        else ()
            message(STATUS "Found a Clang, but AdaptiveCpp's stdpar compiler flag isn't supported (mostly likely because the used Clang isn't AdaptiveCpp)!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    # AdaptiveCpp found
    message(CHECK_PASS "found")
    set_local_and_parent(PLSSVM_STDPAR_BACKEND "ACPP")
else ()
    message(CHECK_FAIL "skipped")
    return()
endif ()

option(PLSSVM_STDPAR_BACKEND_ACPP_USE_GENERIC_SSCP "Enables the generic SSCP target for new AdaptiveCpp versions." ON)

if (PLSSVM_STDPAR_BACKEND_ACPP_USE_GENERIC_SSCP)
    message(STATUS "Using the new AdaptiveCpp SSCP compilation flow.")
    set(ACPP_TARGETS "generic" CACHE STRING "" FORCE)
else ()
    message(STATUS "Using the old AdaptiveCpp compilation flow.")
    # reformat PLSSVM_TARGET_PLATFORMS to be usable with ACPP_TARGETS
    set(ACPP_TARGETS "${PLSSVM_TARGET_PLATFORMS}" CACHE STRING "" FORCE)
    list(TRANSFORM ACPP_TARGETS REPLACE "cpu" "omp.accelerated")
    list(TRANSFORM ACPP_TARGETS REPLACE "nvidia" "cuda")
    list(TRANSFORM ACPP_TARGETS REPLACE "amd" "hip")
    list(TRANSFORM ACPP_TARGETS REPLACE "intel" "spirv")
    # remove CPU and Intel GPU target architectures since they are not supported when using AdaptiveCpp
    if (DEFINED PLSSVM_CPU_TARGET_ARCHS AND PLSSVM_NUM_CPU_TARGET_ARCHS GREATER 0)
        string(REPLACE ";" "," PLSSVM_CPU_TARGET_ARCHS_COMMA "${PLSSVM_CPU_TARGET_ARCHS}")
        string(REPLACE ":${PLSSVM_CPU_TARGET_ARCHS_COMMA}" "" ACPP_TARGETS "${ACPP_TARGETS}")
    endif ()
    if (DEFINED PLSSVM_INTEL_TARGET_ARCHS)
        string(REPLACE ";" "," PLSSVM_INTEL_TARGET_ARCHS_COMMA "${PLSSVM_INTEL_TARGET_ARCHS}")
        string(REPLACE ":${PLSSVM_INTEL_TARGET_ARCHS_COMMA}" "" ACPP_TARGETS "${ACPP_TARGETS}")
    endif ()
endif ()
# propagate ACPP_TARGETS
target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_INTERFACE} INTERFACE PLSSVM_ACPP_TARGETS="${ACPP_TARGETS}")

# add stdpar implementation specific source file to library
append_local_and_parent(PLSSVM_STDPAR_SOURCES ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp)
append_local_and_parent(PLSSVM_STDPAR_SOURCES ${CMAKE_CURRENT_LIST_DIR}/../../SYCL/AdaptiveCpp/detail/utility.cpp)

# set global AdaptiveCpp compiler flags
set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG --acpp-stdpar --acpp-targets="${ACPP_TARGETS}" --acpp-stdpar-unconditional-offload)
# add flag improving CPU performance if only the CPU target is present
if (DEFINED PLSSVM_CPU_TARGET_ARCHS AND NOT (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS OR DEFINED PLSSVM_AMD_TARGET_ARCHS OR DEFINED PLSSVM_INTEL_TARGET_ARCHS))
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} --acpp-stdpar-system-usm)
endif ()

if (PLSSVM_ENABLE_FAST_MATH)
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} -ffast-math)
endif ()
separate_arguments(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG UNIX_COMMAND ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})

# AdaptiveCpp must be linked against TBB
target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_INTERFACE} INTERFACE TBB::tbb)

# set AdaptiveCpp compile definition
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_STDPAR_BACKEND_HAS_ACPP)
target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_INTERFACE} INTERFACE PLSSVM_STDPAR_BACKEND_HAS_ACPP)

if (PLSSVM_STDPAR_BACKEND_ACPP_USE_GENERIC_SSCP)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_ACPP_USE_GENERIC_SSCP)
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_INTERFACE} INTERFACE PLSSVM_STDPAR_BACKEND_ACPP_USE_GENERIC_SSCP)
endif ()