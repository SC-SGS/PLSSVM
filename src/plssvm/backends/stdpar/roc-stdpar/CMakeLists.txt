# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

message(CHECK_START "searching for roc-stdpar")

# check whether the roc-stdpar compiler is available
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # check for AMD's roc-stdpar enabled compiler
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("--hipstdpar" PLSSVM_HAS_HIPSTDPAR_STDPAR_FLAG)
    if (NOT PLSSVM_HAS_HIPSTDPAR_STDPAR_FLAG)
        # the target offload flag couldn't be found!
        if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested stdpar backend using roc-stdpar, but the offloading compiler flag isn't supported!")
        else ()
            message(STATUS "Found stdpar backend using roc-stdpar, but the offloading compiler flag isn't supported!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    # roc-stdpar needs an AMD GPU as target
    if (NOT DEFINED PLSSVM_AMD_TARGET_ARCHS)
        if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested stdpar backend using roc-stdpar, but no \"amd\" targets were specified!")
        else ()
            message(STATUS "Found stdpar backend using roc-stdpar, but \"amd\" targets were specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    # roc-stdpar found
    message(CHECK_PASS "found")
    set_local_and_parent(PLSSVM_STDPAR_BACKEND "roc-stdpar")
endif ()

# add stdpar implementation specific source file to library
append_local_and_parent(PLSSVM_STDPAR_SOURCES ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp)

# set global roc-stdpar compiler flags
set_local_and_parent(
        PLSSVM_STDPAR_BACKEND_COMPILER_FLAG --hipstdpar --offload-arch=${PLSSVM_AMD_TARGET_ARCHS}
)

# be able to set --hipstdpar-interpose-alloc if necessary
set(PLSSVM_STDPAR_BACKEND_ROCSTDPAR_USE_INTERPOSE_ALLOC AUTO CACHE STRING "Use the --hipstdpar-interpose-alloc compiler flag")
set_property(CACHE PLSSVM_STDPAR_BACKEND_ROCSTDPAR_USE_INTERPOSE_ALLOC PROPERTY STRINGS AUTO ON OFF)
# check whether HSA_XNACK is enabled via an environment variable
set(PLSSVM_IS_HSA_XNACK_DISABLED OFF)
if (NOT DEFINED ENV{HSA_XNACK})
    set(PLSSVM_IS_HSA_XNACK_DISABLED ON)
elseif ($ENV{HSA_XNACK} EQUAL 0)
    set(PLSSVM_IS_HSA_XNACK_DISABLED ON)
endif ()
# set compiler flag accordingly
if ((PLSSVM_STDPAR_BACKEND_ROCSTDPAR_USE_INTERPOSE_ALLOC MATCHES "AUTO" AND PLSSVM_IS_HSA_XNACK_DISABLED) OR PLSSVM_STDPAR_BACKEND_ROCSTDPAR_USE_INTERPOSE_ALLOC MATCHES "ON")
    message(STATUS "Using the --hipstdpar-interpose-alloc compiler flag. "
            "Note: this may result in the test not being able to run!")
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG "${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} --hipstdpar-interpose-alloc")
endif ()

if (PLSSVM_ENABLE_FAST_MATH)
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} -ffast-math)
endif ()
separate_arguments(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG UNIX_COMMAND ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})

# set roc-stdpar compile definition
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_STDPAR_BACKEND_HAS_HIPSTDPAR)
target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_INTERFACE} INTERFACE PLSSVM_STDPAR_BACKEND_HAS_HIPSTDPAR)
