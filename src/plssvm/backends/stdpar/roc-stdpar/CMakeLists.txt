## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

message(CHECK_START "searching for roc-stdpar")

find_package(rocthrust REQUIRED)
find_package(rocprim REQUIRED)


# check whether the roc-stdpar include path has been provided
if (DEFINED PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH)
    set(PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH ${PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH} CACHE STRING "The roc-stdpar path (--hipstdpar-path) necessary to compile the roc-stdpar backend." FORCE)
elseif (DEFINED ENV{PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH})
    set(PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH $ENV{PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH} CACHE STRING "The roc-stdpar path (--hipstdpar-path) necessary to compile the roc-stdpar backend." FORCE)
endif ()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH)
    # check for AMD's roc-stdpar enabled compiler -> only possible with the provided path
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-hipstdpar --hipstdpar-path=${PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH} --hipstdpar-thrust-path=${ROCTHRUST_INCLUDE_DIR} --hipstdpar-prim-path=${ROCPRIM__INCLUDE_DIR}" PLSSVM_HAS_HIPSTDPAR_STDPAR_FLAG)
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
else ()
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH)
        message(STATUS "Try checking for the roc-stdpar backend but the roc-stdpar include path wasn't provided via \"PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH\"!")
    endif ()
    message(CHECK_FAIL "skipped")
    return()
endif ()

# add stdpar implementation specific source file to library
append_local_and_parent(PLSSVM_STDPAR_SOURCES ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp)

# set global roc-stdpar compiler flags
# TODO: The flag "--hipstdpar-interpose-alloc" may be necessary for older AMD GPUs
if(NOT HSA_XNACK eq 1 AND NOT ENV{HSA_XNACK} eq 1)
    message (STATUS "HSA_XNACK is not set, enabeling interpose alloc")
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -hipstdpar --hipstdpar-path=${PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH} --offload-arch=${PLSSVM_AMD_TARGET_ARCHS} --hipstdpar-interpose-alloc)
else ()
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -hipstdpar --hipstdpar-path=${PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH} --offload-arch=${PLSSVM_AMD_TARGET_ARCHS})
endif()

set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -hipstdpar --hipstdpar-path=${PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH} --offload-arch=${PLSSVM_AMD_TARGET_ARCHS})

if (PLSSVM_ENABLE_FAST_MATH)
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} -ffast-math)
endif ()
separate_arguments(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG UNIX_COMMAND ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})

# set roc-stdpar compile definition
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_STDPAR_BACKEND_HAS_HIPSTDPAR)
target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_INTERFACE} INTERFACE PLSSVM_STDPAR_BACKEND_HAS_HIPSTDPAR)