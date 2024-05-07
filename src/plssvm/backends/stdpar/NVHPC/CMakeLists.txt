## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

message(CHECK_START "searching for NVHPC (nvc++)")

if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    # check for NVIDIA's nvhpc compiler
    # nvhpc needs either a CPU or an NVIDIA GPU as target
    if (NOT DEFINED PLSSVM_CPU_TARGET_ARCHS AND NOT DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested stdpar backend using NVHPC, but no \"cpu\" or \"nvidia\" targets were specified!")
        else ()
            message(STATUS "Found stdpar backend using NVHPC, but no \"cpu\" or \"nvidia\" targets were specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    # for nvhpc not BOTH targets may be set
    if (DEFINED PLSSVM_CPU_TARGET_ARCHS AND DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested stdpar backend using NVHPC, but both \"cpu\" and \"nvidia\" targets were specified!")
        else ()
            message(STATUS "Found stdpar backend using NVHPC, but both \"cpu\" and \"nvidia\" targets were specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    # nvhpc found
    message(CHECK_PASS "found")
    set_local_and_parent(PLSSVM_STDPAR_BACKEND "NVHPC")
else ()
    message(CHECK_FAIL "skipped")
    return()
endif ()

# add stdpar implementation specific source file to library
target_sources(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp)

## try finding Boost atomic
if (POLICY CMP0144)
    cmake_policy(SET CMP0144 OLD) # suppress cmake warning
endif ()
find_package(Boost 1.73.0 REQUIRED COMPONENTS atomic)
message(STATUS "Found Boost atomic version ${Boost_VERSION}.")
target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC Boost::atomic)
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_boost_VERSION="${Boost_VERSION}")

# set global NVHPC compiler flags
if (DEFINED PLSSVM_CPU_TARGET_ARCHS)
    message(STATUS "CPU target platform provided, setting: \"-stdpar=multicore -tp=native\"")
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -stdpar=multicore)
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} -tp=native)
    
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_NVHPC_CPU)
elseif (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
    message(STATUS "NVIDIA GPU target platform provided, setting: \"-stdpar=gpu\"")
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -stdpar=gpu)
    
    if (PLSSVM_ENABLE_FAST_MATH)
        set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} -gpu=fastmath)
    endif ()
    
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_NVHPC_GPU)
endif ()
separate_arguments(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG UNIX_COMMAND ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})

# NVHPC must be linked against CUDA
enable_language(CUDA)
find_package(CUDAToolkit)

# set compile options
target_link_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
target_link_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
target_link_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
target_compile_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
target_compile_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})

# set NVHPC compile definition
target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_HAS_NVHPC)