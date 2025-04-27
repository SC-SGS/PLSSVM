# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "Kokkos:  ")

# check if Kokkos can be enabled
message(CHECK_START "Checking for Kokkos backend")

find_package(Kokkos)

if (NOT Kokkos_FOUND)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_KOKKOS_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: Kokkos!")
    endif ()
    return()
endif ()
message(CHECK_PASS "found")

# explicitly set sources
set(PLSSVM_KOKKOS_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_wrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
)

# set target properties
set_local_and_parent(PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME plssvm-Kokkos)
add_library(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_KOKKOS_SOURCES})
target_link_libraries(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} PUBLIC Kokkos::kokkos)

if (Kokkos_ENABLE_SYCL)
    # set SYCL (icpx) specific compilation flags
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        message(FATAL_ERROR "For Kokkos::SYCL to work, the compiler must be IntelLLVM, but is ${CMAKE_CXX_COMPILER}!")
    endif ()

    if (PLSSVM_ENABLE_FAST_MATH)
        target_compile_options(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} PRIVATE -ffast-math)
    else ()
        # icpx does compile with fast-math by default -> explicitly disable it if not requested otherwise!
        target_compile_options(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} PRIVATE -fno-fast-math)
    endif ()

    target_compile_options(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} PRIVATE -sycl-std=2020 -fsycl)
    target_link_options(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} PRIVATE -fsycl)
    
    # assemble the icpx specific compiler flags and add them to the Kokkos backend target
    # -> Kokkos sets the values for NVIDIA and AMD correctly, but does NOT set AOT for Intel GPUs!
    include(${PROJECT_SOURCE_DIR}/cmake/assemble_icpx_sycl_target_flags.cmake)
    assemble_icpx_sycl_target_flags(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} PRIVATE)
endif ()

# link base library against Kokkos library
target_link_libraries(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set compile definition that the Kokkos backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_HAS_KOKKOS_BACKEND)
target_compile_definitions(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_KOKKOS_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME})

# mark backend library as install target
append_local_and_parent(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME})

# ~~~
# assemble Kokkos available execution space string
# also set compile definitions -> can't use KOKKOS_ENABLE_* directly inside the "constexpr_available_execution_space.hpp"
# header since we can't include "Kokkos_Core.hpp" there (transitively used in the base library that doesn't know anything about Kokkos
# ~~~
set(PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES "")
if (Kokkos_ENABLE_CUDA)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_KOKKOS_BACKEND_ENABLE_CUDA)
    list(APPEND PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES "Cuda")
endif ()
if (Kokkos_ENABLE_HIP)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_KOKKOS_BACKEND_ENABLE_HIP)
    list(APPEND PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES "HIP")
endif ()
if (Kokkos_ENABLE_SYCL)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_KOKKOS_BACKEND_ENABLE_SYCL)
    list(APPEND PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES "SYCL")
endif ()
if (Kokkos_ENABLE_HPX)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_KOKKOS_BACKEND_ENABLE_HPX)
    list(APPEND PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES "HPX")
endif ()
if (Kokkos_ENABLE_OPENMP)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_KOKKOS_BACKEND_ENABLE_OPENMP)
    list(APPEND PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES "OpenMP")
endif ()
if (Kokkos_ENABLE_OPENMPTARGET)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_KOKKOS_BACKEND_ENABLE_OPENMPTARGET)
    list(APPEND PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES "OpenMPTarget")
endif ()
if (Kokkos_ENABLE_OPENACC)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_KOKKOS_BACKEND_ENABLE_OPENACC)
    list(APPEND PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES "OpenACC")
endif ()
if (Kokkos_ENABLE_THREADS)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_KOKKOS_BACKEND_ENABLE_THREADS)
    list(APPEND PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES "Threads")
endif ()
if (Kokkos_ENABLE_SERIAL)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_KOKKOS_BACKEND_ENABLE_SERIAL)
    list(APPEND PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES "Serial")
endif ()
set(PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES "${PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES}" PARENT_SCOPE)

# ~~~
# also set the number of available Kokkos execution spaces to explicitly set the type of the used std::array
# -> necessary if NO Kokkos execution space is available and, therefore, the size of the std::array would be 0 (can't automatically be deduced)
# ~~~
list(LENGTH PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES PLSSVM_KOKKOS_BACKEND_NUM_AVAILABLE_EXECUTION_SPACES)
target_compile_definitions(
    ${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_KOKKOS_BACKEND_NUM_AVAILABLE_EXECUTION_SPACES=${PLSSVM_KOKKOS_BACKEND_NUM_AVAILABLE_EXECUTION_SPACES}
)

# generate summary string
set(PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_COMPILER " - Kokkos (${PLSSVM_KOKKOS_BACKEND_AVAILABLE_EXECUTION_SPACES}):")
include(${PROJECT_SOURCE_DIR}/cmake/assemble_summary_string.cmake)
assemble_summary_string(PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS)
# do not print any special target architecture information
string(REPLACE " (${PLSSVM_CPU_TARGET_ARCHS})" "" PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS}")
string(REPLACE " (${PLSSVM_NVIDIA_TARGET_ARCHS})" "" PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS}")
string(REPLACE " (${PLSSVM_AMD_TARGET_ARCHS})" "" PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS}")
string(REPLACE " (${PLSSVM_INTEL_TARGET_ARCHS})" "" PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS}")
set(PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING "${PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_COMPILER}${PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS}" PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)
