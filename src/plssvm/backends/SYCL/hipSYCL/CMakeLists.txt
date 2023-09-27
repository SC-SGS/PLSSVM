## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################


message(CHECK_START "Checking for hipSYCL as SYCL implementation")

option(PLSSVM_SYCL_BACKEND_HIPSYCL_USE_GENERIC_SSCP "Enables the generic SSCP target for new hipSYCL versions." ON)

if (PLSSVM_SYCL_BACKEND_HIPSYCL_USE_GENERIC_SSCP)
    message(STATUS "Using the new hipSYCL SSCP compilation flow.")
    set(HIPSYCL_TARGETS "generic" CACHE STRING "" FORCE)
else ()
    message(STATUS "Using the old hipSYCL compilation flow.")
    # reformat PLSSVM_TARGET_PLATFORMS to be usable with HIPSYCL_TARGETS (in case hipSYCL may be available)
    set(HIPSYCL_TARGETS "${PLSSVM_TARGET_PLATFORMS}" CACHE STRING "" FORCE)
    list(TRANSFORM HIPSYCL_TARGETS REPLACE "cpu" "omp.accelerated")
    list(TRANSFORM HIPSYCL_TARGETS REPLACE "nvidia" "cuda")
    list(TRANSFORM HIPSYCL_TARGETS REPLACE "amd" "hip")
    list(TRANSFORM HIPSYCL_TARGETS REPLACE "intel" "spirv")
    # remove CPU and Intel GPU target architectures since they are not supported when using hipSYCL
    if (DEFINED PLSSVM_CPU_TARGET_ARCHS AND PLSSVM_NUM_CPU_TARGET_ARCHS GREATER 0)
        string(REPLACE ";" "," PLSSVM_CPU_TARGET_ARCHS_COMMA "${PLSSVM_CPU_TARGET_ARCHS}")
        string(REPLACE ":${PLSSVM_CPU_TARGET_ARCHS_COMMA}" "" HIPSYCL_TARGETS "${HIPSYCL_TARGETS}")
    endif ()
    if (DEFINED PLSSVM_INTEL_TARGET_ARCHS)
        string(REPLACE ";" "," PLSSVM_INTEL_TARGET_ARCHS_COMMA "${PLSSVM_INTEL_TARGET_ARCHS}")
        string(REPLACE ":${PLSSVM_INTEL_TARGET_ARCHS_COMMA}" "" HIPSYCL_TARGETS "${HIPSYCL_TARGETS}")
    endif ()
endif ()

# check if hipSYCL is used as SYCL compiler
find_package(hipSYCL CONFIG)
if (hipSYCL_FOUND)
    message(CHECK_PASS "found")
    message(STATUS "Setting HIPSYCL_TARGETS to \"${HIPSYCL_TARGETS}\".")
    append_local_and_parent(PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS "hipsycl")

    # set hipSYCL specific targets
    set(PLSSVM_SYCL_HIPSYCL_SOURCES
            ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
            ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
            ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
            )

    # set target properties
    set(PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME plssvm-SYCL_hipsycl CACHE INTERNAL "")
    add_library(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} SHARED ${PLSSVM_SYCL_SOURCES} ${PLSSVM_SYCL_HIPSYCL_SOURCES})
    append_local_and_parent(PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME}")

    # necessary target_*_directories if more than one SYCL implementation could be found
    if (DEFINED ENV{PLSSVM_SYCL_HIPSYCL_INCLUDE_DIR})
        message(STATUS "PLSSVM_SYCL_HIPSYCL_INCLUDE_DIR is \"$ENV{PLSSVM_SYCL_HIPSYCL_INCLUDE_DIR}\"")
        target_include_directories(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PUBLIC $ENV{PLSSVM_SYCL_HIPSYCL_INCLUDE_DIR})
    endif ()

    # add target compile definitions for hipSYCL
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PUBLIC PLSSVM_HAS_SYCL_BACKEND)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PUBLIC PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_COMPILER=1)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_COMPILER_NAME="hipSYCL")

    if (PLSSVM_SYCL_BACKEND_HIPSYCL_USE_GENERIC_SSCP)
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_HIPSYCL_USE_GENERIC_SSCP)
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PUBLIC PLSSVM_SYCL_BACKEND_HIPSYCL_USE_GENERIC_SSCP)
    endif ()

    # hipSYCL's way to add SYCL to a target
    add_sycl_to_target(TARGET ${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} SOURCES ${PROJECT_SOURCE_DIR}/src/plssvm/backends/hipSYCL/csvm.cpp)

    # silence unknown options warnings
    target_compile_options(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PRIVATE -Wno-unknown-warning-option
            $<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>
            )

    # print note that Intel GPU architecture specifications are ignored when using hipSYCL
    if (DEFINED PLSSVM_INTEL_TARGET_ARCHS)
        message(STATUS "Ignoring specified Intel architectures \"${PLSSVM_INTEL_TARGET_ARCHS}\" in favor of SPIR-V when using hipSYCL!")
    endif ()

    target_link_libraries(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})
else ()
    message(CHECK_FAIL "not found")
endif ()