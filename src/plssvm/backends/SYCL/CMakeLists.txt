## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# check if SYCL can be enabled
message(CHECK_START "Checking for SYCL backend")

# reformat PLSSVM_TARGET_PLATFORMS to be usable with HIPSYCL_TARGETS (in case hipSYCL may be available)
set(HIPSYCL_TARGETS "${PLSSVM_TARGET_PLATFORMS}" CACHE STRING "" FORCE)
list(TRANSFORM HIPSYCL_TARGETS REPLACE "cpu" "omp")
list(TRANSFORM HIPSYCL_TARGETS REPLACE "nvidia" "cuda")
list(TRANSFORM HIPSYCL_TARGETS REPLACE "amd" "hip")
list(TRANSFORM HIPSYCL_TARGETS REPLACE "intel" "spirv")
# remove CPU and Intel GPU target architectures since they are not supported when using hipSYCL
if(DEFINED PLSSVM_CPU_TARGET_ARCHS AND PLSSVM_NUM_CPU_TARGET_ARCHS GREATER 0)
    string(REPLACE ";" "," PLSSVM_CPU_TARGET_ARCHS_COMMA "${PLSSVM_CPU_TARGET_ARCHS}")
    string(REPLACE ":${PLSSVM_CPU_TARGET_ARCHS_COMMA}" "" HIPSYCL_TARGETS "${HIPSYCL_TARGETS}")
endif()
if(DEFINED PLSSVM_INTEL_TARGET_ARCHS)
    string(REPLACE ";" "," PLSSVM_INTEL_TARGET_ARCHS_COMMA "${PLSSVM_INTEL_TARGET_ARCHS}")
    string(REPLACE ":${PLSSVM_INTEL_TARGET_ARCHS_COMMA}" "" HIPSYCL_TARGETS "${HIPSYCL_TARGETS}")
endif()

# check if hipSYCL is used as SYCL compiler
find_package(hipSYCL CONFIG)
if(hipSYCL_FOUND)
    set(PLSSVM_SYCL_BACKEND_COMPILER "hipSYCL" CACHE STRING "" FORCE)
    message(CHECK_PASS "found hipSYCL")
    message(STATUS "Setting HIPSYCL_TARGETS to \"${HIPSYCL_TARGETS}\".")
else()
    # if not, check if DPC++ is used instead
    try_compile(PLSSVM_SYCL_BACKEND_CHECK_FOR_DPCPP_COMPILER
            "${CMAKE_BINARY_DIR}/temp"
            "${PROJECT_SOURCE_DIR}/cmake/compile_tests/test_sycl_dpcpp.cpp"
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON)
    if(PLSSVM_SYCL_BACKEND_CHECK_FOR_DPCPP_COMPILER)
        set(PLSSVM_SYCL_BACKEND_COMPILER "DPC++" CACHE STRING "" FORCE)
        message(CHECK_PASS "found DPC++")
    else()
        message(CHECK_FAIL "not found")
        if(PLSSVM_ENABLE_SYCL_BACKEND MATCHES "ON")
            message(FATAL_ERROR "Cannot find requested backend: SYCL!")
        endif()
        return()
    endif()
endif()


# explicitly set sources
set(PLSSVM_SYCL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_device_ptr.cpp
)

# set target properties
set_local_and_parent(PLSSVM_SYCL_BACKEND_LIBRARY_NAME svm-SYCL)
add_library(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_SYCL_SOURCES})

# set additional properties depending on the used SYCL implementation
if("${PLSSVM_SYCL_BACKEND_COMPILER}" STREQUAL "hipSYCL")
    add_sycl_to_target(TARGET ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} SOURCES ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp)
    # set backend compiler to hipSYCL (= 1)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_COMPILER=1)
    # silence unknown options warnings
    target_compile_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -Wno-unknown-warning-option
            $<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>
            )

    # print note that Intel GPU architecture specifications are ignored when using hipSYCL
    if(DEFINED PLSSVM_INTEL_TARGET_ARCHS)
        message(STATUS "Ignoring specified Intel architectures \"${PLSSVM_INTEL_TARGET_ARCHS}\" in favor of SPIR-V when using hipSYCL!")
    endif()
elseif("${PLSSVM_SYCL_BACKEND_COMPILER}" STREQUAL "DPC++")
    # TODO: remove if DPC++ bug is fixed
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(FATAL_ERROR "The SYCL backend when using DPC++ currently does not support the Debug build type. For more information see: https://github.com/intel/llvm/issues/5754")
    endif()

    # enable DPC++ SYCL support
    target_compile_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -sycl-std=2020 -fsycl)
    target_link_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -fsycl)

    set(PLSSVM_DPCPP_FSYCL_TARGETS "")
    # cpu targets
    if(DEFINED PLSSVM_CPU_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND PLSSVM_DPCPP_FSYCL_TARGETS "spir64_x86_64")
        # add target specific flags for AOT
        if(PLSSVM_NUM_CPU_TARGET_ARCHS EQUAL 1)
            target_compile_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=spir64_x86_64 "-march=${PLSSVM_CPU_TARGET_ARCHS}")
            target_link_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=spir64_x86_64 "-march=${PLSSVM_CPU_TARGET_ARCHS}")
        endif()
    endif()
    # nvidia targets
    if(DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND PLSSVM_DPCPP_FSYCL_TARGETS "nvptx64-nvidia-cuda")
        # add target specific flags for AOT
        # add target specific flags for AOT
        if(NOT PLSSVM_NUM_NVIDIA_TARGET_ARCHS EQUAL 1)
            message(FATAL_ERROR "DPC++ currently only supports a single NVIDIA architecture specification for AOT but ${PLSSVM_NUM_NVIDIA_TARGET_ARCHS} were provided!")
        endif()
        target_compile_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=${PLSSVM_NVIDIA_TARGET_ARCHS_STRING})
        target_link_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=${PLSSVM_NVIDIA_TARGET_ARCHS_STRING})
    endif()
    # amd targets
    if(DEFINED PLSSVM_AMD_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND PLSSVM_DPCPP_FSYCL_TARGETS "amdgcn-amd-amdhsa")
        # add target specific flags for AOT
        if(NOT PLSSVM_NUM_AMD_TARGET_ARCHS EQUAL 1)
            message(FATAL_ERROR "DPC++ currently only supports a single AMD architecture specification but ${PLSSVM_NUM_AMD_TARGET_ARCHS} were provided!")
        endif()
        target_compile_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${PLSSVM_AMD_TARGET_ARCHS})
        target_link_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${PLSSVM_AMD_TARGET_ARCHS})
    endif()
    # intel targets
    if(DEFINED PLSSVM_INTEL_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND PLSSVM_DPCPP_FSYCL_TARGETS "spir64_gen")
        # add target specific flags for AOT
        list(JOIN PLSSVM_INTEL_TARGET_ARCHS "," PLSSVM_INTEL_TARGET_ARCHS_STRING)
        target_compile_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=spir64_gen "-device ${PLSSVM_INTEL_TARGET_ARCHS_STRING}")
        target_link_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=spir64_gen "-device ${PLSSVM_INTEL_TARGET_ARCHS_STRING}")
    endif()
    # set -fsycl-targets
    list(JOIN PLSSVM_DPCPP_FSYCL_TARGETS "," PLSSVM_DPCPP_FSYCL_TARGETS_STRING)
    target_compile_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -fsycl-targets=${PLSSVM_DPCPP_FSYCL_TARGETS_STRING})
    target_link_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -fsycl-targets=${PLSSVM_DPCPP_FSYCL_TARGETS_STRING})

    # set backend compiler to DPC++ (= 0)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_COMPILER=0)

    # be able to choose between the Level-Zero and OpenCL DPC++ backend
    option(PLSSVM_SYCL_DPCPP_USE_LEVEL_ZERO "Enable Level-Zero backend in favor of OpenCL when using DPC++." OFF)
    if(PLSSVM_SYCL_BACKEND_DPCPP_USE_LEVEL_ZERO)
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE="level_zero")
    else()
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE="opencl")
    endif()
endif()


# link base library against SYCL library
target_link_libraries(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set compile definition that the SYCL backend is available
target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_SYCL_BACKEND)
target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_SYCL_BACKEND_COMPILER_NAME="${PLSSVM_SYCL_BACKEND_COMPILER}")

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME})

# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_SYCL_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)


# generate summary string
set(PLSSVM_SYCL_BACKEND_SUMMARY_STRING_COMPILER " - SYCL (${PLSSVM_SYCL_BACKEND_COMPILER}):")
include(${PROJECT_SOURCE_DIR}/cmake/assemble_summary_string.cmake)
assemble_summary_string(PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS)
# do not print CPU and Intel GPU target architectures when using hipSYCL
if("${PLSSVM_SYCL_BACKEND_COMPILER}" STREQUAL "hipSYCL")
    string(REPLACE " (${PLSSVM_CPU_TARGET_ARCHS})" "" PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}")
    string(REPLACE " (${PLSSVM_INTEL_TARGET_ARCHS})" "" PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}")
endif()
set(PLSSVM_SYCL_BACKEND_SUMMARY_STRING "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_COMPILER}${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}" PARENT_SCOPE)

