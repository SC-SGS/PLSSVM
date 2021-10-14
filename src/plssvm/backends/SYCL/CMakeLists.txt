## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# check if SYCL can be enabled
message(CHECK_START "Checking for SYCL backend")

# reformat PLSSVM_TARGET_PLATFORMS to be usable with HIPSYCL_TARGETS (in case hipSYCL may be available)
set(HIPSYCL_TARGETS ${PLSSVM_TARGET_PLATFORMS} CACHE STRING "" FORCE)
list(TRANSFORM HIPSYCL_TARGETS REPLACE "cpu" "omp")
list(TRANSFORM HIPSYCL_TARGETS REPLACE "nvidia" "cuda")
list(TRANSFORM HIPSYCL_TARGETS REPLACE "amd" "hip")
list(TRANSFORM HIPSYCL_TARGETS REPLACE "intel" "spirv")

# check if hipSYCL is used as SYCL compiler
find_package(hipSYCL CONFIG)
if(hipSYCL_FOUND)
    set(PLSSVM_SYCL_BACKEND_COMPILER "hipSYCL" CACHE STRING "" FORCE)
    message(CHECK_PASS "found hipSYCL")
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../gpu_csvm.cpp
)

# set target properties
set_local_and_parent(PLSSVM_SYCL_BACKEND_LIBRARY_NAME svm-SYCL)
add_library(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_SYCL_SOURCES})

# set additional properties depending on the used SYCL implementation
if("${PLSSVM_SYCL_BACKEND_COMPILER}" STREQUAL "hipSYCL")
    add_sycl_to_target(TARGET ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} SOURCES ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp)
    # set backend compiler to hipSYCL (= 1)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_COMPILER=1)
elseif("${PLSSVM_SYCL_BACKEND_COMPILER}" STREQUAL "DPC++")
    # enable DPC++ SYCL support
    target_compile_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -fsycl)
    target_link_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -fsycl)

    # nvidia targets
    if(DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        target_compile_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)
        target_link_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)
    endif()
    # amd targets
    if(DEFINED PLSSVM_AMD_TARGET_ARCHS)
        target_compile_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -fsycl-targets=amdgcn-amd-amdhsa)
        target_link_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -fsycl-targets=amdgcn-amd-amdhsa)
        foreach(PLSSVM_AMD_TARGET_ARCH_NAME ${PLSSVM_AMD_TARGET_ARCHS})
            target_compile_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${PLSSVM_AMD_TARGET_ARCH_NAME})
            target_link_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${PLSSVM_AMD_TARGET_ARCH_NAME})
        endforeach()
    endif()

    # set backend compiler to DPC++ (= 0)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_COMPILER=0)

    # be able to choose between the Level-Zero and OpenCL DPC++ backend
    option(PLSSVM_SYCL_DPCPP_USE_LEVEL_ZERO "Enable Level-Zero backend in favor of OpenCL when using DPC++." OFF)
    if(PLSSVM_SYCL_BACKEND_DPCPP_USE_LEVEL_ZERO)
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE="level-zero")
    else()
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE="opencl")
    endif()
endif()


# link base library against SYCL library
target_link_libraries(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set compile definition that the SYCL backend is available
target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_SYCL_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME})

# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_SYCL_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)


# generate summary string
set(PLSSVM_SYCL_BACKEND_SUMMARY_STRING_COMPILER " - SYCL (${PLSSVM_SYCL_BACKEND_COMPILER}):")
include(${PROJECT_SOURCE_DIR}/cmake/assemble_summary_string.cmake)
assemble_summary_string(PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS)
set(PLSSVM_SYCL_BACKEND_SUMMARY_STRING "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_COMPILER}${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}" PARENT_SCOPE)

