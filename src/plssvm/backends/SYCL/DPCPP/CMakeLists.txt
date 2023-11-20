## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################


message(CHECK_START "Checking for DPC++ as SYCL implementation")

# check if current compiler is Intel's clang llvm fork aka DPC++
set(PLSSVM_SYCL_BACKEND_CHECK_FOR_DPCPP_COMPILER OFF)
if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    # intel llvm fork must be clang
    execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --help
            OUTPUT_VARIABLE PLSSVM_DPCPP_CLANG_HELP_OUTPUT)
    # check if help message contains DPC++ specific values
    if ("${PLSSVM_DPCPP_CLANG_HELP_OUTPUT}" MATCHES ".*intel.*" AND "${PLSSVM_DPCPP_CLANG_HELP_OUTPUT}" MATCHES ".*-fsycl.*")
        set(PLSSVM_SYCL_BACKEND_CHECK_FOR_DPCPP_COMPILER ON)
    endif ()
endif ()

if (PLSSVM_SYCL_BACKEND_CHECK_FOR_DPCPP_COMPILER)
    message(CHECK_PASS "found")
    append_local_and_parent(PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS "dpcpp")

    # set DPC++ specific targets
    set(PLSSVM_SYCL_DPCPP_SOURCES
            ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
            ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
            ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    )

    # set target properties
    set(PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME plssvm-SYCL_dpcpp CACHE INTERNAL "")
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

    # enable DPC++ SYCL support
    target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -sycl-std=2020 -fsycl -fno-sycl-id-queries-fit-in-int)
    target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -fsycl)

    set(PLSSVM_DPCPP_FSYCL_TARGETS "")
    # cpu targets
    if (DEFINED PLSSVM_CPU_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND PLSSVM_DPCPP_FSYCL_TARGETS "spir64_x86_64")
    endif ()
    # nvidia targets
    if (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND PLSSVM_DPCPP_FSYCL_TARGETS "nvptx64-nvidia-cuda")
    endif ()
    # amd targets
    if (DEFINED PLSSVM_AMD_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND PLSSVM_DPCPP_FSYCL_TARGETS "amdgcn-amd-amdhsa")
        # add target specific flags for AOT -> must always be specified von amd targets
        if (NOT PLSSVM_NUM_AMD_TARGET_ARCHS EQUAL 1)
            message(SEND_ERROR "DPC++ currently only supports a single AMD architecture specification but ${PLSSVM_NUM_AMD_TARGET_ARCHS} were provided!")
        endif ()
        target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${PLSSVM_AMD_TARGET_ARCHS})
        target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${PLSSVM_AMD_TARGET_ARCHS})
    endif ()
    # intel targets
    if (DEFINED PLSSVM_INTEL_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND PLSSVM_DPCPP_FSYCL_TARGETS "spir64_gen")
    endif ()
    # set -fsycl-targets
    list(JOIN PLSSVM_DPCPP_FSYCL_TARGETS "," PLSSVM_DPCPP_FSYCL_TARGETS_STRING)
    target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -fsycl-targets=${PLSSVM_DPCPP_FSYCL_TARGETS_STRING})
    target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -fsycl-targets=${PLSSVM_DPCPP_FSYCL_TARGETS_STRING})


    # add option for DPC++ Ahead-of-Time (AOT) compilation
    option(PLSSVM_SYCL_BACKEND_DPCPP_ENABLE_AOT "Enables Ahead-of-Time compilation for DPC++." ON)
    if (PLSSVM_SYCL_BACKEND_DPCPP_ENABLE_AOT)
        message(STATUS "Enabled Ahead-of-Time (AOT) compilation with DPC++.")
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_DPCPP_ENABLE_AOT)
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC PLSSVM_SYCL_BACKEND_DPCPP_ENABLE_AOT)
        ## set AOT compiler flags
        # cpu targets
        if (DEFINED PLSSVM_CPU_TARGET_ARCHS)
            # add target specific flags for AOT
            if (PLSSVM_NUM_CPU_TARGET_ARCHS EQUAL 1)
                target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=spir64_x86_64 "-march=${PLSSVM_CPU_TARGET_ARCHS}")
                target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=spir64_x86_64 "-march=${PLSSVM_CPU_TARGET_ARCHS}")
            endif ()
        endif ()
        # nvidia targets
        if (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
            # add target specific flags for AOT
            if (NOT PLSSVM_NUM_NVIDIA_TARGET_ARCHS EQUAL 1)
                message(SEND_ERROR "DPC++ currently only supports a single NVIDIA architecture specification for AOT but ${PLSSVM_NUM_NVIDIA_TARGET_ARCHS} were provided!")
            endif ()
            target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=${PLSSVM_NVIDIA_TARGET_ARCHS})
            target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=${PLSSVM_NVIDIA_TARGET_ARCHS})
        endif ()
        # intel targets
        if (DEFINED PLSSVM_INTEL_TARGET_ARCHS)
            # add target specific flags for AOT
            list(JOIN PLSSVM_INTEL_TARGET_ARCHS "," PLSSVM_INTEL_TARGET_ARCHS_STRING)
            target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=spir64_gen "-device ${PLSSVM_INTEL_TARGET_ARCHS_STRING}")
            target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=spir64_gen "-device ${PLSSVM_INTEL_TARGET_ARCHS_STRING}")
        endif ()
    endif ()

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

    # be able to choose between the HIP and OpenCL DPC++ backend for AMD GPUs
    option(PLSSVM_SYCL_BACKEND_DPCPP_GPU_AMD_USE_HIP "Enable DPC++'s HIP backend in favor of the OpenCL backend for AMD GPUs." ON)
    if (PLSSVM_SYCL_BACKEND_DPCPP_GPU_AMD_USE_HIP)
        message(STATUS "Using DPC++'s HIP backend for AMD GPUs.")
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_DPCPP_GPU_AMD_BACKEND_TYPE="hip")
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC PLSSVM_SYCL_BACKEND_DPCPP_GPU_AMD_BACKEND_TYPE="hip")
    else ()
        message(STATUS "Using DPC++'s OpenCL backend for AMD GPUs.")
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_DPCPP_GPU_AMD_BACKEND_TYPE="opencl")
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC PLSSVM_SYCL_BACKEND_DPCPP_GPU_AMD_BACKEND_TYPE="opencl")
    endif ()

    target_link_libraries(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})
else ()
    message(CHECK_FAIL "not found")
endif ()