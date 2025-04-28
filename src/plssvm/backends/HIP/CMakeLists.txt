# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "HIP:  ")

# check if HIP can be enabled
message(CHECK_START "Checking for HIP backend")

# check if HIP_PLATFORM is provided as environment variable
if (DEFINED ENV{HIP_PLATFORM})
    # the HIP_PLATFORM environment variable is defined
    if ("$ENV{HIP_PLATFORM}" STREQUAL "amd")
        # we want to target AMD GPUs according to the provided HIP_PLATFORM
        if (DEFINED PLSSVM_AMD_TARGET_ARCHS)
            # found AMD target
            set_local_and_parent(PLSSVM_HIP_BACKEND_GPU_RUNTIME "HIP")
        else ()
            # no AMD targets provided
            if (PLSSVM_ENABLE_HIP_BACKEND MATCHES "ON")
                message(SEND_ERROR "The \"HIP_PLATFORM\" is \"amd\" but no \"amd\" target was provided for the requested HIP backend!")
            else ()
                message(STATUS "The \"HIP_PLATFORM\" is \"amd\" but no \"amd\" target was provided for the HIP backend!")
            endif ()
            message(CHECK_FAIL "skipped")
            return()
        endif ()
    elseif ("$ENV{HIP_PLATFORM}" STREQUAL "nvidia")
        # we want to target NVIDIA GPUs according to the provided HIP_PLATFORM
        if (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
            # found NVIDIA target
            set_local_and_parent(PLSSVM_HIP_BACKEND_GPU_RUNTIME "CUDA")
        else ()
            # no NVIDIA targets provided
            if (PLSSVM_ENABLE_HIP_BACKEND MATCHES "ON")
                message(SEND_ERROR "The \"HIP_PLATFORM\" is \"nvidia\" but no \"nvidia\" target was provided for the requested HIP backend!")
            else ()
                message(STATUS "The \"HIP_PLATFORM\" is \"nvidia\" but no \"nvidia\" target was provided for the HIP backend!")
            endif ()
            message(CHECK_FAIL "skipped")
            return()
        endif ()
    else ()
        # invalid HIP_PLATFORM
        if (PLSSVM_ENABLE_HIP_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found invalid \"HIP_PLATFORM\" value \"$ENV{HIP_PLATFORM}\" for the requested HIP backend!")
        else ()
            message(STATUS "Found invalid \"HIP_PLATFORM\" value \"$ENV{HIP_PLATFORM}\" for the HIP backend!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
else ()
    # environment variable not defined -> try inferring it using the provided PLSSVM target platforms
    if (DEFINED PLSSVM_AMD_TARGET_ARCHS AND DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        # we try to target NVIDIA and AMD GPUs
        if (PLSSVM_ENABLE_HIP_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found \"amd\" and \"nvidia\" targets for the requested HIP backend, but only one of them are supported for the HIP backend!")
        else ()
            message(STATUS "Found \"amd\" and \"nvidia\" targets for the HIP backend, but only one of them are supported for the HIP backend.")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    elseif (DEFINED PLSSVM_AMD_TARGET_ARCHS)
        # we target AMD GPUs but NO NVIDIA GPUs -> infer HIP_PLATFORM as amd
        set(ENV{HIP_PLATFORM} "amd")
        set(HIP_PLATFORM "amd" CACHE STRING "set the HIP_PLATFORM to AMD" FORCE)
        set(CMAKE_HIP_PLATFORM "amd" CACHE STRING "set the HIP_PLATFORM to AMD" FORCE)
        set_local_and_parent(PLSSVM_HIP_BACKEND_GPU_RUNTIME "HIP")
    elseif (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        # we target NVIDIA GPUs but NO AMD GPUs -> infer HIP_PLATFORM as nvidia
        set(ENV{HIP_PLATFORM} "nvidia")
        set(HIP_PLATFORM "nvidia" CACHE STRING "set the HIP_PLATFORM to NVIDIA" FORCE)
        set(CMAKE_HIP_PLATFORM "nvidia" CACHE STRING "set the HIP_PLATFORM to NVIDIA" FORCE)
        set_local_and_parent(PLSSVM_HIP_BACKEND_GPU_RUNTIME "CUDA")
    else ()
        # no AMD or NVIDIA GPUs requested -> can't find HIP
        if (PLSSVM_ENABLE_HIP_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found no \"amd\" or \"nvidia\" targets for the requested HIP backend!")
        else ()
            message(STATUS "Found no \"amd\" or \"nvidia\" targets for the HIP backend.")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    message(STATUS "Environment variable \"HIP_PLATFORM\" is not defined. Setting it to \"$ENV{HIP_PLATFORM}\"")
endif ()

include(CheckLanguage)
check_language(${PLSSVM_HIP_BACKEND_GPU_RUNTIME})

if (NOT CMAKE_${PLSSVM_HIP_BACKEND_GPU_RUNTIME}_COMPILER)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_HIP_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: HIP (for GPU runtime ${PLSSVM_HIP_BACKEND_GPU_RUNTIME})!")
    endif ()
    return()
endif ()

enable_language(${PLSSVM_HIP_BACKEND_GPU_RUNTIME})
# always necessary (also with HIP_PLATFORM nvidia) in order to be able to link against hip:: targets
find_package(HIP)
if (NOT HIP_FOUND)
    if (PLSSVM_ENABLE_HIP_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: HIP!")
    else ()
        message(STATUS "Cannot find backend: HIP!")
    endif ()
    message(CHECK_FAIL "skipped")
    return()
endif ()
message(CHECK_PASS "found")

# set necessary flags
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if (DEFINED PLSSVM_AMD_TARGET_ARCHS)
    # hipcc is used to compile the HIP code -> set flags accordingly
    set(CMAKE_HIP_FLAGS
        "${CMAKE_HIP_FLAGS} -std=c++17 -Wall -Wextra -Wdouble-promotion -Wshadow -Wcast-qual -Wnull-dereference -Wextra-semi -Wunreachable-code -Wuninitialized -Wmost -Wconversion"
    )
    # build type specific flags
    set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "${CMAKE_HIP_FLAGS_RELWITHDEBINFO} -O2")
    set(CMAKE_HIP_FLAGS_RELEASE "${CMAKE_HIP_FLAGS_RELEASE} -O3")
    # set fast-math if requested
    if (PLSSVM_ENABLE_FAST_MATH)
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -ffast-math")
    endif ()
elseif (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
    # nvcc is used to compile the HIP code -> set flags accordingly
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-fPIC' -Xcudafe --diag_suppress=unsigned_compare_with_zero -lineinfo")
    # build type specific flags
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xptxas=-v -Xptxas -dlcm=cg -Xcudafe --diag_suppress=unrecognized_gcc_pragma")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -O2")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
    # set fast-math if requested
    if (PLSSVM_ENABLE_FAST_MATH)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    endif ()
    # set OpenMP specific flags
    if (OpenMP_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-fopenmp'")
    endif ()
endif ()

# explicitly set sources
set(PLSSVM_HIP_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.hip
    ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.hip
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.hip
    ${CMAKE_CURRENT_LIST_DIR}/csvm.hip
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.hip
)

# set source file language
set_source_files_properties(${PLSSVM_HIP_SOURCES} PROPERTIES LANGUAGE ${PLSSVM_HIP_BACKEND_GPU_RUNTIME})

# set target properties
set_local_and_parent(PLSSVM_HIP_BACKEND_LIBRARY_NAME plssvm-HIP)
add_library(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_HIP_SOURCES})

# set_target_properties(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTIES HIP_SEPARABLE_COMPILATION ON) #< https://gitlab.kitware.com/cmake/cmake/-/issues/23210

target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC -Wno-sign-compare)

# set target architecture
if (DEFINED PLSSVM_AMD_TARGET_ARCHS)
    set_property(TARGET ${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTY HIP_ARCHITECTURES ${PLSSVM_AMD_TARGET_ARCHS})
elseif (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
    # prepare architectures for CMake's CUDA_ARCHITECTURES
    list(TRANSFORM PLSSVM_NVIDIA_TARGET_ARCHS REPLACE "sm_" "")
    set_property(TARGET ${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES ${PLSSVM_NVIDIA_TARGET_ARCHS})
endif ()

# link base library against HIP library
target_link_libraries(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PRIVATE hip::device hip::host)
set_target_properties(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE ${PLSSVM_HIP_BACKEND_GPU_RUNTIME})
target_link_libraries(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set compile definition that the HIP backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_HAS_HIP_BACKEND)
target_compile_definitions(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_HIP_BACKEND)
if (DEFINED PLSSVM_AMD_TARGET_ARCHS)
    target_compile_definitions(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HIP_BACKEND_USE_HIP_RUNTIME)
elseif (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
    target_compile_definitions(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HIP_BACKEND_USE_CUDA_RUNTIME)
endif ()

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_HIP_BACKEND_LIBRARY_NAME})

# mark backend library as install target
append_local_and_parent(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_HIP_BACKEND_LIBRARY_NAME})

# generate summary string
if (DEFINED PLSSVM_AMD_TARGET_ARCHS)
    set(PLSSVM_HIP_BACKEND_SUMMARY_STRING " - HIP: amd (${PLSSVM_AMD_TARGET_ARCHS})" PARENT_SCOPE)
elseif (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
    list(TRANSFORM PLSSVM_NVIDIA_TARGET_ARCHS PREPEND "sm_")
    set(PLSSVM_HIP_BACKEND_SUMMARY_STRING " - HIP: nvidia (${PLSSVM_NVIDIA_TARGET_ARCHS})" PARENT_SCOPE)
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
