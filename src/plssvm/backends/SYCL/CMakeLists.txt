## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

message(CHECK_START "Checking for SYCL backend")

# explicitly set sources for configuration
set(PLSSVM_SYCL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
)
# explicitly set header for configuration
set(PLSSVM_SYCL_INCLUDES
    ${PROJECT_SOURCE_DIR}/include/plssvm/backends/SYCL/detail/constants.hpp
    ${PROJECT_SOURCE_DIR}/include/plssvm/backends/SYCL/detail/device_ptr.hpp
    ${PROJECT_SOURCE_DIR}/include/plssvm/backends/SYCL/detail/utility.hpp
    ${PROJECT_SOURCE_DIR}/include/plssvm/backends/SYCL/csvm.hpp
    ${PROJECT_SOURCE_DIR}/include/plssvm/backends/SYCL/exceptions.hpp
)


# function used to configure SYCL files
function(configure_sycl include_name result_sources)
    message(STATUS "Configuring ${include_name}.")
    set(sources "")
    foreach(FILENAME ${PLSSVM_SYCL_SOURCES})
        string(REPLACE "SYCL" "${include_name}" NEW_FILENAME ${FILENAME})
        configure_file(${FILENAME} ${NEW_FILENAME} @ONLY)
        list(APPEND sources ${NEW_FILENAME})
        set(${result_sources} ${sources} PARENT_SCOPE)
    endforeach()
    foreach(FILENAME ${PLSSVM_SYCL_INCLUDES})
        string(REPLACE "SYCL" "${include_name}" NEW_FILENAME ${FILENAME})
        configure_file(${FILENAME} ${NEW_FILENAME} @ONLY)
    endforeach()
endfunction()


# set target properties
set_local_and_parent(PLSSVM_SYCL_BACKEND_LIBRARY_NAME plssvm-SYCL)
add_library(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} INTERFACE)

# list of found SYCL implementations
set(PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS "")

# check if SYCL can be enabled
message(CHECK_START "Checking for hipSYCL as SYCL implementation")

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
    message(CHECK_PASS "found")
    message(STATUS "Setting HIPSYCL_TARGETS to \"${HIPSYCL_TARGETS}\".")
    list(APPEND PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS "hipsycl")

    # configure SYCL files
    set(PLSSVM_SYCL_BACKEND_INCLUDE_NAME "hipSYCL")
    set(PLSSVM_SYCL_BACKEND_NAMESPACE_NAME "hipsycl")
    configure_sycl("${PLSSVM_SYCL_BACKEND_INCLUDE_NAME}" PLSSVM_SYCL_HIPSYCL_SOURCES)

    # set target properties
    set_local_and_parent(PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME plssvm-SYCL_hipsycl)
    add_library(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} SHARED ${PLSSVM_SYCL_HIPSYCL_SOURCES} ${CMAKE_CURRENT_LIST_DIR}/../gpu_csvm.cpp)
    list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME}")

    # necessary target_*_directories if more than one SYCL implementation could be found
    if(DEFINED ENV{PLSSVM_SYCL_HIPSYCL_INCLUDE_DIR})
        message(STATUS "PLSSVM_SYCL_HIPSYCL_INCLUDE_DIR is \"$ENV{PLSSVM_SYCL_HIPSYCL_INCLUDE_DIR}\"")
        target_include_directories(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PUBLIC $ENV{PLSSVM_SYCL_HIPSYCL_INCLUDE_DIR})
    endif()

    # add target compile definitions for hipSYCL
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PRIVATE PLSSVM_HAS_SYCL_BACKEND)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PUBLIC PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_COMPILER=1)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_COMPILER_NAME="hipSYCL")

    # hipSYCL's way to add SYCL to a target
    add_sycl_to_target(TARGET ${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} SOURCES ${PROJECT_SOURCE_DIR}/src/plssvm/backends/hipSYCL/csvm.cpp)

    # silence unknown options warnings
    target_compile_options(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PRIVATE -Wno-unknown-warning-option
            $<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>
            )

    # print note that Intel GPU architecture specifications are ignored when using hipSYCL
    if(DEFINED PLSSVM_INTEL_TARGET_ARCHS)
        message(STATUS "Ignoring specified Intel architectures \"${PLSSVM_INTEL_TARGET_ARCHS}\" in favor of SPIR-V when using hipSYCL!")
    endif()

    target_link_libraries(${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})
    target_link_libraries(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} INTERFACE ${PLSSVM_SYCL_BACKEND_HIPSYCL_LIBRARY_NAME})
else()
    message(CHECK_FAIL "not found")
endif()


message(CHECK_START "Checking for DPC++ as SYCL implementation")

# check if current compiler is Intel's clang llvm fork aka DPC++
set(PLSSVM_SYCL_BACKEND_CHECK_FOR_DPCPP_COMPILER OFF)
if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    # intel llvm fork must be clang
    execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --help
            OUTPUT_VARIABLE PLSSVM_DPCPP_CLANG_HELP_OUTPUT)
    # check if help message contains DPC++ specific values
    if("${PLSSVM_DPCPP_CLANG_HELP_OUTPUT}" MATCHES ".*intel.*" AND "${PLSSVM_DPCPP_CLANG_HELP_OUTPUT}" MATCHES ".*-fsycl.*")
        set(PLSSVM_SYCL_BACKEND_CHECK_FOR_DPCPP_COMPILER ON)
    endif()
endif()

if(PLSSVM_SYCL_BACKEND_CHECK_FOR_DPCPP_COMPILER)
    message(CHECK_PASS "found")
    list(APPEND PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS "dpcpp")

    # configure SYCL files
    set(PLSSVM_SYCL_BACKEND_INCLUDE_NAME "DPCPP")
    set(PLSSVM_SYCL_BACKEND_NAMESPACE_NAME "dpcpp")
    configure_sycl("${PLSSVM_SYCL_BACKEND_INCLUDE_NAME}" PLSSVM_SYCL_DPCPP_SOURCES)

    # set target properties
    set_local_and_parent(PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME plssvm-SYCL_dpcpp)
    add_library(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} SHARED ${PLSSVM_SYCL_DPCPP_SOURCES} ${CMAKE_CURRENT_LIST_DIR}/../gpu_csvm.cpp)
    list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME}")

    # necessary target_*_directories if more than one SYCL implementation could be found
    if(DEFINED ENV{PLSSVM_SYCL_DPCPP_INCLUDE_DIR})
        message(STATUS "PLSSVM_SYCL_DPCPP_INCLUDE_DIR is \"$ENV{PLSSVM_SYCL_DPCPP_INCLUDE_DIR}\"")
        target_include_directories(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC $ENV{PLSSVM_SYCL_DPCPP_INCLUDE_DIR})
    endif()

    # add target compile definitions for DPC++
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE PLSSVM_HAS_SYCL_BACKEND)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_COMPILER=0)
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_COMPILER_NAME="DPC++")

    # TODO: remove if DPC++ bug is fixed
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(FATAL_ERROR "The SYCL backend when using DPC++ currently does not support the Debug build type. For more information see: https://github.com/intel/llvm/issues/5754")
    endif()

    # enable DPC++ SYCL support
    target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -sycl-std=2020 -fsycl)
    target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -fsycl)

    set(PLSSVM_DPCPP_FSYCL_TARGETS "")
    # cpu targets
    if(DEFINED PLSSVM_CPU_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND PLSSVM_DPCPP_FSYCL_TARGETS "spir64_x86_64")
    endif()
    # nvidia targets
    if(DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND PLSSVM_DPCPP_FSYCL_TARGETS "nvptx64-nvidia-cuda")
    endif()
    # amd targets
    if(DEFINED PLSSVM_AMD_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND PLSSVM_DPCPP_FSYCL_TARGETS "amdgcn-amd-amdhsa")
        # add target specific flags for AOT -> must always be specified von amd targets
        if(NOT PLSSVM_NUM_AMD_TARGET_ARCHS EQUAL 1)
            message(FATAL_ERROR "DPC++ currently only supports a single AMD architecture specification but ${PLSSVM_NUM_AMD_TARGET_ARCHS} were provided!")
        endif()
        target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${PLSSVM_AMD_TARGET_ARCHS})
        target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${PLSSVM_AMD_TARGET_ARCHS})
    endif()
    # intel targets
    if(DEFINED PLSSVM_INTEL_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND PLSSVM_DPCPP_FSYCL_TARGETS "spir64_gen")
    endif()
    # set -fsycl-targets
    list(JOIN PLSSVM_DPCPP_FSYCL_TARGETS "," PLSSVM_DPCPP_FSYCL_TARGETS_STRING)
    target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -fsycl-targets=${PLSSVM_DPCPP_FSYCL_TARGETS_STRING})
    target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -fsycl-targets=${PLSSVM_DPCPP_FSYCL_TARGETS_STRING})


    # add option for DPC++ Ahead-of-Time (AOT) compilation
    option(PLSSVM_SYCL_BACKEND_DPCPP_ENABLE_AOT "Enables Ahead-of-Time compilation for DPC++." ON)
    if(PLSSVM_SYCL_BACKEND_DPCPP_ENABLE_AOT)
        message(STATUS "Enabled Ahead-of-Time (AOT) compilation with DPC++.")
        ## set AOT compiler flags
        # cpu targets
        if(DEFINED PLSSVM_CPU_TARGET_ARCHS)
            # add target specific flags for AOT
            if(PLSSVM_NUM_CPU_TARGET_ARCHS EQUAL 1)
                target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=spir64_x86_64 "-march=${PLSSVM_CPU_TARGET_ARCHS}")
                target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=spir64_x86_64 "-march=${PLSSVM_CPU_TARGET_ARCHS}")
            endif()
        endif()
        # nvidia targets
        if(DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
            # add target specific flags for AOT
            if(NOT PLSSVM_NUM_NVIDIA_TARGET_ARCHS EQUAL 1)
                message(FATAL_ERROR "DPC++ currently only supports a single NVIDIA architecture specification for AOT but ${PLSSVM_NUM_NVIDIA_TARGET_ARCHS} were provided!")
            endif()
            target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=${PLSSVM_NVIDIA_TARGET_ARCHS})
            target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=${PLSSVM_NVIDIA_TARGET_ARCHS})
        endif()
        # intel targets
        if(DEFINED PLSSVM_INTEL_TARGET_ARCHS)
            # add target specific flags for AOT
            list(JOIN PLSSVM_INTEL_TARGET_ARCHS "," PLSSVM_INTEL_TARGET_ARCHS_STRING)
            target_compile_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=spir64_gen "-device ${PLSSVM_INTEL_TARGET_ARCHS_STRING}")
            target_link_options(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE -Xsycl-target-backend=spir64_gen "-device ${PLSSVM_INTEL_TARGET_ARCHS_STRING}")
        endif()
    endif()

    # be able to choose between the Level-Zero and OpenCL DPC++ backend
    option(PLSSVM_SYCL_BACKEND_DPCPP_USE_LEVEL_ZERO "Enable Level-Zero backend in favor of OpenCL when using DPC++." OFF)
    if(PLSSVM_SYCL_BACKEND_DPCPP_USE_LEVEL_ZERO)
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE="level_zero")
    else()
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE="opencl")
    endif()

    target_link_libraries(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})
    target_link_libraries(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} INTERFACE ${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME})
else()
    message(CHECK_FAIL "not found")
endif()

## check if ANY SYCL implementation has been found
if(NOT PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS)
    message(CHECK_FAIL "not found")
    if(PLSSVM_ENABLE_SYCL_BACKEND MATCHES "ON")
        message(FATAL_ERROR "Cannot find requested backend: SYCL!")
    endif()
    return()
endif()

## check if environment variables are correctly set if more than one SYCL implementation has been found
list(LENGTH PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS PLSSVM_SYCL_BACKEND_NUM_FOUND_IMPLEMENTATIONS)
if(PLSSVM_SYCL_BACKEND_NUM_FOUND_IMPLEMENTATIONS GREATER 1)
    if(NOT DEFINED ENV{PLSSVM_SYCL_HIPSYCL_INCLUDE_DIR})
        message(FATAL_ERROR "Found more than one SYCL implementation, but the environment variable \"PLSSVM_SYCL_HIPSYCL_INCLUDE_DIR\" isn't set!")
    endif()
    if(NOT DEFINED ENV{PLSSVM_SYCL_DPCPP_INCLUDE_DIR})
        message(FATAL_ERROR "Found more than one SYCL implementation, but the environment variable \"PLSSVM_SYCL_DPCPP_INCLUDE_DIR\" isn't set!")
    endif()
endif()
message(CHECK_PASS "found")


# set the preferred SYCL implementation
if(DEFINED PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION)
    # command line value provided
    if(PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION IN_LIST PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS)
        # possible value found -> set preferred SYCL implementation
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} INTERFACE PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION=${PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION})
        message(STATUS "Setting preferred SYCL implementation to \"${PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION}\".")
    else()
        # value not recognized
        message(FATAL_ERROR "\"${PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION}\" as preferred SYCL implementation unrecognized or not available! Possible values are: ${PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS}.")
    endif()
else()
    # no command line value provided -> try to infer preferred SYCL implementation
    list(LENGTH PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS_NUM)
    if(PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS_NUM GREATER 1)
        # more than one SYCL implementation found -> can't infer preferred SYCL implementation
        message(FATAL_ERROR "Found more than one SYCL implementation, but \"PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION\" hasn't been set!")
    else()
        # use only available SYCL implementation as preferred implementation
        target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} INTERFACE PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION=${PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS})
        message(STATUS "Setting preferred SYCL implementation to \"${PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS}\".")
    endif()
endif()

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME})


# the SYCL backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_SYCL_BACKEND)


# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_SYCL_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)


# generate summary string
include(${PROJECT_SOURCE_DIR}/cmake/assemble_summary_string.cmake)
set(PLSSVM_SYCL_BACKEND_SUMMARY_STRINGS "")
foreach(SYCL_IMPLEMENTATION ${PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS})
    # mark preferred SYCL implementation
    get_target_property(PLSSVM_SYCL_BACKEND_LIBRARY_NAME_COMPILE_DEFINITIONS ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} INTERFACE_COMPILE_DEFINITIONS)
    if("${PLSSVM_SYCL_BACKEND_LIBRARY_NAME_COMPILE_DEFINITIONS}" MATCHES ".*${SYCL_IMPLEMENTATION}.*")
        set(SYCL_IMPLEMENTATION "${SYCL_IMPLEMENTATION}*")
    endif()

    assemble_summary_string(PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS)
    # do not print CPU and Intel GPU target architectures when using hipSYCL
    if("${SYCL_IMPLEMENTATION}" STREQUAL "hipsycl")
        string(REPLACE " (${PLSSVM_CPU_TARGET_ARCHS})" "" PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}")
        string(REPLACE " (${PLSSVM_INTEL_TARGET_ARCHS})" "" PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}")
    endif()
    list(APPEND PLSSVM_SYCL_BACKEND_SUMMARY_STRINGS " - SYCL (${SYCL_IMPLEMENTATION}):${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}")
endforeach()
set(PLSSVM_SYCL_BACKEND_SUMMARY_STRINGS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRINGS}" PARENT_SCOPE)

