## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

cmake_minimum_required(VERSION 3.21)

project("PLSSVM - Parallel Least Squares Support Vector Machine"
        VERSION 2.0.0
        LANGUAGES CXX
        DESCRIPTION "A Least Squares Support Vector Machine implementation using different backends.")


## include some generally used utility scripts
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/add_coverage_build_type.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/utility_macros.cmake)

## add custom module files
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

## set default CMake build type
set(PLSSVM_DEFAULT_BUILD_TYPE "RelWithDebInfo")
set(PLSSVM_ALLOWED_BUILD_TYPES "Debug;Release;MinSizeRel;RelWithDebInfo;Coverage")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${PLSSVM_DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${PLSSVM_DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${PLSSVM_ALLOWED_BUILD_TYPES})
else ()
    if ("${CMAKE_BUILD_TYPE}" IN_LIST PLSSVM_ALLOWED_BUILD_TYPES)
        message(STATUS "The build type is '${CMAKE_BUILD_TYPE}'.")
    else ()
        message(FATAL_ERROR "Unrecognized build type '${CMAKE_BUILD_TYPE}'! Allowed build types are: ${PLSSVM_ALLOWED_BUILD_TYPES}")
    endif ()
endif ()


########################################################################################################################
##                                      create and set necessary base properties                                      ##
########################################################################################################################
## set base sources
set(PLSSVM_BASE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/backends/SYCL/kernel_invocation_type.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/backends/SYCL/implementation_type.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/detail/cmd/parser_predict.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/detail/cmd/parser_scale.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/detail/cmd/parser_train.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/detail/io/file_reader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/detail/execution_range.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/detail/layout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/detail/logger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/detail/performance_tracker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/detail/sha256.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/detail/string_utility.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/detail/utility.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/exceptions/exceptions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/version/version.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/version/git_metadata/git_metadata.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/backend_types.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/file_format_types.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/kernel_function_types.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/parameter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/target_platforms.cpp
        )

## create base library: linked against all backend libraries
set(PLSSVM_BASE_LIBRARY_NAME plssvm-base)
add_library(${PLSSVM_BASE_LIBRARY_NAME} STATIC ${PLSSVM_BASE_SOURCES})
## create all library: one target against all backends are linked
set(PLSSVM_ALL_LIBRARY_NAME plssvm-all)
add_library(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE)

## create list of targets to install
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_ALL_LIBRARY_NAME} ${PLSSVM_BASE_LIBRARY_NAME})

## set include directory
target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )

## set library cxx standard
target_compile_features(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC cxx_std_17)
## additional base library compile options
target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wdouble-promotion -fno-common -Wshadow -Wcast-qual
        -Wnull-dereference -Wctor-dtor-privacy -Wnon-virtual-dtor -Wextra-semi -Wunreachable-code -Wuninitialized
        -fPIC>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wsuggest-override -Wstrict-null-sentinel -Wlogical-op -Wduplicated-branches -Wimplicit-fallthrough=5>
        $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wdocumentation -Wmost>
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4 /bigobj /wd4459>>
        )

## nvcc doesn't recognize -Werror=??? option, so only set it when using a CXX compiler
target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Werror=switch -fstrict-enums>
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/we4062 /wd4005>
        )
## enable additional optimization flags only in RELEASE mode
target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
        $<$<AND:$<CONFIG:RELEASE>,$<CXX_COMPILER_ID:GNU,Clang>>:-ffast-math -march=native>
        $<$<AND:$<CONFIG:RELEASE>,$<CXX_COMPILER_ID:MSVC>>:/fp:fast>
        )
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:NOMINMAX>
        )


########################################################################################################################
##                                    check for optional and necessary dependencies                                   ##
########################################################################################################################
# check for OpenMP (not for the backend!)
find_package(OpenMP 4.0 QUIET)
if (OpenMP_FOUND)
    message(STATUS "Found OpenMP ${OpenMP_CXX_VERSION} to speed-up library utilities (like file parsing).")
    set(PLSSVM_FOUND_OPENMP_FOR_UTILITY "${OpenMP_CXX_VERSION}")
    target_link_libraries(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX -fopenmp)
    target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
else ()
    # disable warning for unknown OpenMP pragmas if no OpenMP could be found
    target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
            $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wno-unknown-pragmas>
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4068>
            )
endif ()

## setup dependencies
include(FetchContent)
list(APPEND CMAKE_MESSAGE_INDENT "Dependencies:  ")

## try finding cxxopts
find_package(cxxopts 3.0.0 QUIET)
if (cxxopts_FOUND)
    message(STATUS "Found package cxxopts.")
    target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${cxxopts_INCLUDE_DIR})
else ()
    message(STATUS "Couldn't find package cxxopts. Building from source ...")
    set(PLSSVM_cxxopts_VERSION v3.0.0)
    # fetch command line parser library cxxopts
    FetchContent_Declare(cxxopts
            GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
            GIT_TAG ${PLSSVM_cxxopts_VERSION}
            GIT_SHALLOW TRUE
            QUIET

            set (CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
            set (CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
            )
    FetchContent_MakeAvailable(cxxopts)
    add_dependencies(${PLSSVM_BASE_LIBRARY_NAME} cxxopts)
    target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
            $<BUILD_INTERFACE:${cxxopts_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
            )
    message(STATUS "Installing cxxopts version ${PLSSVM_cxxopts_VERSION}.")
endif ()

## try finding fast_float
find_package(fast_float QUIET)
if (fast_float_FOUND)
    message(STATUS "Found package fast_float.")
    target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${fast_float_INCLUDE_DIR})
else ()
    message(STATUS "Couldn't find package fast_float. Building from source ...")
    set(PLSSVM_fast_float_VERSION v3.4.0)
    # fetch float parsing library fast_float
    FetchContent_Declare(fast_float
            GIT_REPOSITORY https://github.com/fastfloat/fast_float
            GIT_TAG ${PLSSVM_fast_float_VERSION}
            GIT_SHALLOW TRUE
            QUIET

            set (FASTFLOAT_TEST OFF CACHE BOOL "" FORCE)
            set (FASTFLOAT_SANITIZE OFF CACHE BOOL "" FORCE)
            )
    FetchContent_GetProperties(fast_float)
    if (NOT fast_float_POPULATED)
        FetchContent_Populate(fast_float)
        add_subdirectory(${fast_float_SOURCE_DIR} EXCLUDE_FROM_ALL)
    endif ()
    add_dependencies(${PLSSVM_BASE_LIBRARY_NAME} fast_float)
    target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
            $<BUILD_INTERFACE:${fast_float_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
            )
    list(APPEND PLSSVM_TARGETS_TO_INSTALL "fast_float")
    message(STATUS "Installing fast_float version ${PLSSVM_fast_float_VERSION}.")
endif ()

## try finding igor
find_package(igor QUIET)
if (igor_FOUND)
    message(STATUS "Found package igor.")
    target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${igor_INCLUDE_DIR})
else ()
    message(STATUS "Couldn't find package igor. Building from source ...")
    set(PLSSVM_igor_VERSION a5224c60d266974d3f407191583fe266cbe1c93d)
    # fetch named argument library igor
    FetchContent_Declare(igor
            GIT_REPOSITORY https://github.com/bluescarni/igor
            GIT_TAG ${PLSSVM_igor_VERSION}
            GIT_SHALLOW TRUE
            QUIET

            set (IGOR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
            )
    FetchContent_GetProperties(igor)
    if (NOT igor_POPULATED)
        FetchContent_Populate(igor)
        add_subdirectory(${igor_SOURCE_DIR} EXCLUDE_FROM_ALL)
    endif ()
    add_dependencies(${PLSSVM_BASE_LIBRARY_NAME} igor)
    target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
            $<BUILD_INTERFACE:${igor_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
            )
    list(APPEND PLSSVM_TARGETS_TO_INSTALL "igor")
    message(STATUS "Installing igor version ${PLSSVM_igor_VERSION}.")
endif ()

## try finding fmt
find_package(fmt 8.1.1 QUIET)
if (fmt_FOUND)
    message(STATUS "Found package fmt.")
else ()
    message(STATUS "Couldn't find package fmt. Building from source ...")
    set(PLSSVM_fmt_VERSION 8.1.1)
    # fetch string formatting library fmt
    FetchContent_Declare(fmt
            GIT_REPOSITORY https://github.com/fmtlib/fmt.git
            GIT_TAG ${PLSSVM_fmt_VERSION}
            GIT_SHALLOW TRUE
            QUIET

            set (FMT_TEST OFF CACHE BOOL "" FORCE)
            set (FMT_DOC OFF CACHE BOOL "" FORCE)
            set (FMT_INSTALL OFF CACHE BOOL "" FORCE)
            set (FMT_SYSTEM_HEADERS ON CACHE BOOL "" FORCE)
            )
    FetchContent_MakeAvailable(fmt)
    set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)
    add_dependencies(${PLSSVM_BASE_LIBRARY_NAME} fmt)
    target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
            $<BUILD_INTERFACE:${fmt_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
            )
    # append fmt to installed targets if build from source
    list(APPEND PLSSVM_TARGETS_TO_INSTALL "fmt")
    message(STATUS "Installing {fmt} version ${PLSSVM_fmt_VERSION}.")
endif ()
target_link_libraries(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC fmt::fmt)

list(POP_BACK CMAKE_MESSAGE_INDENT)

########################################################################################################################
##                                                 create executables                                                 ##
########################################################################################################################
## create train executable
set(PLSSVM_EXECUTABLE_TRAIN_NAME plssvm-train)
add_executable(${PLSSVM_EXECUTABLE_TRAIN_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main_train.cpp)
## create predict executable
set(PLSSVM_EXECUTABLE_PREDICT_NAME plssvm-predict)
add_executable(${PLSSVM_EXECUTABLE_PREDICT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main_predict.cpp)
## create scale executable
set(PLSSVM_EXECUTABLE_SCALE_NAME plssvm-scale)
add_executable(${PLSSVM_EXECUTABLE_SCALE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main_scale.cpp)

## append executables to installed targets
list(APPEND PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_EXECUTABLE_TRAIN_NAME} ${PLSSVM_EXECUTABLE_PREDICT_NAME} ${PLSSVM_EXECUTABLE_SCALE_NAME})

########################################################################################################################
##                                            setup code coverage analysis                                            ##
########################################################################################################################
## coverage analysis only possible with the Coverage CMAKE_BUILD_TYPE
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (uppercase_CMAKE_BUILD_TYPE MATCHES "COVERAGE")
    # must be linux
    if (WIN32 OR APPLE)
        message(FATAL_ERROR "Only Linux is supported for the coverage analysis.")
    endif ()
    # must be GCC
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        message(FATAL_ERROR "Only GCC is supported for the coverage analysis.")
    endif ()
    # tests must be available for a coverage analysis
    message(STATUS "Enabling tests since they are necessary for the coverage analysis.")
    set(PLSSVM_ENABLE_TESTING ON CACHE BOOL "" FORCE)
    # assertions must be available for a coverage analysis
    message(STATUS "Enabling assertions since they are necessary for the coverage analysis.")
    set(PLSSVM_ENABLE_ASSERTS ON CACHE BOOL "" FORCE)

    message(STATUS "Enable code coverage analysis using lcov.")

    # Create the coverage target. Run coverage tests with 'make coverage'
    add_custom_target(coverage
            COMMAND lcov --zerocounters --directory .
            COMMAND lcov --capture -d . --initial --output-file test_base.info
            COMMAND mkdir -p coverage
            COMMAND ${CMAKE_MAKE_PROGRAM} test || true
            COMMAND lcov --capture -d . --output-file test_test.info
            COMMAND lcov --add-tracefile test_base.info --add-tracefile test_test.info -o test_total.info
            COMMAND lcov --remove test_total.info '/usr/*' '*/build/*' '*/tests/*' '*/_deps/*' -o test_clean.info
            COMMAND genhtml test_clean.info --output-directory coverage --title "PLSSVM Test Coverage" --show-details --legend
            BYPRODUCTS ${CMAKE_BINARY_DIR}/test_base.info ${CMAKE_BINARY_DIR}/test_test.info ${CMAKE_BINARY_DIR}/test_total.info ${CMAKE_BINARY_DIR}/test_clean.info ${CMAKE_BINARY_DIR}/coverage
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            )

    # add executables as coverage target
    add_dependencies(coverage ${PLSSVM_EXECUTABLE_TRAIN_NAME})
    add_dependencies(coverage ${PLSSVM_EXECUTABLE_PREDICT_NAME})
    add_dependencies(coverage ${PLSSVM_EXECUTABLE_SCALE_NAME})

    # add custom target `make clean_coverage` which calls `make clean` and also removes all generate *.gcda and *.gcno files
    add_custom_target(clean_coverage)
    add_custom_command(
            DEPENDS clean
            COMMENT "remove all coverage files"
            COMMAND ${CMAKE_MAKE_PROGRAM} clean
            COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/delete_coverage_files.cmake"
            TARGET clean_coverage
    )
endif ()


########################################################################################################################
##                                         parse target platform information                                          ##
########################################################################################################################
## the target platforms and architectures must either be set using the CMake command line option
## PLSSVM_TARGET_PLATFORMS or the environment variable with the same name
if (DEFINED PLSSVM_TARGET_PLATFORMS)
    set(PLSSVM_TARGET_PLATFORMS ${PLSSVM_TARGET_PLATFORMS} CACHE STRING "The target platforms to compile for." FORCE)
elseif (DEFINED ENV{PLSSVM_TARGET_PLATFORMS})
    set(PLSSVM_TARGET_PLATFORMS $ENV{PLSSVM_TARGET_PLATFORMS} CACHE STRING "The target platforms to compile for." FORCE)
else ()
    # check for Python3 and all necessary libraries
    find_package(Python3 COMPONENTS Interpreter Development)
    if (NOT Python3_FOUND)
        message(FATAL_ERROR "Can't find Python3. Please manually specify PLSSVM_TARGET_PLATFORMS (e.g. -DPLSSVM_TARGET_PLATFORMS=\"cpu;nvidia:sm_70,sm_86;amd:gfx906;intel:skl\"!")
    endif ()

    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_python_libs.cmake)
    set(PLSSVM_TARGET_PLATFORMS_PYTHON_SCRIPT_REQUIRED_LIBS cpuinfo GPUtil pyamdgpuinfo pylspci)
    message(STATUS "Checking required Python3 libraries (${PLSSVM_TARGET_PLATFORMS_PYTHON_SCRIPT_REQUIRED_LIBS}) to automatically determine the PLSSVM_TARGET_PLATFORMS.")
    set(PLSSVM_TARGET_PLATFORMS_PYTHON_SCRIPT_REQUIRED_LIBS_ERROR_MESSAGE "or manually define PLSSVM_TARGET_PLATFORMS (e.g. -DPLSSVM_TARGET_PLATFORMS=\"cpu;nvidia:sm_70,sm_86;amd:gfx906;intel:skl\"!")
    check_python_libs(${PLSSVM_TARGET_PLATFORMS_PYTHON_SCRIPT_REQUIRED_LIBS} ${PLSSVM_TARGET_PLATFORMS_PYTHON_SCRIPT_REQUIRED_LIBS_ERROR_MESSAGE})

    # run our `plssvm_target_platforms.py` script to determine the PLSSVM_TARGET_PLATFORMS string
    execute_process(
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/utility_scripts/plssvm_target_platforms.py" "--quiet"
            RESULT_VARIABLE PLSSVM_PYTHON_TARGET_PLATFORMS_EXIT_CODE
            OUTPUT_VARIABLE PLSSVM_PYTHON_TARGET_PLATFORMS_OUTPUT
    )

    # an error occurred when running our python script
    if (NOT ${PLSSVM_PYTHON_TARGET_PLATFORMS_EXIT_CODE} EQUAL 0)
        message(FATAL_ERROR
                "Error running '${CMAKE_CURRENT_SOURCE_DIR}/utility_scripts/plssvm_target_platforms.py'."
                "Please manually specify PLSSVM_TARGET_PLATFORMS (e.g. -DPLSSVM_TARGET_PLATFORMS=\"cpu;nvidia:sm_70,sm_86;amd:gfx906;intel:skl\"!"
                )
    endif ()

    # set PLSSVM_TARGET_PLATFORMS
    string(STRIP "${PLSSVM_PYTHON_TARGET_PLATFORMS_OUTPUT}" PLSSVM_TARGET_PLATFORMS)
    message(STATUS "Automatically derived PLSSVM_TARGET_PLATFORMS=\"${PLSSVM_TARGET_PLATFORMS}\".")
endif ()

## PLSSVM_TARGET_PLATFORMS must not be empty
if (PLSSVM_TARGET_PLATFORMS STREQUAL "")
    message(FATAL_ERROR "PLSSVM_TARGET_PLATFORMS must not be empty!")
endif ()

## parse provided target platforms
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/parse_architecture_info.cmake)
set(PLSSVM_PLATFORM_NAME_LIST "automatic")
foreach (PLSSVM_PLATFORM ${PLSSVM_TARGET_PLATFORMS})
    if (PLSSVM_PLATFORM MATCHES "^cpu")
        # parse provided CPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_CPU_TARGET_ARCHS PLSSVM_NUM_CPU_TARGET_ARCHS)
        if (PLSSVM_NUM_CPU_TARGET_ARCHS GREATER 1)
            message(FATAL_ERROR "Target platform \"cpu\" must at most have one architecture specification!")
        endif ()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_CPU_TARGET)
        list(APPEND PLSSVM_PLATFORM_NAME_LIST "cpu")
    elseif (PLSSVM_PLATFORM MATCHES "^nvidia")
        # parse provided NVIDIA GPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_NVIDIA_TARGET_ARCHS PLSSVM_NUM_NVIDIA_TARGET_ARCHS)
        if (PLSSVM_NUM_NVIDIA_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"nvidia\" must at least have one architecture specification!")
        endif ()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_NVIDIA_TARGET)
        list(APPEND PLSSVM_PLATFORM_NAME_LIST "gpu_nvidia")
    elseif (PLSSVM_PLATFORM MATCHES "^amd")
        # parse provided AMD GPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_AMD_TARGET_ARCHS PLSSVM_NUM_AMD_TARGET_ARCHS)
        if (PLSSVM_NUM_AMD_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"amd\" must at least have one architecture specification!")
        endif ()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_AMD_TARGET)
        list(APPEND PLSSVM_PLATFORM_NAME_LIST "gpu_amd")
    elseif (PLSSVM_PLATFORM MATCHES "^intel")
        # parse provided Intel GPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_INTEL_TARGET_ARCHS PLSSVM_NUM_INTEL_TARGET_ARCHS)
        if (PLSSVM_NUM_INTEL_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"intel\" must at least have one architecture specification!")
        endif ()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_INTEL_TARGET)
        list(APPEND PLSSVM_PLATFORM_NAME_LIST "gpu_intel")
    else ()
        message(FATAL_ERROR "Unrecognized target platform \"${PLSSVM_PLATFORM}\"! Must be one of: cpu nvidia amd intel")
    endif ()
endforeach ()


########################################################################################################################
##                                            check for and enable backends                                           ##
########################################################################################################################
## check for OpenMP backend
set(PLSSVM_ENABLE_OPENMP_BACKEND AUTO CACHE STRING "Enable OpenMP Backend")
set_property(CACHE PLSSVM_ENABLE_OPENMP_BACKEND PROPERTY STRINGS AUTO ON OFF)
if (PLSSVM_ENABLE_OPENMP_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_OPENMP_BACKEND)
    add_subdirectory(src/plssvm/backends/OpenMP)
endif ()

## check for CUDA backend
set(PLSSVM_ENABLE_CUDA_BACKEND AUTO CACHE STRING "Enable CUDA Backend")
set_property(CACHE PLSSVM_ENABLE_CUDA_BACKEND PROPERTY STRINGS AUTO ON OFF)
if (PLSSVM_ENABLE_CUDA_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_CUDA_BACKEND)
    add_subdirectory(src/plssvm/backends/CUDA)
endif ()

## check for HIP backend
set(PLSSVM_ENABLE_HIP_BACKEND AUTO CACHE STRING "Enable HIP Backend")
set_property(CACHE PLSSVM_ENABLE_HIP_BACKEND PROPERTY STRINGS AUTO ON OFF)
if (PLSSVM_ENABLE_HIP_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_HIP_BACKEND)
    add_subdirectory(src/plssvm/backends/HIP)
endif ()

## check for OpenCL backend
set(PLSSVM_ENABLE_OPENCL_BACKEND AUTO CACHE STRING "Enable OpenCL Backend")
set_property(CACHE PLSSVM_ENABLE_OPENCL_BACKEND PROPERTY STRINGS AUTO ON OFF)
if (PLSSVM_ENABLE_OPENCL_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_OPENCL_BACKEND)
    add_subdirectory(src/plssvm/backends/OpenCL)
endif ()

## check for SYCL backend
set(PLSSVM_ENABLE_SYCL_BACKEND AUTO CACHE STRING "Enable SYCL Backend")
set_property(CACHE PLSSVM_ENABLE_SYCL_BACKEND PROPERTY STRINGS AUTO ON OFF)
if (PLSSVM_ENABLE_SYCL_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_SYCL_BACKEND)
    add_subdirectory(src/plssvm/backends/SYCL)
endif ()

## check if ANY backend is available/has been enabled
get_target_property(PLSSVM_LINKED_BACKENDS ${PLSSVM_ALL_LIBRARY_NAME} INTERFACE_LINK_LIBRARIES)
if (NOT PLSSVM_LINKED_BACKENDS)
    message(FATAL_ERROR "At least one backend must be available!")
endif ()

## link backends against executables
target_link_libraries(${PLSSVM_EXECUTABLE_TRAIN_NAME} ${PLSSVM_ALL_LIBRARY_NAME})
target_link_libraries(${PLSSVM_EXECUTABLE_PREDICT_NAME} ${PLSSVM_ALL_LIBRARY_NAME})
target_link_libraries(${PLSSVM_EXECUTABLE_SCALE_NAME} ${PLSSVM_ALL_LIBRARY_NAME})


########################################################################################################################
##                                         add additional compile definitions                                         ##
########################################################################################################################
## add assert option
option(PLSSVM_ENABLE_ASSERTS "Enables PLSSVM asserts even if NDEBUG is set." OFF)
if (PLSSVM_ENABLE_ASSERTS)
    message(STATUS "Enable additional debugging assertions.")
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_ENABLE_ASSERTS)
endif ()

## set specific thread block sizes if requested
if (DEFINED ENV{PLSSVM_THREAD_BLOCK_SIZE})
    set(PLSSVM_THREAD_BLOCK_SIZE $ENV{PLSSVM_THREAD_BLOCK_SIZE} CACHE STRING "The used thread block size." FORCE)
endif ()
if (DEFINED PLSSVM_THREAD_BLOCK_SIZE)
    if (PLSSVM_THREAD_BLOCK_SIZE MATCHES "^[0-9]+$" AND PLSSVM_THREAD_BLOCK_SIZE GREATER 0)
        message(STATUS "Set THREAD_BLOCK_SIZE to ${PLSSVM_THREAD_BLOCK_SIZE}.")
        # add target definition
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_THREAD_BLOCK_SIZE=${PLSSVM_THREAD_BLOCK_SIZE})
    else ()
        message(SEND_ERROR "PLSSVM_THREAD_BLOCK_SIZE must be an integer greater than 0 but is \"${PLSSVM_THREAD_BLOCK_SIZE}\"!")
    endif ()
endif ()

## set specific internal block sizes of requested
if (DEFINED ENV{PLSSVM_INTERNAL_BLOCK_SIZE})
    set(PLSSVM_INTERNAL_BLOCK_SIZE $ENV{PLSSVM_INTERNAL_BLOCK_SIZE} CACHE STRING "The used internal block size." FORCE)
endif ()
if (DEFINED PLSSVM_INTERNAL_BLOCK_SIZE)
    if (PLSSVM_INTERNAL_BLOCK_SIZE MATCHES "^[0-9]+$" AND PLSSVM_INTERNAL_BLOCK_SIZE GREATER 0)
        message(STATUS "Set INTERNAL_BLOCK_SIZE to ${PLSSVM_INTERNAL_BLOCK_SIZE}.")
        # add target definition
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_INTERNAL_BLOCK_SIZE=${PLSSVM_INTERNAL_BLOCK_SIZE})
    else ()
        message(SEND_ERROR "PLSSVM_INTERNAL_BLOCK_SIZE must be an integer greater than 0 but is \"${PLSSVM_INTERNAL_BLOCK_SIZE}\"!")
    endif ()
endif ()

## set specific OpenMP block sizes of requested
if (DEFINED ENV{PLSSVM_OPENMP_BLOCK_SIZE})
    set(PLSSVM_OPENMP_BLOCK_SIZE $ENV{PLSSVM_OPENMP_BLOCK_SIZE} CACHE STRING "The used block size for the OpenMP kernel." FORCE)
endif ()
if (DEFINED PLSSVM_OPENMP_BLOCK_SIZE)
    if (PLSSVM_OPENMP_BLOCK_SIZE MATCHES "^[0-9]+$" AND PLSSVM_OPENMP_BLOCK_SIZE GREATER 0)
        message(STATUS "Set PLSSVM_OPENMP_BLOCK_SIZE to ${PLSSVM_OPENMP_BLOCK_SIZE}.")
        # add target definition
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_OPENMP_BLOCK_SIZE=${PLSSVM_OPENMP_BLOCK_SIZE})
    else ()
        message(SEND_ERROR "PLSSVM_OPENMP_BLOCK_SIZE must be an integer greater than 0 but is \"${PLSSVM_OPENMP_BLOCK_SIZE}\"!")
    endif ()
endif ()

## enable or disable the performance tracker
option(PLSSVM_ENABLE_PERFORMANCE_TRACKING "Enable performance tracking to YAML files for the PLSSVM executables plssvm-train, plssvm-predict, and plssvm-scale." OFF)
if (PLSSVM_ENABLE_PERFORMANCE_TRACKING)
    message(STATUS "Enabled performance tracking to YAML files for the PLSSVM executables plssvm-train, plssvm-predict, and plssvm-scale.")

    # add target definitions
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_PERFORMANCE_TRACKER_ENABLED)
endif ()


########################################################################################################################
##                                          check for Link Time Optimization                                          ##
########################################################################################################################
## enable Link Time Optimization (LTO)
option(PLSSVM_ENABLE_LTO "Enable Link Time Optimizations." ON)
if (PLSSVM_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT PLSSVM_LTO_SUPPORTED LANGUAGES CXX)
    if (PLSSVM_LTO_SUPPORTED)
        message(STATUS "Interprocedural optimization (IPO/LTO) enabled.")
        set_property(TARGET ${PLSSVM_EXECUTABLE_TRAIN_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_property(TARGET ${PLSSVM_EXECUTABLE_PREDICT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_property(TARGET ${PLSSVM_EXECUTABLE_SCALE_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else ()
        message(STATUS "Interprocedural optimization (IPO/LTO) not supported.")
    endif ()
endif ()


########################################################################################################################
##                                    enable documentation generation via doxygen                                     ##
########################################################################################################################
option(PLSSVM_ENABLE_DOCUMENTATION "Build documentation using doxygen." OFF)
if (PLSSVM_ENABLE_DOCUMENTATION)
    add_subdirectory(docs)
endif ()


########################################################################################################################
##                                           enable testing via google test                                           ##
########################################################################################################################
option(PLSSVM_ENABLE_TESTING "Build tests for all backends." ON)
if (PLSSVM_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif ()


########################################################################################################################
##                                              enable language bindings                                              ##
########################################################################################################################
option(PLSSVM_ENABLE_LANGUAGE_BINDINGS "Build langauge bindings, e.g., for Python." OFF)
if (PLSSVM_ENABLE_LANGUAGE_BINDINGS)
    add_subdirectory(bindings)
endif ()


########################################################################################################################
##                                            print short (backend) summary                                           ##
########################################################################################################################
message(STATUS "")
message(STATUS "Enabled backends with respective targets:")
set(PLSSVM_BACKEND_NAME_LIST "automatic")
if (TARGET ${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME})
    message(STATUS "${PLSSVM_OPENMP_BACKEND_SUMMARY_STRING}")
    list(APPEND PLSSVM_BACKEND_NAME_LIST "openmp")
endif ()
if (TARGET ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME})
    message(STATUS "${PLSSVM_CUDA_BACKEND_SUMMARY_STRING}")
    list(APPEND PLSSVM_BACKEND_NAME_LIST "cuda")
endif ()
if (TARGET ${PLSSVM_HIP_BACKEND_LIBRARY_NAME})
    message(STATUS "${PLSSVM_HIP_BACKEND_SUMMARY_STRING}")
    list(APPEND PLSSVM_BACKEND_NAME_LIST "hip")
endif ()
if (TARGET ${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME})
    message(STATUS "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING}")
    list(APPEND PLSSVM_BACKEND_NAME_LIST "opencl")
endif ()
if (TARGET ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME})
    foreach (SUMMARY_STRING ${PLSSVM_SYCL_BACKEND_SUMMARY_STRINGS})
        message(STATUS "${SUMMARY_STRING}")
    endforeach ()
    list(APPEND PLSSVM_BACKEND_NAME_LIST "sycl")
endif ()
message(STATUS "")


########################################################################################################################
##                                      configure version and git metadata files                                      ##
########################################################################################################################
message(STATUS "Configuring version and git metadata.")
set(PRE_CONFIGURE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/version/git_metadata/git_metadata.cpp.in")
set(POST_CONFIGURE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/version/git_metadata/git_metadata.cpp")
set(GIT_FAIL_IF_NONZERO_EXIT FALSE)
include(cmake/git_watcher.cmake)
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/include/plssvm/version/version.hpp.in
        ${CMAKE_CURRENT_SOURCE_DIR}/include/plssvm/version/version.hpp
        @ONLY
)


########################################################################################################################
##                                                 generate man pages                                                 ##
########################################################################################################################
## assemble the SYCL manpage entry
message(STATUS "Generating manpage files.")
string(REPLACE ";" "|" PLSSVM_PLATFORM_NAME_LIST "${PLSSVM_PLATFORM_NAME_LIST}")
string(REPLACE ";" "|" PLSSVM_BACKEND_NAME_LIST "${PLSSVM_BACKEND_NAME_LIST}")
string(REPLACE ";" "|" PLSSVM_SYCL_BACKEND_NAME_LIST "${PLSSVM_SYCL_BACKEND_NAME_LIST}")
if (TARGET ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME})
    set(PLSSVM_SYCL_IMPLEMENTATION_TYPE_MANPAGE_ENTRY "
.TP
.B --sycl_implementation_type
choose the SYCL implementation to be used in the SYCL backend: ${PLSSVM_SYCL_BACKEND_NAME_LIST} (default: automatic)
")
    set(PLSSVM_SYCL_KERNEL_INVOCATION_TYPE_MANPAGE_ENTRY "
.TP
.B --sycl_kernel_invocation_type
choose the kernel invocation type when using SYCL as backend: automatic|nd_range|hierarchical (default: automatic)
")
endif ()
set(PLSSVM_SYCL_MANPAGE_ENTRY "${PLSSVM_SYCL_KERNEL_INVOCATION_TYPE_MANPAGE_ENTRY}${PLSSVM_SYCL_IMPLEMENTATION_TYPE_MANPAGE_ENTRY}")
## assemble the performance tracker manpage entry
if (PLSSVM_ENABLE_PERFORMANCE_TRACKING)
    set(PLSSVM_PERFORMANCE_TRACKER_MANPAGE_ENTRY "
.TP
.B --performance_tracking
choose the output YAML file where the performance tracking entries should be stored; if not provided, the results are dumped to stderr
")
endif ()

## configure the manpage files
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/plssvm-train.1.in
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/plssvm-train.1
        @ONLY
)
set(PLSSVM_SYCL_MANPAGE_ENTRY "${PLSSVM_SYCL_IMPLEMENTATION_TYPE_MANPAGE_ENTRY}")
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/plssvm-predict.1.in
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/plssvm-predict.1
        @ONLY
)
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/plssvm-scale.1.in
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/plssvm-scale.1
        @ONLY
)


########################################################################################################################
##                                           add support for `make install`                                           ##
########################################################################################################################
include(GNUInstallDirs)
## install all necessary library targets
install(TARGETS ${PLSSVM_TARGETS_TO_INSTALL}
        EXPORT plssvm_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}  # all files that are neither executables, shared lib or headers
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  # all shared lib files
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}  # all executables
        )

## mark header to install via 'make install'
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        )
## install header only libraries fast_float and igor header via 'make install'
install(DIRECTORY "${fast_float_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        )
install(DIRECTORY "${igor_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        )
## only mark fmt headers to install if build via FetchContent
if (NOT fmt_FOUND)
    install(DIRECTORY "${fmt_SOURCE_DIR}/include/"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            )
endif ()
## mark man pages to install
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/docs/plssvm-train.1
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/plssvm-predict.1
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/plssvm-scale.1
        DESTINATION "${CMAKE_INSTALL_MANDIR}/man1"
        )

## manage version comparison
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "plssvmConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

## generate configuration file
configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/plssvmConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/plssvmConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/plssvm/cmake
)

## create and copy install-targets file
install(EXPORT plssvm_Targets
        FILE plssvmTargets.cmake
        NAMESPACE plssvm::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/plssvm/cmake
        )

## create file containing the build configuration and version information
install(FILES "${PROJECT_BINARY_DIR}/plssvmConfig.cmake"
        "${PROJECT_BINARY_DIR}/plssvmConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/plssvm/cmake
        )
