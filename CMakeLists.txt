## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

cmake_minimum_required(VERSION 3.18)

project(ParallelLeastSquaresSupportVectorMachine
        VERSION 0.1.0
        LANGUAGES CXX
        DESCRIPTION "A Support Vector Machine implementation using different backends.")


include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/add_custom_build_type.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/set_local_and_parent.cmake)

## configure version file
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/include/plssvm/version/version.hpp.in
        ${CMAKE_CURRENT_SOURCE_DIR}/include/plssvm/version/version.hpp
        @ONLY
)

########################################################################################################################
##                                      create and set necessary base properties                                      ##
########################################################################################################################
## set base sources
set(PLSSVM_BASE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/csvm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/csvm_IO.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/parameter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/parameter_train.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plssvm/parameter_predict.cpp
)

## create base library: linked against all backend libraries
set(PLSSVM_BASE_LIBRARY_NAME svm-base)
add_library(${PLSSVM_BASE_LIBRARY_NAME} STATIC ${PLSSVM_BASE_SOURCES})
## create all library: one target against all backends are linked
set(PLSSVM_ALL_LIBRARY_NAME svm-all)
add_library(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE)

## create list of targets to install
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_ALL_LIBRARY_NAME} ${PLSSVM_BASE_LIBRARY_NAME})

## set include directory
target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )

# set library cxx standard
target_compile_features(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC cxx_std_17)
## additional base library compile options
target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC -Wall -Wextra)
#target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC -Wdouble-promotion -fno-common -Wconversion -Wshadow) # TODO: make warning free
## nvcc doesn't recognize -Werror=??? option so only set it when using a CXX compiler
target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Werror=switch -fstrict-enums>)
## enable additional optimization flags only in RELEASE mode
target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC $<$<CONFIG:RELEASE>:-ffast-math -march=native>)


########################################################################################################################
##                                    check for optional and necessary dependencies                                   ##
########################################################################################################################
# check for OpenMP (not for the backend!)
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
    message(STATUS "Found OpenMP to speed up file parsing.")
    target_link_libraries(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-fopenmp'")
endif()

## setup dependencies
include(FetchContent)

## try finding cxxopts
find_package(cxxopts QUIET)
if(cxxopts_FOUND)
    message(STATUS "Found package cxxopts.")
else()
    message(STATUS "Couldn't find package cxxopts. Building from source ...")
    # fetch command line parser library cxxopts
    FetchContent_Declare(cxxopts
            GIT_REPOSITORY          https://github.com/jarro2783/cxxopts.git
            GIT_SHALLOW             TRUE

            set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
            set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    )
    FetchContent_MakeAvailable(cxxopts)
    add_dependencies(${PLSSVM_BASE_LIBRARY_NAME} cxxopts)
    target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
            $<BUILD_INTERFACE:${cxxopts_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
            )
endif()

## try finding fast_float
find_package(fast_float QUIET)
if(fast_float_FOUND)
    message(STATUS "Found package fast_float.")
else()
    message(STATUS "Couldn't find package fast_float. Building from source ...")
    # fetch float parsing library fast_float
    FetchContent_Declare(fast_float
            GIT_REPOSITORY          https://github.com/fastfloat/fast_float
            GIT_TAG                 8159e8bcf63c1b92f5a51fb550f966e56624b209
            GIT_SHALLOW             TRUE

            set(FASTFLOAT_TEST OFF CACHE BOOL "" FORCE)
            set(FASTFLOAT_SANITIZE OFF CACHE BOOL "" FORCE)
    )
    FetchContent_GetProperties(fast_float)
    if(NOT fast_float_POPULATED)
        FetchContent_Populate(fast_float)
        add_subdirectory(${fast_float_SOURCE_DIR} EXCLUDE_FROM_ALL)
    endif()
    add_dependencies(${PLSSVM_BASE_LIBRARY_NAME} fast_float)
    target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC
            $<BUILD_INTERFACE:${fast_float_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
            )
endif()

## try finding fmt
find_package(fmt QUIET)
if(fmt_FOUND)
    message(STATUS "Found package fmt.")
else()
    message(STATUS "Couldn't find package fmt. Building from source ...")
    # fetch string formatting library fmt
    FetchContent_Declare(fmt
            GIT_REPOSITORY          https://github.com/fmtlib/fmt.git
            GIT_TAG                 d141cdbeb0fb422a3fb7173b285fd38e0d1772dc
            GIT_SHALLOW             TRUE

            set(FMT_TEST OFF CACHE BOOL "" FORCE)
            set(FMT_DOC OFF CACHE BOOL "" FORCE)
            set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
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
endif()
set_property(TARGET ${PLSSVM_BASE_LIBRARY_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC fmt::fmt)


########################################################################################################################
##                                                 create executables                                                 ##
########################################################################################################################
## create train executable
set(PLSSVM_EXECUTABLE_TRAIN_NAME svm-train)
add_executable(${PLSSVM_EXECUTABLE_TRAIN_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main_train.cpp)
## create predict executable
set(PLSSVM_EXECUTABLE_PREDICT_NAME svm-predict)
add_executable(${PLSSVM_EXECUTABLE_PREDICT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main_predict.cpp)

## append executables to installed targets
list(APPEND PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_EXECUTABLE_TRAIN_NAME} ${PLSSVM_EXECUTABLE_PREDICT_NAME})

########################################################################################################################
##                                            setup code coverage analysis                                            ##
########################################################################################################################
## coverage analysis only possible with the Coverage CMAKE_BUILD_TYPE
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if(uppercase_CMAKE_BUILD_TYPE MATCHES "COVERAGE")
    # must be linux
    if(WIN32 OR APPLE)
        message(FATAL_ERROR "Only Linux is supported for the coverage analysis.")
    endif()
    # must be GCC
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        message(FATAL_ERROR "Only GCC is supported for the coverage analysis.")
    endif()

    message(STATUS "Enable code coverage analysis using lcov.")

    # Create the coverage target. Run coverage tests with 'make coverage'
    add_custom_target(coverage
        COMMAND lcov --zerocounters  --directory .
        COMMAND lcov --capture -d . --initial --output-file test_base.info
        COMMAND mkdir -p coverage
        COMMAND ${CMAKE_MAKE_PROGRAM} test || true
        COMMAND lcov --capture -d . --output-file test_test.info
        COMMAND lcov --add-tracefile test_base.info --add-tracefile test_test.info -o test_total.info
        COMMAND lcov --remove  test_total.info '/usr/*' '*/build/*' '*/tests/*' -o test_clean.info
        COMMAND genhtml test_clean.info --output-directory coverage --title "PLSSVM Test Coverage" --show-details --legend
        BYPRODUCTS ${CMAKE_BINARY_DIR}/test_base.info ${CMAKE_BINARY_DIR}/test_test.info ${CMAKE_BINARY_DIR}/test_total.info ${CMAKE_BINARY_DIR}/test_clean.info ${CMAKE_BINARY_DIR}/coverage
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )

    # add executables as coverage target
    add_dependencies(coverage ${PLSSVM_EXECUTABLE_TRAIN_NAME})
    add_dependencies(coverage ${PLSSVM_EXECUTABLE_PREDICT_NAME})

    # add custom target `make clean_coverage` which calls `make clean` and also removes all generate *.gcda and *.gcno files
    add_custom_target(clean_coverage)
    add_custom_command(
            DEPENDS clean
            COMMENT "remove all coverage files"
            COMMAND ${CMAKE_MAKE_PROGRAM} clean
            COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/delete_coverage_files.cmake"
            TARGET clean_coverage
    )
endif()


########################################################################################################################
##                                         parse target platform information                                          ##
########################################################################################################################
## the target platforms and architectures must either be set using the CMake command line option
## PLSSVM_TARGET_PLATFORMS or the environment variable with the same name
if(DEFINED PLSSVM_TARGET_PLATFORMS)
    set(PLSSVM_TARGET_PLATFORMS ${PLSSVM_TARGET_PLATFORMS} CACHE STRING "The target platforms to compile for." FORCE)
elseif(DEFINED ENV{PLSSVM_TARGET_PLATFORMS})
    set(PLSSVM_TARGET_PLATFORMS $ENV{PLSSVM_TARGET_PLATFORMS} CACHE STRING "The target platforms to compile for." FORCE)
else()
    message(FATAL_ERROR "PLSSVM_TARGET_PLATFORMS must be defined (e.g. -DPLSSVM_TARGET_PLATFORMS=\"cpu;nvidia:sm_70,sm_86;amd:gfx906;intel\"!")
endif()

## PLSSVM_TARGET_PLATFORMS must not be empty
if(PLSSVM_TARGET_PLATFORMS STREQUAL "")
    message(FATAL_ERROR "PLSSVM_TARGET_PLATFORMS must not be empty!")
endif()

## parse provided target platforms
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/parse_architecture_info.cmake)
foreach(PLSSVM_PLATFORM ${PLSSVM_TARGET_PLATFORMS})
    if(PLSSVM_PLATFORM MATCHES "^cpu")
        # parse provided CPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_CPU_TARGET_ARCHS PLSSVM_NUM_CPU_TARGET_ARCHS)
        if(NOT PLSSVM_NUM_CPU_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"cpu\" must not have any architecture specifications!")
        endif()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_CPU_TARGET)
    elseif(PLSSVM_PLATFORM MATCHES "^nvidia")
        # parse provided NVIDIA GPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_NVIDIA_TARGET_ARCHS PLSSVM_NUM_NVIDIA_TARGET_ARCHS)
        if(PLSSVM_NUM_NVIDIA_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"nvidia\" must at least have one architecture specifications!")
        endif()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_NVIDIA_TARGET)
    elseif(PLSSVM_PLATFORM MATCHES "^amd")
        # parse provided AMD GPU architectures
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_AMD_TARGET_ARCHS PLSSVM_NUM_AMD_TARGET_ARCHS)
        if(PLSSVM_NUM_AMD_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"amd\" must at least have one architecture specifications!")
        endif()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_AMD_TARGET)
    elseif(PLSSVM_PLATFORM MATCHES "^intel")
        parse_architecture_info(${PLSSVM_PLATFORM} PLSSVM_INTEL_TARGET_ARCHS PLSSVM_NUM_INTEL_TARGET_ARCHS)
        if(NOT PLSSVM_NUM_INTEL_TARGET_ARCHS EQUAL 0)
            message(FATAL_ERROR "Target platform \"intel\" must not have any architecture specifications!")
        endif()
        target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_INTEL_TARGET)
    else()
        message(FATAL_ERROR "Unrecognized target platform \"${PLSSVM_PLATFORM}\"! Must be one of: cpu nvidia amd intel")
    endif()
endforeach()


########################################################################################################################
##                                            check for and enable backends                                           ##
########################################################################################################################
## check for OpenMP backend
set(PLSSVM_ENABLE_OPENMP_BACKEND AUTO CACHE STRING "Enable OpenMP Backend")
set_property(CACHE PLSSVM_ENABLE_OPENMP_BACKEND PROPERTY STRINGS AUTO ON OFF)
if(PLSSVM_ENABLE_OPENMP_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_OPENMP_BACKEND)
    add_subdirectory(src/plssvm/backends/OpenMP)
endif()

## check for CUDA backend
set(PLSSVM_ENABLE_CUDA_BACKEND AUTO CACHE STRING "Enable CUDA Backend")
set_property(CACHE PLSSVM_ENABLE_CUDA_BACKEND PROPERTY STRINGS AUTO ON OFF)
if(PLSSVM_ENABLE_CUDA_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_CUDA_BACKEND)
    add_subdirectory(src/plssvm/backends/CUDA)
endif()

## check for OpenCL backend
set(PLSSVM_ENABLE_OPENCL_BACKEND AUTO CACHE STRING "Enable OpenCL Backend")
set_property(CACHE PLSSVM_ENABLE_OPENCL_BACKEND PROPERTY STRINGS AUTO ON OFF)
if(PLSSVM_ENABLE_OPENCL_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_OPENCL_BACKEND)
    add_subdirectory(src/plssvm/backends/OpenCL)
endif()

## check for SYCL backend
set(PLSSVM_ENABLE_SYCL_BACKEND AUTO CACHE STRING "Enable SYCL Backend")
set_property(CACHE PLSSVM_ENABLE_SYCL_BACKEND PROPERTY STRINGS AUTO ON OFF)
if(PLSSVM_ENABLE_SYCL_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_SYCL_BACKEND)
    add_subdirectory(src/plssvm/backends/SYCL)
endif()

## check if ANY backend is available/has been enabled
get_target_property(PLSSVM_LINKED_BACKENDS ${PLSSVM_ALL_LIBRARY_NAME} INTERFACE_LINK_LIBRARIES)
if(NOT PLSSVM_LINKED_BACKENDS)
    message(FATAL_ERROR "At least one backend must be available!")
endif()

## link backends against executables
target_link_libraries(${PLSSVM_EXECUTABLE_TRAIN_NAME} ${PLSSVM_ALL_LIBRARY_NAME})
target_link_libraries(${PLSSVM_EXECUTABLE_PREDICT_NAME} ${PLSSVM_ALL_LIBRARY_NAME})


########################################################################################################################
##                                         add additional compile definitions                                         ##
########################################################################################################################
## add assert option
option(PLSSVM_ENABLE_ASSERTS "Enables PLSSVM asserts even if NDEBUG is set." OFF)
if(PLSSVM_ENABLE_ASSERTS)
    message(STATUS "Enable additional debugging assertions.")
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_ENABLE_ASSERTS)
endif()

## set specific thread block sizes if requested
if(DEFINED ENV{PLSSVM_THREAD_BLOCK_SIZE})
    set(PLSSVM_THREAD_BLOCK_SIZE $ENV{PLSSVM_THREAD_BLOCK_SIZE} CACHE STRING "The used thread block size." FORCE)
endif()
if(DEFINED PLSSVM_THREAD_BLOCK_SIZE)
    if (PLSSVM_THREAD_BLOCK_SIZE MATCHES "^[0-9]+$" AND PLSSVM_THREAD_BLOCK_SIZE GREATER 0)
        message(STATUS "Set THREAD_BLOCK_SIZE to ${PLSSVM_THREAD_BLOCK_SIZE}.")
        # add target definition
        target_compile_definitions(${PLSSVM_EXECUTABLE_NAME} PUBLIC PLSSVM_THREAD_BLOCK_SIZE=${PLSSVM_THREAD_BLOCK_SIZE})
    else()
        message(FATAL_ERROR "PLSSVM_THREAD_BLOCK_SIZE must be an integer greater than 0 but is \"${PLSSVM_THREAD_BLOCK_SIZE}\"!")
    endif()
endif()

## set specific internal block sizes of requested
if(DEFINED ENV{PLSSVM_INTERNAL_BLOCK_SIZE})
    set(PLSSVM_INTERNAL_BLOCK_SIZE $ENV{PLSSVM_INTERNAL_BLOCK_SIZE} CACHE STRING "The used internal block size." FORCE)
endif()
if(DEFINED PLSSVM_INTERNAL_BLOCK_SIZE)
    if (PLSSVM_INTERNAL_BLOCK_SIZE MATCHES "^[0-9]+$" AND PLSSVM_INTERNAL_BLOCK_SIZE GREATER 0)
        message(STATUS "Set INTERNAL_BLOCK_SIZE to ${PLSSVM_INTERNAL_BLOCK_SIZE}.")
        # add target definition
        target_compile_definitions(${PLSSVM_EXECUTABLE_NAME} PUBLIC PLSSVM_INTERNAL_BLOCK_SIZE=${PLSSVM_INTERNAL_BLOCK_SIZE})
    else()
        message(FATAL_ERROR "PLSSVM_INTERNAL_BLOCK_SIZE must be an integer greater than 0 but is \"${PLSSVM_INTERNAL_BLOCK_SIZE}\"!")
    endif()
endif()


########################################################################################################################
##                                          check for Link Time Optimization                                          ##
########################################################################################################################
## enable Link Time Optimization (LTO)
option(PLSSVM_ENABLE_LTO "Enable Link Time Optimizations." ON)
if(PLSSVM_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT PLSSVM_LTO_SUPPORTED OUTPUT PLSSVM_LTO_ERROR LANGUAGES CXX)
    if(PLSSVM_LTO_SUPPORTED)
        message(STATUS "Interprocedural optimization (IPO/LTO) enabled.")
        set_property(TARGET ${PLSSVM_EXECUTABLE_TRAIN_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_property(TARGET ${PLSSVM_EXECUTABLE_PREDICT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "Interprocedural optimization (IPO/LTO) requested but not supported: ${PLSSVM_LTO_ERROR}")
    endif()
endif()


########################################################################################################################
##                                    enable documentation generation via doxygen                                     ##
########################################################################################################################
option(PLSSVM_ENABLE_DOCUMENTATION "Build documentation using doxygen." OFF)
if(PLSSVM_ENABLE_DOCUMENTATION)
    add_subdirectory(docs)
endif()


########################################################################################################################
##                                           enable testing via google test                                           ##
########################################################################################################################
option(PLSSVM_ENABLE_TESTING "Build tests for all backends." ON)
if(PLSSVM_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()


########################################################################################################################
##                                            print short (backend) summary                                           ##
########################################################################################################################
message(STATUS "Enabled backends with respective targets:")
if(TARGET ${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME})
    message(STATUS "${PLSSVM_OPENMP_BACKEND_SUMMARY_STRING}")
endif()
if(TARGET ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME})
    message(STATUS "${PLSSVM_CUDA_BACKEND_SUMMARY_STRING}")
endif()
if(TARGET ${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME})
    message(STATUS "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING}")
endif()
if(TARGET ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME})
    message(STATUS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING}")
endif()


########################################################################################################################
##                                           add support for `make install`                                           ##
########################################################################################################################
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
## only mark fmt headers to install if build via FetchContent
if(NOT fmt_FOUND)
    install(DIRECTORY "${fmt_SOURCE_DIR}/include/"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            )
endif()

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





# TODO: separate repo?
## generate scripts
option(PLSSVM_GENERATE_TIMINGSCRIPT "Generate script for timings." ON)
if(PLSSVM_GENERATE_TIMINGSCRIPT)
    configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/time.sh.in
            ${CMAKE_BINARY_DIR}/time.sh
            @ONLY
    )
endif()

