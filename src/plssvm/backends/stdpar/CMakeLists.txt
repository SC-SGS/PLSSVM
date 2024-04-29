## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "stdpar:  ")

include(CheckCXXCompilerFlag)

# check if OpenMP can be enabled
message(CHECK_START "Checking for stdpar backend")

message(CHECK_START "stdpar comiler check: ")

find_package(TBB)

message(WARNING ${CMAKE_CXX_COMPILER_ID})
if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    message(CHECK_PASS "found NVHPC compiler")
    set(PLSSVM_STDPAR_BACKEND "NVHPC")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND TBB_FOUND)
    message(CHECK_PASS "found gcc with TBB")
    set(PLSSVM_STDPAR_BACKEND "GCC_TBB")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND TBB_FOUND)
    check_cxx_compiler_flag("--acpp-stdpar" PLSSVM_HAS_ACPP_STDPAR_FLAG)
    if (PLSSVM_HAS_ACPP_STDPAR_FLAG)
        message(CHECK_PASS "found ACPP")
        set(PLSSVM_STDPAR_BACKEND "ACPP")
    endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    check_cxx_compiler_flag("-fsycl-pstl-offload=gpu" PLSSVM_HAS_INTELLLVM_STDPAR_FLAG)
    message(WARNING ${PLSSVM_HAS_INTELLLVM_STDPAR_FLAG})
    # if (PLSSVM_HAS_INTELLLVM_STDPAR_FLAG) # TODO
        message(CHECK_PASS "found IntelLLVM")
        set(PLSSVM_STDPAR_BACKEND "IntelLLVM")
    # endif ()
endif ()

if (NOT DEFINED PLSSVM_STDPAR_BACKEND)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: stdpar!")
    endif ()
    return()
endif ()

#find_package(OpenMP)
#
#if (NOT OPENMP_FOUND)
#    message(CHECK_FAIL "not found")
#    if (PLSSVM_ENABLE_OPENMP_BACKEND MATCHES "ON")
#        message(SEND_ERROR "Cannot find requested backend: OpenMP!")
#    endif ()
#    return()
#else ()
#    if (NOT DEFINED PLSSVM_CPU_TARGET_ARCHS)
#        if (PLSSVM_ENABLE_OPENMP_BACKEND MATCHES "ON")
#            message(SEND_ERROR "Found requested OpenMP backend, but no \"cpu\" targets were specified!")
#        else ()
#            message(STATUS "Found OpenMP backend, but no \"cpu\" targets were specified!")
#        endif ()
#        message(CHECK_FAIL "skipped")
#        return()
#    endif ()
#endif ()
#message(CHECK_PASS "found ")

# explicitly set sources
set(PLSSVM_STDPAR_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/implementation_types.cpp
)

if (PLSSVM_STDPAR_BACKEND STREQUAL "ACPP")
    list(APPEND PLSSVM_STDPAR_SOURCES ${CMAKE_CURRENT_LIST_DIR}/../SYCL/AdaptiveCpp/detail/utility.cpp)
endif ()

set_local_and_parent(PLSSVM_STDPAR_BACKEND ${PLSSVM_STDPAR_BACKEND})

# set target properties
set_local_and_parent(PLSSVM_STDPAR_BACKEND_LIBRARY_NAME plssvm-stdpar)
add_library(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_STDPAR_SOURCES})

# TODO: fast_math

# TODO: only install if necessary!!!
# TODO: something that isn't Boost? -> https://github.com/boostorg/atomic.git doesn't support floating point values :/

## try finding Boost atomic
if (POLICY CMP0144)
    cmake_policy(SET CMP0144 OLD) # suppress warning
endif ()
find_package(Boost 1.73.0 REQUIRED COMPONENTS atomic)
message(STATUS "Found Boost atomic version ${Boost_VERSION}.")
target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC Boost::atomic)
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_boost_VERSION="${Boost_VERSION}")


# link base library against stdpar library
target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

if (PLSSVM_STDPAR_BACKEND STREQUAL "GCC_TBB")
    target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC TBB::tbb)
endif ()

if (PLSSVM_STDPAR_BACKEND STREQUAL "ACPP")
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG --acpp-stdpar)

    target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC TBB::tbb)
    # TODO: better
    target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_IMPLEMENTATION="AdaptiveCpp")
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_HAS_ACPP)
endif ()

if (PLSSVM_STDPAR_BACKEND STREQUAL "NVHPC")
    enable_language(CUDA)

    find_package(CUDAToolkit)

    # set necessary flags
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    if (DEFINED PLSSVM_CPU_TARGET_ARCHS AND DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        message(FATAL_ERROR "Please specify either CPU or NVIDIA GPU as target platforms but not both!")
    elseif (DEFINED PLSSVM_CPU_TARGET_ARCHS)
        message(STATUS "CPU target platform provided, setting: \"-stdpar=multicore\"")
        set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -stdpar=multicore --restrict)
    elseif (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        message(STATUS "NVIDIA GPU target platform provided, setting: \"-stdpar=gpu\"")
        set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -stdpar=gpu --restrict)
    else ()
        message(FATAL_ERROR "Please specify one of CPU or NVIDIA GPU as target platforms!")
    endif ()
    
    # TODO: better
    target_link_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_link_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_link_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_IMPLEMENTATION="NVHPC")
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_HAS_NVHPC)
endif ()


if (PLSSVM_STDPAR_BACKEND STREQUAL "IntelLLVM")
    # target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC TBB::tbb)
    # TODO: better
    target_link_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC -fsycl-pstl-offload=gpu)
    target_link_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC -fsycl-pstl-offload=gpu)
    target_link_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE -fsycl-pstl-offload=gpu)
    target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC -fsycl-pstl-offload=gpu)
    target_compile_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC -fsycl-pstl-offload=gpu)
    target_compile_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE -fsycl-pstl-offload=gpu)
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_IMPLEMENTATION="IntelLLVM")
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_HAS_INTELLLVM)
endif ()

# set compile definition that the stdpar backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_STDPAR_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME})

# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)

# generate summary string
set(PLSSVM_STDPAR_BACKEND_SUMMARY_STRING " - stdpar: ??? " PARENT_SCOPE) # TODO: acutal values

list(POP_BACK CMAKE_MESSAGE_INDENT)