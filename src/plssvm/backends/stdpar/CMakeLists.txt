## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "stdpar:  ")

# check if stdpar can be enabled
message(STATUS "Checking for stdpar backend")

message(CHECK_START "compiler check")

# try finding Intel TBB -> necessary for GNU's and AdaptiveCpp's stdpar implementation on the CPU
find_package(TBB)

include(CheckCXXCompilerFlag)
if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    # check for NVIDIA's nvhpc compiler
    # nvhpc needs either a CPU or an NVIDIA GPU as target
    if (NOT DEFINED PLSSVM_CPU_TARGET_ARCHS AND NOT DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested stdpar backend using NVHPC, but no \"cpu\" or \"nvidia\" targets were specified!")
        else ()
            message(STATUS "Found stdpar backend using NVHPC, but no \"cpu\" or \"nvidia\" targets were specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    # for nvhpc not BOTH targets may be set
    if (DEFINED PLSSVM_CPU_TARGET_ARCHS AND DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested stdpar backend using NVHPC, but both \"cpu\" and \"nvidia\" targets were specified!")
        else ()
            message(STATUS "Found stdpar backend using NVHPC, but both \"cpu\" and \"nvidia\" targets were specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    # nvhpc found
    message(CHECK_PASS "found NVHPC")
    set(PLSSVM_STDPAR_BACKEND "NVHPC")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # check for Intel's icpx compiler
    check_cxx_compiler_flag("-fsycl-pstl-offload" PLSSVM_HAS_INTEL_LLVM_STDPAR_FLAG)
    if (NOT PLSSVM_HAS_INTEL_LLVM_STDPAR_FLAG)
        # the target offload flag couldn't be found!
        if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested stdpar backend using IntelLLVM, but the offloading compiler flag isn't supported!")
        else ()
            message(STATUS "Found stdpar backend using IntelLLVM, but the offloading compiler flag isn't supported!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    # for Intel LLVM either CPU or any GPU target may be set, but not both!
    if (DEFINED PLSSVM_CPU_TARGET_ARCHS AND (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS OR DEFINED PLSSVM_AMD_TARGET_ARCHS OR DEFINED PLSSVM_INTEL_TARGET_ARCHS))
        if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested stdpar backend using IntelLLVM, but both \"cpu\" and at least one GPU target was specified!")
        else ()
            message(STATUS "Found stdpar backend using NVHPC, but both \"cpu\" and at least one GPU target was specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    # IntelLLVM found
    message(CHECK_PASS "found IntelLLVM")
    set(PLSSVM_STDPAR_BACKEND "IntelLLVM")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND TBB_FOUND)
    # AdaptiveCpp also needs TBB
    # check for the AdaptiveCpp compiler
    check_cxx_compiler_flag("--acpp-stdpar" PLSSVM_HAS_ACPP_STDPAR_FLAG)
    if (NOT PLSSVM_HAS_ACPP_STDPAR_FLAG)
        # the stdpar flag couldn't be found!
        if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found a Clang, but AdaptiveCpp's stdpar compiler flag isn't supported (mostly likely because the used Clang isn't AdaptiveCpp)!")
        else ()
            message(STATUS "Found a Clang, but AdaptiveCpp's stdpar compiler flag isn't supported (mostly likely because the used Clang isn't AdaptiveCpp)!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    # AdaptiveCpp found
    message(CHECK_PASS "found AdaptiveCpp")
    set(PLSSVM_STDPAR_BACKEND "ACPP")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND TBB_FOUND)
    # GNU GCC also needs TBB
    # GNU GCC + TBB needs either a CPU target
    if (NOT DEFINED PLSSVM_CPU_TARGET_ARCHS)
        if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested stdpar backend using GNU GCC + TBB, but no \"cpu\" targets were specified!")
        else ()
            message(STATUS "Found stdpar backend using GNU GCC + TBB, but no \"cpu\" targets were specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    # GNU GCC + TBB found
    message(CHECK_PASS "found GNU GCC with TBB")
    set(PLSSVM_STDPAR_BACKEND "GNU_TBB")
else ()
    # couldn't find any supported stdpar implementation!
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: stdpar!")
    endif ()
    return()
endif ()
# set variable also in the parent scope
set_local_and_parent(PLSSVM_STDPAR_BACKEND ${PLSSVM_STDPAR_BACKEND})

# explicitly set sources
set(PLSSVM_STDPAR_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/implementation_types.cpp
)

# add additional source files depending on the used stdpar implementation
if (PLSSVM_STDPAR_BACKEND STREQUAL "ACPP")
    list(APPEND PLSSVM_STDPAR_SOURCES ${CMAKE_CURRENT_LIST_DIR}/../SYCL/AdaptiveCpp/detail/utility.cpp)
elseif (PLSSVM_STDPAR_BACKEND STREQUAL "IntelLLVM")
    list(APPEND PLSSVM_STDPAR_SOURCES ${CMAKE_CURRENT_LIST_DIR}/../SYCL/DPCPP/detail/utility.cpp)
endif ()

# set target properties
set_local_and_parent(PLSSVM_STDPAR_BACKEND_LIBRARY_NAME plssvm-stdpar)
add_library(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_STDPAR_SOURCES})

# link base library against stdpar library
target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# only NVHPC and GNU GCC + TBB need the boost::atomic_ref
if (PLSSVM_STDPAR_BACKEND STREQUAL "NVHPC" OR PLSSVM_STDPAR_BACKEND STREQUAL "GNU_TBB")
    ## try finding Boost atomic
    if (POLICY CMP0144)
        cmake_policy(SET CMP0144 OLD) # suppress cmake warning
    endif ()
    find_package(Boost 1.73.0 REQUIRED COMPONENTS atomic)
    message(STATUS "Found Boost atomic version ${Boost_VERSION}.")
    target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC Boost::atomic)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_boost_VERSION="${Boost_VERSION}")
endif ()

if (PLSSVM_STDPAR_BACKEND STREQUAL "NVHPC")
    # set global NVHPC compiler flags
    if (DEFINED PLSSVM_CPU_TARGET_ARCHS)
        message(STATUS "CPU target platform provided, setting: \"-stdpar=multicore -tp=native\"")
        set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -stdpar=multicore -tp=native)
        
        target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_NVHPC_CPU)
    elseif (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        message(STATUS "NVIDIA GPU target platform provided, setting: \"-stdpar=gpu\"")
        set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -stdpar=gpu)
        
        if (PLSSVM_ENABLE_FAST_MATH)
            set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} -gpu=fastmath)
        endif ()
        
        target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_NVHPC_GPU)
    endif ()
    separate_arguments(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG UNIX_COMMAND ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    
    # NVHPC must be linked against CUDA
    enable_language(CUDA)
    find_package(CUDAToolkit)
    
    # set compile options
    target_link_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_link_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_link_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    
    # set NVHPC compile definition
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_HAS_NVHPC)
elseif (PLSSVM_STDPAR_BACKEND STREQUAL "IntelLLVM")
    # set global IntelLLVM compiler flags
    if (DEFINED PLSSVM_CPU_TARGET_ARCHS)
        message(STATUS "CPU target platform provided, setting: \"-fsycl-pstl-offload=cpu\"")
        set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -fsycl -fsycl-pstl-offload=cpu -fsycl-targets=spir64_x86_64)
        if (PLSSVM_NUM_CPU_TARGET_ARCHS EQUAL 1)
            set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} -Xsycl-target-backend=spir64_x86_64 "-march=${PLSSVM_CPU_TARGET_ARCHS}")
        endif ()
    else ()
        message(STATUS "A GPU target platform provided, setting: \"-fsycl-pstl-offload=gpu\"")
        set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -fsycl -fsycl-pstl-offload=gpu)
        
        if (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
            set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=${PLSSVM_NVIDIA_TARGET_ARCHS})
        elseif (DEFINED PLSSVM_AMD_TARGET_ARCHS)
            set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${PLSSVM_AMD_TARGET_ARCHS})
        elseif (DEFINED PLSSVM_INTEL_TARGET_ARCHS)
            set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device ${PLSSVM_INTEL_TARGET_ARCHS}")
        endif ()
    endif ()
    if (PLSSVM_ENABLE_FAST_MATH)
        set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} -ffast-math)
    endif ()
    separate_arguments(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG UNIX_COMMAND ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    
    # Intel LLVM must be linked against oneDPL
    find_package(oneDPL REQUIRED) # causes -qopenmp warning
    target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC oneDPL)
    target_link_libraries(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC oneDPL)
    
    # set compile options
    target_link_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_link_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_link_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    
    # set Intel LLVM compile definition
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_HAS_INTEL_LLVM)
elseif (PLSSVM_STDPAR_BACKEND STREQUAL "ACPP")
    # set global AdaptiveCpp compiler flags
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG --acpp-stdpar)
    if (PLSSVM_ENABLE_FAST_MATH)
        set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} -ffast-math)
    endif ()
    separate_arguments(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG UNIX_COMMAND ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    
    # AdaptiveCpp must be linked against TBB
    target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC TBB::tbb)
    
    # set compile options
    target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    
    # set AdaptiveCpp compile definition
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_HAS_ACPP)
elseif (PLSSVM_STDPAR_BACKEND STREQUAL "GNU_TBB")
    if (PLSSVM_ENABLE_FAST_MATH)
        set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -ffast-math)
    endif ()
    
    # GNU GCC must be linked against TBB
    target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC TBB::tbb)
    
    # set compile options
    if (PLSSVM_STDPAR_BACKEND_COMPILER_FLAG)
        target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
        target_compile_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
        target_compile_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    endif ()
    
    # set GNU GCC + TBB compile definition
    target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_HAS_GNU_TBB)
endif ()

# set compile definition that the stdpar backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_STDPAR_BACKEND)
target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_STDPAR_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME})

# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)

# generate summary string
include(${PROJECT_SOURCE_DIR}/cmake/assemble_summary_string.cmake)
assemble_summary_string(PLSSVM_STDPAR_BACKEND_SUMMARY_STRING_ARCHS)
set(PLSSVM_STDPAR_BACKEND_SUMMARY_STRING " - stdpar (${PLSSVM_STDPAR_BACKEND}):${PLSSVM_STDPAR_BACKEND_SUMMARY_STRING_ARCHS}" PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)