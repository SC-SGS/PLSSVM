## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

message(CHECK_START "searching for Intel LLVM (icpx)")

if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # check for Intel's icpx compiler
    include(CheckCXXCompilerFlag)
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
    message(CHECK_PASS "found")
    set_local_and_parent(PLSSVM_STDPAR_BACKEND "IntelLLVM")
else ()
    message(CHECK_FAIL "skipped")
    return()
endif ()

# add stdpar implementation specific source file to library
target_sources(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp)
target_sources(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../SYCL/DPCPP/detail/utility.cpp)

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
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_STDPAR_BACKEND_HAS_INTEL_LLVM)
target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_HAS_INTEL_LLVM)