## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if the stdpar backend is available
if (TARGET plssvm::plssvm-stdpar)
    # enable stdpar based on the used stdpar implementation
    include(CheckCXXCompilerFlag)
    if (PLSSVM_STDPAR_BACKEND MATCHES "NVHPC")
        enable_language(CUDA)
        find_dependency(CUDAToolkit)
        if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
            set(plssvm_FOUND OFF)
            set(plssvm_stdpar_FOUND OFF)
            set(plssvm_NOT_FOUND_MESSAGE "The CMAKE_CXX_COMPILER must be set to NVIDIA's HPC SDK compiler (nvc++) in user code in order to use plssvm::stdpar!")
            return()
        endif ()
    elseif (PLSSVM_STDPAR_BACKEND MATCHES "roc-stdpar")
        check_cxx_compiler_flag("-hipstdpar --hipstdpar-path=${PLSSVM_STDPAR_BACKEND_HIPSTDPAR_PATH}" PLSSVM_HAS_HIPSTDPAR_STDPAR_FLAG)
        if (NOT PLSSVM_HAS_HIPSTDPAR_STDPAR_FLAG)
            set(plssvm_FOUND OFF)
            set(plssvm_stdpar_FOUND OFF)
            set(plssvm_NOT_FOUND_MESSAGE "The CMAKE_CXX_COMPILER must be set to the hipstdpar patched LLVM compiler (acpp) in user code in order to use plssvm::stdpar!")
            return()
        endif ()
    elseif (PLSSVM_STDPAR_BACKEND MATCHES "IntelLLVM")
        find_dependency(oneDPL)
        check_cxx_compiler_flag("-fsycl-pstl-offload" PLSSVM_HAS_INTEL_LLVM_STDPAR_FLAG)
        if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" AND NOT PLSSVM_HAS_INTEL_LLVM_STDPAR_FLAG)
            set(plssvm_FOUND OFF)
            set(plssvm_stdpar_FOUND OFF)
            set(plssvm_NOT_FOUND_MESSAGE "The CMAKE_CXX_COMPILER must be set to the Intel LLVM compiler (icpx) in user code in order to use plssvm::stdpar!")
            return()
        endif ()
    elseif (PLSSVM_STDPAR_BACKEND MATCHES "ACPP")
        find_dependency(TBB)
        check_cxx_compiler_flag("--acpp-stdpar" PLSSVM_HAS_ACPP_STDPAR_FLAG)
        if (NOT PLSSVM_HAS_ACPP_STDPAR_FLAG)
            set(plssvm_FOUND OFF)
            set(plssvm_stdpar_FOUND OFF)
            set(plssvm_NOT_FOUND_MESSAGE "The CMAKE_CXX_COMPILER must be set to the AdaptiveCpp compiler (acpp) in user code in order to use plssvm::stdpar!")
            return()
        endif ()
    elseif (PLSSVM_STDPAR_BACKEND MATCHES "GNU_TBB")
        find_dependency(TBB)
        find_dependency(Boost COMPONENTS atomic)
        if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(plssvm_FOUND OFF)
            set(plssvm_stdpar_FOUND OFF)
            set(plssvm_NOT_FOUND_MESSAGE "The CMAKE_CXX_COMPILER must be set to GNU GCC in user code in order to use plssvm::stdpar!")
            return()
        endif ()
    endif ()
    
    # set alias targets
    add_library(plssvm::stdpar ALIAS plssvm::plssvm-stdpar)
    # set COMPONENT to be found
    set(plssvm_stdpar_FOUND ON)
else ()
    # set COMPONENT to be NOT found
    set(plssvm_stdpar_FOUND OFF)
endif ()