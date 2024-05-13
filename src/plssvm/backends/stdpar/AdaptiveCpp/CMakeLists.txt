## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

message(CHECK_START "searching for AdaptiveCpp")

# try finding Intel TBB
find_package(TBB QUIET)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND TBB_FOUND)
    # AdaptiveCpp also needs TBB
    # check for the AdaptiveCpp compiler
    include(CheckCXXCompilerFlag)
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
    message(CHECK_PASS "found")
    set_local_and_parent(PLSSVM_STDPAR_BACKEND "ACPP")
else ()
    message(CHECK_FAIL "skipped")
    return()
endif ()

# add stdpar implementation specific source file to library
target_sources(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp)
target_sources(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../SYCL/AdaptiveCpp/detail/utility.cpp)

# set global AdaptiveCpp compiler flags
set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG --acpp-stdpar)
# add flag improving CPU performance if only the CPU target is present # TODO: comment in if bug is fixed (already in current develop but not in the latest Release)
#if (DEFINED PLSSVM_CPU_TARGET_ARCHS AND NOT (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS OR DEFINED PLSSVM_AMD_TARGET_ARCHS OR DEFINED PLSSVM_INTEL_TARGET_ARCHS))
#    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG} --acpp-stdpar-system-usm)
#endif ()

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
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_STDPAR_BACKEND_HAS_ACPP)
target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_HAS_ACPP)