## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "stdpar:  ")

# check if OpenMP can be enabled
message(CHECK_START "Checking for stdpar backend")

message(CHECK_START "stdpar comiler check: ")


find_package(TBB)

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
)

# set target properties
set_local_and_parent(PLSSVM_STDPAR_BACKEND_LIBRARY_NAME plssvm-stdpar)
add_library(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_STDPAR_SOURCES})
# special command line options for MSVC:
target_compile_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:"/Zc:__cplusplus">)
target_compile_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE $<$<CXX_COMPILER_ID:NVHPC>:"-stdpar">)

# link base library against stdpar library
target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

if (PLSSVM_STDPAR_BACKEND STREQUAL "GCC_TBB")
    target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC TBB::tbb)
endif ()

if (PLSSVM_STDPAR_BACKEND STREQUAL "ACPP")
    target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC TBB::tbb)
    target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC --acpp-stdpar)
    target_compile_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC --acpp-stdpar)
    target_compile_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE --acpp-stdpar)
endif ()

# set compile definition that the stdpar backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_STDPAR_BACKEND)


# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME})

# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)

# generate summary string
set(PLSSVM_STDPAR_BACKEND_SUMMARY_STRING " - stdpar: cpu " PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)