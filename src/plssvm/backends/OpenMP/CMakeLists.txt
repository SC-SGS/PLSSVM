## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "OpenMP:  ")

# check if OpenMP can be enabled
message(CHECK_START "Checking for OpenMP backend")

find_package(OpenMP)

if (NOT OPENMP_FOUND)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_OPENMP_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: OpenMP!")
    endif ()
    return()
else ()
    if (NOT DEFINED PLSSVM_CPU_TARGET_ARCHS)
        if (PLSSVM_ENABLE_OPENMP_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested OpenMP backend, but no \"cpu\" targets were specified!")
        else ()
            message(STATUS "Found OpenMP backend, but no \"cpu\" targets were specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
endif ()
message(CHECK_PASS "found ")

# explicitly set sources
set(PLSSVM_OPENMP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
        ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cg_explicit/blas.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cg_explicit/kernel_matrix_assembly.cpp
)


# set target properties
set_local_and_parent(PLSSVM_OPENMP_BACKEND_LIBRARY_NAME plssvm-OpenMP)
add_library(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_OPENMP_SOURCES})
target_link_libraries(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:OpenMP::OpenMP_CXX>)
# special command line options for MSVC:
# -openmp:llvm -> enables unsigned loop indexes in OpenMP parallel for loops
# -openmp:experimental -> enables OpenMP's SIMD instructions
target_compile_options(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:-openmp:llvm -openmp:experimental>)

# link base library against OpenMP library
target_link_libraries(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set compile definition that the OpenMP backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_OPENMP_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME})

# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)


# generate summary string
set(PLSSVM_OPENMP_BACKEND_SUMMARY_STRING " - OpenMP: cpu " PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)