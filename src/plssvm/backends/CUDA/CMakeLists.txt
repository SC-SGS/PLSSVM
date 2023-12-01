## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "CUDA:  ")

# check if CUDA can be enabled
message(CHECK_START "Checking for CUDA backend")

include(CheckLanguage)
check_language(CUDA)

if (NOT CMAKE_CUDA_COMPILER)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_CUDA_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: CUDA!")
    endif ()
    return()
else ()
    if (NOT DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        if (PLSSVM_ENABLE_CUDA_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested CUDA backend, but no \"nvidia\" targets were specified!")
        else ()
            message(STATUS "Found CUDA backend, but no \"nvidia\" targets were specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
endif ()
message(CHECK_PASS "found")

enable_language(CUDA)

find_package(CUDAToolkit)

# set necessary flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # use nvcc to compile CUDA code
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-fPIC' -Xcudafe --diag_suppress=unsigned_compare_with_zero -lineinfo")
    # build type specific flags
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xptxas=-v -Xptxas -dlcm=cg -Xcudafe --diag_suppress=unrecognized_gcc_pragma")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -O2 --use_fast_math")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")
    # set OpenMP specific flags
    if (OpenMP_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-fopenmp'")
    endif ()
elseif (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    # use clang to compile CUDA code
    # build type specific flags
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -O2 -ffast-math")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3 -ffast-math")
    # set OpenMP specific flags
    if (OpenMP_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fopenmp")
    endif ()
endif ()

# explicitly set sources
set(PLSSVM_CUDA_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/cg_explicit/blas.cu
        ${CMAKE_CURRENT_LIST_DIR}/cg_explicit/kernel_matrix_assembly.cu
#        ${CMAKE_CURRENT_LIST_DIR}/cg_implicit/svm_kernel.cu
        ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cu
        ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cu
        ${CMAKE_CURRENT_LIST_DIR}/csvm.cu
        ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
        ${CMAKE_CURRENT_LIST_DIR}/predict_kernel.cu
        ${CMAKE_CURRENT_LIST_DIR}/../gpu_device_ptr.cpp
        )

# set target properties
set_local_and_parent(PLSSVM_CUDA_BACKEND_LIBRARY_NAME plssvm-CUDA)
add_library(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_CUDA_SOURCES})
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set_target_properties(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()
if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC -fPIC)
    target_link_options(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC -lcudart)
endif ()

target_link_libraries(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC CUDA::cudart)

# prepare architectures for CMake's CUDA_ARCHITECTURES
list(TRANSFORM PLSSVM_NVIDIA_TARGET_ARCHS REPLACE "sm_" "")
# set target architecture
set_property(TARGET ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES ${PLSSVM_NVIDIA_TARGET_ARCHS})

# link base library against CUDA library
target_link_libraries(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set compile definition that the CUDA backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_HAS_CUDA_BACKEND)
target_compile_definitions(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_CUDA_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME})

# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_CUDA_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)

# generate summary string
list(TRANSFORM PLSSVM_NVIDIA_TARGET_ARCHS PREPEND "sm_")
set(PLSSVM_CUDA_BACKEND_SUMMARY_STRING " - CUDA: nvidia (${PLSSVM_NVIDIA_TARGET_ARCHS})" PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)