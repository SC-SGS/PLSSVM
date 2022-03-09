## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# check if HIP can be enabled
message(CHECK_START "Checking for HIP backend")

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH}/hip $ENV{ROCM_PATH})
find_package(hip REQUIRED)

#if(NOT DEFINED HIP_VERSION)
#    message(CHECK_FAIL "not found")
#    if(PLSSVM_ENABLE_HIP_BACKEND MATCHES "ON")
#        message(FATAL_ERROR "Cannot find requested backend: HIP!")
#    endif()
#    return()
#else()
#    if(NOT DEFINED PLSSVM_AMD_TARGET_ARCHS)
#        message(STATUS "Found requested HIP backend, but no \"amd\" targets were specified!")
#        message(CHECK_FAIL "skipped")
#        return()
#    endif()
#endif()
message(CHECK_PASS "found")


## set necessary flags
#set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#
#if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
#    # use nvcc to compile CUDA code
#    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -lineinfo -Xptxas=-v -Xptxas -dlcm=cg" )
#    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Wno-attributes")
#elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
#    # use clang to compile CUDA code
#    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3 -ffast-math")
#endif()

# explicitly set sources
set(PLSSVM_HIP_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.hip.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.hip.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm.hip.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_csvm.cpp
)

# set target properties
set_local_and_parent(PLSSVM_HIP_BACKEND_LIBRARY_NAME svm-HIP)
add_library(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_HIP_SOURCES})
#if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
#    set_target_properties(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#endif()

# prepare architectures for CMake's CUDA_ARCHITECTURES
#list(TRANSFORM PLSSVM_NVIDIA_TARGET_ARCHS REPLACE "sm_" "")
# set target architecture
set_property(TARGET ${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTY HIP_ARCHITECTURES ${PLSSVM_AMD_TARGET_ARCHS})

# link base library against HIP library
target_link_libraries(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})
target_link_libraries(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PUBLIC hip::device)

# set compile definition that the HIP backend is available
target_compile_definitions(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_HIP_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_HIP_BACKEND_LIBRARY_NAME})

# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_HIP_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)


# generate summary string
#list(TRANSFORM PLSSVM_NVIDIA_TARGET_ARCHS PREPEND "sm_")
set(PLSSVM_HIP_BACKEND_SUMMARY_STRING " - HIP: amd (${PLSSVM_AMD_TARGET_ARCHS})" PARENT_SCOPE)
