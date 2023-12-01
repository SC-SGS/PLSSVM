## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# TODO: only tested on AMD GPUs -> test on NVIDIA GPUs

list(APPEND CMAKE_MESSAGE_INDENT "HIP:  ")

# check if HIP can be enabled
message(CHECK_START "Checking for HIP backend")

include(CheckLanguage)
check_language(HIP)

if (NOT CMAKE_HIP_COMPILER)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_HIP_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: HIP!")
    endif ()
    return()
else ()
    if (NOT DEFINED PLSSVM_AMD_TARGET_ARCHS)
        if (PLSSVM_ENABLE_HIP_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested HIP backend, but no \"amd\" targets were specified!")
        else ()
            message(STATUS "Found HIP backend, but no \"amd\" targets were specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
endif ()
message(CHECK_PASS "found")

enable_language(HIP)
find_package(HIP REQUIRED)


## set necessary flags
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -std=c++17")
set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "${CMAKE_HIP_FLAGS_RELWITHDEBINFO} -O2 -ffast-math")
set(CMAKE_HIP_FLAGS_RELEASE "${CMAKE_HIP_FLAGS_RELEASE} -O3 -ffast-math")

# explicitly set sources
set(PLSSVM_HIP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.hip.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/utility.hip.cpp
        ${CMAKE_CURRENT_LIST_DIR}/csvm.hip.cpp
        ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../gpu_device_ptr.cpp
)

# set target properties
set_local_and_parent(PLSSVM_HIP_BACKEND_LIBRARY_NAME plssvm-HIP)
add_library(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_HIP_SOURCES})

# https://gitlab.kitware.com/cmake/cmake/-/issues/23210
#set_target_properties(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTIES HIP_SEPARABLE_COMPILATION ON)

target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC -Wno-sign-compare)

# set target architecture
set_property(TARGET ${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTY HIP_ARCHITECTURES ${PLSSVM_AMD_TARGET_ARCHS})

# link base library against HIP library
target_link_libraries(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME} hip::device hip::host)

# set compile definition that the HIP backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_HAS_HIP_BACKEND)
target_compile_definitions(${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_HIP_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_HIP_BACKEND_LIBRARY_NAME})

# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_HIP_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)


# generate summary string
set(PLSSVM_HIP_BACKEND_SUMMARY_STRING " - HIP: amd (${PLSSVM_AMD_TARGET_ARCHS})" PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)
