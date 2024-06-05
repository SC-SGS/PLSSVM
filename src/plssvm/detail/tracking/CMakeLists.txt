## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "Tracking:  ")

message(STATUS "Enabled performance tracking to YAML files for the PLSSVM executables plssvm-train, plssvm-predict, and plssvm-scale.")

target_sources(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/performance_tracker.cpp>)

## add target definitions
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_PERFORMANCE_TRACKER_ENABLED)

## explicitly set sources
set(PLSSVM_HARDWARE_TRACKING_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/nvml_hardware_sampler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hardware_sampler.cpp
)

## set target properties
set(PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME plssvm-tracking)
add_library(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} SHARED ${PLSSVM_HARDWARE_TRACKING_SOURCES})

# TODO: logic
find_package(CUDAToolkit REQUIRED)
message(STATUS "Enable sampling of NVIDIA GPU information using NVML.")
target_link_libraries(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC CUDA::nvml)

## link base library against CUDA library
target_link_libraries(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

## link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME})

## mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)