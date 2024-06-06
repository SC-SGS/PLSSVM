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

option(PLSSVM_ENABLE_HARDWARE_SAMPLING "Enable hardware sampling (like clock frequency or power draw)." OFF)
if (PLSSVM_ENABLE_HARDWARE_SAMPLING)
    message(STATUS "Enable hardware sampling to YAML files for the PLSSVM executables plssvm-train, plssvm-predict, and plssvm-scale.")
    
    ## explicitly set sources
    set(PLSSVM_HARDWARE_TRACKING_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/hardware_sampler.cpp
    )
    
    ## set target properties
    set(PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME plssvm-tracking)
    add_library(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} SHARED ${PLSSVM_HARDWARE_TRACKING_SOURCES})
    
    target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_SAMPLING_ENABLED)
    
    # TODO: enable/disable error checking
    
    ## option for the used sampling interval
    set(PLSSVM_HARDWARE_SAMPLING_INTERVAL "100" CACHE STRING "The interval in milliseconds in which the hardware information (like clock frequency or power draw) are queried.")
    if (NOT PLSSVM_HARDWARE_SAMPLING_INTERVAL MATCHES "^[0-9]+$" OR PLSSVM_HARDWARE_SAMPLING_INTERVAL LESS_EQUAL 0)
        message(FATAL_ERROR "The PLSSVM_HARDWARE_SAMPLING_INTERVAL must be a natural number greater 0, but is \"${PLSSVM_HARDWARE_SAMPLING_INTERVAL}\"!")
    endif ()
    message(STATUS "Setting the hardware sampler interval to ${PLSSVM_HARDWARE_SAMPLING_INTERVAL}ms.")
    target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_SAMPLING_INTERVAL=${PLSSVM_HARDWARE_SAMPLING_INTERVAL}ms)
    
    ## nvidia targets are provided -> enable NVIDIA GPU hardware samping using NVML
    if (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        # find libraries necessary for NVML and link against them
        find_package(CUDAToolkit REQUIRED)
        message(STATUS "Enable sampling of NVIDIA GPU information using NVML.")
        target_link_libraries(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC CUDA::nvml)
        
        # add source file to source file list
        target_sources(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nvml_hardware_sampler.cpp>)
        
        # add compile definition
        target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE PLSSVM_HARDWARE_TRACKING_VIA_NVML_ENABLED)
    endif ()
    
    ## link base library against hardware sampling library
    target_link_libraries(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})
    
    ## link against interface library
    target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME})
    
    ## mark backend library as install target
    list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME}")
    set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)