## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "Tracking:  ")

message(STATUS "Enabled performance tracking to YAML files for the PLSSVM executables plssvm-train, plssvm-predict, and plssvm-scale.")

target_sources(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE $<BUILD_INTERFACE:
               ${CMAKE_CURRENT_SOURCE_DIR}/performance_tracker.cpp;
               ${CMAKE_CURRENT_SOURCE_DIR}/events.cpp;
               >)

## add target definitions
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_PERFORMANCE_TRACKER_ENABLED)

option(PLSSVM_ENABLE_HARDWARE_SAMPLING "Enable hardware sampling (like clock frequency or power draw)." OFF)
if (PLSSVM_ENABLE_HARDWARE_SAMPLING)
    ## enable hardware sampling
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_SAMPLING_ENABLED)
    
    include(${PROJECT_SOURCE_DIR}/cmake/utility_macros.cmake)
    ## option for the used sampling interval
    set(PLSSVM_HARDWARE_SAMPLING_INTERVAL "100" CACHE STRING "The interval in milliseconds in which the hardware information (like clock frequency or power draw) are queried.")
    check_integer(PLSSVM_HARDWARE_SAMPLING_INTERVAL)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_SAMPLING_INTERVAL=${PLSSVM_HARDWARE_SAMPLING_INTERVAL}ms)
    
    include(FetchContent)
    find_package(hardware_sampling QUIET)
    if (hardware_sampling_FOUND)
        message(STATUS "Found package hws.")
    else ()
        message(STATUS "Couldn't find package hws. Building from source.")
        set(HWS_ENABLE_ERROR_CHECKS ${PLSSVM_ENABLE_ASSERTS} CACHE INTERNAL "" FORCE)
        set(HWS_SAMPLING_INTERVAL ${PLSSVM_HARDWARE_SAMPLING_INTERVAL} CACHE INTERNAL "" FORCE)
        set(HWS_ENABLE_PYTHON_BINDINGS OFF CACHE INTERNAL "" FORCE)
        FetchContent_Declare(hws
                             GIT_REPOSITORY https://github.com/SC-SGS/hardware_sampling
                             GIT_TAG unification # TODO: use release tag after one was created
                             QUIET
        )
        FetchContent_MakeAvailable(hws)
        add_dependencies(${PLSSVM_BASE_LIBRARY_NAME} hws)
    endif ()
    target_link_libraries(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC hws::hws)
    
    ## mark backend library as install target
    list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME}")
    set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)