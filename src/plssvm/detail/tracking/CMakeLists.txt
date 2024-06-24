## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "Tracking:  ")

message(STATUS "Enabled performance tracking to YAML files for the PLSSVM executables plssvm-train, plssvm-predict, and plssvm-scale.")

target_sources(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE $<BUILD_INTERFACE:
               ${CMAKE_CURRENT_SOURCE_DIR}/performance_tracker.cpp;
               ${CMAKE_CURRENT_SOURCE_DIR}/events.cpp
               >)

## add target definitions
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_PERFORMANCE_TRACKER_ENABLED)

option(PLSSVM_ENABLE_HARDWARE_SAMPLING "Enable hardware sampling (like clock frequency or power draw)." OFF)
if (PLSSVM_ENABLE_HARDWARE_SAMPLING)
    message(STATUS "Enable hardware sampling to YAML files for the PLSSVM executables plssvm-train, plssvm-predict, and plssvm-scale.")
    
    ## explicitly set sources
    set(PLSSVM_HARDWARE_TRACKING_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/hardware_sampler.cpp
        ${CMAKE_CURRENT_LIST_DIR}/hardware_sampler_factory.cpp
    )
    
    ## set target properties
    set(PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME plssvm-tracking)
    add_library(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} SHARED ${PLSSVM_HARDWARE_TRACKING_SOURCES})
    target_sources(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hardware_sampler.cpp>)
    
    target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_SAMPLING_ENABLED)
    target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_SAMPLING_ENABLED)
    
    ## add option to enable/disable error checking in the hardware sampling functions
    option(PLSSVM_HARDWARE_SAMPLING_ENABLE_ERROR_CHECKS "Enable error checking for the hardware sampling functions. May be problematic with smaller sample intervals." OFF)
    if (PLSSVM_HARDWARE_SAMPLING_ENABLE_ERROR_CHECKS)
        message(STATUS "Enable error checks for the hardware sampling functions.")
        target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE PLSSVM_HARDWARE_SAMPLING_ERROR_CHECKS_ENABLED)
    endif ()
    
    ## option for the used sampling interval
    set(PLSSVM_HARDWARE_SAMPLING_INTERVAL "100" CACHE STRING "The interval in milliseconds in which the hardware information (like clock frequency or power draw) are queried.")
    if (NOT PLSSVM_HARDWARE_SAMPLING_INTERVAL MATCHES "^[0-9]+$" OR PLSSVM_HARDWARE_SAMPLING_INTERVAL LESS_EQUAL 0)
        message(FATAL_ERROR "The PLSSVM_HARDWARE_SAMPLING_INTERVAL must be a natural number greater 0, but is \"${PLSSVM_HARDWARE_SAMPLING_INTERVAL}\"!")
    endif ()
    message(STATUS "Setting the hardware sampler interval to ${PLSSVM_HARDWARE_SAMPLING_INTERVAL}ms.")
    target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_SAMPLING_INTERVAL=${PLSSVM_HARDWARE_SAMPLING_INTERVAL}ms)
    
    ####################################################################################################################
    ##                                                CPU measurements                                                ##
    ####################################################################################################################
    ## check whether lscpu could be found -> used for the CPU targets as well as for ALL host measurements
    ## -> checked even if no CPU targets where provided
    ## LINUX only
    find_program(PLSSVM_LSCPU_FOUND lscpu)
    if (PLSSVM_LSCPU_FOUND)
        message(STATUS "Enable sampling of CPU information using lscpu.")
        target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_TRACKING_VIA_LSCPU_ENABLED)
    endif ()

    ## check whether free could be found -> used for the CPU targets as well as for ALL host measurements
    ## -> checked even if no CPU targets where provided
    ## LINUX only
    find_program(PLSSVM_FREE_FOUND free)
    if (PLSSVM_FREE_FOUND)
        message(STATUS "Enable sampling of CPU information using free.")
        target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_TRACKING_VIA_FREE_ENABLED)
    endif ()
    
    ## check whether turbostat could be found -> used for the CPU targets as well as for ALL host measurements
    ## -> checked even if no CPU targets where provided
    find_program(PLSSVM_TURBOSTAT_FOUND turbostat)
    if (PLSSVM_TURBOSTAT_FOUND)
        ## check if the turbostat command works as intended
        execute_process(COMMAND turbostat -n 1 -S -q
                        RESULT_VARIABLE PLSSVM_TURBOSTAT_WITHOUT_ROOT
                        OUTPUT_QUIET
                        ERROR_QUIET)
        if (NOT PLSSVM_TURBOSTAT_WITHOUT_ROOT EQUAL 0)
            message(STATUS "Can't use turbostat without root permissions! Try using with sudo...")
            execute_process(COMMAND sudo -n turbostat -n 1 -S -q
                            RESULT_VARIABLE PLSSVM_TURBOSTAT_ROOT_PASSWORD_REQUIRED
                            OUTPUT_QUIET
                            ERROR_QUIET)
            if (NOT PLSSVM_TURBOSTAT_ROOT_PASSWORD_REQUIRED EQUAL 0)
                message(STATUS "Can't use turbostat with root if a sudo password is required! Please add turbostat to the sudoer group.")
                message(STATUS "Disabling turbostat support!")
            else ()
                execute_process(COMMAND sudo turbostat -n 1 -S -q
                                RESULT_VARIABLE PLSSVM_TURBOSTAT_WITH_ROOT
                                OUTPUT_QUIET
                                ERROR_QUIET)
                if (NOT PLSSVM_TURBOSTAT_WITH_ROOT EQUAL 0)
                    message(STATUS "Can't use turbostat with root even if no sudo password is required!")
                    message(STATUS "Disabling turbostat support!")
                else ()
                    message(STATUS "Enable sampling of CPU information using turbostat.")
                    
                    set(PLSSVM_TURBOSTAT_EXECUTION_TYPE "root")
                    # add compile definitions
                    target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_TRACKING_VIA_TURBOSTAT_ENABLED)
                    target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_TRACKING_VIA_TURBOSTAT_ROOT)
                endif ()
            endif ()
        else ()
            set(PLSSVM_TURBOSTAT_EXECUTION_TYPE "without_root")
            # add compile definitions
            target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_TRACKING_VIA_TURBOSTAT_ENABLED)
        endif ()
    endif ()
    
    ## check if the CPU hardware tracker can be used
    if (PLSSVM_LSCPU_FOUND OR PLSSVM_FREE_FOUND OR PLSSVM_TURBOSTAT_EXECUTION_TYPE)
        ## try finding subprocess.h
        set(PLSSVM_subprocess_VERSION b6e1611d430e3019c423d2af26bb162e7ed5c3ae)
        find_package(subprocess QUIET)
        if (subprocess_FOUND)
            message(STATUS "Found package subprocess.h.")
            target_include_directories(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE ${subprocess_INCLUDE_DIR})
        else ()
            message(STATUS "Couldn't find package subprocess.h. Building version ${PLSSVM_subprocess_VERSION} from source.")
            target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_subprocess_VERSION="${PLSSVM_subprocess_VERSION}")
            # fetch subprocess library subprocess.h
            FetchContent_Declare(subprocess
                                 GIT_REPOSITORY https://github.com/sheredom/subprocess.h.git
                                 GIT_TAG ${PLSSVM_subprocess_VERSION}
                                 QUIET
                                 CONFIGURE_COMMAND ""
                                 BUILD_COMMAND ""
            )
            FetchContent_MakeAvailable(subprocess)
            target_include_directories(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE $<BUILD_INTERFACE:${subprocess_SOURCE_DIR}>)
        endif ()
        
        # add source file to source file list
        target_sources(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE
                       $<BUILD_INTERFACE:
                       ${CMAKE_CURRENT_SOURCE_DIR}/cpu/hardware_sampler.cpp;
                       ${CMAKE_CURRENT_SOURCE_DIR}/cpu/cpu_samples.cpp;
                       ${CMAKE_CURRENT_SOURCE_DIR}/cpu/utility.cpp;
                       >)
        
        # add compile definitions
        target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
    endif ()
    
    ####################################################################################################################
    ##                  nvidia targets are provided -> enable NVIDIA GPU hardware samping using NVML                  ##
    ####################################################################################################################
    if (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        # find libraries necessary for NVML and link against them
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE CUDA::nvml)
        
        message(STATUS "Enable sampling of NVIDIA GPU information using NVML.")
        
        # add source file to source file list
        target_sources(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE
                       $<BUILD_INTERFACE:
                       ${CMAKE_CURRENT_SOURCE_DIR}/gpu_nvidia/hardware_sampler.cpp;
                       ${CMAKE_CURRENT_SOURCE_DIR}/gpu_nvidia/nvml_samples.cpp
                       >)
        
        # add compile definition
        target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_TRACKING_FOR_NVIDIA_GPUS_ENABLED)
    endif ()
    
    ####################################################################################################################
    ##                   amd targets are provided -> enable AMD GPU hardware samping using ROCm SMI                   ##
    ####################################################################################################################
    if (DEFINED PLSSVM_AMD_TARGET_ARCHS)
        ## try finding ROCm SMI
        find_package(rocm_smi REQUIRED)
        target_link_libraries(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE -lrocm_smi64)
        target_include_directories(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE ${ROCM_SMI_INCLUDE_DIR})
        
        message(STATUS "Enable sampling of AMD GPU information using ROCm SMI (${ROCM_SMI_INCLUDE_DIR}).")
        
        # add source file to source file list
        target_sources(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE
                       $<BUILD_INTERFACE:
                       ${CMAKE_CURRENT_SOURCE_DIR}/gpu_amd/hardware_sampler.cpp;
                       ${CMAKE_CURRENT_SOURCE_DIR}/gpu_amd/rocm_smi_samples.cpp
                       >)
        
        # add compile definition
        target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_TRACKING_FOR_AMD_GPUS_ENABLED)
    endif ()
    
    ####################################################################################################################
    ##                intel targets are provided -> enable Intel GPU hardware samping using Level Zero                ##
    ####################################################################################################################
    if (DEFINED PLSSVM_INTEL_TARGET_ARCHS)
        # try finding Level Zero
        # find_package(LevelZero REQUIRED)
        target_link_libraries(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE LevelZero::LevelZero)
        target_include_directories(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE ${LevelZero_INCLUDE_DIRS})
        
        message(STATUS "Enable sampling of Intel GPU information using Level Zero.")
        
        # add source file to source file list
        target_sources(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PRIVATE
                       $<BUILD_INTERFACE:
                       ${CMAKE_CURRENT_SOURCE_DIR}/gpu_intel/hardware_sampler.cpp;
                       ${CMAKE_CURRENT_SOURCE_DIR}/gpu_intel/utility.cpp
                       >)
        
        # add compile definition
        target_compile_definitions(${PLSSVM_HARDWARE_TRACKING_LIBRARY_NAME} PUBLIC PLSSVM_HARDWARE_TRACKING_FOR_INTEL_GPUS_ENABLED)
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