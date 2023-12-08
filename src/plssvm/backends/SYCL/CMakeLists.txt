## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeDependentOption)

list(APPEND CMAKE_MESSAGE_INDENT "SYCL:  ")

message(CHECK_START "Checking for SYCL backend")

# explicitly set sources shared between all SYCL implementations
set(PLSSVM_SYCL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../gpu_device_ptr.cpp
)

# list of found SYCL implementations
set(PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS "")


# check if SYCL can be enabled
## add AdaptiveCpp
set(PLSSVM_ENABLE_SYCL_ADAPTIVECPP_BACKEND ${PLSSVM_ENABLE_SYCL_BACKEND} CACHE STRING "Enable AdaptiveCpp as SYCL Backend")
set_property(CACHE PLSSVM_ENABLE_SYCL_ADAPTIVECPP_BACKEND PROPERTY STRINGS AUTO ON OFF)
if (PLSSVM_ENABLE_SYCL_ADAPTIVECPP_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_SYCL_ADAPTIVECPP_BACKEND)
    add_subdirectory(AdaptiveCpp)
endif ()
## add DPC++
set(PLSSVM_ENABLE_SYCL_DPCPP_BACKEND ${PLSSVM_ENABLE_SYCL_BACKEND} CACHE STRING "Enable DPC++ as SYCL Backend")
set_property(CACHE PLSSVM_ENABLE_SYCL_DPCPP_BACKEND PROPERTY STRINGS AUTO ON OFF)
if (PLSSVM_ENABLE_SYCL_DPCPP_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_SYCL_DPCPP_BACKEND)
    add_subdirectory(DPCPP)
endif ()


## check if ANY SYCL implementation has been found
if (NOT PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_SYCL_BACKEND MATCHES "ON")
        if (PLSSVM_ENABLE_SYCL_ADAPTIVECPP_BACKEND MATCHES "OFF" AND PLSSVM_ENABLE_SYCL_DPCPP_BACKEND MATCHES "OFF")
            message(SEND_ERROR "Set \"PLSSVM_ENABLE_SYCL_BACKEND\" to \"ON\" but disabled all supported SYCL implementations!")
        else ()
            message(SEND_ERROR "Cannot find requested backend: SYCL!")
        endif ()
    endif ()
    return()
endif ()


# set target properties
set_local_and_parent(PLSSVM_SYCL_BACKEND_LIBRARY_NAME plssvm-SYCL)
add_library(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} INTERFACE)

if (TARGET ${PLSSVM_SYCL_BACKEND_ADAPTIVECPP_LIBRARY_NAME})
    target_link_libraries(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} INTERFACE ${PLSSVM_SYCL_BACKEND_ADAPTIVECPP_LIBRARY_NAME})
endif ()
if (TARGET ${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME})
    target_link_libraries(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} INTERFACE ${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME})
endif ()

## check if environment variables are correctly set if more than one SYCL implementation has been found
list(LENGTH PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS PLSSVM_SYCL_BACKEND_NUM_FOUND_IMPLEMENTATIONS)
if (PLSSVM_SYCL_BACKEND_NUM_FOUND_IMPLEMENTATIONS GREATER 1)
    if (NOT DEFINED ENV{PLSSVM_SYCL_ADAPTIVECPP_INCLUDE_DIR})
        message(SEND_ERROR "Found more than one SYCL implementation, but the environment variable \"PLSSVM_SYCL_ADAPTIVECPP_INCLUDE_DIR\" isn't set!")
    endif ()
    if (NOT DEFINED ENV{PLSSVM_SYCL_DPCPP_INCLUDE_DIR})
        message(SEND_ERROR "Found more than one SYCL implementation, but the environment variable \"PLSSVM_SYCL_DPCPP_INCLUDE_DIR\" isn't set!")
    endif ()
endif ()
message(CHECK_PASS "found")


# set the preferred SYCL implementation
if (DEFINED PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION)
    # command line value provided
    if (PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION IN_LIST PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS)
        # possible value found -> set preferred SYCL implementation
        message(STATUS "Setting preferred SYCL implementation to \"${PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION}\".")
    else ()
        # value not recognized
        message(SEND_ERROR "\"${PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION}\" as preferred SYCL implementation unrecognized or not available! Possible values are: ${PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS}.")
    endif ()
else ()
    # no command line value provided -> try to infer preferred SYCL implementation
    list(LENGTH PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS_NUM)
    if (PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS_NUM GREATER 1)
        # more than one SYCL implementation found -> can't infer preferred SYCL implementation
        message(SEND_ERROR "Found more than one SYCL implementation, but \"PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION\" hasn't been set!")
    else ()
        # use only available SYCL implementation as preferred implementation
        set(PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION ${PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS})
        message(STATUS "Setting preferred SYCL implementation to \"${PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION}\".")
    endif ()
endif ()
target_compile_definitions(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} INTERFACE PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION=${PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION})
if (TARGET ${PLSSVM_SYCL_BACKEND_ADAPTIVECPP_LIBRARY_NAME})
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_ADAPTIVECPP_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION=${PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION})
endif ()
if (TARGET ${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME})
    target_compile_definitions(${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME} PRIVATE PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION=${PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION})
endif ()


# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME})


# the SYCL backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_HAS_SYCL_BACKEND)


# mark backend library as install target
list(APPEND PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_SYCL_BACKEND_LIBRARY_NAME}")
set(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_TARGETS_TO_INSTALL} PARENT_SCOPE)

# set manpage string
set_local_and_parent(PLSSVM_SYCL_BACKEND_NAME_LIST "automatic;${PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS}")


# generate summary string
include(${PROJECT_SOURCE_DIR}/cmake/assemble_summary_string.cmake)
set(PLSSVM_SYCL_BACKEND_SUMMARY_STRINGS "")
foreach (SYCL_IMPLEMENTATION ${PLSSVM_SYCL_BACKEND_FOUND_IMPLEMENTATIONS})
    # mark preferred SYCL implementation
    get_target_property(PLSSVM_SYCL_BACKEND_LIBRARY_NAME_COMPILE_DEFINITIONS ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} INTERFACE_COMPILE_DEFINITIONS)
    if ("${PLSSVM_SYCL_BACKEND_LIBRARY_NAME_COMPILE_DEFINITIONS}" MATCHES ".*${SYCL_IMPLEMENTATION}.*")
        set(SYCL_IMPLEMENTATION "${SYCL_IMPLEMENTATION}*")
    endif ()

    assemble_summary_string(PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS)
    # do not print CPU and Intel GPU target architectures when using AdaptiveCpp
    if ("${SYCL_IMPLEMENTATION}" MATCHES "^adaptivecpp")
        if (PLSSVM_SYCL_BACKEND_ADAPTIVECPP_USE_GENERIC_SSCP)

            assemble_summary_string(PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS)
            # do not print any special target architecture information
            string(REPLACE " (${PLSSVM_CPU_TARGET_ARCHS})" "" PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}")
            string(REPLACE " (${PLSSVM_NVIDIA_TARGET_ARCHS})" "" PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}")
            string(REPLACE " (${PLSSVM_AMD_TARGET_ARCHS})" "" PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}")
            string(REPLACE " (${PLSSVM_INTEL_TARGET_ARCHS})" "" PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}")

            # remove first whitespace from string
            string(LENGTH "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}" PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS_LENGTH)
            MATH(EXPR PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS_LENGTH "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS_LENGTH} - 1")
            string(SUBSTRING "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}" 1 ${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS_LENGTH} PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS)

            set(PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS " generic (${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS})")
        else ()
            string(REPLACE " (${PLSSVM_CPU_TARGET_ARCHS})" "" PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}")
            string(REPLACE " (${PLSSVM_INTEL_TARGET_ARCHS})" "" PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}")
        endif ()
    endif ()
    list(APPEND PLSSVM_SYCL_BACKEND_SUMMARY_STRINGS " - SYCL (${SYCL_IMPLEMENTATION}):${PLSSVM_SYCL_BACKEND_SUMMARY_STRING_ARCHS}")
endforeach ()
set(PLSSVM_SYCL_BACKEND_SUMMARY_STRINGS "${PLSSVM_SYCL_BACKEND_SUMMARY_STRINGS}" PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)