## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "Kokkos:  ")

# check if Kokkos can be enabled
message(CHECK_START "Checking for Kokkos backend")

find_package(Kokkos)

if (NOT Kokkos_FOUND)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_KOKKOS_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: Kokkos!")
    endif ()
    return()
endif ()
message(CHECK_PASS "found")

# explicitly set sources
set(PLSSVM_KOKKOS_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/execution_space.cpp
)

# set target properties
set_local_and_parent(PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME plssvm-Kokkos)
add_library(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_KOKKOS_SOURCES})
target_link_libraries(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} PUBLIC Kokkos::kokkos)

# link base library against Kokkos library
target_link_libraries(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set whether the kernel source should be compiled with fast math enabled or not # TODO: enable fast-math
#if (PLSSVM_ENABLE_FAST_MATH)
#    target_compile_definitions(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_ENABLE_FAST_MATH)
#endif ()

# set compile definition that the Kokkos backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PRIVATE PLSSVM_HAS_KOKKOS_BACKEND)
target_compile_definitions(${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_KOKKOS_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME})

# mark backend library as install target
append_local_and_parent(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_KOKKOS_BACKEND_LIBRARY_NAME})

# generate summary string
set(PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_COMPILER " - Kokkos:")
include(${PROJECT_SOURCE_DIR}/cmake/assemble_summary_string.cmake)
assemble_summary_string(PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS)
# do not print any special target architecture information
string(REPLACE " (${PLSSVM_CPU_TARGET_ARCHS})" "" PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS}")
string(REPLACE " (${PLSSVM_NVIDIA_TARGET_ARCHS})" "" PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS}")
string(REPLACE " (${PLSSVM_AMD_TARGET_ARCHS})" "" PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS}")
string(REPLACE " (${PLSSVM_INTEL_TARGET_ARCHS})" "" PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS "${PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS}")
set(PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING "${PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_COMPILER}${PLSSVM_KOKKOS_BACKEND_SUMMARY_STRING_ARCHS}" PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)