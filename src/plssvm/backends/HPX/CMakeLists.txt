## Authors: Alexander Van Craen, Marcel Breyer, Alexander Strack
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "HPX:  ")

# check if HPX can be enabled
message(CHECK_START "Checking for HPX backend")

find_package(HPX)

if (NOT HPX_FOUND)
    message(CHECK_FAIL "not found")
    if (PLSSVM_ENABLE_HPX_BACKEND MATCHES "ON")
        message(SEND_ERROR "Cannot find requested backend: HPX!")
    endif ()
    return()
else ()
    if (NOT DEFINED PLSSVM_CPU_TARGET_ARCHS)
        if (PLSSVM_ENABLE_HPX_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested HPX backend, but no \"cpu\" targets were specified!")
        else ()
            message(STATUS "Found HPX backend, but no \"cpu\" targets were specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
endif ()
message(CHECK_PASS "found ")

# explicitly set sources
set(PLSSVM_HPX_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
)

# set target properties
set_local_and_parent(PLSSVM_HPX_BACKEND_LIBRARY_NAME plssvm-HPX)
add_library(${PLSSVM_HPX_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_HPX_SOURCES})
target_link_libraries(${PLSSVM_HPX_BACKEND_LIBRARY_NAME} PUBLIC HPX::hpx HPX::wrap_main HPX::iostreams_component)

# additional compilation flags
target_compile_options(${PLSSVM_HPX_BACKEND_LIBRARY_NAME} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wconversion>)

# link base library against HPX library
target_link_libraries(${PLSSVM_HPX_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set compile definition that the HPX backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_HPX_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_HPX_BACKEND_LIBRARY_NAME})

# mark backend library as install target
append_local_and_parent(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_HPX_BACKEND_LIBRARY_NAME})

# generate summary string
set(PLSSVM_HPX_BACKEND_SUMMARY_STRING " - HPX: cpu " PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)
