## Authors: Alexander Van Craen, Marcel Breyer, Alexander Strack
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

## create HPX tests
set(PLSSVM_HPX_TEST_NAME HPX_tests)

# list all necessary sources
set(PLSSVM_HPX_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hpx_csvm.cpp
)

# check if HPX can be enabled
message(CHECK_START "Checking for HPX backend")

find_package(HPX 1.9.0)

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


# add test executable
add_executable(${PLSSVM_HPX_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../hpx_main.cpp ${PLSSVM_HPX_TEST_SOURCES})

# link against test library
target_link_libraries(${PLSSVM_HPX_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME} HPX::hpx HPX::wrap_main)

# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_HPX_TEST_NAME})

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_HPX_TEST_NAME})
endif ()
