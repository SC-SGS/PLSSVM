## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

## create HIP tests
set(PLSSVM_HIP_TEST_NAME HIP_tests)

# list all necessary sources
set(PLSSVM_HIP_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/fill_kernel.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
        ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
        ${CMAKE_CURRENT_LIST_DIR}/hip_csvm.cpp
)

# add test executable
add_executable(${PLSSVM_HIP_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../main.cpp ${PLSSVM_HIP_TEST_SOURCES})
target_include_directories(${PLSSVM_HIP_TEST_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)

# link against test library
target_link_libraries(${PLSSVM_HIP_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})


# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_HIP_TEST_NAME})


# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_HIP_TEST_NAME})
endif ()