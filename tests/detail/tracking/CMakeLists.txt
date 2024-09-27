## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

## create specific hardware sampler tests
set(PLSSVM_PERFORMANCE_TRACKING_TEST_NAME performance_tracking_tests)

# list all necessary sources
set(PLSSVM_PERFORMANCE_TRACKING_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/events.cpp;
    ${CMAKE_CURRENT_LIST_DIR}/performance_tracker.cpp;
    ${CMAKE_CURRENT_LIST_DIR}/utility.cpp;
)

# add test executable
add_executable(${PLSSVM_PERFORMANCE_TRACKING_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../main.cpp ${PLSSVM_PERFORMANCE_TRACKING_TEST_SOURCES})

# link against test library
target_link_libraries(${PLSSVM_PERFORMANCE_TRACKING_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})

# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_PERFORMANCE_TRACKING_TEST_NAME})

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_PERFORMANCE_TRACKING_TEST_NAME})
endif ()



