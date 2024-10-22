## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

## create Kokkos tests
set(PLSSVM_KOKKOS_TEST_NAME Kokkos_tests)

# list all necessary sources
set(PLSSVM_KOKKOS_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/typedefs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/standard_layout_tuple.cpp
#    ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.cpp
#    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cu
    ${CMAKE_CURRENT_LIST_DIR}/kokkos_csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/execution_space.cpp
)

find_package(Kokkos REQUIRED)

# add test executable
add_executable(${PLSSVM_KOKKOS_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../kokkos_main.cpp ${PLSSVM_KOKKOS_TEST_SOURCES})

# link against test library
target_link_libraries(${PLSSVM_KOKKOS_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})

# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_KOKKOS_TEST_NAME})

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_KOKKOS_TEST_NAME})
endif ()