## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

### create SYCL tests
set(PLSSVM_SYCL_TEST_NAME SYCL_tests)

# list all necessary sources
set(PLSSVM_SYCL_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
## since the SYCL implementation_type and kernel_invocation_type enumerations are used even if no SYCL backend
## is available these are also tested in the base library
#        ${CMAKE_CURRENT_LIST_DIR}/implementation_type.cpp
#        ${CMAKE_CURRENT_LIST_DIR}/kernel_invocation_type.cpp
)

if (TARGET ${PLSSVM_SYCL_BACKEND_ADAPTIVECPP_LIBRARY_NAME})
    # add AdaptiveCpp specific test files
    list(APPEND PLSSVM_SYCL_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/AdaptiveCpp/detail/device_ptr.cpp)
    list(APPEND PLSSVM_SYCL_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/AdaptiveCpp/detail/utility.cpp)
    list(APPEND PLSSVM_SYCL_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/AdaptiveCpp/sycl_csvm.cpp)
endif ()

if (TARGET ${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME})
    # add DPC++ specific test files
    list(APPEND PLSSVM_SYCL_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/DPCPP/detail/device_ptr.cpp)
    list(APPEND PLSSVM_SYCL_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/DPCPP/detail/utility.cpp)
    list(APPEND PLSSVM_SYCL_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/DPCPP/sycl_csvm.cpp)
endif ()

# add test executable
add_executable(${PLSSVM_SYCL_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../main.cpp ${PLSSVM_SYCL_TEST_SOURCES})
target_include_directories(${PLSSVM_SYCL_TEST_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)

# link against test library
target_link_libraries(${PLSSVM_SYCL_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})


# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_SYCL_TEST_NAME})


# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_SYCL_TEST_NAME})
endif ()