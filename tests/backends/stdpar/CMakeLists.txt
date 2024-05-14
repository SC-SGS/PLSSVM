## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

## create stdpar tests
set(PLSSVM_STDPAR_TEST_NAME stdpar_tests)

# list all necessary sources
set(PLSSVM_STDPAR_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/implementation_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/stdpar_csvm.cpp
)

# add stdpar implementation specific test files containing tests for the stdpar::csvm constructors
if (PLSSVM_STDPAR_BACKEND STREQUAL "NVHPC")
    list(APPEND PLSSVM_STDPAR_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/NVHPC/stdpar_csvm.cpp)
elseif (PLSSVM_STDPAR_BACKEND STREQUAL "roc-stdpar")
    list(APPEND PLSSVM_STDPAR_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/roc-stdpar/stdpar_csvm.cpp)
elseif (PLSSVM_STDPAR_BACKEND STREQUAL "IntelLLVM")
    list(APPEND PLSSVM_STDPAR_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/IntelLLVM/stdpar_csvm.cpp)
elseif (PLSSVM_STDPAR_BACKEND STREQUAL "ACPP")
    list(APPEND PLSSVM_STDPAR_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/AdaptiveCpp/stdpar_csvm.cpp)
elseif (PLSSVM_STDPAR_BACKEND STREQUAL "GNU_TBB")
    list(APPEND PLSSVM_STDPAR_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/GNU_TBB/stdpar_csvm.cpp)
endif ()

# add test executable
add_executable(${PLSSVM_STDPAR_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../main.cpp ${PLSSVM_STDPAR_TEST_SOURCES})

# link against test library
target_link_libraries(${PLSSVM_STDPAR_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})

# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_STDPAR_TEST_NAME})

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_STDPAR_TEST_NAME})
endif ()