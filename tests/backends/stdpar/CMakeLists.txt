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

# add test executable
add_executable(${PLSSVM_STDPAR_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../main.cpp ${PLSSVM_STDPAR_TEST_SOURCES})

# link against test library
target_link_libraries(${PLSSVM_STDPAR_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})
if (PLSSVM_STDPAR_BACKEND MATCHES "ACPP")
    target_compile_options(${PLSSVM_STDPAR_TEST_NAME} PRIVATE ${PLSSVM_STDPAR_BACKEND_ACPP_COMPILER_FLAG})
endif ()

# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_STDPAR_TEST_NAME})

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_STDPAR_TEST_NAME})
endif ()