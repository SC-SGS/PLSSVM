## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

## create HIP tests
set(PLSSVM_HIP_TEST_NAME HIP_tests)

# list all necessary sources
set(PLSSVM_HIP_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/hip_csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device_ptr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
)

# add test executable
add_executable(${PLSSVM_HIP_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../main.cpp ${PLSSVM_HIP_TEST_SOURCES})

# link against test library
target_link_libraries(${PLSSVM_HIP_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})


# add tests to google test
include(GoogleTest)
gtest_add_tests(TARGET ${PLSSVM_HIP_TEST_NAME} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)

# add test as coverage dependency
if(TARGET coverage)
    add_dependencies(coverage ${PLSSVM_HIP_TEST_NAME})
endif()