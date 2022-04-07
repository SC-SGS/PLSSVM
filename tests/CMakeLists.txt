## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

## setup testing wit GoogleTest
find_package(GTest QUIET)
if(GTEST_FOUND)
    message(STATUS "Found package GTest.")
else()
    message(STATUS "Couldn't find package GTest. Building from source...")
    # fetch testing framework GoogleTest
    include(FetchContent)
    FetchContent_Declare(googletest
            GIT_REPOSITORY          https://github.com/google/googletest
            GIT_TAG                 e2239ee6043f73722e7aa812a459f54a28552929
            GIT_SHALLOW             TRUE
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_SOURCE_DIR} EXCLUDE_FROM_ALL)
    endif()
endif()


# create larger test files if desired
option(PLSSVM_GENERATE_TEST_FILE "Generate larger test file" ON)
if(PLSSVM_GENERATE_TEST_FILE)
    # set default sizes
    set(PLSSVM_TEST_FILE_NUM_DATA_POINTS "5000" CACHE STRING "Number of data points for the test file generator")
    set(PLSSVM_TEST_FILE_NUM_FEATURES "2000" CACHE STRING "Number of features for each data point for the test file generator")

    # set test file name based on requested sizes
    set(PLSSVM_TEST_FILE "${PROJECT_SOURCE_DIR}/tests/data/${PLSSVM_TEST_FILE_NUM_DATA_POINTS}x${PLSSVM_TEST_FILE_NUM_FEATURES}.libsvm" CACHE STRING "Path including name of the test file" FORCE)

    # check if the requested test file already exists (otherwise create new test file)
    if(EXISTS "${PLSSVM_TEST_FILE}")
        message(STATUS "Skipped test file generation since it already exists (${PLSSVM_TEST_FILE})!")
    else()
        # find necessary Python3 packages to use the test file generator
        find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

        include(${PROJECT_SOURCE_DIR}/cmake/check_python_libs.cmake)
        set(PLSSVM_TEST_REQUIRED_LIBS argparse sklearn)
        message(STATUS "Checking required Python3 libraries (${PLSSVM_TEST_REQUIRED_LIBS}) to to generate the test file.")
        check_python_libs(${PLSSVM_TEST_REQUIRED_LIBS} "or set PLSSVM_GENERATE_TEST_FILE to OFF.")

        # generate test file
        message(STATUS "Generating test file: ${PLSSVM_TEST_FILE}.")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/utility_scripts/generate_data.py" "--output" "${PLSSVM_TEST_FILE}" "--format" "libsvm" "--samples" "${PLSSVM_TEST_FILE_NUM_DATA_POINTS}" "--features" "${PLSSVM_TEST_FILE_NUM_FEATURES}"
            OUTPUT_QUIET)
    endif()
else()
    # set test file to default test file
    set(PLSSVM_TEST_FILE "${PROJECT_SOURCE_DIR}/tests/data/libsvm/500x200.libsvm" CACHE STRING "Path including name of the test file")
endif()


## create base test library
set(PLSSVM_BASE_TEST_LIBRARY_NAME base_tests_lib)

# list all necessary sources
set(PLSSVM_BASE_TEST_LIBRARY_NAME_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backends/compare.cpp
)

# create base test library
add_library(${PLSSVM_BASE_TEST_LIBRARY_NAME} STATIC ${PLSSVM_BASE_TEST_LIBRARY_NAME_SOURCES})

# set necessary include directories
target_include_directories(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# add necessary compile definitions
target_compile_definitions(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC PLSSVM_TEST_PATH="${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC PLSSVM_TEST_FILE="${PLSSVM_TEST_FILE}")

# link against plssvm library
target_link_libraries(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC ${PLSSVM_ALL_LIBRARY_NAME})

# link against gtest
target_link_libraries(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC GTest::gtest_main GTest::gmock)



## create base tests
set(PLSSVM_BASE_TEST_NAME Base_tests)

# list all necessary sources
set(PLSSVM_BASE_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/base_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/parameter_test.cpp
)

# create test executable
add_executable(${PLSSVM_BASE_TEST_NAME} ${PLSSVM_BASE_TEST_SOURCES})

# link against base library
target_link_libraries(${PLSSVM_BASE_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})


# add tests to google test
include(GoogleTest)
gtest_add_tests(TARGET ${PLSSVM_BASE_TEST_NAME} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)


# add test for help messages
add_test(NAME ParameterTrain/plssvm-train_help COMMAND plssvm-train --help)
add_test(NAME ParameterPredict/plssvm-predict_help COMMAND plssvm-predict --help)


# add test as coverage dependency
if(TARGET coverage)
    add_dependencies(coverage ${PLSSVM_BASE_TEST_NAME})
endif()


# add tests for the available backends
add_subdirectory(backends)
