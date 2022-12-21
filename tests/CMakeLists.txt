## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "Tests:  ")

## setup testing wit GoogleTest
find_package(GTest 1.11.0 QUIET)
if (GTEST_FOUND)
    message(STATUS "Found package GTest.")
else ()
    message(STATUS "Couldn't find package GTest. Building from source...")
    set(PLSSVM_googletest_VERSION release-1.11.0)
    # fetch testing framework GoogleTest
    include(FetchContent)
    FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest
            GIT_TAG ${PLSSVM_googletest_VERSION}
            GIT_SHALLOW TRUE
            QUIET
            )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_GetProperties(googletest)
    if (NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_SOURCE_DIR} EXCLUDE_FROM_ALL)
    endif ()
    message(STATUS "Installed GoogleTest version ${PLSSVM_googletest_VERSION}.")
endif ()


# create larger test files if desired
option(PLSSVM_GENERATE_TEST_FILE "Generate larger test file" ON)
if (PLSSVM_GENERATE_TEST_FILE)
    # set default sizes
    set(PLSSVM_TEST_FILE_NUM_DATA_POINTS "5000" CACHE STRING "Number of data points for the test file generator")
    set(PLSSVM_TEST_FILE_NUM_FEATURES "2000" CACHE STRING "Number of features for each data point for the test file generator")

    # set test file name based on requested sizes
    set(PLSSVM_TEST_FILE "${PROJECT_SOURCE_DIR}/tests/data/${PLSSVM_TEST_FILE_NUM_DATA_POINTS}x${PLSSVM_TEST_FILE_NUM_FEATURES}.libsvm" CACHE STRING "Path including name of the test file" FORCE)

    # check if the requested test file already exists (otherwise create new test file)
    if (EXISTS "${PLSSVM_TEST_FILE}")
        message(STATUS "Skipped test file generation since it already exists (${PLSSVM_TEST_FILE})!")
    else ()
        # find necessary Python3 packages to use the test file generator
        find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

        include(${PROJECT_SOURCE_DIR}/cmake/check_python_libs.cmake)
        set(PLSSVM_TEST_REQUIRED_LIBS argparse sklearn humanize)
        message(STATUS "Checking required Python3 libraries (${PLSSVM_TEST_REQUIRED_LIBS}) to to generate the test file.")
        check_python_libs("${PLSSVM_TEST_REQUIRED_LIBS}" "or set PLSSVM_GENERATE_TEST_FILE to OFF.")

        # generate test file
        message(STATUS "Generating test file: ${PLSSVM_TEST_FILE}.")
        execute_process(
                COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/utility_scripts/generate_data.py" "--output" "${PLSSVM_TEST_FILE}" "--format" "libsvm" "--samples" "${PLSSVM_TEST_FILE_NUM_DATA_POINTS}" "--features" "${PLSSVM_TEST_FILE_NUM_FEATURES}"
                OUTPUT_QUIET)
    endif ()
else ()
    # set test file to default test file
    set(PLSSVM_TEST_FILE "${PROJECT_SOURCE_DIR}/tests/data/libsvm/500x200.libsvm" CACHE STRING "Path including name of the test file")
endif ()


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
## since the SYCL implementation_type and kernel_invocation_type enumerations are used even if no SYCL backend
## is available these are also tested in the base library
        ${CMAKE_CURRENT_LIST_DIR}/backends/SYCL/implementation_type.cpp
        ${CMAKE_CURRENT_LIST_DIR}/backends/SYCL/kernel_invocation_type.cpp

        ${CMAKE_CURRENT_LIST_DIR}/detail/cmd/data_set_variants.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/cmd/parser_predict.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/cmd/parser_scale.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/cmd/parser_train.cpp

        ${CMAKE_CURRENT_LIST_DIR}/detail/io/arff_parsing.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/io/file_reader.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/io/libsvm_model_parsing.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/io/libsvm_parsing.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/io/scaling_factors_parsing.cpp

        ${CMAKE_CURRENT_LIST_DIR}/detail/arithmetic_type_name.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/assert.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/execution_range.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/layout.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/operators.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/sha256.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/string_conversion.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/string_utility.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/type_traits.cpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp

        ${CMAKE_CURRENT_LIST_DIR}/exceptions/exceptions.cpp
        ${CMAKE_CURRENT_LIST_DIR}/exceptions/source_location.cpp

        ${CMAKE_CURRENT_LIST_DIR}/version/git_metadata/git_metadata.cpp
        ${CMAKE_CURRENT_LIST_DIR}/version/version.cpp

        ${CMAKE_CURRENT_LIST_DIR}/csvm_factory.cpp
        ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
        ${CMAKE_CURRENT_LIST_DIR}/backend_types.cpp
        ${CMAKE_CURRENT_LIST_DIR}/data_set.cpp
        ${CMAKE_CURRENT_LIST_DIR}/default_value.cpp
        ${CMAKE_CURRENT_LIST_DIR}/file_format_types.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernel_function_types.cpp
        ${CMAKE_CURRENT_LIST_DIR}/model.cpp
        ${CMAKE_CURRENT_LIST_DIR}/parameter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/target_platforms.cpp
        )

# create test executable
add_executable(${PLSSVM_BASE_TEST_NAME} main.cpp ${PLSSVM_BASE_TEST_SOURCES})

# link against base library
target_link_libraries(${PLSSVM_BASE_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})


# add tests to google test
include(GoogleTest)
string(TOUPPER "${CMAKE_BUILD_TYPE}" PLSSVM_CMAKE_BUILD_TYPE_UPPER_CASE)
set(PLSSVM_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_${PLSSVM_CMAKE_BUILD_TYPE_UPPER_CASE}}")
if (PLSSVM_ENABLE_ASSERTS OR NOT "${PLSSVM_CMAKE_CXX_FLAGS}" MATCHES ".*NDEBUG.*")
    # assertions are enabled -> enable Google death tests
    message(STATUS "Enabling GTest's death tests.")
    gtest_discover_tests(${PLSSVM_BASE_TEST_NAME})
else ()
    # assertions are disabled -> disable Google death tests
    message(STATUS "Disabling GTest's death tests.")
    gtest_discover_tests(${PLSSVM_BASE_TEST_NAME} EXTRA_ARGS --gtest_filter -*DeathTest)
endif ()


# add test for help messages
add_test(NAME MainTrain/executable_help COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} --help)
add_test(NAME MainPredict/executable_help COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} --help)
add_test(NAME MainScale/executable_help COMMAND ${PLSSVM_EXECUTABLE_SCALE_NAME} --help)

# add test for version messages
add_test(NAME MainTrain/executable_version COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} --version)
add_test(NAME MainPredict/executable_version COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} --version)
add_test(NAME MainScale/executable_version COMMAND ${PLSSVM_EXECUTABLE_SCALE_NAME} --version)

# add minimal run test
add_test(NAME MainTrain/executable_minimal COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} "${PLSSVM_TEST_FILE}" "${CMAKE_CURRENT_BINARY_DIR}/test.libsvm.model")
add_test(NAME MainPredict/executable_minimal COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME}
        "${CMAKE_CURRENT_LIST_DIR}/data/predict/500x200_test.libsvm"          # test file
        "${CMAKE_CURRENT_LIST_DIR}/data/predict/500x200_linear.libsvm.model"  # model file
        "${CMAKE_CURRENT_BINARY_DIR}/test.libsvm.predict"                     # predict file (result)
        )
add_test(NAME MainScale/executable_minimal COMMAND ${PLSSVM_EXECUTABLE_SCALE_NAME} "${PLSSVM_TEST_FILE}" "${CMAKE_CURRENT_BINARY_DIR}/scaled.libsvm.model")

# add failing test (must return with a non-zero exit code)
add_test(NAME MainTrain/executable_fail COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} " ")
set_tests_properties(MainTrain/executable_fail PROPERTIES WILL_FAIL TRUE)
add_test(NAME MainPredict/executable_fail COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME}  " " " ")
set_tests_properties(MainPredict/executable_fail PROPERTIES WILL_FAIL TRUE)
add_test(NAME MainScale/executable_fail COMMAND ${PLSSVM_EXECUTABLE_SCALE_NAME} " ")
set_tests_properties(MainScale/executable_fail PROPERTIES WILL_FAIL TRUE)

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_BASE_TEST_NAME})
endif ()


# add tests for the available backends
add_subdirectory(backends)

list(POP_BACK CMAKE_MESSAGE_INDENT)
