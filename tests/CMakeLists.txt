# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "Tests:  ")

# setup testing wit GoogleTest
set(PLSSVM_googletest_VERSION v1.15.2)
find_package(GTest 1.15.2 QUIET)
if (GTEST_FOUND)
    message(STATUS "Found package GTest.")
else ()
    message(STATUS "Couldn't find package GTest. Building version ${PLSSVM_googletest_VERSION} from source.")
    set(CMAKE_CXX_FLAGS_OLD "${CMAKE_CXX_FLAGS}")
    if (PLSSVM_ENABLE_STL_DEBUG_MODE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PLSSVM_STL_DEBUG_MODE_FLAGS}")
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    # fetch testing framework GoogleTest
    include(FetchContent)
    # TODO: really use custom fork? -> try finding another GoogleTest fix for MSVC on Windows
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/breyerml/googletest
        GIT_TAG windows-msvc-fix
        GIT_SHALLOW TRUE
        QUIET
    )
    FetchContent_MakeAvailable(googletest)
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_OLD}")
endif ()

# create larger test files if desired
option(PLSSVM_GENERATE_TEST_FILES "Generate test files with user defined sizes." ON)
if (PLSSVM_GENERATE_TEST_FILES)
    include(${PROJECT_SOURCE_DIR}/cmake/utility_macros.cmake)
    # set default sizes
    set(PLSSVM_TEST_FILE_NUM_DATA_POINTS "5000" CACHE STRING "Number of data points for the test file generator")
    check_integer(PLSSVM_TEST_FILE_NUM_DATA_POINTS)
    set(PLSSVM_TEST_FILE_NUM_FEATURES "2000" CACHE STRING "Number of features for each data point for the test file generator")
    check_integer(PLSSVM_TEST_FILE_NUM_FEATURES)
    set(PLSSVM_TEST_FILE_NUM_CLASSES "4" CACHE STRING "Number of classes (classification task only) for the test file generator")
    check_integer(PLSSVM_TEST_FILE_NUM_CLASSES)

    # set test file name based on requested sizes
    set(PLSSVM_CLASSIFICATION_TEST_FILE
        "${PROJECT_SOURCE_DIR}/tests/data/libsvm/classification/${PLSSVM_TEST_FILE_NUM_DATA_POINTS}x${PLSSVM_TEST_FILE_NUM_FEATURES}_${PLSSVM_TEST_FILE_NUM_CLASSES}.libsvm"
        CACHE STRING "Path including name of the classification test file" FORCE
    )
    set(PLSSVM_REGRESSION_TEST_FILE "${PROJECT_SOURCE_DIR}/tests/data/libsvm/regression/${PLSSVM_TEST_FILE_NUM_DATA_POINTS}x${PLSSVM_TEST_FILE_NUM_FEATURES}.libsvm"
        CACHE STRING "Path including name of the regression test file" FORCE
    )

    # check if the requested test file already exists and is not zero based indexed (otherwise create new test file)
    check_test_file_validity(${PLSSVM_CLASSIFICATION_TEST_FILE})
    check_test_file_validity(${PLSSVM_REGRESSION_TEST_FILE})

    # check for possible Python dependencies
    if (NOT EXISTS "${PLSSVM_CLASSIFICATION_TEST_FILE}" OR NOT EXISTS "${PLSSVM_REGRESSION_TEST_FILE}")
        # find necessary Python3 packages to use the test file generator
        find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

        include(${PROJECT_SOURCE_DIR}/cmake/check_python_libs.cmake)
        set(PLSSVM_TEST_REQUIRED_LIBS argparse sklearn humanize numpy)
        message(STATUS "Checking required Python3 libraries (${PLSSVM_TEST_REQUIRED_LIBS}) to to generate the test file.")
        check_python_libs("${PLSSVM_TEST_REQUIRED_LIBS}" "or set PLSSVM_GENERATE_TEST_FILES to OFF.")
    endif ()

    if (NOT EXISTS "${PLSSVM_CLASSIFICATION_TEST_FILE}")
        # generate the classification test file if necessary
        message(STATUS "Generating the classification test file: ${PLSSVM_CLASSIFICATION_TEST_FILE}.")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/utility_scripts/generate_data.py" "--output" "${PLSSVM_CLASSIFICATION_TEST_FILE}" "--format"
                    "libsvm" "--samples" "${PLSSVM_TEST_FILE_NUM_DATA_POINTS}" "--features" "${PLSSVM_TEST_FILE_NUM_FEATURES}" "classification" "--classes"
                    "${PLSSVM_TEST_FILE_NUM_CLASSES}" OUTPUT_QUIET
        )
    endif ()
    if (NOT EXISTS "${PLSSVM_REGRESSION_TEST_FILE}")
        # generate the regression test file if necessary
        message(STATUS "Generating the regression test file: ${PLSSVM_REGRESSION_TEST_FILE}.")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/utility_scripts/generate_data.py" "--output" "${PLSSVM_REGRESSION_TEST_FILE}" "--format"
                    "libsvm" "--samples" "${PLSSVM_TEST_FILE_NUM_DATA_POINTS}" "--features" "${PLSSVM_TEST_FILE_NUM_FEATURES}" "regression" OUTPUT_QUIET
        )
    endif ()
else ()
    # set test file to default test file
    set(PLSSVM_CLASSIFICATION_TEST_FILE "${PROJECT_SOURCE_DIR}/tests/data/libsvm/classification/500x200_4.libsvm"
        CACHE STRING "Path including name of the classification test file"
    )
    set(PLSSVM_REGRESSION_TEST_FILE "${PROJECT_SOURCE_DIR}/tests/data/libsvm/regression/500x200.libsvm" CACHE STRING
                                                                                                              "Path including name of the regression test file"
    )
endif ()

# warn if PLSSVM_ENABLE_FAST_MATH is set and tests are enabled
if (PLSSVM_ENABLE_FAST_MATH)
    message(
        WARNING "PLSSVM_ENABLE_FAST_MATH is set while also testing is enabled. This may result in tests failing due to increased floating point inaccuracies."
    )
endif ()

# create base test library
set(PLSSVM_BASE_TEST_LIBRARY_NAME base_tests_lib)

# list all necessary sources
set(PLSSVM_BASE_TEST_LIBRARY_NAME_SOURCES ${CMAKE_CURRENT_LIST_DIR}/backends/ground_truth.cpp)

# create base test library
add_library(${PLSSVM_BASE_TEST_LIBRARY_NAME} SHARED ${PLSSVM_BASE_TEST_LIBRARY_NAME_SOURCES})

# optionally use a reduced number of label types in the tests
option(PLSSVM_TEST_WITH_REDUCED_LABEL_TYPES "Reduce the number of tested label types to speed up compilation and actual tests." OFF)
if (PLSSVM_TEST_WITH_REDUCED_LABEL_TYPES)
    message(STATUS "Reducing the number of tested label types.")
    target_compile_definitions(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC PLSSVM_TEST_WITH_REDUCED_LABEL_TYPES)
endif ()

# set necessary include directories
target_include_directories(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")

# add necessary compile definitions
target_compile_definitions(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC PLSSVM_TEST_PATH="${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC PLSSVM_CLASSIFICATION_TEST_FILE="${PLSSVM_CLASSIFICATION_TEST_FILE}")
target_compile_definitions(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC PLSSVM_REGRESSION_TEST_FILE="${PLSSVM_REGRESSION_TEST_FILE}")

# link against plssvm library
target_link_libraries(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC ${PLSSVM_ALL_LIBRARY_NAME})

# link against gtest
target_link_libraries(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC GTest::gmock)

# create base tests
set(PLSSVM_BASE_TEST_NAME Base_tests)

# list all necessary sources
set(PLSSVM_BASE_TEST_SOURCES
    # it is unnecessary to test the execution range for each backend
    ${CMAKE_CURRENT_LIST_DIR}/backends/execution_range.cpp
    # since the SYCL implementation_type and kernel_invocation_type enumerations are used even if no SYCL backend is available these are also tested in the base
    # library
    ${CMAKE_CURRENT_LIST_DIR}/backends/SYCL/implementation_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backends/SYCL/kernel_invocation_types.cpp
    # since the stdpar implementation_type enumeration is used even if no stdpar backend is available this is also tested in the base library
    ${CMAKE_CURRENT_LIST_DIR}/backends/stdpar/implementation_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_set/classification/constructors.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_set/classification/getter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_set/classification/label_mapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_set/classification/save.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_set/regression/constructors.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_set/regression/getter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_set/regression/save.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_set/min_max_scaler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/cmd/data_set_variants.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/cmd/parser_predict.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/cmd/parser_scale.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/cmd/parser_train.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/classification_libsvm_model_parsing/data_parse_invalid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/classification_libsvm_model_parsing/data_parse_valid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/classification_libsvm_model_parsing/data_write.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/classification_libsvm_model_parsing/header_parse_invalid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/classification_libsvm_model_parsing/header_parse_valid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/classification_libsvm_model_parsing/header_write.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/classification_libsvm_model_parsing/utility_functions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/regression_libsvm_model_parsing/data_parse_invalid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/regression_libsvm_model_parsing/data_parse_valid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/regression_libsvm_model_parsing/data_write.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/regression_libsvm_model_parsing/header_parse_invalid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/regression_libsvm_model_parsing/header_parse_valid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/regression_libsvm_model_parsing/header_write.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/arff_parsing.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/file_reader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/libsvm_parsing.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/io/scaling_factors_parsing.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/arithmetic_type_name.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/assert.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/data_distribution.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/logging.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/logging_without_performance_tracking.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/memory_size.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/move_only_any.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/operators.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/sha256.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/string_conversion.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/string_utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/type_list.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/type_traits.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions/exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions/source_location.cpp
    ${CMAKE_CURRENT_LIST_DIR}/svm/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/svm/csvc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/svm/csvr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/version/git_metadata/git_metadata.cpp
    ${CMAKE_CURRENT_LIST_DIR}/version/version.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/classification_model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/classification_report.cpp
    ${CMAKE_CURRENT_LIST_DIR}/classification_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvc_factory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvr_factory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/file_format_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gamma.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_function_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_functions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matrix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/parameter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/regression_model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shape.cpp
    ${CMAKE_CURRENT_LIST_DIR}/solver_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/svm_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/target_platforms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/verbosity_levels.cpp
)

# create test executable
add_executable(${PLSSVM_BASE_TEST_NAME} main.cpp ${PLSSVM_BASE_TEST_SOURCES})

# link against base library
target_link_libraries(${PLSSVM_BASE_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})

# check whether death tests should be activated or not
string(TOUPPER "${CMAKE_BUILD_TYPE}" PLSSVM_CMAKE_BUILD_TYPE_UPPER_CASE)
if (PLSSVM_ENABLE_ASSERTS)
    # assertions are enabled -> enable Google death tests
    message(STATUS "Enabling GTest's death tests.")
    set(PLSSVM_ENABLE_DEATH_TESTS ON)
    target_compile_definitions(${PLSSVM_BASE_TEST_LIBRARY_NAME} PUBLIC PLSSVM_DEATH_TESTS_ENABLED)
else ()
    # assertions are disabled -> disable Google death tests
    message(STATUS "Disabling GTest's death tests.")
    set(PLSSVM_ENABLE_DEATH_TESTS OFF)
endif ()

# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_BASE_TEST_NAME})

# add test for help messages
add_test(NAME MainTrain/executable_help COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} --help)
add_test(NAME MainPredict/executable_help COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} --help)
add_test(NAME MainScale/executable_help COMMAND ${PLSSVM_EXECUTABLE_SCALE_NAME} --help)

# add test for version messages
add_test(NAME MainTrain/executable_version COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} --version)
add_test(NAME MainPredict/executable_version COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} --version)
add_test(NAME MainScale/executable_version COMMAND ${PLSSVM_EXECUTABLE_SCALE_NAME} --version)

# add minimal run test for scaling
add_test(NAME MainScale/executable_minimal COMMAND ${PLSSVM_EXECUTABLE_SCALE_NAME} "${PLSSVM_CLASSIFICATION_TEST_FILE}"
                                                   "${CMAKE_CURRENT_BINARY_DIR}/scaled.libsvm.model"
)
add_test(NAME MainScale/executable_save_scaling_factors
         COMMAND ${PLSSVM_EXECUTABLE_SCALE_NAME} -s "${CMAKE_CURRENT_BINARY_DIR}/scaling_parameter.txt" # the file the scaling parameters are saved to
                 "${PLSSVM_CLASSIFICATION_TEST_FILE}" # the file to scale
                 "${CMAKE_CURRENT_BINARY_DIR}/scaled.libsvm.model" # the scaled file (result)
)

# add minimal run test for the classification task
add_test(NAME MainTrainClassification/executable_minimal COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} "${PLSSVM_CLASSIFICATION_TEST_FILE}"
                                                                 "${CMAKE_CURRENT_BINARY_DIR}/test_classification.libsvm.model"
)
add_test(NAME MainPredictClassification/executable_minimal
         COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} "${CMAKE_CURRENT_LIST_DIR}/data/libsvm/classification/5x4.libsvm" # test file
                 "${CMAKE_CURRENT_LIST_DIR}/data/model/classification/5x4.libsvm.model" # model file
                 "${CMAKE_CURRENT_BINARY_DIR}/test_classification.libsvm.predict" # predict file (result)
)
# add minimal run test for the regression task
add_test(NAME MainTrainRegression/executable_minimal COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} "${PLSSVM_REGRESSION_TEST_FILE}"
                                                             "${CMAKE_CURRENT_BINARY_DIR}/test_regression.libsvm.model"
)
add_test(NAME MainPredictRegression/executable_minimal
         COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} "${CMAKE_CURRENT_LIST_DIR}/data/libsvm/regression/5x4.libsvm" # test file
                 "${CMAKE_CURRENT_LIST_DIR}/data/model/regression/5x4.libsvm.model" # model file
                 "${CMAKE_CURRENT_BINARY_DIR}/test_regression.libsvm.predict" # predict file (result)
)

# add failing test (must return with a non-zero exit code)
add_test(NAME MainTrain/executable_fail COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} " ")
set_tests_properties(MainTrain/executable_fail PROPERTIES WILL_FAIL TRUE)
add_test(NAME MainPredict/executable_fail COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} " " " ")
set_tests_properties(MainPredict/executable_fail PROPERTIES WILL_FAIL TRUE)
add_test(NAME MainScale/executable_fail COMMAND ${PLSSVM_EXECUTABLE_SCALE_NAME} " ")
set_tests_properties(MainScale/executable_fail PROPERTIES WILL_FAIL TRUE)

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_BASE_TEST_NAME})
endif ()

# add tests for the available backends
add_subdirectory(backends)

# add tests for the hardware samplers
if (PLSSVM_ENABLE_PERFORMANCE_TRACKING)
    add_subdirectory(detail/tracking)
endif ()

# fix shared library linker error on Windows
if (NOT GTEST_FOUND)
    foreach (target gmock gtest)
        target_compile_definitions(${target} INTERFACE GTEST_LINKED_AS_SHARED_LIBRARY=1)
    endforeach ()
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
