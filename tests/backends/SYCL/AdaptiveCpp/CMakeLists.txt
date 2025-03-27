# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# create SYCL tests
set(PLSSVM_SYCL_ADAPTIVECPP_TEST_NAME SYCL_AdaptiveCpp_tests)

# list all necessary sources
set(PLSSVM_SYCL_ADAPTIVECPP_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/../exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/adaptivecpp_csvm.cpp
    # ~~~
    # since the SYCL implementation_type and kernel_invocation_type enumerations are used even if no SYCL backend
    # is available these are also tested in the base library
    # ${CMAKE_CURRENT_LIST_DIR}/../implementation_types.cpp
    # ${CMAKE_CURRENT_LIST_DIR}/../kernel_invocation_types.cpp
    # ~~~
)

# add test executable
add_executable(${PLSSVM_SYCL_ADAPTIVECPP_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../../main.cpp ${PLSSVM_SYCL_ADAPTIVECPP_TEST_SOURCES})

# AdaptiveCpp's way to add SYCL to a target
find_package(AdaptiveCpp CONFIG)
add_sycl_to_target(TARGET ${PLSSVM_SYCL_ADAPTIVECPP_TEST_NAME} SOURCES ${PLSSVM_SYCL_ADAPTIVECPP_TEST_SOURCES})

# link against test library
target_link_libraries(${PLSSVM_SYCL_ADAPTIVECPP_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})

# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_SYCL_ADAPTIVECPP_TEST_NAME})

# add minimal run test for the classification task with the AdaptiveCpp SYCL backend
add_test(NAME MainTrainClassificationSYCLAdaptiveCpp/executable_minimal
         COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} -s 0 -b sycl --sycl_implementation_type acpp "${PLSSVM_CLASSIFICATION_TEST_FILE}"
                 "${CMAKE_CURRENT_BINARY_DIR}/test_classification.libsvm.model"
)
add_test(
    NAME MainPredictClassificationSYCLAdaptiveCpp/executable_minimal
    COMMAND
        ${PLSSVM_EXECUTABLE_PREDICT_NAME} -b sycl --sycl_implementation_type acpp
        "${CMAKE_CURRENT_LIST_DIR}/../../../data/libsvm/classification/5x4.libsvm" # test file
        "${CMAKE_CURRENT_LIST_DIR}/../../../data/model/classification/5x4.libsvm.model" # model file
        "${CMAKE_CURRENT_BINARY_DIR}/test_classification.libsvm.predict" # predict file (result)
)

# add minimal run test for the regression task with the AdaptiveCpp SYCL backend
add_test(NAME MainTrainRegressionSYCLAdaptiveCpp/executable_minimal
         COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} -s 1 -b sycl --sycl_implementation_type acpp "${PLSSVM_REGRESSION_TEST_FILE}"
                 "${CMAKE_CURRENT_BINARY_DIR}/test_regression.libsvm.model"
)
add_test(
    NAME MainPredictRegressionSYCLAdaptiveCpp/executable_minimal
    COMMAND
        ${PLSSVM_EXECUTABLE_PREDICT_NAME} -b sycl --sycl_implementation_type acpp
        "${CMAKE_CURRENT_LIST_DIR}/../../../data/libsvm/regression/5x4.libsvm" # test file
        "${CMAKE_CURRENT_LIST_DIR}/../../../data/model/regression/5x4.libsvm.model" # model file
        "${CMAKE_CURRENT_BINARY_DIR}/test_regression.libsvm.predict" # predict file (result)
)

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_SYCL_ADAPTIVECPP_TEST_NAME})
endif ()
