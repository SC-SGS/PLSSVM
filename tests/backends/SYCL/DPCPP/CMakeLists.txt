# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# create SYCL tests
set(PLSSVM_SYCL_DPCPP_TEST_NAME SYCL_DPCPP_tests)

# list all necessary sources
set(PLSSVM_SYCL_DPCPP_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/../exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dpcpp_csvm.cpp
    # ~~~
    # since the SYCL implementation_type and data_parallel_kernel enumerations are used even if no SYCL backend
    # is available these are also tested in the base library
    # ${CMAKE_CURRENT_LIST_DIR}/../implementation_types.cpp
    # ${CMAKE_CURRENT_LIST_DIR}/../data_parallel_kernels.cpp
    # ~~~
)

# add test executable
add_executable(${PLSSVM_SYCL_DPCPP_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../../main.cpp ${PLSSVM_SYCL_DPCPP_TEST_SOURCES})

# enable DPC++ SYCL support
target_compile_options(${PLSSVM_SYCL_DPCPP_TEST_NAME} PRIVATE -sycl-std=2020 -fsycl)
target_link_options(${PLSSVM_SYCL_DPCPP_TEST_NAME} PRIVATE -fsycl)

# link against test library
target_link_libraries(${PLSSVM_SYCL_DPCPP_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})

# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_SYCL_DPCPP_TEST_NAME})

# add minimal run test for the classification task with the DPC++ SYCL backend
add_test(NAME MainTrainClassificationSYCLDPCPP/ExecutableMinimal
         COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} -s 0 -b sycl --sycl_implementation_type dpcpp "${PLSSVM_CLASSIFICATION_TEST_FILE}"
                 "${CMAKE_CURRENT_BINARY_DIR}/test_classification.libsvm.model"
)
add_test(
    NAME MainPredictClassificationSYCLDPCPP/ExecutableMinimal
    COMMAND
        ${PLSSVM_EXECUTABLE_PREDICT_NAME} -b sycl --sycl_implementation_type dpcpp
        "${CMAKE_CURRENT_LIST_DIR}/../../../data/libsvm/classification/5x4.libsvm" # test file
        "${CMAKE_CURRENT_LIST_DIR}/../../../data/model/classification/5x4.libsvm.model" # model file
        "${CMAKE_CURRENT_BINARY_DIR}/test_classification.libsvm.predict" # predict file (result)
)

# add minimal run test for the regression task with the DPC++ SYCL backend
add_test(NAME MainTrainRegressionSYCLDPCPP/ExecutableMinimal
         COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} -s 1 -b sycl --sycl_implementation_type dpcpp "${PLSSVM_REGRESSION_TEST_FILE}"
                 "${CMAKE_CURRENT_BINARY_DIR}/test_regression.libsvm.model"
)
add_test(
    NAME MainPredictRegressionSYCLDPCPP/ExecutableMinimal
    COMMAND
        ${PLSSVM_EXECUTABLE_PREDICT_NAME} -b sycl --sycl_implementation_type dpcpp
        "${CMAKE_CURRENT_LIST_DIR}/../../../data/libsvm/regression/5x4.libsvm" # test file
        "${CMAKE_CURRENT_LIST_DIR}/../../../data/model/regression/5x4.libsvm.model" # model file
        "${CMAKE_CURRENT_BINARY_DIR}/test_regression.libsvm.predict" # predict file (result)
)

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_SYCL_DPCPP_TEST_NAME})
endif ()
