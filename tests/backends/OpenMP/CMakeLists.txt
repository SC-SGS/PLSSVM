# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# create OpenMP tests
set(PLSSVM_OPENMP_TEST_NAME OpenMP_tests)

# list all necessary sources
set(PLSSVM_OPENMP_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp ${CMAKE_CURRENT_LIST_DIR}/openmp_csvm.cpp)

# add test executable
add_executable(${PLSSVM_OPENMP_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../main.cpp ${PLSSVM_OPENMP_TEST_SOURCES})

# link against test library
target_link_libraries(${PLSSVM_OPENMP_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})

# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_OPENMP_TEST_NAME})

# add minimal run test for the classification task with the OpenMP backend
add_test(NAME MainTrainClassificationOpenMP/executable_minimal COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} -s 0 -b openmp "${PLSSVM_CLASSIFICATION_TEST_FILE}"
         "${CMAKE_CURRENT_BINARY_DIR}/test_classification.libsvm.model"
)
add_test(NAME MainPredictClassificationOpenMP/executable_minimal
         COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} -b openmp "${CMAKE_CURRENT_LIST_DIR}/../../data/libsvm/classification/5x4.libsvm" # test file
         "${CMAKE_CURRENT_LIST_DIR}/../../data/model/classification/5x4.libsvm.model" # model file
         "${CMAKE_CURRENT_BINARY_DIR}/test_classification.libsvm.predict" # predict file (result)
)

# add minimal run test for the regression task with the OpenMP backend
add_test(NAME MainTrainRegressionOpenMP/executable_minimal COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} -s 1 -b openmp "${PLSSVM_REGRESSION_TEST_FILE}"
         "${CMAKE_CURRENT_BINARY_DIR}/test_regression.libsvm.model"
)
add_test(NAME MainPredictRegressionOpenMP/executable_minimal
         COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} -b openmp "${CMAKE_CURRENT_LIST_DIR}/../../data/libsvm/regression/5x4.libsvm" # test file
         "${CMAKE_CURRENT_LIST_DIR}/../../data/model/regression/5x4.libsvm.model" # model file
         "${CMAKE_CURRENT_BINARY_DIR}/test_regression.libsvm.predict" # predict file (result)
)

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_OPENMP_TEST_NAME})
endif ()
