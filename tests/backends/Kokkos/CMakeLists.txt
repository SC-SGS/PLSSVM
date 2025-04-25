# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# create Kokkos tests
set(PLSSVM_KOKKOS_TEST_NAME Kokkos_tests)

# list all necessary sources
set(PLSSVM_KOKKOS_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/constexpr_available_execution_spaces.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_view_wrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_wrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/standard_layout_tuple.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kokkos_csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/execution_space_type_traits.cpp
)

find_package(Kokkos REQUIRED)

# add test executable
add_executable(${PLSSVM_KOKKOS_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../main.cpp ${PLSSVM_KOKKOS_TEST_SOURCES})

if (Kokkos_ENABLE_CUDA AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # tests won't compile with nvcc
    if (NOT PLSSVM_TEST_WITH_REDUCED_LABEL_TYPES)
        message(
            FATAL_ERROR
                "Due to template instantiation limits within nvcc, only reduced label type tests are currently supported! Alternatively, compile Kokkos CUDA with clang. "
        )
    endif ()
endif ()

# increase recursive template instantiation limit
target_compile_options(${PLSSVM_KOKKOS_TEST_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang,IntelLLVM>:-ftemplate-depth=2048>>)

# link against test library
target_link_libraries(${PLSSVM_KOKKOS_TEST_NAME} PRIVATE ${PLSSVM_BASE_TEST_LIBRARY_NAME})

# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_KOKKOS_TEST_NAME})

# add minimal run test for the classification task with the Kokkos backend
add_test(NAME MainTrainClassificationKokkos/executable_minimal COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} -s 0 -b kokkos "${PLSSVM_CLASSIFICATION_TEST_FILE}"
                                                                       "${CMAKE_CURRENT_BINARY_DIR}/test_classification.libsvm.model"
)
add_test(NAME MainPredictClassificationKokkos/executable_minimal
         COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} -b kokkos "${CMAKE_CURRENT_LIST_DIR}/../../data/libsvm/classification/5x4.libsvm" # test file
                 "${CMAKE_CURRENT_LIST_DIR}/../../data/model/classification/5x4.libsvm.model" # model file
                 "${CMAKE_CURRENT_BINARY_DIR}/test_classification.libsvm.predict" # predict file (result)
)

# add minimal run test for the regression task with the Kokkos backend
add_test(NAME MainTrainRegressionKokkos/executable_minimal COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} -s 1 -b kokkos "${PLSSVM_REGRESSION_TEST_FILE}"
                                                                   "${CMAKE_CURRENT_BINARY_DIR}/test_regression.libsvm.model"
)
add_test(NAME MainPredictRegressionKokkos/executable_minimal
         COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} -b kokkos "${CMAKE_CURRENT_LIST_DIR}/../../data/libsvm/regression/5x4.libsvm" # test file
                 "${CMAKE_CURRENT_LIST_DIR}/../../data/model/regression/5x4.libsvm.model" # model file
                 "${CMAKE_CURRENT_BINARY_DIR}/test_regression.libsvm.predict" # predict file (result)
)

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_KOKKOS_TEST_NAME})
endif ()
