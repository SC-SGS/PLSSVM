# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# create HIP tests
set(PLSSVM_HIP_TEST_NAME HIP_tests)

if (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
    # NOTE: not sure why that should be necessary!
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ftemplate-depth=2048")
endif ()

include(CheckLanguage)
check_language(${PLSSVM_HIP_BACKEND_GPU_RUNTIME})
enable_language(${PLSSVM_HIP_BACKEND_GPU_RUNTIME})
find_package(HIP REQUIRED)

# list all necessary sources
set(PLSSVM_HIP_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.hip
    ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.hip
    ${CMAKE_CURRENT_LIST_DIR}/kernel/detail/fill_kernel.hip
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.hip
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.hip
    ${CMAKE_CURRENT_LIST_DIR}/hip_csvm.hip
)

# set source file language
set_source_files_properties(${PLSSVM_HIP_TEST_SOURCES} PROPERTIES LANGUAGE ${PLSSVM_HIP_BACKEND_GPU_RUNTIME})

# set target properties
set_local_and_parent(PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME plssvm-HIP_test)
add_library(${PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME} SHARED ${PLSSVM_HIP_TEST_SOURCES})

# set target architecture
if (DEFINED PLSSVM_AMD_TARGET_ARCHS)
    set_property(TARGET ${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTY HIP_ARCHITECTURES ${PLSSVM_AMD_TARGET_ARCHS})
elseif (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
    # prepare architectures for CMake's CUDA_ARCHITECTURES
    list(TRANSFORM PLSSVM_NVIDIA_TARGET_ARCHS REPLACE "sm_" "")
    set_property(TARGET ${PLSSVM_HIP_BACKEND_LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES ${PLSSVM_NVIDIA_TARGET_ARCHS})
endif ()

# link base library against HIP library
target_link_libraries(${PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME} PRIVATE hip::device hip::host)
set_target_properties(${PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE ${PLSSVM_HIP_BACKEND_GPU_RUNTIME})

target_link_libraries(${PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_TEST_LIBRARY_NAME})

# add test executable
add_executable(${PLSSVM_HIP_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../main.cpp)
target_link_libraries(${PLSSVM_HIP_TEST_NAME} PRIVATE ${PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME})

# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_HIP_TEST_NAME})

# add minimal run test for the classification task with the HIP backend
add_test(NAME MainTrainClassificationHIP/executable_minimal COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} -s 0 -b hip "${PLSSVM_CLASSIFICATION_TEST_FILE}"
                                                                    "${CMAKE_CURRENT_BINARY_DIR}/test_classification.libsvm.model"
)
add_test(NAME MainPredictClassificationHIP/executable_minimal
         COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} -b hip "${CMAKE_CURRENT_LIST_DIR}/../../data/libsvm/classification/5x4.libsvm" # test file
                 "${CMAKE_CURRENT_LIST_DIR}/../../data/model/classification/5x4.libsvm.model" # model file
                 "${CMAKE_CURRENT_BINARY_DIR}/test_classification.libsvm.predict" # predict file (result)
)

# add minimal run test for the regression task with the HIP backend
add_test(NAME MainTrainRegressionHIP/executable_minimal COMMAND ${PLSSVM_EXECUTABLE_TRAIN_NAME} -s 1 -b hip "${PLSSVM_REGRESSION_TEST_FILE}"
                                                                "${CMAKE_CURRENT_BINARY_DIR}/test_regression.libsvm.model"
)
add_test(NAME MainPredictRegressionHIP/executable_minimal
         COMMAND ${PLSSVM_EXECUTABLE_PREDICT_NAME} -b hip "${CMAKE_CURRENT_LIST_DIR}/../../data/libsvm/regression/5x4.libsvm" # test file
                 "${CMAKE_CURRENT_LIST_DIR}/../../data/model/regression/5x4.libsvm.model" # model file
                 "${CMAKE_CURRENT_BINARY_DIR}/test_regression.libsvm.predict" # predict file (result)
)

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_HIP_TEST_NAME})
endif ()
