set(PLSSVM_CUDA_TEST_NAME CUDA_tests)

add_executable(
  ${PLSSVM_CUDA_TEST_NAME}
  test.cpp
  ${PROJECT_SOURCE_DIR}/src/plssvm/IO.cpp
  ${PROJECT_SOURCE_DIR}/src/plssvm/CSVM.cpp
  ../compare.cpp
  ) # TODO: link against svm as library

target_include_directories(${PLSSVM_CUDA_TEST_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PLSSVM_CUDA_TEST_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)

target_link_libraries(${PLSSVM_CUDA_TEST_NAME} PUBLIC gtest_main)
target_link_libraries(${PLSSVM_CUDA_TEST_NAME} PUBLIC gmock)

target_compile_definitions(${PLSSVM_CUDA_TEST_NAME} PRIVATE TESTPATH="${PROJECT_SOURCE_DIR}/tests")
target_compile_definitions(${PLSSVM_CUDA_TEST_NAME} PRIVATE TESTFILE="${TESTFILE}")



get_directory_property(PLSSVM_CUDA_BACKEND_LIBRARY_NAME DIRECTORY ${PROJECT_SOURCE_DIR}/src/plssvm/backends/CUDA DEFINITION PLSSVM_CUDA_BACKEND_LIBRARY_NAME)
target_link_libraries(${PLSSVM_CUDA_TEST_NAME} PUBLIC ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME})

include(GoogleTest)
gtest_discover_tests(${PLSSVM_CUDA_TEST_NAME} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)