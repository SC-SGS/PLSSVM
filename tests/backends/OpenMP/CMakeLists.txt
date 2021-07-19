set(PLSSVM_OpenMP_TEST_NAME OpenMP_tests)

add_executable(
  ${PLSSVM_OpenMP_TEST_NAME}
  test.cpp
  ../compare.cpp
  ) # TODO: link against svm as library

target_include_directories(${PLSSVM_OpenMP_TEST_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PLSSVM_OpenMP_TEST_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)

target_link_libraries(${PLSSVM_OpenMP_TEST_NAME} PUBLIC gtest_main)
target_link_libraries(${PLSSVM_OpenMP_TEST_NAME} PUBLIC gmock)

target_compile_definitions(${PLSSVM_OpenMP_TEST_NAME} PRIVATE TESTPATH="${PROJECT_SOURCE_DIR}/tests")
target_compile_definitions(${PLSSVM_OpenMP_TEST_NAME} PRIVATE TESTFILE="${TESTFILE}")



get_directory_property(PLSSVM_OPENMP_BACKEND_LIBRARY_NAME DIRECTORY ${PROJECT_SOURCE_DIR}/src/plssvm/backends/OpenMP DEFINITION PLSSVM_OPENMP_BACKEND_LIBRARY_NAME)
target_link_libraries(${PLSSVM_OpenMP_TEST_NAME} PUBLIC ${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME})

include(GoogleTest)
gtest_discover_tests(${PLSSVM_OpenMP_TEST_NAME} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)