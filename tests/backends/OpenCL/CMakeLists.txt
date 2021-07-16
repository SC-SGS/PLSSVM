set(PLSSVM_OPENCL_TEST_NAME OpenCL_tests)

add_executable(
  ${PLSSVM_OPENCL_TEST_NAME}
  test.cpp
  ${PROJECT_SOURCE_DIR}/src/plssvm/IO.cpp
  ${PROJECT_SOURCE_DIR}/src/plssvm/CSVM.cpp
  ${PROJECT_SOURCE_DIR}/src/plssvm/parameter.cpp
  ../compare.cpp
  ) # TODO: link against svm as library

target_include_directories(${PLSSVM_OPENCL_TEST_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PLSSVM_OPENCL_TEST_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_include_directories(${PLSSVM_OPENCL_TEST_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src/plssvm/backends/OpenCL)
target_include_directories(${PLSSVM_OPENCL_TEST_NAME} PUBLIC ${fmt_SOURCE_DIR}/include)
target_include_directories(${PLSSVM_OPENCL_TEST_NAME} PUBLIC ${fast_float_SOURCE_DIR}/include)
target_link_libraries(${PLSSVM_OPENCL_TEST_NAME} PUBLIC fmt::fmt)

target_link_libraries(${PLSSVM_OPENCL_TEST_NAME} PUBLIC gtest_main)
target_link_libraries(${PLSSVM_OPENCL_TEST_NAME} PUBLIC gmock)

target_compile_definitions(${PLSSVM_OPENCL_TEST_NAME} PRIVATE TESTPATH="${PROJECT_SOURCE_DIR}/tests")
target_compile_definitions(${PLSSVM_OPENCL_TEST_NAME} PRIVATE TESTFILE="${TESTFILE}")


get_directory_property(PLSSVM_OpenCL_BACKEND_LIBRARY_NAME DIRECTORY ${PROJECT_SOURCE_DIR}/src/plssvm/backends/OpenCL DEFINITION PLSSVM_OpenCL_BACKEND_LIBRARY_NAME)
target_link_libraries(${PLSSVM_OPENCL_TEST_NAME} PUBLIC ${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME})

include(GoogleTest)
gtest_discover_tests(${PLSSVM_OPENCL_TEST_NAME} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)