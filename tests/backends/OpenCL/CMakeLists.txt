## create OpenCL tests
set(PLSSVM_OPENCL_TEST_NAME OpenCL_tests)

# list all necessary sources
set(PLSSVM_OPENCL_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../compare.cpp
)

# add test executable
add_executable(${PLSSVM_OPENCL_TEST_NAME} ${PLSSVM_OPENCL_TEST_SOURCES})

# set necessary include directories
target_include_directories(${PLSSVM_OPENCL_TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# add necessary compile definitions
target_compile_definitions(${PLSSVM_OPENCL_TEST_NAME} PRIVATE TEST_PATH="${PROJECT_SOURCE_DIR}/tests")
target_compile_definitions(${PLSSVM_OPENCL_TEST_NAME} PRIVATE TEST_FILE="${PLSSVM_TEST_FILE}")

# link against OpenCL backend
target_link_libraries(${PLSSVM_OPENCL_TEST_NAME} PRIVATE ${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME})


include(GoogleTest)

# link against gtest
target_link_libraries(${PLSSVM_OPENCL_TEST_NAME} PRIVATE GTest::gtest_main GTest::gmock)

# add tests
gtest_add_tests(TARGET ${PLSSVM_OPENCL_TEST_NAME} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)

# add to coverage as dependencie
if(TARGET gcov)
    add_dependencies(gcov ${PLSSVM_OPENCL_TEST_NAME})
endif()