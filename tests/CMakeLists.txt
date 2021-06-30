include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
  hello_test test.cpp ${PROJECT_SOURCE_DIR}/src/plssvm/IO.cpp
             ${PROJECT_SOURCE_DIR}/src/plssvm/CSVM.cpp) # TODO: link against
# svm as library
target_link_libraries(hello_test gtest_main)
target_link_libraries(hello_test gmock)
target_include_directories(hello_test PUBLIC ${GTEST_INCLUDE_DIRS}
                                             ${GMOCK_INCLUDE_DIRS})
target_include_directories(hello_test PUBLIC ${PROJECT_SOURCE_DIR}/include/)
target_compile_definitions(hello_test
                           PRIVATE TESTPATH="${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(hello_test PUBLIC ${fmt_SOURCE_DIR}/include)# TODO: better?
target_link_libraries(hello_test fmt::fmt)

if(TARGET svm-CPU)
  target_link_libraries(hello_test svm-CPU)
  target_compile_definitions(hello_test PUBLIC PLSSVM_HAS_OPENMP_BACKEND)
endif()
if(TARGET svm-OCL)
  target_link_libraries(hello_test svm-OCL)
  target_compile_definitions(hello_test PUBLIC PLSSVM_HAS_OPENCL_BACKEND)

endif()
if(TARGET svm-CUDA)
  target_link_libraries(hello_test svm-CUDA)
  target_compile_definitions(hello_test PUBLIC PLSSVM_HAS_CUDA_BACKEND)
endif()

include(GoogleTest)
gtest_discover_tests(hello_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME Parameter COMMAND svm-train --help)
