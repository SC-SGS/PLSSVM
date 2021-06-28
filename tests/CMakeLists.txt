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
             ${PROJECT_SOURCE_DIR}/src/plssvm/CSVM.cpp) # TODO: link against svm
                                                        # as library
target_link_libraries(hello_test gtest_main)
target_link_libraries(hello_test gmock)
target_include_directories(hello_test PUBLIC ${GTEST_INCLUDE_DIRS}
                                             ${GMOCK_INCLUDE_DIRS})
target_include_directories(hello_test PUBLIC ${PROJECT_SOURCE_DIR}/include/)

include(GoogleTest)
gtest_discover_tests(hello_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME Parameter COMMAND svm-train --help)
