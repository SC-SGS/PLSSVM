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
  main_tests
  test.cpp
  backends/compare.cpp
  ${PROJECT_SOURCE_DIR}/src/plssvm/IO.cpp
  ${PROJECT_SOURCE_DIR}/src/plssvm/CSVM.cpp) # TODO: link against
                                                           # svm
# as library
target_link_libraries(main_tests gtest_main)
target_link_libraries(main_tests gmock)
# target_include_directories(main_tests PUBLIC ${GTEST_INCLUDE_DIRS}
#                                              ${GMOCK_INCLUDE_DIRS})
target_include_directories(main_tests PUBLIC ${PROJECT_SOURCE_DIR}/include/)
target_compile_definitions(main_tests
                           PRIVATE TESTPATH="${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(main_tests PUBLIC ${fmt_SOURCE_DIR}/include) # TODO: better?
target_link_libraries(main_tests fmt::fmt)
target_include_directories(main_tests PUBLIC ${fast_float_SOURCE_DIR}/include)



include(GoogleTest)
gtest_discover_tests(main_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME Parameter COMMAND svm-train --help)

add_subdirectory(backends)