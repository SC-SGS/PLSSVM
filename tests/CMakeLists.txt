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
  test.cpp backends/compare.cpp ${PROJECT_SOURCE_DIR}/src/plssvm/IO.cpp
  ${PROJECT_SOURCE_DIR}/src/plssvm/CSVM.cpp) # TODO: link against
# svm as library
target_link_libraries(main_tests gtest_main)
target_link_libraries(main_tests gmock)
# target_include_directories(main_tests PUBLIC ${GTEST_INCLUDE_DIRS}
# ${GMOCK_INCLUDE_DIRS})
target_include_directories(main_tests PUBLIC ${PROJECT_SOURCE_DIR}/include/)
target_compile_definitions(main_tests
                           PRIVATE TESTPATH="${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(main_tests PUBLIC ${fmt_SOURCE_DIR}/include
)# TODO: better?
target_link_libraries(main_tests fmt::fmt)
target_include_directories(main_tests PUBLIC ${fast_float_SOURCE_DIR}/include)

option(GENERATE_TEST_FILE "Generate larger test file" ON)
if(${GENERATE_TEST_FILE})
  set(NUMBER_DATAPOINTS
      5000
      CACHE STRING "Number of data points for the test file generator")
  set(NUMBER_FEATURES
      2000
      CACHE STRING
            "Number of features for each data point for the test file generator"
  )
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

  foreach(
    lib IN
    ITEMS argparse
          numpy
          pandas
          sklearn
          arff
          matplotlib
          mpl_toolkits)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import ${lib}"
      RESULT_VARIABLE EXIT_CODE
      OUTPUT_QUIET)
    message(STATUS "Searching for ${lib} Python3 package")
    if(NOT ${EXIT_CODE} EQUAL 0)
      message(
        FATAL_ERROR
          "The \"${lib}\" Python3 package is not installed. Please install it using the following command: \"${Python3_EXECUTABLE} -m pip install ${lib}\"\n or set GENERATE_TEST_FILE=OFF ."
      )
    endif()
  endforeach(lib in LISTS ${PYTHON_LIBS})

  set(TESTFILE
      "${PROJECT_SOURCE_DIR}/tests/data/${NUMBER_DATAPOINTS}x${NUMBER_FEATURES}.libsvm"
      CACHE STRING "Path including name of the test file")
  if(EXISTS "${TESTFILE}")
    message(WARNING "Skipped test file generation it already exists!")
  else()
    message(STATUS "Generating test file")
    execute_process(
      COMMAND
        ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/data/generate_data.py"
        "--output" "${TESTFILE}" "--format" "libsvm" "--samples"
        "${NUMBER_DATAPOINTS}" "--features" "${NUMBER_FEATURES}"
      OUTPUT_QUIET)
  endif()
  else()
  set(TESTFILE
      "${PROJECT_SOURCE_DIR}/tests/data/500x200.libsvm"
      CACHE STRING "Path including name of the test file")

endif()

include(GoogleTest)
gtest_discover_tests(main_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME Parameter COMMAND svm-train --help)

add_subdirectory(backends)
