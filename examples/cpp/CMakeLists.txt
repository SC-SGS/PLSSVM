cmake_minimum_required(VERSION 3.16)

project(LibraryUsageExample LANGUAGES CXX)

find_package(plssvm CONFIG REQUIRED)

# classification executable example
add_executable(classification main_classification.cpp)
target_compile_features(classification PUBLIC cxx_std_17)
target_link_libraries(classification PUBLIC plssvm::plssvm-all)

# regression executable example
add_executable(regression main_regression.cpp)
target_compile_features(regression PUBLIC cxx_std_17)
target_link_libraries(regression PUBLIC plssvm::plssvm-all)