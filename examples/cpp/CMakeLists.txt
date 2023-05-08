cmake_minimum_required(VERSION 3.16)

project(LibraryUsageExample LANGUAGES CXX)

find_package(plssvm CONFIG REQUIRED)

add_executable(prog main.cpp)

target_compile_features(prog PUBLIC cxx_std_17)
target_link_libraries(prog PUBLIC plssvm::plssvm-all)