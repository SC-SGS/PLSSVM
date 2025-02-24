cmake_minimum_required(VERSION 3.16)

project(LibraryUsageExamples LANGUAGES CXX)

find_package(plssvm CONFIG REQUIRED)

# classification executable example
add_executable(classification main_classification.cpp)
# classification executable example using MPI
add_executable(classification_mpi main_classification_mpi.cpp)
# regression executable example
add_executable(regression main_regression.cpp)
# regression executable example using MPI
add_executable(regression_mpi main_regression_mpi.cpp)

# link PLSSVM against executables
foreach (target classification classification_mpi regression regression_mpi)
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_link_libraries(${target} PUBLIC plssvm::plssvm-all)
endforeach ()
