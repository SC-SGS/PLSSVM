cmake_minimum_required(VERSION 3.10.0)
project(bachelor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)  
set(CMAKE_MAKEFILE_DEPENDS "CMakeLists.txt")
set(CXX nvcc)

 
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O3 -lineinfo -Xptxas=-v -Xptxas -dlcm=cg")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -gencode arch=compute_60,code=sm_60 -Xcompiler '-fopenmp'")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
include_directories(include)


set(SOURCES_SVM_GPU main.cpp src/CSVM.cu src/operators.cpp src/svm-kernel.cu src/cuda-kernel.cu src/IO.cpp src/predict.cu)
set(SOURCES_SVM_CPU main.cpp src/CSVM.cpp src/operators.cpp src/IO.cpp)

add_executable(svm-train-gpu ${SOURCES_SVM_GPU})
add_executable(svm-train-cpu ${SOURCES_SVM_CPU})