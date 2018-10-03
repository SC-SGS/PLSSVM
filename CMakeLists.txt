cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_CXX_STANDARD 14) 

option(WITH_CUDA "Compile CUDA svm" ON)
option(WITH_OPENCL "Compile CUDA svm" ON)

set(CMAKE_MAKEFILE_DEPENDS "CMakeLists.txt")

if(WITH_CUDA)
  project (SupportVectorMachine LANGUAGES CXX CUDA)
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -lineinfo -Xptxas=-v -Xptxas -dlcm=cg")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 -gencode arch=compute_60,code=sm_60 -Xcompiler '-fopenmp' --use_fast_math")
  add_compile_definitions(WITH_CUDA)
else(WITH_CUDA)
  project (SupportVectorMachine LANGUAGES CXX)
endif(WITH_CUDA)

if(WITH_OPENCL)
  add_compile_definitions(WITH_OPENCL)
endif(WITH_OPENCL)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


include_directories(include)


set(SOURCES
  src/operators.cpp 
  src/IO.cpp
  )

if(WITH_CUDA)
  set(SOURCES_CUDA 
    src/CUDA/CSVM.cu
    src/CUDA/svm-kernel.cu
    src/CUDA/cuda-kernel.cu
    src/CUDA/predict.cu
    )
  set(SOURCES ${SOURCES} ${SOURCES_CUDA})
endif(WITH_CUDA)

if(WITH_OPENCL)
  set(SOURCES_OCL
    src/OpenCL/manager/manager.cpp
    src/OpenCL/manager/error_codes.cpp
    src/OpenCL/manager/run_kernel.cpp
    src/OpenCL/manager/configuration.cpp
    src/OpenCL/manager/platform_wrapper.cpp
    src/OpenCL/manager/json/dict_node.cpp
    src/OpenCL/manager/json/id_node.cpp
    src/OpenCL/manager/json/json.cpp
    src/OpenCL/manager/json/json_exception.cpp
    src/OpenCL/manager/json/list_node.cpp
    src/OpenCL/manager/json/node.cpp
    src/OpenCL/manager/json/text_node.cpp
    )
  set(SOURCES ${SOURCES} ${SOURCES_OCL})
endif(WITH_OPENCL)

if(!WITH_CUDA AND !WITH_OPENCL)
  set(SOURCES_CPU
    src/CPU/CSVM.cpp
    )
  set(SOURCES ${SOURCES} ${SOURCES_CPU})
endif(!WITH_CUDA AND !WITH_OPENCL)



add_library(svm SHARED ${SOURCES})
target_compile_options(svm PUBLIC) # -march=native -mtune=native
if(WITH_OPENCL)
  target_link_libraries(svm OpenCL)
endif(WITH_OPENCL)

if(WITH_CUDA)
  add_executable(svm-train-gpu src/main.cpp)
  target_compile_options(svm-train-gpu PUBLIC) # -march=native -mtune=native
  target_link_libraries(svm-train-gpu svm)
endif(WITH_CUDA)
add_executable(svm-train-cpu src/main.cpp)
target_compile_options(svm-train-cpu PUBLIC) # -march=native -mtune=native
target_link_libraries(svm-train-cpu svm)

#set_target_properties(prog PROPERTIES CUDA_SEPARABLE_COMPILATION ON)



#set(CXX nvcc)

 
# find_package(OpenMP)
# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O3 -lineinfo -Xptxas=-v -Xptxas -dlcm=cg")
# SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -gencode arch=compute_60,code=sm_60 -Xcompiler '-fopenmp'")


# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
# include_directories(include)


# set(SOURCES_SVM_GPU main.cpp src/CSVM.cu src/operators.cpp src/svm-kernel.cu src/cuda-kernel.cu src/IO.cpp src/predict.cu)
# set(SOURCES_SVM_CPU main.cpp src/CSVM.cpp src/operators.cpp src/IO.cpp)

# add_executable(svm-train-gpu ${SOURCES_SVM_GPU})
# add_executable(svm-train-cpu ${SOURCES_SVM_CPU})