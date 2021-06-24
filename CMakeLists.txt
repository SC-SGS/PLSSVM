cmake_minimum_required(VERSION 3.14.0)
set(CMAKE_CXX_STANDARD 17) 

option(WITH_CUDA "Compile CUDA svm" OFF)
option(WITH_OPENCL "Compile OPENCL svm" OFF)

set(CMAKE_MAKEFILE_DEPENDS "CMakeLists.txt")

if(WITH_CUDA)
  project (SupportVectorMachine LANGUAGES CXX CUDA)
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O3 -lineinfo -Xptxas=-v -Xptxas -dlcm=cg")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14 -gencode arch=compute_60,code=sm_60 -Xcompiler '-fopenmp' --use_fast_math")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -gencode arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86  -Xcompiler '-fopenmp' --use_fast_math")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11  -Xcompiler '-fopenmp' --use_fast_math")
  if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath, 
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET particle_test 
                PROPERTY
                BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif(APPLE)
else(WITH_CUDA)
  project (SupportVectorMachine LANGUAGES CXX)
endif(WITH_CUDA)

if(WITH_OPENCL)
  find_package(OpenCL REQUIRED)
  include_directories( ${OpenCL_INCLUDE_DIRS} )
endif(WITH_OPENCL)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include(FetchContent)
FetchContent_Declare(cxxopts
    # Compared to release tag or branch, commit hash avoids network connection
    # if the local clone already has that particular commit.
    GIT_REPOSITORY          https://github.com/jarro2783/cxxopts.git
    GIT_TAG                 df229cff0d5b96e146f3f11441f714e8e240cad0
    GIT_SHALLOW             TRUE

set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
)
FetchContent_MakeAvailable(cxxopts)

include_directories(include)

set(SOURCES
  src/CSVM.cpp
  src/IO.cpp
  src/CPU/CPU_CSVM.cpp
  )

if(WITH_CUDA)
  set(SOURCES_CUDA 
    src/IO.cpp
    src/CUDA/CUDA_CSVM.cu
    src/CUDA/svm-kernel.cu
    src/CUDA/cuda-kernel.cu
    src/OpenCL/cuda-kernel.cpp #TODO:
    src/CUDA/predict.cu
    src/CSVM.cpp
    )
endif(WITH_CUDA)

if(WITH_OPENCL)
  set(SOURCES_OCL
    src/IO.cpp
    src/OpenCL/OCL_CSVM.cpp
    src/OpenCL/cuda-kernel.cpp
    src/OpenCL/kernels/svm-kernel-linear.cl
    src/OpenCL/kernels/kernel_q.cl
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
    src/CSVM.cpp
    )
endif(WITH_OPENCL)


if(WITH_OPENCL)
  add_library(svm-OCL SHARED ${SOURCES_OCL})
  target_compile_definitions(svm-OCL PUBLIC WITH_OPENCL)
  target_compile_options(svm-OCL PUBLIC) # -march=native -mtune=native
  target_link_libraries(svm-OCL OpenCL)
  add_executable(svm-train-OCL src/OCL_small.cpp)
  target_compile_options(svm-train-OCL PUBLIC) # -march=native -mtune=native
  target_link_libraries(svm-train-OCL svm-OCL)
  # enable Link Time Optimization (LTO)
 include(CheckIPOSupported)
 check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
 if(lto_supported)
   message(STATUS "IPO / LTO enabled for svm-train-OCL")
   set_property(TARGET svm-train-OCL PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
 else()
   message(STATUS "IPO / LTO not supported: <${lto_error}>")
 endif()
endif(WITH_OPENCL)

if(WITH_CUDA)
  add_library(svm-CUDA SHARED ${SOURCES_CUDA})
  target_compile_definitions(svm-CUDA PUBLIC WITH_CUDA)
  add_executable(svm-train-CUDA src/CUDA_small.cpp)
  target_compile_options(svm-train-CUDA PUBLIC) # -march=native -mtune=native
  target_link_libraries(svm-train-CUDA svm-CUDA)
  # enable Link Time Optimization (LTO)
 include(CheckIPOSupported)
 check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
 if(lto_supported)
   message(STATUS "IPO / LTO enabled for svm-CUDA")
   set_property(TARGET svm-CUDA PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
 else()
   message(STATUS "IPO / LTO not supported: <${lto_error}>")
 endif()
endif(WITH_CUDA)

add_library(svm SHARED ${SOURCES})
# enable Link Time Optimization (LTO)
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
if(lto_supported)
  message(STATUS "IPO / LTO enabled for svm")
  set_property(TARGET svm PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
  message(STATUS "IPO / LTO not supported: <${lto_error}>")
endif()
target_compile_options(svm PUBLIC) # -march=native -mtune=native
add_executable(svm-train-cpu src/main.cpp)
# enable Link Time Optimization (LTO)
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
if(lto_supported)
  message(STATUS "IPO / LTO enabled for svm-train-cpu")
  set_property(TARGET svm-train-cpu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
  message(STATUS "IPO / LTO not supported: <${lto_error}>")
endif()
target_compile_options(svm-train-cpu PUBLIC) # -march=native -mtune=native
target_link_libraries(svm-train-cpu svm)
target_include_directories(svm-train-cpu PUBLIC ${cxxopts_SOURCE_DIR}/include)
add_dependencies(svm-train-cpu cxxopts)





