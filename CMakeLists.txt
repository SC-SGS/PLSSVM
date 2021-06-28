cmake_minimum_required(VERSION 3.14.0)
set(CMAKE_CXX_STANDARD 17)

set(ENABLE_CPU_BACKEND AUTO CACHE STRING "Enable CPU Backend")
set_property(CACHE ENABLE_CPU_BACKEND PROPERTY STRINGS AUTO ON OFF)
set(ENABLE_CUDA_BACKEND AUTO CACHE STRING "Enable CUDA Backend")
set_property(CACHE ENABLE_CUDA_BACKEND PROPERTY STRINGS AUTO ON OFF)
set(ENABLE_OPENCL_BACKEND AUTO CACHE STRING "Enable OpenCL Backend")
set_property(CACHE ENABLE_OPENCL_BACKEND PROPERTY STRINGS AUTO ON OFF)

project (SupportVectorMachine LANGUAGES CXX)

if(ENABLE_CPU_BACKEND STREQUAL "AUTO")
   set(ENABLE_CPU_BACKEND ON CACHE STRING "Enable CPU Backend" FORCE)
endif()

include(CheckLanguage)
if(ENABLE_CUDA_BACKEND STREQUAL "AUTO")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(ENABLE_CUDA_BACKEND ON CACHE STRING "Enable CUDA Backend" FORCE)
  else()
    set(ENABLE_CUDA_BACKEND OFF CACHE STRING "Enable CUDA Backend" FORCE)
    message(STATUS "No CUDA backend support")
  endif()
elseif(ENABLE_CUDA_BACKEND)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "Can not find CUDA")
  endif()
endif()



if(ENABLE_OPENCL_BACKEND STREQUAL "AUTO")
  find_package(OpenCL)
  if(OpenCL_FOUND)
    set(ENABLE_OPENCL_BACKEND ON CACHE STRING "Enable OpenCL Backend" FORCE)
  else()
    set(ENABLE_OPENCL_BACKEND OFF CACHE STRING "Enable OpenCL Backend" FORCE)
  endif()
elseif(ENABLE_OPENCL_BACKEND)
find_package(OpenCL)
  if(OpenCL_FOUND)

  else()
    message(FATAL_ERROR "Can not find OPenCL")
  endif()
endif()



if(ENABLE_CUDA_BACKEND)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O3 -lineinfo -Xptxas=-v -Xptxas -dlcm=cg" )
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
endif(ENABLE_CUDA_BACKEND)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-fopenmp' ")
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

set(CSVM_SOURCES
        src/plssvm/CSVM.cpp
        src/plssvm/IO.cpp
  )

set(CPU_SOURCES
        src/plssvm/OpenMP/CPU_CSVM.cpp)


set(CUDA_SOURCES
        src/plssvm/CUDA/CUDA_CSVM.cu
        src/plssvm/CUDA/svm-kernel.cu
        src/plssvm/CUDA/cuda-kernel.cu
        src/plssvm/OpenCL/cuda-kernel.cpp #TODO:
        src/plssvm/CUDA/predict.cu
    )


set(OCL_SOURCES
        src/plssvm/IO.cpp
        src/plssvm/OpenCL/OCL_CSVM.cpp
        src/plssvm/OpenCL/cuda-kernel.cpp
        src/plssvm/OpenCL/kernels/svm-kernel-linear.cl
        src/plssvm/OpenCL/kernels/kernel_q.cl
        src/plssvm/OpenCL/manager/manager.cpp
        src/plssvm/OpenCL/manager/error_codes.cpp
        src/plssvm/OpenCL/manager/run_kernel.cpp
        src/plssvm/OpenCL/manager/configuration.cpp
        src/plssvm/OpenCL/manager/platform_wrapper.cpp
        src/plssvm/OpenCL/manager/json/dict_node.cpp
        src/plssvm/OpenCL/manager/json/id_node.cpp
        src/plssvm/OpenCL/manager/json/json.cpp
        src/plssvm/OpenCL/manager/json/json_exception.cpp
        src/plssvm/OpenCL/manager/json/list_node.cpp
        src/plssvm/OpenCL/manager/json/node.cpp
        src/plssvm/OpenCL/manager/json/text_node.cpp
        src/plssvm/CSVM.cpp
  )

set(EXECUTABLE_NAME svm-train)
add_executable(${EXECUTABLE_NAME} src/main.cpp)
add_dependencies(${EXECUTABLE_NAME} cxxopts)
target_include_directories(${EXECUTABLE_NAME} PUBLIC ${cxxopts_SOURCE_DIR}/include)

if(ENABLE_OPENCL_BACKEND)
  set(OPENCL_BACKEND_LIBRARY_NAME svm-OCL)
  add_library(${OPENCL_BACKEND_LIBRARY_NAME} SHARED ${CSVM_SOURCES} ${OCL_SOURCES})
  target_include_directories(${OPENCL_BACKEND_LIBRARY_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} )
  target_compile_definitions(${OPENCL_BACKEND_LIBRARY_NAME} PUBLIC WITH_OPENCL)
  target_link_libraries(${OPENCL_BACKEND_LIBRARY_NAME} OpenCL)
  target_compile_definitions(${EXECUTABLE_NAME} PUBLIC HAS_OPENCL_BACKEND)
  target_link_libraries(${EXECUTABLE_NAME} ${OPENCL_BACKEND_LIBRARY_NAME})
endif(ENABLE_OPENCL_BACKEND)

if(ENABLE_CUDA_BACKEND)
  set(CUDA_BACKEND_LIBRARY_NAME svm-CUDA)
  add_library(${CUDA_BACKEND_LIBRARY_NAME} SHARED ${CSVM_SOURCES} ${CUDA_SOURCES})
  set_property(TARGET ${CUDA_BACKEND_LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES 72 80 86 )
  target_compile_definitions(${CUDA_BACKEND_LIBRARY_NAME} PUBLIC WITH_CUDA)
  target_compile_definitions(${EXECUTABLE_NAME} PUBLIC HAS_CUDA_BACKEND)
  target_link_libraries(${EXECUTABLE_NAME} ${CUDA_BACKEND_LIBRARY_NAME})
endif(ENABLE_CUDA_BACKEND)

if(ENABLE_CPU_BACKEND)
  set(CPU_BACKEND_LIBRARY_NAME svm-CPU)
  add_library(${CPU_BACKEND_LIBRARY_NAME} SHARED ${CSVM_SOURCES} ${CPU_SOURCES})
  target_compile_definitions(${EXECUTABLE_NAME} PUBLIC HAS_CPU_BACKEND)
  target_link_libraries(${EXECUTABLE_NAME} ${CPU_BACKEND_LIBRARY_NAME})
endif(ENABLE_CPU_BACKEND)


# enable Link Time Optimization (LTO)
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
if(lto_supported)
  message(STATUS "IPO / LTO enabled for svm-train")
  set_property(TARGET ${EXECUTABLE_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
  message(STATUS "IPO / LTO not supported: <${lto_error}>")
endif()