cmake_minimum_required(VERSION 3.18)

project (SupportVectorMachine LANGUAGES CXX)

include(FetchContent)

## fetch command line parser library cxxopts
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

## fetch string formatting library fmt
set(CMAKE_CXX_FLAGS_OLD ${CMAKE_CXX_FLAGS}) # TODO: better?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
FetchContent_Declare(fmt
  GIT_REPOSITORY          https://github.com/fmtlib/fmt.git
  GIT_TAG                 9e8b86fd2d9806672cc73133d21780dd182bfd24
  GIT_SHALLOW             TRUE

  set(FMT_TEST OFF CACHE BOOL "" FORCE)
  set(FMT_DOC OFF CACHE BOOL "" FORCE)
  set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
)
FetchContent_MakeAvailable(fmt)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_OLD})


## set C++ standard globally
set(CMAKE_CXX_STANDARD 17)
## set include directory globally
include_directories(include)


## check for OpenMP (not for the backend!) TODO: better?
find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-fopenmp'")
endif()


## set main sources
set(PLSSVM_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/src/plssvm/CSVM.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/plssvm/IO.cpp
)


## create main executable
set(PLSSVM_EXECUTABLE_NAME svm-train)
add_executable(${PLSSVM_EXECUTABLE_NAME} ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp)
add_dependencies(${PLSSVM_EXECUTABLE_NAME} cxxopts)
target_include_directories(${PLSSVM_EXECUTABLE_NAME} PUBLIC ${cxxopts_SOURCE_DIR}/include)
#target_include_directories(${PLSSVM_EXECUTABLE_NAME} PUBLIC ${fmt_SOURCE_DIR}/include) # TODO: better?
#target_link_libraries(${PLSSVM_EXECUTABLE_NAME} fmt::fmt)


## check for OpenMP backend
set(PLSVM_ENABLE_OpenMP_BACKEND AUTO CACHE STRING "Enable OpenMP Backend")
set_property(CACHE PLSVM_ENABLE_OpenMP_BACKEND PROPERTY STRINGS AUTO ON OFF)
if(PLSVM_ENABLE_OpenMP_BACKEND MATCHES "AUTO" OR PLSVM_ENABLE_OpenMP_BACKEND)
  add_subdirectory(src/plssvm/OpenMP)
endif()

## check for CUDA backend
set(PLSSVM_ENABLE_CUDA_BACKEND AUTO CACHE STRING "Enable CUDA Backend")
set_property(CACHE PLSSVM_ENABLE_CUDA_BACKEND PROPERTY STRINGS AUTO ON OFF)
if(PLSSVM_ENABLE_CUDA_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_CUDA_BACKEND)
  add_subdirectory(src/plssvm/CUDA)
endif()

## check for OpenCL backend
set(PLSSVM_ENABLE_OPENCL_BACKEND AUTO CACHE STRING "Enable OpenCL Backend")
set_property(CACHE PLSSVM_ENABLE_OPENCL_BACKEND PROPERTY STRINGS AUTO ON OFF)
if(PLSSVM_ENABLE_OPENCL_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_OPENCL_BACKEND)
  add_subdirectory(src/plssvm/OpenCL)
endif()


## enable Link Time Optimization (LTO)
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
if(lto_supported)
  message(STATUS "IPO / LTO enabled.")
  set_property(TARGET ${PLSSVM_EXECUTABLE_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
  message(STATUS "IPO / LTO not supported: <${lto_error}>")
endif()

enable_testing()
add_subdirectory(tests)