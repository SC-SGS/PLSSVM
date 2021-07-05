cmake_minimum_required(VERSION 3.18)

project (SupportVectorMachine
         VERSION 0.1.0
         LANGUAGES CXX
         DESCRIPTION "A Support Vector Machine implementation using different backends.")

## configure version file
set(CONFIG_IN ${CMAKE_CURRENT_SOURCE_DIR}/include/plssvm/version.hpp.in)
set(CONFIG_OUT ${CMAKE_CURRENT_SOURCE_DIR}/include/plssvm/version.hpp)
configure_file(${CONFIG_IN} ${CONFIG_OUT} @ONLY)

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

## fetch float parsing library fast_float
FetchContent_Declare(fast_float
  GIT_REPOSITORY          https://github.com/fastfloat/fast_float
  GIT_TAG                 8159e8bcf63c1b92f5a51fb550f966e56624b209
  GIT_SHALLOW             TRUE

  set(FASTFLOAT_TEST OFF CACHE BOOL "" FORCE)
  set(FASTFLOAT_SANITIZE OFF CACHE BOOL "" FORCE)
)
FetchContent_MakeAvailable(fast_float)


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
add_executable(${PLSSVM_EXECUTABLE_NAME} ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp ${PLSSVM_SOURCES})
add_dependencies(${PLSSVM_EXECUTABLE_NAME} cxxopts)
target_include_directories(${PLSSVM_EXECUTABLE_NAME} PUBLIC ${cxxopts_SOURCE_DIR}/include)
target_include_directories(${PLSSVM_EXECUTABLE_NAME} PUBLIC ${fast_float_SOURCE_DIR}/include)
#target_include_directories(${PLSSVM_EXECUTABLE_NAME} PUBLIC ${fmt_SOURCE_DIR}/include) # TODO: better?
#target_link_libraries(${PLSSVM_EXECUTABLE_NAME} fmt::fmt)


## check for OpenMP backend
set(PLSVM_ENABLE_OpenMP_BACKEND AUTO CACHE STRING "Enable OpenMP Backend")
set_property(CACHE PLSVM_ENABLE_OpenMP_BACKEND PROPERTY STRINGS AUTO ON OFF)
if(PLSVM_ENABLE_OpenMP_BACKEND MATCHES "AUTO" OR PLSVM_ENABLE_OpenMP_BACKEND)
  add_subdirectory(src/plssvm/backends/OpenMP)
endif()

## check for CUDA backend
set(PLSSVM_ENABLE_CUDA_BACKEND AUTO CACHE STRING "Enable CUDA Backend")
set_property(CACHE PLSSVM_ENABLE_CUDA_BACKEND PROPERTY STRINGS AUTO ON OFF)
if(PLSSVM_ENABLE_CUDA_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_CUDA_BACKEND)
  add_subdirectory(src/plssvm/backends/CUDA)
endif()

## check for OpenCL backend
set(PLSSVM_ENABLE_OPENCL_BACKEND AUTO CACHE STRING "Enable OpenCL Backend")
set_property(CACHE PLSSVM_ENABLE_OPENCL_BACKEND PROPERTY STRINGS AUTO ON OFF)
if(PLSSVM_ENABLE_OPENCL_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_OPENCL_BACKEND)
  add_subdirectory(src/plssvm/backends/OpenCL)
endif()


## enable Link Time Optimization (LTO)
option(PLSSVM_ENABLE_LTO "Enable Link Time Optimizations." ON) # TODO: default ON/OFF?
if(PLSSVM_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
  if(lto_supported)
    message(STATUS "IPO / LTO enabled.")
    set_property(TARGET ${PLSSVM_EXECUTABLE_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "IPO / LTO not supported: <${lto_error}>")
  endif()
endif()


## enable testing
option(PLSSVM_ENABLE_TESTING "Build tests for all backends." ON) # TODO: default ON/OFF?
if(PLSSVM_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()


## enable documentation generation with doxygen
option(PLSSVM_ENABLE_DOCU "Build documentation using doxygen." ON) # TODO: default ON/OFF?
if(PLSSVM_ENABLE_DOCU)
  add_subdirectory(docs)
endif()