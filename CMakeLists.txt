cmake_minimum_required(VERSION 3.18)

project(ParallelLeastSquaresSupportVectorMachine
        VERSION 0.1.0
        LANGUAGES CXX
        DESCRIPTION "A Support Vector Machine implementation using different backends.")


## configure version file
set(CONFIG_IN ${CMAKE_SOURCE_DIR}/include/plssvm/version/version.hpp.in)
set(CONFIG_OUT ${CMAKE_SOURCE_DIR}/include/plssvm/version/version.hpp)
configure_file(${CONFIG_IN} ${CONFIG_OUT} @ONLY)


## set base sources
set(PLSSVM_BASE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/plssvm/CSVM.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/plssvm/IO.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/plssvm/parameter.cpp
)

## create base library
set(PLSSVM_BASE_LIBRARY_NAME svm-base)
add_library(${PLSSVM_BASE_LIBRARY_NAME} STATIC ${PLSSVM_BASE_SOURCES})

# set library cxx standard and include directory
target_compile_features(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC cxx_std_17)
target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

# check for OpenMP (not for the backend!)
find_package(OpenMP)
if(OPENMP_FOUND)
    target_link_libraries(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-fopenmp'")
endif()

## setup dependencies
include(FetchContent)

# fetch command line parser library cxxopts
FetchContent_Declare(cxxopts
        GIT_REPOSITORY          https://github.com/jarro2783/cxxopts.git
        GIT_TAG                 df229cff0d5b96e146f3f11441f714e8e240cad0
        GIT_SHALLOW             TRUE

        set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        )
FetchContent_MakeAvailable(cxxopts)
add_dependencies(${PLSSVM_BASE_LIBRARY_NAME} cxxopts)
target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${cxxopts_SOURCE_DIR}/include)

# fetch float parsing library fast_float
FetchContent_Declare(fast_float
        GIT_REPOSITORY          https://github.com/fastfloat/fast_float
        GIT_TAG                 8159e8bcf63c1b92f5a51fb550f966e56624b209
        GIT_SHALLOW             TRUE

        set(FASTFLOAT_TEST OFF CACHE BOOL "" FORCE)
        set(FASTFLOAT_SANITIZE OFF CACHE BOOL "" FORCE)
        )
FetchContent_MakeAvailable(fast_float)
add_dependencies(${PLSSVM_BASE_LIBRARY_NAME} fast_float)
target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${fast_float_SOURCE_DIR}/include)

# fetch string formatting library fmt
FetchContent_Declare(fmt
        GIT_REPOSITORY          https://github.com/fmtlib/fmt.git
        GIT_TAG                 9e8b86fd2d9806672cc73133d21780dd182bfd24
        GIT_SHALLOW             TRUE

        set(FMT_TEST OFF CACHE BOOL "" FORCE)
        set(FMT_DOC OFF CACHE BOOL "" FORCE)
        set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
        )
FetchContent_MakeAvailable(fmt)
set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)
add_dependencies(${PLSSVM_BASE_LIBRARY_NAME} fmt)
set_property(TARGET ${PLSSVM_BASE_LIBRARY_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${fmt_SOURCE_DIR}/include)
target_link_libraries(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC fmt::fmt)


## create main executable
set(PLSSVM_EXECUTABLE_NAME svm-train)
add_executable(${PLSSVM_EXECUTABLE_NAME} ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp)


## check for OpenMP backend
set(PLSSVM_ENABLE_OPENMP_BACKEND AUTO CACHE STRING "Enable OpenMP Backend")
set_property(CACHE PLSSVM_ENABLE_OPENMP_BACKEND PROPERTY STRINGS AUTO ON OFF)
if(PLSSVM_ENABLE_OPENMP_BACKEND MATCHES "AUTO" OR PLSSVM_ENABLE_OPENMP_BACKEND)
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


## check if ANY backend is available/has been enabled
get_target_property(PLSSVM_LINKED_BACKENDS svm-train LINK_LIBRARIES)
if(NOT PLSSVM_LINKED_BACKENDS)
    message(FATAL_ERROR "At least one backend must be available!")
endif()


## enable Link Time Optimization (LTO)
option(PLSSVM_ENABLE_LTO "Enable Link Time Optimizations." ON)
if(PLSSVM_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT PLSSVM_LTO_SUPPORTED OUTPUT PLSSVM_LTO_ERROR LANGUAGES CXX)
    if(PLSSVM_LTO_SUPPORTED)
        message(STATUS "Interprocedural optimization (IPO/LTO) enabled.")
        set_property(TARGET ${PLSSVM_EXECUTABLE_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "Interprocedural optimization (IPO/LTO) not supported: ${PLSSVM_LTO_ERROR}")
    endif()
endif()


## enable documentation generation with doxygen
option(PLSSVM_ENABLE_DOCUMENTATION "Build documentation using doxygen." OFF)
if(PLSSVM_ENABLE_DOCUMENTATION)
    add_subdirectory(docs)
endif()


## enable testing
option(PLSSVM_ENABLE_TESTING "Build tests for all backends." ON)
if(PLSSVM_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
