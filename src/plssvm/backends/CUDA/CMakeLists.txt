# check if CUDA can be enabled
include(CheckLanguage)
check_language(CUDA)

# create message depending on requested mode
if(NOT CMAKE_CUDA_COMPILER)
  if(PLSSVM_ENABLE_CUDA_BACKEND MATCHES "ON")
    message(FATAL_ERROR "Cannot find requested backend: CUDA!")
  endif()
  if(PLSSVM_ENABLE_CUDA_BACKEND MATCHES "AUTO")
    message(STATUS "CUDA backend not found.")
    return()
  endif()
endif()
message(STATUS "CUDA backend enabled.")

enable_language(CUDA)

# set necessary flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O3 -lineinfo -Xptxas=-v -Xptxas -dlcm=cg" )
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# explicitly set sources
set(PLSSVM_CUDA_SOURCES
  CUDA_CSVM.cu
  svm-kernel.cu
  cuda-kernel.cu
  cuda-kernel.cpp # TODO: remove if add_mult_ is on gpu again
  predict.cu
)

# set target properties
set(PLSSVM_CUDA_BACKEND_LIBRARY_NAME svm-CUDA)
add_library(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_CUDA_SOURCES})
target_include_directories(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC ${fmt_SOURCE_DIR}/include)
target_include_directories(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC ${fast_float_SOURCE_DIR}/include)
target_include_directories(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC ${cxxopts_SOURCE_DIR}/include)
target_link_libraries(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} fmt::fmt)
set_property(TARGET ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES 72 80 86) # TODO: maybe better?
set_target_properties(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(${PLSSVM_EXECUTABLE_NAME} PUBLIC PLSSVM_HAS_CUDA_BACKEND)
target_link_libraries(${PLSSVM_EXECUTABLE_NAME} ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME})