# check if OpenCL can be enabled
find_package(OpenCL)

if(NOT OpenCL_FOUND)
  if(PLSSVM_ENABLE_OPENCL_BACKEND MATCHES "ON")
    message(FATAL_ERROR "Cannot find requested backend: OpenCL!")
  endif()
  if (PLSSVM_ENABLE_OPENCL_BACKEND MATCHES "AUTO")
    message(STATUS "OpenCL backend not found.")
    return()
  endif()
endif()
message(STATUS "OpenCL backend enabled.")


# explicitly set sources
set(PLSSVM_OpenCL_SOURCES
    OpenCL_CSVM.cpp
    kernels/svm-kernel-linear.cl
    kernels/kernel_q.cl
    manager/manager.cpp
    manager/error_codes.cpp
    manager/run_kernel.cpp
    manager/configuration.cpp
    manager/platform_wrapper.cpp
    manager/json/dict_node.cpp
    manager/json/id_node.cpp
    manager/json/json.cpp
    manager/json/json_exception.cpp
    manager/json/list_node.cpp
    manager/json/node.cpp
    manager/json/text_node.cpp
)

# set target properties
set(PLSSVM_OpenCL_BACKEND_LIBRARY_NAME svm-OCL)
add_library(${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_OpenCL_SOURCES})
target_include_directories(${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME} PUBLIC ${fmt_SOURCE_DIR}/include)
target_include_directories(${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME} PUBLIC ${fast_float_SOURCE_DIR}/include)
target_link_libraries(${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME} fmt::fmt)
target_include_directories(${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME} OpenCL)
target_compile_definitions(${PLSSVM_EXECUTABLE_NAME} PUBLIC PLSSVM_HAS_OPENCL_BACKEND)
target_link_libraries(${PLSSVM_EXECUTABLE_NAME} ${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME})