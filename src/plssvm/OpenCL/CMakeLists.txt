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
  ${CMAKE_CURRENT_LIST_DIR}/OCL_CSVM.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cuda-kernel.cpp
  ${CMAKE_CURRENT_LIST_DIR}/kernels/svm-kernel-linear.cl
  ${CMAKE_CURRENT_LIST_DIR}/kernels/kernel_q.cl
  ${CMAKE_CURRENT_LIST_DIR}/manager/manager.cpp
  ${CMAKE_CURRENT_LIST_DIR}/manager/error_codes.cpp
  ${CMAKE_CURRENT_LIST_DIR}/manager/run_kernel.cpp
  ${CMAKE_CURRENT_LIST_DIR}/manager/configuration.cpp
  ${CMAKE_CURRENT_LIST_DIR}/manager/platform_wrapper.cpp
  ${CMAKE_CURRENT_LIST_DIR}/manager/json/dict_node.cpp
  ${CMAKE_CURRENT_LIST_DIR}/manager/json/id_node.cpp
  ${CMAKE_CURRENT_LIST_DIR}/manager/json/json.cpp
  ${CMAKE_CURRENT_LIST_DIR}/manager/json/json_exception.cpp
  ${CMAKE_CURRENT_LIST_DIR}/manager/json/list_node.cpp
  ${CMAKE_CURRENT_LIST_DIR}/manager/json/node.cpp
  ${CMAKE_CURRENT_LIST_DIR}/manager/json/text_node.cpp
)

# set target properties
set(PLSSVM_OpenCL_BACKEND_LIBRARY_NAME svm-OCL)
add_library(${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_SOURCES} ${PLSSVM_OpenCL_SOURCES})
target_include_directories(${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME} OpenCL)
target_compile_definitions(${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME} PUBLIC WITH_OPENCL) # TODO: remove later
target_compile_definitions(${PLSSVM_EXECUTABLE_NAME} PUBLIC PLSSVM_HAS_OPENCL_BACKEND)
target_link_libraries(${PLSSVM_EXECUTABLE_NAME} ${PLSSVM_OpenCL_BACKEND_LIBRARY_NAME})