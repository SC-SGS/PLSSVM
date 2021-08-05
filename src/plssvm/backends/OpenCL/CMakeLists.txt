# check if OpenCL can be enabled
message(CHECK_START "Checking for OpenCL backend")

# TODO: rewrite to use PLSSVM_TARGET_PLATFORMS

find_package(OpenCL)

if(NOT OpenCL_FOUND)
    message(CHECK_FAIL "not found")
    if(PLSSVM_ENABLE_OPENCL_BACKEND MATCHES "ON")
        message(FATAL_ERROR "Cannot find requested backend: OpenCL!")
    endif()
    return()
endif()
message(CHECK_PASS "found")


# explicitly set sources
set(PLSSVM_OPENCL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/error_code.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/svm-kernel.cl
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
set_local_and_parent(PLSSVM_OPENCL_BACKEND_LIBRARY_NAME svm-OpenCL)
add_library(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} STATIC ${PLSSVM_OPENCL_SOURCES})
target_include_directories(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC OpenCL)

# link base library against OpenCL library
target_link_libraries(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# link OpenMP library against executable
target_compile_definitions(${PLSSVM_EXECUTABLE_NAME} PUBLIC PLSSVM_HAS_OPENCL_BACKEND)
target_link_libraries(${PLSSVM_EXECUTABLE_NAME} ${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME})


# generate summary string
set(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_COMPILER " - OpenCL:")
include(${PROJECT_SOURCE_DIR}/cmake/assemble_summary_string.cmake)
assemble_summary_string(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS)
set(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_COMPILER}${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}" PARENT_SCOPE)