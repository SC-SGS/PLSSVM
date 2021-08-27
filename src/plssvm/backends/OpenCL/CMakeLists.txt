# check if OpenCL can be enabled
message(CHECK_START "Checking for OpenCL backend")

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
    ${CMAKE_CURRENT_LIST_DIR}/q_kernel.cl
    ${CMAKE_CURRENT_LIST_DIR}/svm_kernel.cl
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detail/error_code.cpp
)

# set target properties
set_local_and_parent(PLSSVM_OPENCL_BACKEND_LIBRARY_NAME svm-OpenCL)
add_library(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} STATIC ${PLSSVM_OPENCL_SOURCES})
target_include_directories(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC OpenCL)

# link base library against OpenCL library
target_link_libraries(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set kernel file directory
target_compile_definitions(${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY="${CMAKE_CURRENT_LIST_DIR}/")

# link OpenCL library against executable
target_compile_definitions(${PLSSVM_EXECUTABLE_TRAIN_NAME} PUBLIC PLSSVM_HAS_OPENCL_BACKEND)
target_link_libraries(${PLSSVM_EXECUTABLE_TRAIN_NAME} ${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME})
target_compile_definitions(${PLSSVM_EXECUTABLE_PREDICT_NAME} PUBLIC PLSSVM_HAS_OPENCL_BACKEND)
target_link_libraries(${PLSSVM_EXECUTABLE_PREDICT_NAME} ${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME})


# generate summary string
set(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_COMPILER " - OpenCL:")
include(${PROJECT_SOURCE_DIR}/cmake/assemble_summary_string.cmake)
assemble_summary_string(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS)
set(PLSSVM_OPENCL_BACKEND_SUMMARY_STRING "${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_COMPILER}${PLSSVM_OPENCL_BACKEND_SUMMARY_STRING_ARCHS}" PARENT_SCOPE)