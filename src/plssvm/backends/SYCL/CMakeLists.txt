# check if SYCL can be enabled
message(CHECK_START "Checking for SYCL backend")

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(CHECK_FAIL "not found")
    return()
endif()

# check if the compiler supports SYCL
#try_compile(PLSSVM_IS_SYCL_COMPILER
#        "${CMAKE_BINARY_DIR}/temp"
#        "${PROJECT_SOURCE_DIR}/cmake/compile_tests/test_sycl_base.cpp"
#        CMAKE_FLAGS "-DCMAKE_CXX_STANDARD=17" "-DCMAKE_EXE_LINKER_FLAGS=-fopenmp")
#message(STATUS "SYCL: ${PLSSVM_IS_SYCL_COMPILER}")
#
#if(NOT PLSSVM_IS_SYCL_COMPILER)
#    message(CHECK_FAIL "not found")
#    if(PLSSVM_ENABLE_SYCL_BACKEND MATCHES "ON")
#        message(FATAL_ERROR "Cannot find requested backend: SYCL!")
#    endif()
#    return()
#endif()

message(CHECK_PASS "found")

# explicitly set sources
set(PLSSVM_SYCL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
)

# set target properties
set(PLSSVM_SYCL_BACKEND_LIBRARY_NAME svm-SYCL)
add_library(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_SYCL_SOURCES})

target_compile_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice --cuda-gpu-arch=sm_80 --cuda-path=/usr/local.nfs/sw/cuda/cuda-11.2.2)
target_link_options(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -fsycl --cuda-gpu-arch=sm_80 --cuda-path=/usr/local.nfs/sw/cuda/cuda-11.2.2)
target_link_libraries(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PRIVATE -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)

# link base library against SYCL library
target_link_libraries(${PLSSVM_SYCL_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# link SYCL library against executable
target_compile_definitions(${PLSSVM_EXECUTABLE_NAME} PUBLIC PLSSVM_HAS_SYCL_BACKEND)
target_link_libraries(${PLSSVM_EXECUTABLE_NAME} ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME})