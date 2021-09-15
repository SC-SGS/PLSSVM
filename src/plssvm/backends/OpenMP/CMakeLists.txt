# check if OpenMP can be enabled
message(CHECK_START "Checking for OpenMP backend")

find_package(OpenMP)

if(NOT OPENMP_FOUND)
    message(CHECK_FAIL "not found")
    if(PLSSVM_ENABLE_OPENMP_BACKEND MATCHES "ON")
        message(FATAL_ERROR "Cannot find requested backend: OpenMP!")
    endif()
    return()
else()
    if(NOT DEFINED PLSSVM_CPU_TARGET_ARCHS)
        message(STATUS "Found requested OpenMP backend, but no \"cpu\" targets were specified!")
        message(CHECK_FAIL "skipped")
        return()
    endif()
endif()
message(CHECK_PASS "found")

# explicitly set sources
set(PLSSVM_OPENMP_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/q_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/svm_kernel.cpp
)


# set target properties
set_local_and_parent(PLSSVM_OPENMP_BACKEND_LIBRARY_NAME svm-OpenMP)
add_library(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} STATIC ${PLSSVM_OPENMP_SOURCES})
target_link_libraries(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)

# link base library against OpenMP library
target_link_libraries(${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# link OpenMP library against executable
target_compile_definitions(${PLSSVM_EXECUTABLE_TRAIN_NAME} PUBLIC PLSSVM_HAS_OPENMP_BACKEND)
target_link_libraries(${PLSSVM_EXECUTABLE_TRAIN_NAME} ${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME})
target_compile_definitions(${PLSSVM_EXECUTABLE_PREDICT_NAME} PUBLIC PLSSVM_HAS_OPENMP_BACKEND)
target_link_libraries(${PLSSVM_EXECUTABLE_PREDICT_NAME} ${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME})


# generate summary string
set(PLSSVM_OPENMP_BACKEND_SUMMARY_STRING " - OpenMP: cpu" PARENT_SCOPE)