# check if OpenMP can be enabled
find_package(OpenMP)

if(NOT OPENMP_FOUND)
    if(PLSVM_ENABLE_OpenMP_BACKEND MATCHES "ON")
        message(FATAL_ERROR "Cannot find requested backend: OpenMP!")
    endif()
    if (PLSVM_ENABLE_OpenMP_BACKEND MATCHES "AUTO")
        message(STATUS "OpenMP backend not found.")
        return()
    endif()
endif()
message(STATUS "OpenMP backend enabled.")

# explicitly set sources
set(PLSSVM_OpenMP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/OpenMP_CSVM.cpp
        ${CMAKE_CURRENT_LIST_DIR}/svm-kernel.cpp
)

# set target properties
set(PLSSVM_OpenMP_BACKEND_LIBRARY_NAME svm-OpenMP)
add_library(${PLSSVM_OpenMP_BACKEND_LIBRARY_NAME} STATIC ${PLSSVM_OpenMP_SOURCES})
target_include_directories(${PLSSVM_OpenMP_BACKEND_LIBRARY_NAME} PUBLIC ${fmt_SOURCE_DIR}/include)
target_include_directories(${PLSSVM_OpenMP_BACKEND_LIBRARY_NAME} PUBLIC ${fast_float_SOURCE_DIR}/include)
target_include_directories(${PLSSVM_OpenMP_BACKEND_LIBRARY_NAME} PUBLIC ${cxxopts_SOURCE_DIR}/include)
target_link_libraries(${PLSSVM_OpenMP_BACKEND_LIBRARY_NAME} PUBLIC fmt::fmt)
target_compile_definitions(${PLSSVM_EXECUTABLE_NAME} PUBLIC PLSSVM_HAS_OPENMP_BACKEND)
target_link_libraries(${PLSSVM_EXECUTABLE_NAME} ${PLSSVM_OpenMP_BACKEND_LIBRARY_NAME})