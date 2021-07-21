# check if CUDA can be enabled
include(CheckLanguage)
check_language(CUDA) # TODO: CUDA via clang?

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
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O3 -lineinfo -Xptxas=-v -Xptxas -dlcm=cg" )
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# explicitly set sources
set(PLSSVM_CUDA_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.cu
    ${CMAKE_CURRENT_LIST_DIR}/CUDA_CSVM.cu
    ${CMAKE_CURRENT_LIST_DIR}/predict.cu
    ${CMAKE_CURRENT_LIST_DIR}/q-kernel.cu
    ${CMAKE_CURRENT_LIST_DIR}/svm-kernel.cu
)

# set target properties
set(PLSSVM_CUDA_BACKEND_LIBRARY_NAME svm-CUDA)
add_library(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} SHARED ${PLSSVM_CUDA_SOURCES})
set_target_properties(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# also specific architectures can be set using a CMake option
set(PLSSVM_CUDA_ARCHITECTURES "" CACHE STRING "List of CUDA architectures to compile for.")
# auto detect available CUDA architectures and append to list
include(${PROJECT_SOURCE_DIR}/cmake/auto_detect_cuda_arch.cmake)
auto_detect_cuda_arch(PLSSVM_CUDA_ARCHITECTURES_AUTO)
set(PLSSVM_CUDA_ARCHITECTURES ${PLSSVM_CUDA_ARCHITECTURES} ${PLSSVM_CUDA_ARCHITECTURES_AUTO})
# check that at least one architecture has been defined
if(NOT PLSSVM_CUDA_ARCHITECTURES)
    message(STATUS "No CUDA architecture specified! Setting CUDA_ARCHITECTURES to OFF")
    set_property(TARGET ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES OFF.)
else()
    # set architectures for target
    message(STATUS "CUDA architectures to compile for: ${PLSSVM_CUDA_ARCHITECTURES}")
    set_property(TARGET ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES ${PLSSVM_CUDA_ARCHITECTURES})
endif()

# link base library against CUDA library
target_link_libraries(${PLSSVM_CUDA_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# link OpenMP library against executable
target_compile_definitions(${PLSSVM_EXECUTABLE_NAME} PUBLIC PLSSVM_HAS_CUDA_BACKEND)
target_link_libraries(${PLSSVM_EXECUTABLE_NAME} ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME})