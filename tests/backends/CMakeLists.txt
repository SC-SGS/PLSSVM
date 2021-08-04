# create OpenMP tests if the OpenMP backend is available
if(TARGET ${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME})
    add_subdirectory(OpenMP)
endif()

# create CUDA tests if the CUDA backend is available
if(TARGET ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME})
    add_subdirectory(CUDA)
endif()

# create OpenCL tests if the OpenCL backend is available
if(TARGET ${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME})
    add_subdirectory(OpenCL)
endif()

# create SYCL tests if the SYCL backend is available
if(TARGET ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME})
    add_subdirectory(SYCL)
endif()