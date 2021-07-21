# create OpenMP tests if the OpenMP backend is available
if(TARGET svm-OpenMP)
    add_subdirectory(OpenMP)
endif()

# create OpenCL tests if the OpenCL backend is available
if(TARGET svm-OCL)
    add_subdirectory(OpenCL)
endif()

# create CUDA tests if the CUDA backend is available
if(TARGET svm-CUDA)
    add_subdirectory(CUDA)
endif()