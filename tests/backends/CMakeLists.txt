## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# create OpenMP tests if the OpenMP backend is available
if (TARGET ${PLSSVM_OPENMP_BACKEND_LIBRARY_NAME})
    add_subdirectory(OpenMP)
endif ()

# create CUDA tests if the CUDA backend is available
if (TARGET ${PLSSVM_CUDA_BACKEND_LIBRARY_NAME})
    add_subdirectory(CUDA)
endif ()

# create HIP tests if the HIP backend is available
if (TARGET ${PLSSVM_HIP_BACKEND_LIBRARY_NAME})
    add_subdirectory(HIP)
endif ()

# create OpenCL tests if the OpenCL backend is available
if (TARGET ${PLSSVM_OPENCL_BACKEND_LIBRARY_NAME})
    add_subdirectory(OpenCL)
endif ()

# create SYCL tests if the SYCL backend is available
if (TARGET ${PLSSVM_SYCL_BACKEND_LIBRARY_NAME})
    add_subdirectory(SYCL)
endif ()