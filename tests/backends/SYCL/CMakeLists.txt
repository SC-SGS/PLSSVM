## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# create AdaptiveCpp tests if the SYCL implementation backend is available
if (TARGET ${PLSSVM_SYCL_BACKEND_ADAPTIVECPP_LIBRARY_NAME})
    add_subdirectory(AdaptiveCpp)
endif ()

# create DPC++ tests if the SYCL implementation backend is available
if (TARGET ${PLSSVM_SYCL_BACKEND_DPCPP_LIBRARY_NAME})
    add_subdirectory(DPCPP)
endif ()