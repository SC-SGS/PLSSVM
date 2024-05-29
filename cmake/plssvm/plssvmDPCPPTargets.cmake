## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if the AdaptiveCpp backend is available
if (TARGET plssvm::plssvm-SYCL_dpcpp)
    # set alias targets
    add_library(plssvm::DPCPP ALIAS plssvm::plssvm-SYCL_dpcpp)
    add_library(plssvm::dpcpp ALIAS plssvm::plssvm-SYCL_dpcpp)
    # set COMPONENT to be found
    set(plssvm_DPCPP_FOUND ON)
else ()
    # set COMPONENT to be NOT found
    set(plssvm_DPCPP_FOUND OFF)
endif ()