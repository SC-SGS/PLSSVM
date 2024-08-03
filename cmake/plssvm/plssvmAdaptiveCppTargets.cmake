## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if the AdaptiveCpp backend is available
if (TARGET plssvm::plssvm-SYCL_adaptivecpp)
    # enable AdaptiveCpp
    find_dependency(AdaptiveCpp CONFIG)
    # set alias targets
    add_library(plssvm::AdaptiveCpp ALIAS plssvm::plssvm-SYCL_adaptivecpp)
    add_library(plssvm::adaptivecpp ALIAS plssvm::plssvm-SYCL_adaptivecpp)
    # set COMPONENT to be found
    set(plssvm_AdaptiveCpp_FOUND ON)
else ()
    # set COMPONENT to be NOT found
    set(plssvm_AdaptiveCpp_FOUND OFF)
endif ()