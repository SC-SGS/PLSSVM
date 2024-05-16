## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if the HIP backend is available
if (TARGET plssvm::plssvm-HIP)
    # enable HIP
    enable_language(HIP)
    find_dependency(HIP REQUIRED)
    # set alias targets
    add_library(plssvm::HIP ALIAS plssvm::plssvm-HIP)
    add_library(plssvm::hip ALIAS plssvm::plssvm-HIP)
    # set COMPONENT to be found
    set(plssvm_HIP_FOUND ON)
else ()
    # set COMPONENT to be NOT found
    set(plssvm_HIP_FOUND OFF)
endif ()