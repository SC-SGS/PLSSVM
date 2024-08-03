## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if the OpenCL backend is available
if (TARGET plssvm::plssvm-OpenCL)
    # enable OpenCL
    find_dependency(OpenCL)
    # set alias targets
    add_library(plssvm::OpenCL ALIAS plssvm::plssvm-OpenCL)
    add_library(plssvm::opencl ALIAS plssvm::plssvm-OpenCL)
    # set COMPONENT to be found
    set(plssvm_OpenCL_FOUND ON)
else ()
    # set COMPONENT to be NOT found
    set(plssvm_OpenCL_FOUND OFF)
endif ()