## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if the OpenCL backend is available
if (TARGET plssvm::plssvm-OpenMP)
    # enable OpenMP
    find_dependency(OpenMP)
    # set alias targets
    add_library(plssvm::OpenMP ALIAS plssvm::plssvm-OpenMP)
    add_library(plssvm::openmp ALIAS plssvm::plssvm-OpenMP)
    # set COMPONENT to be found
    set(plssvm_OpenMP_FOUND ON)
else ()
    # set COMPONENT to be NOT found
    set(plssvm_OpenMP_FOUND OFF)
endif ()