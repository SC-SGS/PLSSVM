## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if the CUDA backend is available
if (TARGET plssvm::plssvm-CUDA)
    # enable CUDA
    enable_language(CUDA)
    find_dependency(CUDAToolkit)
    # set alias targets
    add_library(plssvm::CUDA ALIAS plssvm::plssvm-CUDA)
    add_library(plssvm::cuda ALIAS plssvm::plssvm-CUDA)
    # set COMPONENT to be found
    set(plssvm_CUDA_FOUND ON)
else ()
    # set COMPONENT to be NOT found
    set(plssvm_CUDA_FOUND OFF)
endif ()