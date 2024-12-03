## Authors: Alexander Van Craen, Marcel Breyer, Alexander Strack
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if the HPX backend is available
if (TARGET plssvm::plssvm-HPX)
    # enable HPX
    find_dependency(HPX)
    # set alias targets
    add_library(plssvm::HPX ALIAS plssvm::plssvm-HPX)
    add_library(plssvm::hpx ALIAS plssvm::plssvm-HPX)
    # set COMPONENT to be found
    set(plssvm_HPX_FOUND ON)
else ()
    # set COMPONENT to be NOT found
    set(plssvm_HPX_FOUND OFF)
endif ()
