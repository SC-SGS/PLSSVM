# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if the Kokkos backend is available
if (TARGET plssvm::plssvm-Kokkos)
    # enable Kokkos
    find_dependency(Kokkos CONFIG)
    # set alias targets
    add_library(plssvm::Kokkos ALIAS plssvm::plssvm-Kokkos)
    add_library(plssvm::kokkos ALIAS plssvm::plssvm-Kokkos)
    # set COMPONENT to be found
    set(plssvm_Kokkos_FOUND ON)
else ()
    # set COMPONENT to be NOT found
    set(plssvm_Kokkos_FOUND OFF)
endif ()
