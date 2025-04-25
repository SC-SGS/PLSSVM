# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if Kokkos is an optional component
is_component_optional(Kokkos)

# check if the Kokkos backend is available
if (TARGET plssvm::plssvm-Kokkos)
    # enable Kokkos
    find_dependency(Kokkos CONFIG)
    
    # set alias targets
    add_library(plssvm::Kokkos ALIAS plssvm::plssvm-Kokkos)
    add_library(plssvm::kokkos ALIAS plssvm::plssvm-Kokkos)
    
    # set COMPONENT to be found
    set(plssvm_Kokkos_FOUND ON)
    if (NOT plssvm_FIND_QUIETLY)
        if (plssvm_FIND_OPTIONAL_Kokkos)
            message(STATUS "Found optional component \"Kokkos\".")
        else ()
            message(STATUS "Found component \"Kokkos\".")
        endif ()
    endif ()
else ()
    # set COMPONENT to be NOT found
    set(plssvm_Kokkos_FOUND OFF)
    # PLSSVM is only not found if Kokkos is NOT an optional component
    if (NOT plssvm_FIND_OPTIONAL_Kokkos)
        set(plssvm_FOUND OFF)
    endif ()
    
    # if REQUIRED was set in the find_package call, fail
    if (plssvm_FIND_REQUIRED AND NOT plssvm_FIND_OPTIONAL_Kokkos)
        set(plssvm_NOT_FOUND_MESSAGE "Couldn't find required component \"Kokkos\".")
        return()
    elseif (NOT plssvm_FIND_QUIETLY)
        if (plssvm_FIND_OPTIONAL_Kokkos)
            message(STATUS "Couldn't find optional component \"Kokkos\".")
        else ()
            message(STATUS "Couldn't find component \"Kokkos\".")
        endif ()
    endif ()
endif ()
