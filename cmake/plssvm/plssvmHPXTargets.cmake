# Authors: Alexander Van Craen, Marcel Breyer, Alexander Strack
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if HPX is an optional component
is_component_optional(HPX)

# check if the HPX backend is available
if (TARGET plssvm::plssvm-HPX)
    # enable HPX
    find_dependency(HPX)

    # set alias targets
    add_library(plssvm::HPX ALIAS plssvm::plssvm-HPX)
    add_library(plssvm::hpx ALIAS plssvm::plssvm-HPX)

    # set COMPONENT to be found
    set(plssvm_HPX_FOUND ON)
    if (NOT plssvm_FIND_QUIETLY)
        if (plssvm_FIND_OPTIONAL_HPX)
            message(STATUS "Found optional component \"HPX\".")
        else ()
            message(STATUS "Found component \"HPX\".")
        endif ()
    endif ()
else ()
    # set COMPONENT to be NOT found
    set(plssvm_HPX_FOUND OFF)
    # PLSSVM is only not found if HPX is NOT an optional component
    if (NOT plssvm_FIND_OPTIONAL_HPX)
        set(plssvm_FOUND OFF)
    endif ()

    # if REQUIRED was set in the find_package call, fail
    if (plssvm_FIND_REQUIRED AND NOT plssvm_FIND_OPTIONAL_HPX)
        set(plssvm_NOT_FOUND_MESSAGE "Couldn't find required component \"HPX\".")
        return()
    elseif (NOT plssvm_FIND_QUIETLY)
        if (plssvm_FIND_OPTIONAL_HPX)
            message(STATUS "Couldn't find optional component \"HPX\".")
        else ()
            message(STATUS "Couldn't find component \"HPX\".")
        endif ()
    endif ()
endif ()
