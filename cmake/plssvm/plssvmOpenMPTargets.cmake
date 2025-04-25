# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if OpenMP is an optional component
is_component_optional(OpenMP)

# check if the OpenMP backend is available
if (TARGET plssvm::plssvm-OpenMP)
    # enable OpenMP
    find_dependency(OpenMP)

    # set alias targets
    add_library(plssvm::OpenMP ALIAS plssvm::plssvm-OpenMP)
    add_library(plssvm::openmp ALIAS plssvm::plssvm-OpenMP)

    # set COMPONENT to be found
    set(plssvm_OpenMP_FOUND ON)
    if (NOT plssvm_FIND_QUIETLY)
        if (plssvm_FIND_OPTIONAL_OpenMP)
            message(STATUS "Found optional component \"OpenMP\".")
        else ()
            message(STATUS "Found component \"OpenMP\".")
        endif ()
    endif ()
else ()
    # set COMPONENT to be NOT found
    set(plssvm_OpenMP_FOUND OFF)
    # PLSSVM is only not found if OpenMP is NOT an optional component
    if (NOT plssvm_FIND_OPTIONAL_OpenMP)
        set(plssvm_FOUND OFF)
    endif ()

    # if REQUIRED was set in the find_package call, fail
    if (plssvm_FIND_REQUIRED AND NOT plssvm_FIND_OPTIONAL_OpenMP)
        set(plssvm_NOT_FOUND_MESSAGE "Couldn't find required component \"OpenMP\".")
        return()
    elseif (NOT plssvm_FIND_QUIETLY)
        if (plssvm_FIND_OPTIONAL_OpenMP)
            message(STATUS "Couldn't find optional component \"OpenMP\".")
        else ()
            message(STATUS "Couldn't find component \"OpenMP\".")
        endif ()
    endif ()
endif ()
