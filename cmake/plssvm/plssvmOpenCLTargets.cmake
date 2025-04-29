# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if OpenCL is an optional component
is_component_optional(OpenCL)

# check if the OpenCL backend is available
if (TARGET plssvm::plssvm-OpenCL)
    # enable OpenCL
    find_dependency(OpenCL)

    # set alias targets
    add_library(plssvm::OpenCL ALIAS plssvm::plssvm-OpenCL)
    add_library(plssvm::opencl ALIAS plssvm::plssvm-OpenCL)

    # set COMPONENT to be found
    set(plssvm_OpenCL_FOUND ON)
    if (NOT plssvm_FIND_QUIETLY)
        if (plssvm_FIND_OPTIONAL_OpenCL)
            message(STATUS "Found optional component \"OpenCL\".")
        else ()
            message(STATUS "Found component \"OpenCL\".")
        endif ()
    endif ()
else ()
    # set COMPONENT to be NOT found
    set(plssvm_OpenCL_FOUND OFF)
    # PLSSVM is only not found if OpenCL is NOT an optional component
    if (NOT plssvm_FIND_OPTIONAL_OpenCL)
        set(plssvm_FOUND OFF)
    endif ()

    # if REQUIRED was set in the find_package call, fail
    if (plssvm_FIND_REQUIRED AND NOT plssvm_FIND_OPTIONAL_OpenCL)
        set(plssvm_NOT_FOUND_MESSAGE "Couldn't find required component \"OpenCL\".")
        return()
    elseif (NOT plssvm_FIND_QUIETLY)
        if (plssvm_FIND_OPTIONAL_OpenCL)
            message(STATUS "Couldn't find optional component \"OpenCL\".")
        else ()
            message(STATUS "Couldn't find component \"OpenCL\".")
        endif ()
    endif ()
endif ()
