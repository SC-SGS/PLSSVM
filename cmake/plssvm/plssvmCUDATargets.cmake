# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if CUDA is an optional component
is_component_optional(CUDA)

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
    if (NOT plssvm_FIND_QUIETLY)
        if (plssvm_FIND_OPTIONAL_CUDA)
            message(STATUS "Found optional component \"CUDA\".")
        else ()
            message(STATUS "Found component \"CUDA\".")
        endif ()
    endif ()
else ()
    # set COMPONENT to be NOT found
    set(plssvm_CUDA_FOUND OFF)
    # PLSSVM is only not found if CUDA is NOT an optional component
    if (NOT plssvm_FIND_OPTIONAL_CUDA)
        set(plssvm_FOUND OFF)
    endif ()

    # if REQUIRED was set in the find_package call, fail
    if (plssvm_FIND_REQUIRED AND NOT plssvm_FIND_OPTIONAL_CUDA)
        set(plssvm_NOT_FOUND_MESSAGE "Couldn't find required component \"CUDA\".")
        return()
    elseif (NOT plssvm_FIND_QUIETLY)
        if (plssvm_FIND_OPTIONAL_CUDA)
            message(STATUS "Couldn't find optional component \"CUDA\".")
        else ()
            message(STATUS "Couldn't find component \"CUDA\".")
        endif ()
    endif ()
endif ()
