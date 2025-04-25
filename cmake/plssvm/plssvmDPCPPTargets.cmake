# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

include(CMakeFindDependencyMacro)

# check if DPCPP is an optional component
is_component_optional(DPCPP)

# check if the AdaptiveCpp backend is available
if (TARGET plssvm::plssvm-SYCL_DPCPP)
    # set alias targets
    add_library(plssvm::DPCPP ALIAS plssvm::plssvm-SYCL_DPCPP)
    add_library(plssvm::dpcpp ALIAS plssvm::plssvm-SYCL_DPCPP)
    
    # set COMPONENT to be found
    set(plssvm_DPCPP_FOUND ON)
    if (NOT plssvm_FIND_QUIETLY)
        if (plssvm_FIND_OPTIONAL_DPCPP)
            message(STATUS "Found optional component \"DPCPP\".")
        else ()
            message(STATUS "Found component \"DPCPP\".")
        endif ()
    endif ()
else ()
    # set COMPONENT to be NOT found
    set(plssvm_DPCPP_FOUND OFF)
    # PLSSVM is only not found if DPCPP is NOT an optional component
    if (NOT plssvm_FIND_OPTIONAL_DPCPP)
        set(plssvm_FOUND OFF)
    endif ()
    
    # if REQUIRED was set in the find_package call, fail
    if (plssvm_FIND_REQUIRED AND NOT plssvm_FIND_OPTIONAL_DPCPP)
        set(plssvm_NOT_FOUND_MESSAGE "Couldn't find required component \"DPCPP\".")
        return()
    elseif (NOT plssvm_FIND_QUIETLY)
        if (plssvm_FIND_OPTIONAL_DPCPP)
            message(STATUS "Couldn't find optional component \"DPCPP\".")
        else ()
            message(STATUS "Couldn't find component \"DPCPP\".")
        endif ()
    endif ()
endif ()
