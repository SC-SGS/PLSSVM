## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

function(parse_architecture_info target_platform target_archs num_archs)
    # transform platforms to list (e.g "nvidia:sm_70,sm_80" -> "nvidia;sm_70,sm_80")
    string(REPLACE ":" ";" ARCH_LIST ${target_platform})
    # remove platform from list (e.g. "nvidia;sm_70,sm_80" -> "sm_70,sm_80")
    list(POP_FRONT ARCH_LIST)

    if (ARCH_LIST STREQUAL "")
        # immediately return if architecture list is empty
        set(${target_archs} "" PARENT_SCOPE)
        set(${num_archs} 0 PARENT_SCOPE)
    else ()
        # transform architectures to list and set output-variable (e.g. "sm_70,sm_80" -> "sm_70;sm_80")
        string(REPLACE "," ";" ARCH_LIST ${ARCH_LIST})
        set(${target_archs} ${ARCH_LIST} PARENT_SCOPE)

        # get number of architectures and set output-variable (e.g. "sm_70;sm_80" -> 2)
        list(LENGTH ARCH_LIST LEN)
        set(${num_archs} ${LEN} PARENT_SCOPE)
    endif ()
endfunction()

