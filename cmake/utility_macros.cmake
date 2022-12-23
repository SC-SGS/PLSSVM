## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# set variable in the local and parent scope
macro(set_local_and_parent NAME VALUE)
    set(${ARGV0} ${ARGV1})
    set(${ARGV0} ${ARGV1} PARENT_SCOPE)
endmacro()

macro(append_local_and_parent LIST_NAME VALUE)
    list(APPEND ${ARGV0} ${ARGV1})
    set(${ARGV0} ${${ARGV0}} PARENT_SCOPE)
endmacro()