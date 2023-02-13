## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
## Based on: https://gist.github.com/CodeFinder2/40864be863c887e0a6dabf4f3a1fa93b
########################################################################################################################

find_package(PkgConfig)

# check if the include directory is given using an environment variable
if (DEFINED ENV{fast_float_INCLUDE_DIR} AND NOT EXISTS "${fast_float_INCLUDE_DIR}")
    set(fast_float_INCLUDE_DIR $ENV{fast_float_INCLUDE_DIR})
endif ()

# try to automatically find the header files in the standard directories
if (NOT EXISTS "${fast_float_INCLUDE_DIR}")
    find_path(fast_float_INCLUDE_DIR
            NAMES fast_float.hpp
            DOC "fast_float header-only library header files"
            )
endif ()

# allow the user to specify the include directory in the CMake call (if provided, used instead of the environment variable)
if (EXISTS "${fast_float_INCLUDE_DIR}")
    include(FindPackageHandleStandardArgs)
    mark_as_advanced(fast_float_INCLUDE_DIR)
else ()
    #message(WARNING "Can't find required package fast_float!")
endif ()

# set the _FOUND variable to the correct value
if (EXISTS "${fast_float_INCLUDE_DIR}")
    set(fast_float_FOUND 1)
else ()
    set(fast_float_FOUND 0)
endif ()
