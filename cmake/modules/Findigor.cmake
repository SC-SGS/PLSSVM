## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
## Based on: https://gist.github.com/CodeFinder2/40864be863c887e0a6dabf4f3a1fa93b
########################################################################################################################

find_package(PkgConfig)

# check if the include directory is given using an environment variable
if (DEFINED ENV{igor_INCLUDE_DIR} AND NOT EXISTS "${igor_INCLUDE_DIR}")
    set(igor_INCLUDE_DIR $ENV{igor_INCLUDE_DIR})
endif ()

# try to automatically find the header files in the standard directories
if (NOT EXISTS "${igor_INCLUDE_DIR}")
    find_path(igor_INCLUDE_DIR
            NAMES igor.hpp
            DOC "igor header-only library header files"
            )
endif ()

# allow the user to specify the include directory in the CMake call (if provided, used instead of the environment variable)
if (EXISTS "${igor_INCLUDE_DIR}")
    include(FindPackageHandleStandardArgs)
    mark_as_advanced(igor_INCLUDE_DIR)
else ()
    #message(WARNING "Can't find required package igor!")
endif ()

# set the _FOUND variable to the correct value
if (EXISTS "${igor_INCLUDE_DIR}")
    set(igor_FOUND 1)
else ()
    set(igor_FOUND 0)
endif ()
