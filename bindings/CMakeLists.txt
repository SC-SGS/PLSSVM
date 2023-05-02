## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################


include(CMakeDependentOption)

list(APPEND CMAKE_MESSAGE_INDENT "Bindings:  ")


# enable Python langauge bindings
cmake_dependent_option(PLSSVM_ENABLE_PYTHON_BINDINGS "Build langauge bindings for Python." ON "PLSSVM_ENABLE_LANGUAGE_BINDINGS" OFF)
if (PLSSVM_ENABLE_PYTHON_BINDINGS)
    add_subdirectory(Python)
endif ()


# explicitly update install targets in parent scope
set(PLSSVM_TARGETS_TO_INSTALL "${PLSSVM_TARGETS_TO_INSTALL}" PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)