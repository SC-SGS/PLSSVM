## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "Tracking:  ")

message(STATUS "Enabled performance tracking to YAML files for the PLSSVM executables plssvm-train, plssvm-predict, and plssvm-scale.")

target_sources(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/performance_tracker.cpp>)

list(POP_BACK CMAKE_MESSAGE_INDENT)