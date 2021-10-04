## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

file(GLOB_RECURSE PLSSVM_COVERAGE_GCDA_FILES ${CMAKE_BINARY_DIR}/*.gcda)
file(REMOVE ${PLSSVM_COVERAGE_GCDA_FILES})
file(GLOB_RECURSE PLSSVM_COVERAGE_GCNO_FILES ${CMAKE_BINARY_DIR}/*.gcno)
file(REMOVE ${PLSSVM_COVERAGE_GCNO_FILES})