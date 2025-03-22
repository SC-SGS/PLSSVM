# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# define the used compiler and linker flags for the coverage target
set(PLSSVM_COVERAGE_COMPILER_FLAGS "-O0 -g --coverage -fprofile-abs-path -fno-inline -fno-inline-functions -fno-inline-small-functions -fno-elide-constructors -fno-common -ffunction-sections -fno-omit-frame-pointer")
set(PLSSVM_COVERAGE_LINKER_FLAGS "-O0 -g --coverage -fno-lto -lgcov")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)

# add new coverage build type
set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} ${PLSSVM_COVERAGE_COMPILER_FLAGS}" CACHE STRING "Flags used by the C++ compiler during coverage builds."
                                                                                             FORCE
)
set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_C_FLAGS_DEBUG} ${PLSSVM_COVERAGE_COMPILER_FLAGS}" CACHE STRING "Flags used by the C compiler during coverage builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${PLSSVM_COVERAGE_LINKER_FLAGS} -lgcov"
    CACHE STRING "Flags used for linking binaries during coverage builds." FORCE
)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} ${PLSSVM_COVERAGE_LINKER_FLAGS} -lgcov"
    CACHE STRING "Flags used by the shared libraries linker during coverage builds." FORCE
)
mark_as_advanced(CMAKE_CXX_FLAGS_COVERAGE CMAKE_C_FLAGS_COVERAGE CMAKE_EXE_LINKER_FLAGS_COVERAGE CMAKE_SHARED_LINKER_FLAGS_COVERAGE)

# update the documentation string of CMAKE_BUILD_TYPE for GUIs
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE)
set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS
             "Debug"
             "Release"
             "RelWithDebInfo"
             "MinSizeRel"
             "Coverage"
)
