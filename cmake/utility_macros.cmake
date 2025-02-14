# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# set variable in the local and parent scope
macro (set_local_and_parent NAME VALUE)
    if (${ARGC} GREATER 2)
        set(PLSSVM_TEMP_ARGN "${ARGN}")
        list(JOIN PLSSVM_TEMP_ARGN " " PLSSVM_REMAINING_FLAGS)
        set(${ARGV0} "${ARGV1} ${PLSSVM_REMAINING_FLAGS}")
        set(${ARGV0} "${ARGV1} ${PLSSVM_REMAINING_FLAGS}" PARENT_SCOPE)
    else ()
        set(${ARGV0} "${ARGV1}")
        set(${ARGV0} "${ARGV1}" PARENT_SCOPE)
    endif ()
endmacro ()

macro (append_local_and_parent LIST_NAME VALUE)
    list(APPEND ${ARGV0} ${ARGV1})
    set(${ARGV0} ${${ARGV0}} PARENT_SCOPE)
endmacro ()

# test whether the provided variable contains a natural number greater than zero
macro (check_integer VARIABLE)
    if (NOT ${${VARIABLE}} MATCHES "^[0-9]+$" OR ${${VARIABLE}} LESS_EQUAL 0)
        message(FATAL_ERROR "The ${VARIABLE} must be a natural number greater 0, but is \"${${VARIABLE}}\"!")
    endif ()
endmacro ()

# check if the requested test file already exists and is not zero based indexed (otherwise create new test file)
macro (check_test_file_validity FILE_NAME)
    if (EXISTS "${FILE_NAME}")
        file(READ "${FILE_NAME}" FILE_CONTENT)
        if (FILE_CONTENT MATCHES " 0:[0-9]")
            message(STATUS "Test file \"${FILE_NAME}\" already exists but is zero based indexed. Generating new test file.")
            file(REMOVE "${FILE_NAME}")
        else ()
            message(STATUS "Skipped test file generation since it already exists (${FILE_NAME})!")
        endif ()
    endif ()
endmacro ()