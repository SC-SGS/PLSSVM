## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

function(check_python_libs required_libraries error_string)
    foreach (PLSSVM_PYTHON_LIB ${required_libraries})
        # search for Python package
        execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import ${PLSSVM_PYTHON_LIB}"
                RESULT_VARIABLE PLSSVM_PYTHON_LIB_EXIT_CODE
                OUTPUT_QUIET)

        # emit error if package couldn't be found
        if (NOT ${PLSSVM_PYTHON_LIB_EXIT_CODE} EQUAL 0)
            message(FATAL_ERROR
                    "The '${PLSSVM_PYTHON_LIB}' Python3 package is not installed. "
                    "Please install it using the following command: '${Python3_EXECUTABLE} -m pip install ${PLSSVM_PYTHON_LIB}'\n "
                    "${error_string}"
                    )
        endif ()
    endforeach ()
endfunction()