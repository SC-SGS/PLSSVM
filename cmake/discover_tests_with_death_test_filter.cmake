## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

function(discover_tests_with_death_test_filter test_executable_name)
    if (PLSSVM_ENABLE_DEATH_TESTS)
        # assertions are enabled -> enable Google death tests
        gtest_discover_tests(${test_executable_name})
    else ()
        # assertions are disabled -> disable Google death tests
        gtest_discover_tests(${test_executable_name} TEST_FILTER -*DeathTest*)
    endif ()
endfunction()