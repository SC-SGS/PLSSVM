## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

function(discover_tests_with_death_test_filter test_executable_name)
    if (PLSSVM_ENABLE_DEATH_TESTS)
        # assertions are enabled -> enable Google death tests
        gtest_discover_tests(${test_executable_name} PROPERTIES
                DISCOVERY_TIMEOUT 600
                DISCOVERY_MODE PRE_TEST
                WORKING_DIRECTORY $<TARGET_FILE_DIR:${test_executable_name}>)
    else ()
        # assertions are disabled -> disable Google death tests
        gtest_discover_tests(${test_executable_name} TEST_FILTER -*DeathTest* PROPERTIES
                DISCOVERY_TIMEOUT 600
                DISCOVERY_MODE PRE_TEST
                WORKING_DIRECTORY $<TARGET_FILE_DIR:${test_executable_name}>)
    endif ()
    if (WIN32)
        add_custom_command(
                TARGET ${test_executable_name}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${test_executable_name}> $<TARGET_FILE_DIR:${test_executable_name}>
                COMMAND_EXPAND_LISTS
        )
    endif ()
endfunction()
