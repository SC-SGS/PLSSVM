/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Contains the googletest main function. Sets the DeathTest to "threadsafe" execution instead of "fast".
 */

#include "Kokkos_Core.hpp"  // Kokkos::initialize, Kokkos::finalize

#include "gtest/gtest.h"  // GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST, RUN_ALL_TESTS, ::testing::{InitGoogleTest, GTEST_FLAG}

#include "main.hpp"  // GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST definitions

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // initialize Kokkos
    Kokkos::initialize(argc, argv);

    // prevent problems with fork() in the presence of multiple threads
    // https://github.com/google/googletest/blob/main/docs/advanced.md#death-tests-and-threads
    // NOTE: may reduce performance of the (death) tests
#if !defined(_WIN32)
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
#endif

    // run all tests
    const int return_code = RUN_ALL_TESTS();

    // finalize Kokkos
    Kokkos::finalize();

    return return_code;
}
