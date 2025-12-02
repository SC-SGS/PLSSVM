/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the OpenMP backend.
 */

#include "plssvm/backends/OpenMP/detail/utility.hpp"

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_FALSE

#include <string>  // std::string

TEST(OpenMPUtility, GetNumThreads) {
    EXPECT_GT(plssvm::openmp::detail::get_num_threads(), 0);
}

TEST(OpenMPUtility, GetOpenMPVersion) {
    EXPECT_FALSE(plssvm::openmp::detail::get_openmp_version().empty());
}
