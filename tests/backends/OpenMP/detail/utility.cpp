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

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_NE, EXPECT_NO_THROW, EXPECT_FALSE

#include <regex>   // std::regex, std::regex::extended, std::regex_match
#include <string>  // std::string

TEST(OpenMPUtility, get_num_threads) {
    EXPECT_GT(plssvm::openmp::detail::get_num_threads(), 0);
}

TEST(OpenMPUtility, get_openmp_version) {
    EXPECT_FALSE(plssvm::openmp::detail::get_openmp_version().empty());
}
