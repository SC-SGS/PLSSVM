/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the HPX backend.
 */

#include "plssvm/backends/HPX/detail/utility.hpp"

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_NE, EXPECT_NO_THROW, EXPECT_FALSE

#include <string>  // std::string

TEST(HPXUtility, get_num_threads) {
    EXPECT_GT(plssvm::hpx::detail::get_num_threads(), 0);
}

TEST(HPXUtility, get_hpx_version) {
    EXPECT_FALSE(plssvm::hpx::detail::get_hpx_version().empty());
}
