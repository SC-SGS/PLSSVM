/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the stdpar backend.
 */

#include "plssvm/backends/stdpar/detail/utility.hpp"

#include "gtest/gtest.h"  // TEST, EXPECT_NE

TEST(stdparUtility, get_stdpar_implementation) {
    // add new implementations here
    EXPECT_NE(plssvm::stdpar::detail::get_stdpar_version(), "unknown");
}
