/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the SYCL backends with AdaptiveCpp as SYCL implementation.
 */

#include "plssvm/backends/SYCL/AdaptiveCpp/detail/utility.hpp"  // plssvm::adaptivecpp::detail::get_device_list
#include "plssvm/target_platforms.hpp"                          // plssvm::target_platform

#include "gtest/gtest.h"  // TEST, EXPECT_NE, EXPECT_FALSE

TEST(AdaptiveCppUtility, get_device_list) {
    const auto &[queues, actual_target] = plssvm::adaptivecpp::detail::get_device_list(plssvm::target_platform::automatic);
    // at least one queue must be provided
    EXPECT_FALSE(queues.empty());
    // the returned target must not be the automatic one
    EXPECT_NE(actual_target, plssvm::target_platform::automatic);
}
