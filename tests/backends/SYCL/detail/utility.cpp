/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Tests for the custom utility functions related to the SYCL backend.
*/

#include "plssvm/backends/SYCL/detail/utility.hpp"
#include "plssvm/target_platforms.hpp"

#include "sycl/sycl.hpp"

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT

#include "gtest/gtest.h"           // TEST, EXPECT_EQ, EXPECT_NE, EXPECT_NO_THROW, EXPECT_FALSE

TEST(SYCLUtility, get_device_list) {
    const auto &[queues, actual_target] = plssvm::sycl::detail::get_device_list(plssvm::target_platform::automatic);
    // at least one queue must be provided
    EXPECT_FALSE(queues.empty());
    // the returned target must not be the automatic one
    EXPECT_NE(actual_target, plssvm::target_platform::automatic);
}

// https://stackoverflow.com/questions/20836622/how-do-i-put-some-code-into-multiple-namespaces-without-duplicating-this-code