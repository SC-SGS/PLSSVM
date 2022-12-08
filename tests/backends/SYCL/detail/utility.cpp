/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the SYCL backends.
 */

#include "plssvm/target_platforms.hpp"

#include "sycl/sycl.hpp"

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_NE, EXPECT_NO_THROW, EXPECT_FALSE

#if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    #include "plssvm/backends/SYCL/DPCPP/detail/utility.hpp"

TEST(DPCPPUtility, get_device_list) {
    const auto &[queues, actual_target] = plssvm::dpcpp::detail::get_device_list(plssvm::target_platform::automatic);
    // at least one queue must be provided
    EXPECT_FALSE(queues.empty());
    // the returned target must not be the automatic one
    EXPECT_NE(actual_target, plssvm::target_platform::automatic);
}
#endif

#if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
    #include "plssvm/backends/SYCL/hipSYCL//detail/utility.hpp"

TEST(hipSYCLUtility, get_device_list) {
    const auto &[queues, actual_target] = plssvm::hipsycl::detail::get_device_list(plssvm::target_platform::automatic);
    // at least one queue must be provided
    EXPECT_FALSE(queues.empty());
    // the returned target must not be the automatic one
    EXPECT_NE(actual_target, plssvm::target_platform::automatic);
}
#endif
