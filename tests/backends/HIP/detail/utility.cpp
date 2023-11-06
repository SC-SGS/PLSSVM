/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the HIP backend.
 */

#include "plssvm/backends/HIP/detail/utility.hip.hpp"  // PLSSVM_HIP_ERROR_CHECK, plssvm::hip::detail::{gpu_assert, get_device_count, set_device, device_synchronize}

#include "plssvm/backends/HIP/exceptions.hpp"  // plssvm::hip::backend_exception

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_THROW_WHAT_MATCHER

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::StartsWith
#include "gtest/gtest.h"           // TEST, EXPECT_GE, EXPECT_NO_THROW

#if __has_include("hip/hip_runtime.h") && __has_include("hip/hip_runtime_api.h")

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

TEST(HIPUtility, gpu_assert) {
    // hipSuccess must not throw
    EXPECT_NO_THROW(PLSSVM_HIP_ERROR_CHECK(hipSuccess));
    EXPECT_NO_THROW(plssvm::hip::detail::gpu_assert(hipSuccess));

    // any other code must throw
    EXPECT_THROW_WHAT_MATCHER(PLSSVM_HIP_ERROR_CHECK(hipErrorInvalidValue),
                              plssvm::hip::backend_exception,
                              ::testing::StartsWith("HIP assert 'hipErrorInvalidValue' (1):"));
    EXPECT_THROW_WHAT_MATCHER(plssvm::hip::detail::gpu_assert(hipErrorInvalidValue),
                              plssvm::hip::backend_exception,
                              ::testing::StartsWith("HIP assert 'hipErrorInvalidValue' (1):"));
}

TEST(HIPUtility, get_device_count) {
    // must not return a negative number
    EXPECT_GE(plssvm::hip::detail::get_device_count(), 0);
}

TEST(HIPUtility, set_device) {
    // exception must be thrown if an illegal device ID has been provided
    EXPECT_THROW_WHAT(plssvm::hip::detail::set_device(plssvm::hip::detail::get_device_count()),
                      plssvm::hip::backend_exception,
                      fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}!", plssvm::hip::detail::get_device_count(), plssvm::hip::detail::get_device_count()));
}

TEST(HIPUtility, device_synchronize) {
    // exception must be thrown if an illegal device ID has been provided
    EXPECT_THROW_WHAT(plssvm::hip::detail::device_synchronize(plssvm::hip::detail::get_device_count()),
                      plssvm::hip::backend_exception,
                      fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}!", plssvm::hip::detail::get_device_count(), plssvm::hip::detail::get_device_count()));
}

#endif