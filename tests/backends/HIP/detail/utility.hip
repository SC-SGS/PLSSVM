/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the HIP backend.
 */

#include "plssvm/backends/HIP/detail/utility.hip.hpp"
// main header

#include "plssvm/backends/execution_range.hpp"  // plssvm::detail::dim_type
#include "plssvm/backends/HIP/exceptions.hpp"   // plssvm::hip::backend_exception

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_THROW_WHAT_MATCHER

#include "fmt/format.h"   // fmt::format
#include "gmock/gmock.h"  // ::testing::StartsWith
#include "gtest/gtest.h"  // TEST, EXPECT_GE, EXPECT_NO_THROW

#include <regex>  // std::regex, std::regex::extended, std::regex_match

TEST(HIPUtility, error_check) {
    // hipSuccess must not throw
    EXPECT_NO_THROW(PLSSVM_HIP_ERROR_CHECK(hipSuccess));

    // any other code must throw
#if defined(PLSSVM_HIP_BACKEND_USE_HIP_RUNTIME)
    EXPECT_THROW_WHAT_MATCHER(PLSSVM_HIP_ERROR_CHECK(hipErrorInvalidValue),
                              plssvm::hip::backend_exception,
                              ::testing::StartsWith("HIP assert 'hipErrorInvalidValue' (1):"));
#elif defined(PLSSVM_HIP_BACKEND_USE_CUDA_RUNTIME)
    EXPECT_THROW_WHAT_MATCHER(PLSSVM_HIP_ERROR_CHECK(hipErrorInvalidValue),
                              plssvm::hip::backend_exception,
                              ::testing::StartsWith("HIP assert 'cudaErrorInvalidValue' (1):"));
#endif
}

TEST(HIPUtility, dim_type_to_native) {
    // create a dim_type
    constexpr plssvm::detail::dim_type dim{ 128ull, 64ull, 32ull };

    // convert it to a HIP dim3
    const dim3 native_dim = plssvm::hip::detail::dim_type_to_native(dim);

    // check values for correctness
    EXPECT_EQ(native_dim.x, dim.x);
    EXPECT_EQ(native_dim.y, dim.y);
    EXPECT_EQ(native_dim.z, dim.z);
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

TEST(HIPUtility, get_runtime_version) {
    const std::regex reg{ "[0-9]+\\.[0-9]+", std::regex::extended };
    EXPECT_TRUE(std::regex_match(plssvm::hip::detail::get_runtime_version(), reg));
}

TEST(HIPUtility, get_runtime) {
    const std::regex reg{ "HIP|CUDA", std::regex::extended };
    EXPECT_TRUE(std::regex_match(plssvm::hip::detail::get_runtime(), reg));
}
