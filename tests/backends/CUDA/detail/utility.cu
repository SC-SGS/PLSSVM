/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the CUDA backend.
 */

#include "plssvm/backends/CUDA/detail/utility.cuh"

#include "plssvm/backends/CUDA/exceptions.hpp"  // plssvm::cuda::backend_exception
#include "plssvm/backends/execution_range.hpp"  // plssvm::detail::dim_type

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_THROW_WHAT_MATCHER

#include "fmt/format.h"   // fmt::format
#include "gmock/gmock.h"  // ::testing::StartsWith
#include "gtest/gtest.h"  // TEST, EXPECT_GE, EXPECT_NO_THROW

#include <regex>  // std::regex, std::regex::extended, std::regex_match

TEST(CUDAUtility, error_check) {
    // cudaSuccess must not throw
    EXPECT_NO_THROW(PLSSVM_CUDA_ERROR_CHECK(cudaSuccess));

    // any other code must throw
    EXPECT_THROW_WHAT_MATCHER(PLSSVM_CUDA_ERROR_CHECK(cudaErrorInvalidValue),
                              plssvm::cuda::backend_exception,
                              ::testing::StartsWith("CUDA assert 'cudaErrorInvalidValue' (1):"));
}

TEST(CUDAUtility, dim_type_to_native) {
    // create a dim_type
    constexpr plssvm::detail::dim_type dim{ 128ull, 64ull, 32ull };

    // convert it to a CUDA dim3
    const dim3 native_dim = plssvm::cuda::detail::dim_type_to_native(dim);

    // check values for correctness
    EXPECT_EQ(native_dim.x, dim.x);
    EXPECT_EQ(native_dim.y, dim.y);
    EXPECT_EQ(native_dim.z, dim.z);
}

TEST(CUDAUtility, get_device_count) {
    // must not return a negative number
    EXPECT_GE(plssvm::cuda::detail::get_device_count(), 0);
}

TEST(CUDAUtility, set_device) {
    // exception must be thrown if an illegal device ID has been provided
    EXPECT_THROW_WHAT(plssvm::cuda::detail::set_device(plssvm::cuda::detail::get_device_count()),
                      plssvm::cuda::backend_exception,
                      fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}!", plssvm::cuda::detail::get_device_count(), plssvm::cuda::detail::get_device_count()));
}

TEST(CUDAUtility, device_synchronize) {
    // exception must be thrown if an illegal device ID has been provided
    EXPECT_THROW_WHAT(plssvm::cuda::detail::device_synchronize(plssvm::cuda::detail::get_device_count()),
                      plssvm::cuda::backend_exception,
                      fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}!", plssvm::cuda::detail::get_device_count(), plssvm::cuda::detail::get_device_count()));
}

TEST(CUDAUtility, get_runtime_version) {
    const std::regex reg{ "[0-9]+\\.[0-9]+", std::regex::extended };
    EXPECT_TRUE(std::regex_match(plssvm::cuda::detail::get_runtime_version(), reg));
}
