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

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_THROW_WHAT_MATCHER

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::StartsWith
#include "gtest/gtest.h"           // TEST, EXPECT_GE, EXPECT_NO_THROW

#if __has_include("cuda_runtime.h")

TEST(CUDAUtility, gpu_assert) {
    // cudaSuccess must not throw
    EXPECT_NO_THROW(PLSSVM_CUDA_ERROR_CHECK(cudaSuccess));
    EXPECT_NO_THROW(plssvm::cuda::detail::gpu_assert(cudaSuccess));

    // any other code must throw
    EXPECT_THROW_WHAT_MATCHER(PLSSVM_CUDA_ERROR_CHECK(cudaErrorInvalidValue),
                              plssvm::cuda::backend_exception,
                              ::testing::StartsWith("CUDA assert 'cudaErrorInvalidValue' (1):"));
    EXPECT_THROW_WHAT_MATCHER(plssvm::cuda::detail::gpu_assert(cudaErrorInvalidValue),
                              plssvm::cuda::backend_exception,
                              ::testing::StartsWith("CUDA assert 'cudaErrorInvalidValue' (1):"));
}

#endif

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