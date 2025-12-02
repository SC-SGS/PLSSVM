/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the JIT info struct necessary for the OpenCL backend.
 */

#include "plssvm/backends/OpenCL/detail/jit_info.hpp"  // plssvm::opencl::detail::jit_info

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::HasSubstr
#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

#include <chrono>  // std::chrono literals
#include <string>  // std::string

// check whether the plssvm::opencl::detail::jit_info::caching_status -> std::string conversions are correct
TEST(OpenCLJITInfoCachingStatus, ToString) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::opencl::detail::jit_info::caching_status::success, "success");
    EXPECT_CONVERSION_TO_STRING(plssvm::opencl::detail::jit_info::caching_status::error_no_cached_files, "no cached files exist (checksum missmatch)");
    EXPECT_CONVERSION_TO_STRING(plssvm::opencl::detail::jit_info::caching_status::error_invalid_number_of_cached_files, "invalid number of cached files");
}

TEST(OpenCLJITInfoCachingStatus, ToStringUnknown) {
    // check conversions to std::string from unknown caching_status
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::opencl::detail::jit_info::caching_status>(3), "unknown");
}

TEST(OpenCLJITInfo, DefaultConstruct) {
    // default construct a JIT info struct
    const plssvm::opencl::detail::jit_info info{};

    EXPECT_FALSE(info.use_ptx_inline);
    EXPECT_EQ(info.cache_state, plssvm::opencl::detail::jit_info::caching_status::success);
    EXPECT_EQ(info.cache_dir, std::string{});
    EXPECT_EQ(info.duration, std::chrono::milliseconds{});
}

TEST(OpenCLJITInfo, Construct) {
    using namespace std::chrono_literals;

    // construct a JIT info struct
    const plssvm::opencl::detail::jit_info info{
        true,
        plssvm::opencl::detail::jit_info::caching_status::error_no_cached_files,
        "jit/file/path",
        250ms
    };

    EXPECT_TRUE(info.use_ptx_inline);
    EXPECT_EQ(info.cache_state, plssvm::opencl::detail::jit_info::caching_status::error_no_cached_files);
    EXPECT_EQ(info.cache_dir, std::string{ "jit/file/path" });
    EXPECT_EQ(info.duration, 250ms);
}

TEST(OpenCLJITInfo, CreateJitReport) {
    using namespace std::chrono_literals;

    // construct a JIT info struct
    const plssvm::opencl::detail::jit_info info{
        true,
        plssvm::opencl::detail::jit_info::caching_status::error_invalid_number_of_cached_files,
        "jit/file/path",
        250ms
    };

    EXPECT_THAT(plssvm::opencl::detail::create_jit_report(info), ::testing::HasSubstr("250ms; PTX inline; cache: invalid number of cached files (jit/file/path)"));
}
