/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the SYCL backends with DPC++ as SYCL implementation.
 */

#include "plssvm/backends/SYCL/DPCPP/detail/utility.hpp"

#include "plssvm/backends/execution_range.hpp"  // plssvm::detail::dim_type
#include "plssvm/target_platforms.hpp"          // plssvm::target_platform

#include "sycl/sycl.hpp"  // sycl::range

#include "gtest/gtest.h"  // TEST, EXPECT_NE, EXPECT_FALSE

#include <regex>   // std::regex, std::regex::extended, std::regex_match
#include <string>  // std::string

TEST(DPCPPUtility, dim_type_to_native_1) {
    // create a dim_type
    constexpr plssvm::detail::dim_type dim{ 128ull, 64ull, 32ull };

    // convert it to a SYCL sycl::range
    const ::sycl::range native_dim = plssvm::dpcpp::detail::dim_type_to_native<1>(dim);

    // check values for correctness
    ::testing::StaticAssertTypeEq<std::remove_const_t<decltype(native_dim)>, ::sycl::range<1>>();
    EXPECT_EQ(native_dim[0], dim.x);
}

TEST(DPCPPUtility, dim_type_to_native_2) {
    // create a dim_type
    constexpr plssvm::detail::dim_type dim{ 128ull, 64ull, 32ull };

    // convert it to a SYCL sycl::range
    const ::sycl::range native_dim = plssvm::dpcpp::detail::dim_type_to_native<2>(dim);

    // check values for correctness -> account for inversed iteration range in SYCL!
    ::testing::StaticAssertTypeEq<std::remove_const_t<decltype(native_dim)>, ::sycl::range<2>>();
    EXPECT_EQ(native_dim[0], dim.y);
    EXPECT_EQ(native_dim[1], dim.x);
}

TEST(DPCPPUtility, dim_type_to_native_3) {
    // create a dim_type
    constexpr plssvm::detail::dim_type dim{ 128ull, 64ull, 32ull };

    // convert it to a SYCL sycl::range
    const ::sycl::range native_dim = plssvm::dpcpp::detail::dim_type_to_native<3>(dim);

    // check values for correctness -> account for inversed iteration range in SYCL!
    ::testing::StaticAssertTypeEq<std::remove_const_t<decltype(native_dim)>, ::sycl::range<3>>();
    EXPECT_EQ(native_dim[0], dim.z);
    EXPECT_EQ(native_dim[1], dim.y);
    EXPECT_EQ(native_dim[2], dim.x);
}

TEST(DPCPPUtility, get_device_list) {
    const auto &[queues, actual_target] = plssvm::dpcpp::detail::get_device_list(plssvm::target_platform::automatic);
    // at least one queue must be provided
    EXPECT_FALSE(queues.empty());
    // the returned target must not be the automatic one
    EXPECT_NE(actual_target, plssvm::target_platform::automatic);
}

TEST(AdaptiveCppUtility, get_dpcpp_version) {
    const std::regex reg{ "[0-9]+\\.[0-9]+\\.[0-9]+", std::regex::extended };
    EXPECT_TRUE(std::regex_match(plssvm::dpcpp::detail::get_dpcpp_version(), reg));
}

TEST(AdaptiveCppUtility, get_dpcpp_timestamp_version) {
    const std::string version = plssvm::dpcpp::detail::get_dpcpp_timestamp_version();
    EXPECT_FALSE(version.empty());
}
