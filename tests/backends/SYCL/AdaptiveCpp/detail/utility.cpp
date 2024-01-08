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

#include "plssvm/backends/SYCL/detail/utility.hpp"          // plssvm::sycl::detail::calculate_execution_range
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "sycl/sycl.hpp"  // sycl::range, sycl::nd_range

#include "tests/naming.hpp"  // naming::pretty_print_execution_range

#include "gtest/gtest.h"  // TEST, EXPECT_NE, EXPECT_FALSE, TEST_P, INSTANTIATE_TEST_SUITE_P, ::testing::TestWithParam, ::testing::Values

#include <regex>   // std::regex, std::regex::extended, std::regex_match
#include <string>  // std::string
#include <tuple>   // std::tuple, std::make_tuple

TEST(AdaptiveCppUtility, get_device_list) {
    const auto &[queues, actual_target] = plssvm::adaptivecpp::detail::get_device_list(plssvm::target_platform::automatic);
    // at least one queue must be provided
    EXPECT_FALSE(queues.empty());
    // the returned target must not be the automatic one
    EXPECT_NE(actual_target, plssvm::target_platform::automatic);
}

TEST(AdaptiveCppUtility, get_adaptivecpp_version_short) {
    const std::regex reg{ "[0-9]+\\.[0-9]+\\.[0-9]+", std::regex::extended };
    EXPECT_TRUE(std::regex_match(plssvm::adaptivecpp::detail::get_adaptivecpp_version_short(), reg));
}

TEST(AdaptiveCppUtility, get_adaptivecpp_version) {
    const std::string version = plssvm::adaptivecpp::detail::get_adaptivecpp_version();
    EXPECT_FALSE(version.empty());
}

class AdaptiveCppDetailUtility : public ::testing::TestWithParam<std::tuple<::sycl::range<2>, plssvm::sycl::kernel_invocation_type, ::sycl::nd_range<2>>> {
  public:
    /**
     * @brief The correct block size equals a two-dimensional sycl::range of size `plssvm::THREAD_BLOCK_SIZE`.
     * @return the correct SYCL block size (`[[nodiscard]]`)
     */
    [[nodiscard]] static ::sycl::range<2> correct_block() {
        return { plssvm::THREAD_BLOCK_SIZE, plssvm::THREAD_BLOCK_SIZE };
    }

    /**
     * @brief The correct grid size when using nd_range kernels.
     * @return the correct SYCL grid size (`[[nodiscard]]`)
     */
    [[nodiscard]] static ::sycl::range<2> correct_grid_nd_range(const unsigned long long x, const unsigned long long y) {
        const auto block = AdaptiveCppDetailUtility::correct_block();
        return { static_cast<std::size_t>(std::ceil(static_cast<double>(x) / static_cast<double>(block[0] * plssvm::INTERNAL_BLOCK_SIZE))) * block[0],
                 static_cast<std::size_t>(std::ceil(static_cast<double>(y) / static_cast<double>(block[1] * plssvm::INTERNAL_BLOCK_SIZE))) * block[1] };
    }

    /**
     * @brief The correct block size when using hierarchical or scoped parallelism kernels.
     * @return the correct SYCL grid size (`[[nodiscard]]`)
     */
    [[nodiscard]] static ::sycl::range<2> correct_grid(const unsigned long long x, const unsigned long long y) {
        const auto block = AdaptiveCppDetailUtility::correct_block();
        return { static_cast<std::size_t>(std::ceil(static_cast<double>(x) / static_cast<double>(block[0] * plssvm::INTERNAL_BLOCK_SIZE))),
                 static_cast<std::size_t>(std::ceil(static_cast<double>(y) / static_cast<double>(block[1] * plssvm::INTERNAL_BLOCK_SIZE))) };
    }
};

TEST_P(AdaptiveCppDetailUtility, calculate_execution_range) {
    // get generated parameter
    const auto &[iteration_range, invocation, result] = GetParam();

    // calculate execution_range
    const ::sycl::nd_range<2> execution_range = plssvm::sycl::detail::calculate_execution_range(iteration_range, invocation);

    // check values for correctness
    EXPECT_EQ(execution_range.get_global_range()[0], result.get_global_range()[0]);
    EXPECT_EQ(execution_range.get_global_range()[1], result.get_global_range()[1]);
    EXPECT_EQ(execution_range.get_local_range()[0], result.get_local_range()[0]);
    EXPECT_EQ(execution_range.get_local_range()[1], result.get_local_range()[1]);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(AdaptiveCppDetailUtility, AdaptiveCppDetailUtility, ::testing::Values(
        std::make_tuple(::sycl::range<2>{ 42, 42 }, plssvm::sycl::kernel_invocation_type::nd_range, ::sycl::nd_range<2>{ AdaptiveCppDetailUtility::correct_grid_nd_range(42, 42), AdaptiveCppDetailUtility::correct_block() }),
        std::make_tuple(::sycl::range<2>{ 42, 42 }, plssvm::sycl::kernel_invocation_type::hierarchical, ::sycl::nd_range<2>{ AdaptiveCppDetailUtility::correct_grid(42, 42), AdaptiveCppDetailUtility::correct_block() }),
        std::make_tuple(::sycl::range<2>{ 42, 42 }, plssvm::sycl::kernel_invocation_type::scoped, ::sycl::nd_range<2>{ AdaptiveCppDetailUtility::correct_grid(42, 42), AdaptiveCppDetailUtility::correct_block() })),
        naming::pretty_print_execution_range<AdaptiveCppDetailUtility>);
// clang-format on
