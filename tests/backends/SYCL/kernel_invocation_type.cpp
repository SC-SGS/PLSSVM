/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different SYCL kernel invocation types.
 */

#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"

#include "utility.hpp"  // util::{convert_to_string, convert_from_string}

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE

#include <sstream>  // std::istringstream

// check whether the plssvm::sycl::kernel_invocation_type -> std::string conversions are correct
TEST(SYCLKernelInvocationType, to_string) {
    // check conversions to std::string
    EXPECT_EQ(util::convert_to_string(plssvm::sycl::kernel_invocation_type::automatic), "automatic");
    EXPECT_EQ(util::convert_to_string(plssvm::sycl::kernel_invocation_type::nd_range), "nd_range");
    EXPECT_EQ(util::convert_to_string(plssvm::sycl::kernel_invocation_type::hierarchical), "hierarchical");
}
TEST(SYCLKernelInvocationType, to_string_unknown) {
    // check conversions to std::string from unknown file_format_type
    EXPECT_EQ(util::convert_to_string(static_cast<plssvm::sycl::kernel_invocation_type>(3)), "unknown");
}

// check whether the std::string -> plssvm::sycl::kernel_invocation_type conversions are correct
TEST(SYCLKernelInvocationType, from_string) {
    // check conversion from std::string
    EXPECT_EQ(util::convert_from_string<plssvm::sycl::kernel_invocation_type>("automatic"), plssvm::sycl::kernel_invocation_type::automatic);
    EXPECT_EQ(util::convert_from_string<plssvm::sycl::kernel_invocation_type>("AUTOMATIC"), plssvm::sycl::kernel_invocation_type::automatic);
    EXPECT_EQ(util::convert_from_string<plssvm::sycl::kernel_invocation_type>("nd_range"), plssvm::sycl::kernel_invocation_type::nd_range);
    EXPECT_EQ(util::convert_from_string<plssvm::sycl::kernel_invocation_type>("ND_RANGE"), plssvm::sycl::kernel_invocation_type::nd_range);
    EXPECT_EQ(util::convert_from_string<plssvm::sycl::kernel_invocation_type>("hierarchical"), plssvm::sycl::kernel_invocation_type::hierarchical);
    EXPECT_EQ(util::convert_from_string<plssvm::sycl::kernel_invocation_type>("HIERARCHICAL"), plssvm::sycl::kernel_invocation_type::hierarchical);
}
TEST(SYCLKernelInvocationType, from_string_unknown) {
    // foo isn't a valid file_format_type
    std::istringstream input{ "foo" };
    plssvm::sycl::kernel_invocation_type invocation_type;
    input >> invocation_type;
    EXPECT_TRUE(input.fail());
}
