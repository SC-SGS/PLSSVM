/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different SYCL kernel invocation types.
 */

#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl::kernel_invocation_type

#include "custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gtest/gtest.h"  // TEST, EXPECT_TRUE

#include <sstream>  // std::istringstream

// check whether the plssvm::sycl::kernel_invocation_type -> std::string conversions are correct
TEST(SYCLKernelInvocationType, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::kernel_invocation_type::automatic, "automatic");
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::kernel_invocation_type::nd_range, "nd_range");
}
TEST(SYCLKernelInvocationType, to_string_unknown) {
    // check conversions to std::string from unknown file_format_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::sycl::kernel_invocation_type>(3), "unknown");
}

// check whether the std::string -> plssvm::sycl::kernel_invocation_type conversions are correct
TEST(SYCLKernelInvocationType, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("automatic", plssvm::sycl::kernel_invocation_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("AUTOMATIC", plssvm::sycl::kernel_invocation_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("nd_range", plssvm::sycl::kernel_invocation_type::nd_range);
    EXPECT_CONVERSION_FROM_STRING("ND_RANGE", plssvm::sycl::kernel_invocation_type::nd_range);
}
TEST(SYCLKernelInvocationType, from_string_unknown) {
    // foo isn't a valid file_format_type
    std::istringstream input{ "foo" };
    plssvm::sycl::kernel_invocation_type invocation_type{};
    input >> invocation_type;
    EXPECT_TRUE(input.fail());
}
