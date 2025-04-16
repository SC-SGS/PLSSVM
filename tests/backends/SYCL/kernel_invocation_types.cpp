/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different SYCL kernel invocation types.
 */

#include "plssvm/backends/SYCL/kernel_invocation_types.hpp"  // plssvm::sycl::kernel_invocation_type

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gmock/gmock.h"  // EXPECT_THAT; ::testing::Contains
#include "gtest/gtest.h"  // TEST, EXPECT_TRUE, EXPECT_GE

#include <sstream>  // std::istringstream

// check whether the plssvm::sycl::kernel_invocation_type -> std::string conversions are correct
TEST(SYCLKernelInvocationType, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::kernel_invocation_type::automatic, "automatic");
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::kernel_invocation_type::basic, "basic");
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::kernel_invocation_type::work_group, "work_group");
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::kernel_invocation_type::hierarchical, "hierarchical");
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::kernel_invocation_type::scoped, "scoped");
}

TEST(SYCLKernelInvocationType, to_string_unknown) {
    // check conversions to std::string from unknown file_format_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::sycl::kernel_invocation_type>(5), "unknown");
}

// check whether the std::string -> plssvm::sycl::kernel_invocation_type conversions are correct
TEST(SYCLKernelInvocationType, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("automatic", plssvm::sycl::kernel_invocation_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("AUTOMATIC", plssvm::sycl::kernel_invocation_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("auto", plssvm::sycl::kernel_invocation_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("AUTO", plssvm::sycl::kernel_invocation_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("basic", plssvm::sycl::kernel_invocation_type::basic);
    EXPECT_CONVERSION_FROM_STRING("BASIC", plssvm::sycl::kernel_invocation_type::basic);
    EXPECT_CONVERSION_FROM_STRING("work_group", plssvm::sycl::kernel_invocation_type::work_group);
    EXPECT_CONVERSION_FROM_STRING("WORK-GROUP", plssvm::sycl::kernel_invocation_type::work_group);
    EXPECT_CONVERSION_FROM_STRING("nd_range", plssvm::sycl::kernel_invocation_type::work_group);
    EXPECT_CONVERSION_FROM_STRING("ND-RANGE", plssvm::sycl::kernel_invocation_type::work_group);
    EXPECT_CONVERSION_FROM_STRING("hierarchical", plssvm::sycl::kernel_invocation_type::hierarchical);
    EXPECT_CONVERSION_FROM_STRING("HIERARCHICAL", plssvm::sycl::kernel_invocation_type::hierarchical);
    EXPECT_CONVERSION_FROM_STRING("scoped", plssvm::sycl::kernel_invocation_type::scoped);
    EXPECT_CONVERSION_FROM_STRING("SCOPED", plssvm::sycl::kernel_invocation_type::scoped);
}

TEST(SYCLKernelInvocationType, from_string_unknown) {
    // foo isn't a valid file_format_type
    std::istringstream input{ "foo" };
    plssvm::sycl::kernel_invocation_type invocation_type{};
    input >> invocation_type;
    EXPECT_TRUE(input.fail());
}

TEST(SYCLKernelInvocationType, minimal_available_sycl_kernel_invocation_types) {
    const std::vector<plssvm::sycl::kernel_invocation_type> invocation_type = plssvm::sycl::list_available_sycl_kernel_invocation_types();

    // at least three must be available (automatic, basic, and work_group)!
    EXPECT_GE(invocation_type.size(), 3);

    // check for the kernel invocation types that must always be present
    EXPECT_THAT(invocation_type, ::testing::Contains(plssvm::sycl::kernel_invocation_type::automatic));
    EXPECT_THAT(invocation_type, ::testing::Contains(plssvm::sycl::kernel_invocation_type::basic));
    EXPECT_THAT(invocation_type, ::testing::Contains(plssvm::sycl::kernel_invocation_type::work_group));
}
