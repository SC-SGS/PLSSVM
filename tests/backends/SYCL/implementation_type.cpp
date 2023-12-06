/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different SYCL implementation types.
 */

#include "plssvm/backends/SYCL/implementation_type.hpp"  // plssvm::sycl::implementation_type

#include "custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gtest/gtest.h"  // TEST, EXPECT_TRUE

#include <sstream>  // std::istringstream

// check whether the plssvm::sycl::implementation_type -> std::string conversions are correct
TEST(SYCLImplementationType, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::implementation_type::automatic, "automatic");
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::implementation_type::adaptivecpp, "adaptivecpp");
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::implementation_type::dpcpp, "dpcpp");
}
TEST(SYCLImplementationType, to_string_unknown) {
    // check conversions to std::string from unknown file_format_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::sycl::implementation_type>(3), "unknown");
}

// check whether the std::string -> plssvm::sycl::implementation_type conversions are correct
TEST(SYCLImplementationType, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("automatic", plssvm::sycl::implementation_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("AUTOMATIC", plssvm::sycl::implementation_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("AdaptiveCpp", plssvm::sycl::implementation_type::adaptivecpp);
    EXPECT_CONVERSION_FROM_STRING("ADAPTIVECPP", plssvm::sycl::implementation_type::adaptivecpp);
    EXPECT_CONVERSION_FROM_STRING("ACPP", plssvm::sycl::implementation_type::adaptivecpp);
    EXPECT_CONVERSION_FROM_STRING("dpcpp", plssvm::sycl::implementation_type::dpcpp);
    EXPECT_CONVERSION_FROM_STRING("DPCPP", plssvm::sycl::implementation_type::dpcpp);
    EXPECT_CONVERSION_FROM_STRING("dpc++", plssvm::sycl::implementation_type::dpcpp);
    EXPECT_CONVERSION_FROM_STRING("DPC++", plssvm::sycl::implementation_type::dpcpp);
}
TEST(SYCLImplementationType, from_string_unknown) {
    // foo isn't a valid file_format_type
    std::istringstream input{ "foo" };
    plssvm::sycl::implementation_type impl{};
    input >> impl;
    EXPECT_TRUE(input.fail());
}

TEST(SYCLImplementationType, minimal_available_sycl_implementation_type) {
    const std::vector<plssvm::sycl::implementation_type> implementation_type = plssvm::sycl::list_available_sycl_implementations();

    // at least one must be available (automatic)!
    EXPECT_GE(implementation_type.size(), 1);

    // the automatic backend must always be present
    EXPECT_THAT(implementation_type, ::testing::Contains(plssvm::sycl::implementation_type::automatic));
}