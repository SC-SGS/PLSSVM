/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different stdpar implementations.
 */

#include "plssvm/backends/stdpar/implementation_types.hpp"  // plssvm::stdpar::implementation_type

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gtest/gtest.h"  // TEST, EXPECT_TRUE

#include <sstream>  // std::istringstream

// check whether the plssvm::stdpar::implementation_type -> std::string conversions are correct
TEST(stdparImplementationType, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::stdpar::implementation_type::nvhpc, "nvhpc");
    EXPECT_CONVERSION_TO_STRING(plssvm::stdpar::implementation_type::intel_llvm, "intel_llvm");
    EXPECT_CONVERSION_TO_STRING(plssvm::stdpar::implementation_type::adaptivecpp, "adaptivecpp");
    EXPECT_CONVERSION_TO_STRING(plssvm::stdpar::implementation_type::gnu_tbb, "gnu_tbb");
}

TEST(stdparImplementationType, to_string_unknown) {
    // check conversions to std::string from unknown implementation_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::stdpar::implementation_type>(4), "unknown");
}

// check whether the std::string -> plssvm::stdpar::implementation_type conversions are correct
TEST(stdparImplementationType, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("NVHPC", plssvm::stdpar::implementation_type::nvhpc);
    EXPECT_CONVERSION_FROM_STRING("nvcpp", plssvm::stdpar::implementation_type::nvhpc);
    EXPECT_CONVERSION_FROM_STRING("nvc++", plssvm::stdpar::implementation_type::nvhpc);

    EXPECT_CONVERSION_FROM_STRING("intel_llvm", plssvm::stdpar::implementation_type::intel_llvm);
    EXPECT_CONVERSION_FROM_STRING("icpx", plssvm::stdpar::implementation_type::intel_llvm);
    EXPECT_CONVERSION_FROM_STRING("dpcpp", plssvm::stdpar::implementation_type::intel_llvm);
    EXPECT_CONVERSION_FROM_STRING("dpc++", plssvm::stdpar::implementation_type::intel_llvm);

    EXPECT_CONVERSION_FROM_STRING("AdaptiveCpp", plssvm::stdpar::implementation_type::adaptivecpp);
    EXPECT_CONVERSION_FROM_STRING("ACPP", plssvm::stdpar::implementation_type::adaptivecpp);

    EXPECT_CONVERSION_FROM_STRING("gnu_tbb", plssvm::stdpar::implementation_type::gnu_tbb);
    EXPECT_CONVERSION_FROM_STRING("gcc_tbb", plssvm::stdpar::implementation_type::gnu_tbb);
    EXPECT_CONVERSION_FROM_STRING("g++_tbb", plssvm::stdpar::implementation_type::gnu_tbb);
    EXPECT_CONVERSION_FROM_STRING("GNU", plssvm::stdpar::implementation_type::gnu_tbb);
    EXPECT_CONVERSION_FROM_STRING("GCC", plssvm::stdpar::implementation_type::gnu_tbb);
    EXPECT_CONVERSION_FROM_STRING("g++", plssvm::stdpar::implementation_type::gnu_tbb);
}

TEST(stdparImplementationType, from_string_unknown) {
    // foo isn't a valid implementation_type
    std::istringstream input{ "foo" };
    plssvm::stdpar::implementation_type impl{};
    input >> impl;
    EXPECT_TRUE(input.fail());
}
