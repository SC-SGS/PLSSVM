/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Tests for the custom exception classes related to the CUDA backend.
*/

#include "plssvm/backends/CUDA/exceptions.hpp"  // plssvm::cuda::backend_exception

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::split

#include "../../custom_test_macros.hpp"  // EXPECT_THROW_WHAT

#include "gmock/gmock-matchers.h"  // ::testing::{HasSubstr, ContainsRegex}
#include "gtest/gtest.h"           // TEST, EXPECT_EQ, EXPECT_THAT, EXPECT_TRUE, ASSERT_EQ, ::testing::Test, ::testing::Types

#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

// helper function returning an exception used to be able to name the source location function
plssvm::cuda::backend_exception dummy(const std::string &msg) {
   return plssvm::cuda::backend_exception{ msg };
}

// check whether throwing exceptions works as intended
TEST(OpenMPExceptions, throwing_excpetion) {
   // throw the specified exception
   EXPECT_THROW_WHAT(throw plssvm::cuda::backend_exception{ "exception message" }, plssvm::cuda::backend_exception, "exception message");
}

// check whether the source location information are populated correctly
TEST(OpenMPExceptions, exception_source_location) {
   const plssvm::cuda::backend_exception exc = dummy("exception message");

   EXPECT_EQ(exc.loc().file_name(), __FILE__);
   EXPECT_THAT(exc.loc().function_name(), ::testing::HasSubstr("dummy"));
   EXPECT_EQ(exc.loc().line(), 26);   // attention: hardcoded line!
   EXPECT_EQ(exc.loc().column(), 0);  // attention: always 0!
}

// check whether what message including the source location information is assembled correctly
TEST(OpenMPExceptions, exception_what_with_source_location) {
   const plssvm::cuda::backend_exception exc = dummy("exception message");

   // get exception message with source location information split into a vector of separate lines
   const std::string what = exc.what_with_loc();
   const std::vector<std::string_view> what_lines = plssvm::detail::split(what, '\n');

   // check the number of lines in the "what" message
   ASSERT_EQ(what_lines.size(), 5);

   // check the "what" message content
   EXPECT_EQ(what_lines[0], "exception message");
   EXPECT_EQ(what_lines[1], "cuda::backend_exception thrown:");
   EXPECT_EQ(what_lines[2], "  in file      " __FILE__);
   EXPECT_THAT(what_lines[3], ::testing::ContainsRegex("  in function  .*dummy.*"));
   EXPECT_EQ(what_lines[4], "  @ line       26");
}