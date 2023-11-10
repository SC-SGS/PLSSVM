/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Generic exception tests for all backends to reduce code duplication.
 */

#ifndef PLSSVM_TESTS_BACKENDS_GENERIC_EXCEPTIONS_TESTS_HPP_
#define PLSSVM_TESTS_BACKENDS_GENERIC_EXCEPTIONS_TESTS_HPP_
#pragma once

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::split

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::{HasSubstr, ContainsRegex}
#include "gtest/gtest.h"           // TEST, EXPECT_EQ, EXPECT_THAT, EXPECT_TRUE, ASSERT_EQ, ::testing::Test

#include <cstdint>      // std::uint_least32_t
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

// helper function returning an exception used to be able to name the source location function
template <typename ExceptionType>
ExceptionType dummy(const std::string &msg) {
    return ExceptionType{ msg };
}

template <typename T>
class Exception : public ::testing::Test {};
TYPED_TEST_SUITE_P(Exception);

// check whether throwing exceptions works as intended
TYPED_TEST_P(Exception, throwing_excpetion) {
    using exception_type = typename TypeParam::exception_type;

    // throw the specified exception
    const auto dummy = []() { throw exception_type{ "exception message" }; };
    EXPECT_THROW_WHAT(dummy(), exception_type, "exception message");
}

// check whether the source location information are populated correctly
TYPED_TEST_P(Exception, exception_source_location) {
    using exception_type = typename TypeParam::exception_type;

    const exception_type exc = dummy<exception_type>("exception message");

    EXPECT_EQ(exc.loc().file_name(), __FILE__);
    EXPECT_THAT(exc.loc().function_name(), ::testing::HasSubstr("dummy"));
    EXPECT_EQ(exc.loc().line(), std::uint_least32_t{ 32 });   // attention: hardcoded line!
    EXPECT_EQ(exc.loc().column(), std::uint_least32_t{ 0 });  // attention: always 0!
}

// check whether what message including the source location information is assembled correctly
TYPED_TEST_P(Exception, exception_what_with_source_location) {
    using exception_type = typename TypeParam::exception_type;
    constexpr std::string_view exception_name = TypeParam::name;

    const exception_type exc = dummy<exception_type>("exception message");

    // get exception message with source location information split into a vector of separate lines
    const std::string what = exc.what_with_loc();
    const std::vector<std::string_view> what_lines = plssvm::detail::split(what, '\n');

    // check the number of lines in the "what" message
    ASSERT_EQ(what_lines.size(), 5);

    // check the "what" message content
    EXPECT_EQ(what_lines[0], "exception message");
    EXPECT_EQ(what_lines[1], fmt::format("{} thrown:", exception_name));
    EXPECT_EQ(what_lines[2], "  in file      " __FILE__);
    EXPECT_THAT(std::string{ what_lines[3] }, ::testing::ContainsRegex("  in function  .*dummy.*"));
    EXPECT_EQ(what_lines[4], "  @ line       32");
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(Exception,
                            throwing_excpetion, exception_source_location, exception_what_with_source_location);
// clang-format on

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_EXCEPTIONS_TESTS_HPP_