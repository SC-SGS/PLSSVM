/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functions in the string utility header.
 */

#include "plssvm/detail/string_utility.hpp"

#include "naming.hpp"  // naming::pretty_print_escaped_string

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TEST, ASSERT_EQ, EXPECT_EQ

#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::make_tuple
#include <vector>       // std::vector

TEST(StringUtility, starts_with_string) {
    EXPECT_TRUE(plssvm::detail::starts_with("abc", "abc"));
    EXPECT_TRUE(plssvm::detail::starts_with("abc", "ab"));
    EXPECT_FALSE(plssvm::detail::starts_with("abc", "abcd"));
    EXPECT_FALSE(plssvm::detail::starts_with("abc", "bc"));
}
TEST(StringUtility, starts_with_char) {
    EXPECT_TRUE(plssvm::detail::starts_with("abc", 'a'));
    EXPECT_FALSE(plssvm::detail::starts_with("abc", 'c'));
    EXPECT_FALSE(plssvm::detail::starts_with("abc", 'd'));
}

TEST(StringUtility, ends_with_string) {
    EXPECT_TRUE(plssvm::detail::ends_with("abc", "abc"));
    EXPECT_FALSE(plssvm::detail::ends_with("abc", "ab"));
    EXPECT_FALSE(plssvm::detail::ends_with("abc", "abcd"));
    EXPECT_TRUE(plssvm::detail::ends_with("abc", "bc"));
}
TEST(StringUtility, ends_with_char) {
    EXPECT_FALSE(plssvm::detail::ends_with("abc", 'a'));
    EXPECT_TRUE(plssvm::detail::ends_with("abc", 'c'));
    EXPECT_FALSE(plssvm::detail::ends_with("abc", 'd'));
}

TEST(StringUtility, contains_string) {
    EXPECT_TRUE(plssvm::detail::contains("abc", "abc"));
    EXPECT_TRUE(plssvm::detail::contains("abc", "ab"));
    EXPECT_FALSE(plssvm::detail::contains("abc", "abcd"));
    EXPECT_TRUE(plssvm::detail::contains("abc", "bc"));
}
TEST(StringUtility, contains_char) {
    EXPECT_TRUE(plssvm::detail::contains("abc", 'a'));
    EXPECT_TRUE(plssvm::detail::contains("abc", 'c'));
    EXPECT_FALSE(plssvm::detail::contains("abc", 'd'));
}

class StringUtilityBase : public ::testing::TestWithParam<std::tuple<std::string, std::string_view>> {};

// test trim left only
class StringUtilityTrimLeft : public StringUtilityBase {};
TEST_P(StringUtilityTrimLeft, trim_left) {
    auto [input, output] = GetParam();
    EXPECT_EQ(plssvm::detail::trim_left(input), output);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(StringUtility, StringUtilityTrimLeft, ::testing::Values(
                std::make_tuple("", ""), std::make_tuple("abc", "abc"),
                std::make_tuple("  abc", "abc"), std::make_tuple("abc   ", "abc   "),
                std::make_tuple(" abc  ", "abc  "), std::make_tuple(" a b c ", "a b c ")),
                naming::pretty_print_escaped_string<StringUtilityTrimLeft>);
// clang-format on

// test trim right only
class StringUtilityTrimRight : public StringUtilityBase {};
TEST_P(StringUtilityTrimRight, trim_right) {
    auto [input, output] = GetParam();
    EXPECT_EQ(plssvm::detail::trim_right(input), output);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(StringUtility, StringUtilityTrimRight, ::testing::Values(
                std::make_tuple("", ""), std::make_tuple("abc", "abc"),
                std::make_tuple("  abc", "  abc"), std::make_tuple("abc   ", "abc"),
                std::make_tuple(" abc  ", " abc"), std::make_tuple(" a b c ", " a b c")),
                naming::pretty_print_escaped_string<StringUtilityTrimRight>);
// clang-format on

// test trim
class StringUtilityTrim : public StringUtilityBase {};
TEST_P(StringUtilityTrim, trim) {
    auto [input, output] = GetParam();
    EXPECT_EQ(plssvm::detail::trim(input), output);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(StringUtility, StringUtilityTrim, ::testing::Values(
                std::make_tuple("", ""), std::make_tuple("abc", "abc"),
                std::make_tuple("  abc", "abc"), std::make_tuple("abc   ", "abc"),
                std::make_tuple(" abc  ", "abc"), std::make_tuple(" a b c ", "a b c")),
                naming::pretty_print_escaped_string<StringUtilityTrim>);
// clang-format on

// test conversion to lower case
class StringUtilityConvertLowerCase : public StringUtilityBase {};
TEST_P(StringUtilityConvertLowerCase, to_lower_case) {
    auto [input, output] = GetParam();
    EXPECT_EQ(plssvm::detail::to_lower_case(input), output);
}
TEST_P(StringUtilityConvertLowerCase, as_lower_case) {
    auto [input, output] = GetParam();
    EXPECT_EQ(plssvm::detail::as_lower_case(input), output);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(StringUtility, StringUtilityConvertLowerCase, ::testing::Values(
                std::make_tuple("", ""), std::make_tuple("abc", "abc"),
                std::make_tuple("ABC", "abc"), std::make_tuple(" AbC 1", " abc 1")),
                naming::pretty_print_escaped_string<StringUtilityConvertLowerCase>);
// clang-format on

// test conversion to upper case
class StringUtilityConvertUpperCase : public StringUtilityBase {};
TEST_P(StringUtilityConvertUpperCase, to_upper_case) {
    auto [input, output] = GetParam();
    EXPECT_EQ(plssvm::detail::to_upper_case(input), output);
}
TEST_P(StringUtilityConvertUpperCase, as_upper_case) {
    auto [input, output] = GetParam();
    EXPECT_EQ(plssvm::detail::as_upper_case(input), output);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(StringUtility, StringUtilityConvertUpperCase, ::testing::Values(
                std::make_tuple("", ""), std::make_tuple("abc", "ABC"),
                std::make_tuple("ABC", "ABC"), std::make_tuple(" AbC 1", " ABC 1")),
                naming::pretty_print_escaped_string<StringUtilityConvertUpperCase>);
// clang-format on

// test replace_all
class StringUtilityReplace : public ::testing::TestWithParam<std::tuple<std::string, std::string_view, std::string_view, std::string_view>> {};
TEST_P(StringUtilityReplace, replace_all) {
    auto [input, what, with, output] = GetParam();
    EXPECT_EQ(plssvm::detail::replace_all(input, what, with), output);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(StringUtility, StringUtilityReplace, ::testing::Values(
                std::make_tuple("", "", "", ""), std::make_tuple("aaa", "a", "b", "bbb"),
                std::make_tuple("aaa", "", "b", "aaa"), std::make_tuple("aaa", "b", "c", "aaa"),
                std::make_tuple("aaa", "aa", "b", "ba"), std::make_tuple("a a b c d aa", "a", "", "  b c d "),
                std::make_tuple("a", "aa", "b", "a")),
                naming::pretty_print_replace<StringUtilityReplace>);
// clang-format on

TEST(StringUtility, split_default_delimiter) {
    // split string using the default delimiter
    const std::string string_to_split = "1.5 2.0 -3.5 4.0 5.0 -6.0  7.5";

    const std::vector<std::string_view> split_correct = { "1.5", "2.0", "-3.5", "4.0", "5.0", "-6.0", "", "7.5" };
    const std::vector<std::string_view> split = plssvm::detail::split(string_to_split);
    ASSERT_EQ(split.size(), split_correct.size());
    for (typename std::vector<double>::size_type i = 0; i < split_correct.size(); ++i) {
        EXPECT_EQ(split[i], split_correct[i]) << fmt::format("pos: {}, split: {}, correct: {}", i, split[i], split_correct[i]);
    }
}
TEST(StringUtility, split_custom_delimiter) {
    // split string using a custom delimiter
    const std::string string_to_split = "1.5,2.0,-3.5,4.0,5.0,-6.0,,7.5";

    const std::vector<std::string_view> split_correct = { "1.5", "2.0", "-3.5", "4.0", "5.0", "-6.0", "", "7.5" };
    const std::vector<std::string_view> split = plssvm::detail::split(string_to_split, ',');
    ASSERT_EQ(split.size(), split_correct.size());
    for (typename std::vector<double>::size_type i = 0; i < split_correct.size(); ++i) {
        EXPECT_EQ(split[i], split_correct[i]) << fmt::format("pos: {}, split: {}, correct: {}", i, split[i], split_correct[i]);
    }
}
TEST(StringUtility, split_single_value) {
    // split string containing a single value
    const std::vector<std::string_view> split = plssvm::detail::split("42");
    ASSERT_EQ(split.size(), 1);
    EXPECT_EQ(split.front(), "42") << fmt::format("split: {}, correct: 42", split.front());
}
TEST(StringUtility, split_empty_string) {
    // split the empty string
    const std::vector<std::string_view> split = plssvm::detail::split("");
    EXPECT_TRUE(split.empty());
}
