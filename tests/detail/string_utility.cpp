/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functions in the string utility header.
 */

#include "plssvm/detail/string_utility.hpp"                // plssvm::detail::sha256

#include "gtest/gtest.h"  // TEST, ASSERT_EQ, EXPECT_EQ

#include <string>  // std::string
#include <vector>  // std::vector


TEST(Base_Detail, starts_with) {
    EXPECT_TRUE(plssvm::detail::starts_with("abc", "abc"));
    EXPECT_TRUE(plssvm::detail::starts_with("abc", "ab"));
    EXPECT_FALSE(plssvm::detail::starts_with("abc", "abcd"));
    EXPECT_FALSE(plssvm::detail::starts_with("abc", "bc"));
    EXPECT_TRUE(plssvm::detail::starts_with("abc", 'a'));
    EXPECT_FALSE(plssvm::detail::starts_with("abc", 'c'));
    EXPECT_FALSE(plssvm::detail::starts_with("abc", 'd'));
}

TEST(Base_Detail, ends_with) {
    EXPECT_TRUE(plssvm::detail::ends_with("abc", "abc"));
    EXPECT_FALSE(plssvm::detail::ends_with("abc", "ab"));
    EXPECT_FALSE(plssvm::detail::ends_with("abc", "abcd"));
    EXPECT_TRUE(plssvm::detail::ends_with("abc", "bc"));
    EXPECT_FALSE(plssvm::detail::ends_with("abc", 'a'));
    EXPECT_TRUE(plssvm::detail::ends_with("abc", 'c'));
    EXPECT_FALSE(plssvm::detail::ends_with("abc", 'd'));
}

TEST(Base_Detail, contains) {
    EXPECT_TRUE(plssvm::detail::contains("abc", "abc"));
    EXPECT_TRUE(plssvm::detail::contains("abc", "ab"));
    EXPECT_FALSE(plssvm::detail::contains("abc", "abcd"));
    EXPECT_TRUE(plssvm::detail::contains("abc", "bc"));
    EXPECT_TRUE(plssvm::detail::contains("abc", 'a'));
    EXPECT_TRUE(plssvm::detail::contains("abc", 'c'));
    EXPECT_FALSE(plssvm::detail::contains("abc", 'd'));
}

TEST(Base_Detail, trim) {
    // trim left
    EXPECT_EQ(plssvm::detail::trim_left(""), "");
    EXPECT_EQ(plssvm::detail::trim_left("abc"), "abc");
    EXPECT_EQ(plssvm::detail::trim_left("  abc"), "abc");
    EXPECT_EQ(plssvm::detail::trim_left("abc   "), "abc   ");
    EXPECT_EQ(plssvm::detail::trim_left(" abc  "), "abc  ");
    EXPECT_EQ(plssvm::detail::trim_left(" a b c "), "a b c ");

    // trim right
    EXPECT_EQ(plssvm::detail::trim_right(""), "");
    EXPECT_EQ(plssvm::detail::trim_right("abc"), "abc");
    EXPECT_EQ(plssvm::detail::trim_right("  abc"), "  abc");
    EXPECT_EQ(plssvm::detail::trim_right("abc   "), "abc");
    EXPECT_EQ(plssvm::detail::trim_right(" abc  "), " abc");
    EXPECT_EQ(plssvm::detail::trim_right(" a b c "), " a b c");

    // trim
    EXPECT_EQ(plssvm::detail::trim(""), "");
    EXPECT_EQ(plssvm::detail::trim("abc"), "abc");
    EXPECT_EQ(plssvm::detail::trim("  abc"), "abc");
    EXPECT_EQ(plssvm::detail::trim("abc   "), "abc");
    EXPECT_EQ(plssvm::detail::trim(" abc  "), "abc");
    EXPECT_EQ(plssvm::detail::trim(" a b c "), "a b c");
}

TEST(Base_Detail, replace_all) {
    std::string input;
    EXPECT_EQ(plssvm::detail::replace_all(input, "", ""), "");

    input = "aaa";
    EXPECT_EQ(plssvm::detail::replace_all(input, "a", "b"), "bbb");

    input = "aaa";
    EXPECT_EQ(plssvm::detail::replace_all(input, "", "b"), "aaa");

    input = "aaa";
    EXPECT_EQ(plssvm::detail::replace_all(input, "b", "c"), "aaa");

    input = "aaa";
    EXPECT_EQ(plssvm::detail::replace_all(input, "aa", "b"), "ba");

    input = "a a b c d aa";
    EXPECT_EQ(plssvm::detail::replace_all(input, "a", ""), "  b c d ");

    input = "a";
    EXPECT_EQ(plssvm::detail::replace_all(input, "aa", "b"), "a");
}

TEST(Base_Detail, to_lower_case) {
    std::string input;
    EXPECT_EQ(plssvm::detail::to_lower_case(input), "");

    input = "abc";
    EXPECT_EQ(plssvm::detail::to_lower_case(input), "abc");

    input = "ABC";
    EXPECT_EQ(plssvm::detail::to_lower_case(input), "abc");

    input = " AbC 1";
    EXPECT_EQ(plssvm::detail::to_lower_case(input), " abc 1");
}

TEST(Base_Detail, as_lower_case) {
    EXPECT_EQ(plssvm::detail::as_lower_case(""), "");
    EXPECT_EQ(plssvm::detail::as_lower_case("abc"), "abc");
    EXPECT_EQ(plssvm::detail::as_lower_case("ABC"), "abc");
    EXPECT_EQ(plssvm::detail::as_lower_case(" AbC 1"), " abc 1");
}

TEST(Base_Detail, to_upper_case) {
    std::string input;
    EXPECT_EQ(plssvm::detail::to_upper_case(input), "");

    input = "abc";
    EXPECT_EQ(plssvm::detail::to_upper_case(input), "ABC");

    input = "ABC";
    EXPECT_EQ(plssvm::detail::to_upper_case(input), "ABC");

    input = " AbC 1";
    EXPECT_EQ(plssvm::detail::to_upper_case(input), " ABC 1");
}

TEST(Base_Detail, as_upper_case) {
    EXPECT_EQ(plssvm::detail::as_upper_case(""), "");
    EXPECT_EQ(plssvm::detail::as_upper_case("abc"), "ABC");
    EXPECT_EQ(plssvm::detail::as_upper_case("ABC"), "ABC");
    EXPECT_EQ(plssvm::detail::as_upper_case(" AbC 1"), " ABC 1");
}

TEST(Base_Detail, split) {
    using namespace plssvm::detail;

    // split string using the default delimiter
    {
        const std::string string_to_split = "1.5 2.0 -3.5 4.0 5.0 -6.0  7.5";

        const std::vector<std::string_view> splitted_correct = { "1.5", "2.0", "-3.5", "4.0", "5.0", "-6.0", "", "7.5" };
        const std::vector<std::string_view> splitted = split(string_to_split);
        ASSERT_EQ(splitted.size(), splitted_correct.size());
        for (typename std::vector<double>::size_type i = 0; i < splitted_correct.size(); ++i) {
            EXPECT_EQ(splitted[i], splitted_correct[i]) << "split position: " << i << ", splitted: " << splitted[i] << ", correct: " << splitted_correct[i];
        }
    }

    // split string using a custom delimiter
    {
        const std::string string_to_split = "1.5,2.0,-3.5,4.0,5.0,-6.0,,7.5";

        const std::vector<std::string_view> splitted_correct = { "1.5", "2.0", "-3.5", "4.0", "5.0", "-6.0", "", "7.5" };
        const std::vector<std::string_view> splitted = split(string_to_split, ',');
        ASSERT_EQ(splitted.size(), splitted_correct.size());
        for (typename std::vector<double>::size_type i = 0; i < splitted_correct.size(); ++i) {
            EXPECT_EQ(splitted[i], splitted_correct[i]) << "split position: " << i << ", splitted: " << splitted[i] << ", correct: " << splitted_correct[i];
        }
    }

    // split string containing a single value
    {
        const std::vector<std::string_view> splitted = split("42");
        ASSERT_EQ(splitted.size(), 1);
        EXPECT_EQ(splitted.front(), "42") << "splitted: " << splitted.front() << ", correct: "
                                          << "42";
    }

    // split empty string
    {
        const std::vector<std::string_view> splitted = split("");
        EXPECT_TRUE(splitted.empty());
    }
}