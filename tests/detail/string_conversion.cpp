/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the sha256 implementation.
 */

#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to

#include "../utility.hpp"  // EXPECT_THROW_WHAT

#include "gtest/gtest.h"  // TEST, ASSERT_EQ, EXPECT_EQ, EXPECT_TRUE

#include <stdexcept>  // std::invalid_argument
#include <string>     // std::string
#include <vector>     // std::vector


template <typename T>
void check_convert_to(const std::vector<std::string_view> &input, const std::vector<T> &correct_output) {
    ASSERT_EQ(input.size(), correct_output.size());

    for (std::vector<std::string_view>::size_type i = 0; i < input.size(); ++i) {
        const T conv = plssvm::detail::convert_to<T>(input[i]);
        EXPECT_EQ(conv, correct_output[i]) << "input: \"" << input[i] << "\", output: \"" << conv << "\", correct: \"" << correct_output[i] << '\"';
    }
}

TEST(Base_Detail, string_conversion) {
    using namespace plssvm::detail;

    std::vector<std::string_view> input = { "-3", "-1.5", "0.0", "1.5", "3", "   5", "  6 ", "7  " };
    std::vector<std::string_view> input_unsigned = { "0.0", "1.5", "3", "   5", "  6 ", "7  " };
    std::vector<std::string_view> input_char = { "0", "48", "65.2", "66", "122", "   119", "  120 ", "121  " };

    // boolean
    // std::from_chars seems to not support bool
//    check_convert_to(input, std::vector<bool>{ true, true, false, true, true, true, true, true });

    // character types
    check_convert_to(input_char, std::vector<char>{ '\0', '0', 'A', 'B', 'z', 'w', 'x', 'y' });
    check_convert_to(input_char, std::vector<signed char>{ '\0', '0', 'A', 'B', 'z', 'w', 'x', 'y' });
    check_convert_to(input_char, std::vector<unsigned char>{ '\0', '0', 'A', 'B', 'z', 'w', 'x', 'y' });
    // std::from_chars seems to not support char16_t, char32_t, and wchar_t
//    check_convert_to(input_char, std::vector<char16_t>{ '\0', '0', 'A', 'B', 'z', 'w', 'x', 'y' });
//    check_convert_to(input_char, std::vector<char32_t>{ '\0', '0', 'A', 'B', 'z', 'w', 'x', 'y' });
//    check_convert_to(input_char, std::vector<wchar_t>{ '\0', '0', 'A', 'B', 'z', 'w', 'x', 'y' });

    // integer types
    check_convert_to(input, std::vector<short>{ -3, -1, 0, 1, 3, 5, 6, 7 });
    check_convert_to(input_unsigned, std::vector<unsigned short>{ 0, 1, 3, 5, 6, 7 });
    check_convert_to(input, std::vector<int>{ -3, -1, 0, 1, 3, 5, 6, 7 });
    check_convert_to(input_unsigned, std::vector<unsigned int>{ 0, 1, 3, 5, 6, 7 });
    check_convert_to(input, std::vector<long>{ -3, -1, 0, 1, 3, 5, 6, 7 });
    check_convert_to(input_unsigned, std::vector<unsigned long>{ 0, 1, 3, 5, 6, 7 });
    check_convert_to(input, std::vector<long long>{ -3, -1, 0, 1, 3, 5, 6, 7 });
    check_convert_to(input_unsigned, std::vector<unsigned long long>{ 0, 1, 3, 5, 6, 7 });

    // floating point types
    check_convert_to(input, std::vector<float>{ -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 5.0f, 6.0f, 7.0f });
    check_convert_to(input, std::vector<double>{ -3.0, -1.5, 0.0, 1.5, 3.0, 5.0, 6.0, 7.0 });
    // fast_float::from_chars seems to not support long double
//    check_convert_to(input, std::vector<long double>{ -3.0l, -1.5l, 0.0l, 1.5l, 3.0l, 5.0l, 6.0l, 7.0l });

    // std::string
    check_convert_to(input, std::vector<std::string>{ "-3", "-1.5", "0.0", "1.5", "3", "5", "6", "7" });
}

TEST(Base_Detail, string_conversion_exception) {
    using namespace plssvm::detail;

    {
        [[maybe_unused]] double res;
        EXPECT_THROW_WHAT(res = convert_to<double>("a"), std::runtime_error, "Can't convert 'a' to a value of type double!");
        EXPECT_THROW_WHAT(res = convert_to<double>("  abc 1"), std::runtime_error, "Can't convert '  abc 1' to a value of type double!");

        EXPECT_THROW_WHAT((res = convert_to<double, std::invalid_argument>("a")), std::invalid_argument, "Can't convert 'a' to a value of type double!");
    }
    {
        [[maybe_unused]] unsigned int res;
        EXPECT_THROW_WHAT(res = convert_to<unsigned int>("a"), std::runtime_error, "Can't convert 'a' to a value of type unsigned int!");
        EXPECT_THROW_WHAT(res = convert_to<unsigned int>("  abc 1"), std::runtime_error, "Can't convert '  abc 1' to a value of type unsigned int!");
        EXPECT_THROW_WHAT(res = convert_to<unsigned int>("-1"), std::runtime_error, "Can't convert '-1' to a value of type unsigned int!");

        EXPECT_THROW_WHAT((res = convert_to<unsigned int, std::invalid_argument>("a")), std::invalid_argument, "Can't convert 'a' to a value of type unsigned int!");
    }
}

TEST(Base_Detail, extract_first_integer_from_string) {
    using namespace plssvm::detail;

    EXPECT_EQ(extract_first_integer_from_string<int>("111"), 111);
    EXPECT_EQ(extract_first_integer_from_string<int>("111 222"), 111);
    EXPECT_EQ(extract_first_integer_from_string<int>("-111 222"), 111);
    EXPECT_EQ(extract_first_integer_from_string<int>(" 111 222 333"), 111);
    EXPECT_EQ(extract_first_integer_from_string<int>("abcd 111"), 111);
    EXPECT_EQ(extract_first_integer_from_string<int>("abcd111 222"), 111);
    EXPECT_EQ(extract_first_integer_from_string<int>("111_222"), 111);
    EXPECT_EQ(extract_first_integer_from_string<int>("111 abcd 222"), 111);
    EXPECT_EQ(extract_first_integer_from_string<int>("abc123def456"), 123);
}

TEST(Base_Detail, extract_first_integer_from_string_exception) {
    using namespace plssvm::detail;

    [[maybe_unused]] int res;
    EXPECT_THROW_WHAT(res = extract_first_integer_from_string<int>("abc"), std::runtime_error, "String \"abc\" doesn't contain any integer!");
    EXPECT_THROW_WHAT(res = extract_first_integer_from_string<int>(""), std::runtime_error, "String \"\" doesn't contain any integer!");
}

TEST(Base_Detail, split_as) {
    using namespace plssvm::detail;

    // split string using the default delimiter
    {
        const std::string string_to_split = "1.5 2.0 -3.5 4.0 5.0 -6.0 7.5";

        const std::vector<double> splitted_double_correct = { 1.5, 2.0, -3.5, 4.0, 5.0, -6.0, 7.5 };
        const std::vector<double> splitted_double = split_as<double>(string_to_split);
        ASSERT_EQ(splitted_double.size(), splitted_double_correct.size());
        for (typename std::vector<double>::size_type i = 0; i < splitted_double_correct.size(); ++i) {
            EXPECT_EQ(splitted_double[i], splitted_double_correct[i]) << "split position: " << i << ", splitted: " << splitted_double[i] << ", correct: " << splitted_double_correct[i];
        }

        const std::vector<int> splitted_int_correct = { 1, 2, -3, 4, 5, -6, 7 };
        const std::vector<int> splitted_int = split_as<int>(string_to_split);
        ASSERT_EQ(splitted_int.size(), splitted_int_correct.size());
        for (typename std::vector<double>::size_type i = 0; i < splitted_int_correct.size(); ++i) {
            EXPECT_EQ(splitted_int[i], splitted_int_correct[i]) << "split position: " << i << ", splitted: " << splitted_int[i] << ", correct: " << splitted_int_correct[i];
        }
    }

    // split string using a custom delimiter
    {
        const std::string string_to_split = "1.5,2.0,-3.5,4.0,5.0,-6.0,7.5";

        const std::vector<double> splitted_double_correct = { 1.5, 2.0, -3.5, 4.0, 5.0, -6.0, 7.5 };
        const std::vector<double> splitted_double = split_as<double>(string_to_split, ',');
        ASSERT_EQ(splitted_double.size(), splitted_double_correct.size());
        for (typename std::vector<double>::size_type i = 0; i < splitted_double_correct.size(); ++i) {
            EXPECT_EQ(splitted_double[i], splitted_double_correct[i]) << "split position: " << i << ", splitted: " << splitted_double[i] << ", correct: " << splitted_double_correct[i];
        }

        const std::vector<int> splitted_int_correct = { 1, 2, -3, 4, 5, -6, 7 };
        const std::vector<int> splitted_int = split_as<int>(string_to_split, ',');
        ASSERT_EQ(splitted_int.size(), splitted_int_correct.size());
        for (typename std::vector<double>::size_type i = 0; i < splitted_int_correct.size(); ++i) {
            EXPECT_EQ(splitted_int[i], splitted_int_correct[i]) << "split position: " << i << ", splitted: " << splitted_int[i] << ", correct: " << splitted_int_correct[i];
        }
    }

    // split string containing a single value
    {
        const std::vector<int> splitted = split_as<int>("42");
        ASSERT_EQ(splitted.size(), 1);
        EXPECT_EQ(splitted.front(), 42) << "splitted: " << splitted.front() << ", correct: " << 42;
    }

    // split empty string
    {
        const std::vector<int> splitted = split_as<int>("");
        EXPECT_TRUE(splitted.empty());
    }
}