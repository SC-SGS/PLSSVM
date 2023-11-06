/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functions in the string conversion header.
 */

#include "plssvm/detail/string_conversion.hpp"

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::{real_type_to_name, pretty_print_escaped_string}
#include "types_to_test.hpp"       // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, test_parameter_type_at_t}

#include "fmt/format.h"   // fmt::format
#include "gtest/gtest.h"  // TEST, ASSERT_EQ, EXPECT_EQ, EXPECT_TRUE, TYPED_TEST, TYPED_TEST_SUITE, TEST_P, INSTANTIATE_TEST_SUITE_P
                          // ::testing::{Test, TestWithParam, Types, Values}

#include <stdexcept>    // std::invalid_argument, std::runtime_error
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::ignore
#include <utility>      // std::pair, std::make_pair
#include <vector>       // std::vector

/**
 * @brief Checks the plssvm::detail::convert_to function.
 * @tparam T the type to convert the input values to
 * @param[in] input the input values to convert
 * @param[in] correct_output the correct output values
 */
template <typename T>
void check_convert_to(const std::vector<std::string_view> &input, const std::vector<T> &correct_output) {
    ASSERT_EQ(input.size(), correct_output.size());

    for (std::vector<std::string_view>::size_type i = 0; i < input.size(); ++i) {
        const T conv = plssvm::detail::convert_to<T>(input[i]);
        EXPECT_EQ(conv, correct_output[i]) << fmt::format(R"(input: "{}", output: "{}", correct: "{}")", input[i], conv, correct_output[i]);
    }
}

TEST(StringConversion, string_conversion) {
    using namespace plssvm::detail;

    const std::vector<std::string_view> input = { "-3", "-1.5", "0.0", "1.5", "3", "   5", "  6 ", "7  " };
    const std::vector<std::string_view> input_unsigned = { "0.0", "1.5", "3", "   5", "  6 ", "7  " };
    const std::vector<std::string_view> input_signed_unsigned_char = { "0", "48", "65.2", "66", "122", "   119", "  120 ", "121  " };
    const std::vector<std::string_view> input_char = { "0", "a", "A", "z", "w", "   x", "  Y ", "9  " };
    const std::vector<std::string_view> input_bool = { "true", "false", "True", "False", "TRUE", "FALSE" };

    // boolean
    // std::from_chars seems to not support bool
    check_convert_to(input, std::vector<bool>{ true, true, false, true, true, true, true, true });
    check_convert_to(input_bool, std::vector<bool>{ true, false, true, false, true, false });

    // boolean
    check_convert_to(input, std::vector<bool>{ true, true, false, true, true, true, true, true });
    check_convert_to(input_bool, std::vector<bool>{ true, false, true, false, true, false });

    // character types
    check_convert_to(input_char, std::vector<char>{ '0', 'a', 'A', 'z', 'w', 'x', 'Y', '9' });
    check_convert_to(input_signed_unsigned_char, std::vector<signed char>{ '\0', '0', 'A', 'B', 'z', 'w', 'x', 'y' });
    check_convert_to(input_signed_unsigned_char, std::vector<unsigned char>{ '\0', '0', 'A', 'B', 'z', 'w', 'x', 'y' });
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

using string_conversion_exception_types = std::tuple<short, unsigned char, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double>;
using string_conversion_exception_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<string_conversion_exception_types>>;

template <typename T>
class StringConversionException : public ::testing::Test {
  protected:
    using fixture_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(StringConversionException, string_conversion_exception_types_gtest, naming::test_parameter_to_name);

TYPED_TEST(StringConversionException, string_conversion_exception) {
    using type = typename TestFixture::fixture_type;

    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::convert_to<type>("a"),
                      std::runtime_error,
                      fmt::format("Can't convert 'a' to a value of type {}!", plssvm::detail::arithmetic_type_name<type>()));
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::convert_to<type>("  abc 1"),
                      std::runtime_error,
                      fmt::format("Can't convert '  abc 1' to a value of type {}!", plssvm::detail::arithmetic_type_name<type>()));
    EXPECT_THROW_WHAT((std::ignore = plssvm::detail::convert_to<type, std::invalid_argument>("a")),
                      std::invalid_argument,
                      fmt::format("Can't convert 'a' to a value of type {}!", plssvm::detail::arithmetic_type_name<type>()));
}
TEST(StringConversionException, string_conversion_exception_bool) {
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::convert_to<bool>("a"),
                      std::runtime_error,
                      "Can't convert 'a' to a value of type long long!");
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::convert_to<bool>("  abc 1"),
                      std::runtime_error,
                      "Can't convert '  abc 1' to a value of type long long!");
    EXPECT_THROW_WHAT((std::ignore = plssvm::detail::convert_to<bool, std::invalid_argument>("a")),
                      std::invalid_argument,
                      "Can't convert 'a' to a value of type long long!");
}
TEST(StringConversionException, string_conversion_exception_char) {
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::convert_to<char>("42"),
                      std::runtime_error,
                      "Can't convert '42' to a value of type char!");
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::convert_to<char>("  abc 1"),
                      std::runtime_error,
                      "Can't convert '  abc 1' to a value of type char!");
    EXPECT_THROW_WHAT((std::ignore = plssvm::detail::convert_to<char, std::invalid_argument>("")),
                      std::invalid_argument,
                      "Can't convert '' to a value of type char!");
}

class StringConversionExtract : public ::testing::TestWithParam<std::tuple<std::string_view, int>> {};
TEST_P(StringConversionExtract, extract_first_integer_from_string) {
    auto [input, output] = GetParam();
    EXPECT_EQ(plssvm::detail::extract_first_integer_from_string<int>(input), output);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(StringUtility, StringConversionExtract, ::testing::Values(
                std::make_tuple("111", 111), std::make_tuple("111 222", 111),
                std::make_tuple("-111 222", 111), std::make_tuple(" 111 222 333", 111),
                std::make_tuple("abcd 111", 111), std::make_tuple("abcd111 222", 111),
                std::make_tuple("111_222", 111), std::make_tuple("111 abcd 222", 111),
                std::make_tuple("abc123def456", 123)),
                naming::pretty_print_escaped_string<StringConversionExtract>);
// clang-format on

TEST(StringConversion, extract_first_integer_from_string_exception) {
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::extract_first_integer_from_string<int>("abc"), std::runtime_error, R"(String "abc" doesn't contain any integer!)");
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::extract_first_integer_from_string<int>(""), std::runtime_error, R"(String "" doesn't contain any integer!)");
}

using split_as_types = std::tuple<short, int, long, long long, float, double>;
using split_as_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<split_as_types>>;

template <typename T>
class StringConversionSplitAs : public ::testing::Test {
  protected:
    using fixture_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(StringConversionSplitAs, split_as_types_gtest, naming::test_parameter_to_name);

TYPED_TEST(StringConversionSplitAs, split_default_delimiter) {
    using type = typename TestFixture::fixture_type;

    // split string using the default delimiter
    const std::string string_to_split = "1.5 2.0 -3.5 4.0 5.0 -6.0 7.5";

    const std::vector<type> split_correct = { static_cast<type>(1.5), static_cast<type>(2.0), static_cast<type>(-3.5), static_cast<type>(4.0), static_cast<type>(5.0), static_cast<type>(-6.0), static_cast<type>(7.5) };
    const std::vector<type> split = plssvm::detail::split_as<type>(string_to_split);
    ASSERT_EQ(split.size(), split_correct.size());
    for (typename std::vector<TypeParam>::size_type i = 0; i < split_correct.size(); ++i) {
        EXPECT_EQ(split[i], split_correct[i]) << fmt::format("pos: {}, split: {}, correct: {}", i, split[i], split_correct[i]);
    }
}
TYPED_TEST(StringConversionSplitAs, split_custom_delimiter) {
    using type = typename TestFixture::fixture_type;

    // split string using a custom delimiter
    const std::string string_to_split = "1.5,2.0,-3.5,4.0,5.0,-6.0,7.5";

    const std::vector<type> split_correct = { static_cast<type>(1.5), static_cast<type>(2.0), static_cast<type>(-3.5), static_cast<type>(4.0), static_cast<type>(5.0), static_cast<type>(-6.0), static_cast<type>(7.5) };
    const std::vector<type> split = plssvm::detail::split_as<type>(string_to_split, ',');
    ASSERT_EQ(split.size(), split_correct.size());
    for (typename std::vector<TypeParam>::size_type i = 0; i < split_correct.size(); ++i) {
        EXPECT_EQ(split[i], split_correct[i]) << fmt::format("pos: {}, split: {}, correct: {}", i, split[i], split_correct[i]);
    }
}
TYPED_TEST(StringConversionSplitAs, split_single_value) {
    using type = typename TestFixture::fixture_type;

    // split string containing a single value
    const std::vector<type> split = plssvm::detail::split_as<type>("42");
    ASSERT_EQ(split.size(), 1);
    EXPECT_EQ(split.front(), static_cast<type>(42)) << fmt::format("split: {}, correct: {}", split.front(), static_cast<type>(42));
}
TYPED_TEST(StringConversionSplitAs, split_empty_string) {
    using type = typename TestFixture::fixture_type;

    // split the empty string
    const std::vector<type> split = plssvm::detail::split_as<type>("");
    EXPECT_TRUE(split.empty());
}