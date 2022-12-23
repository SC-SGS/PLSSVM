/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the implementation converting arithmetic types to their name as string.
 */

#include "plssvm/detail/arithmetic_type_name.hpp"

#include "gtest/gtest.h"  // TEST, EXPECT_EQ

TEST(ArithmeticTypeName, type) {
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<bool>(), "bool");

    // character types
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<char>(), "char");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<signed char>(), "signed char");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<unsigned char>(), "unsigned char");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<char16_t>(), "char16_t");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<char32_t>(), "char32_t");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<wchar_t>(), "wchar_t");

    // integral types
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<short>(), "short");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<unsigned short>(), "unsigned short");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<int>(), "int");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<unsigned int>(), "unsigned int");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<long>(), "long");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<unsigned long>(), "unsigned long");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<long long>(), "long long");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<unsigned long long>(), "unsigned long long");

    // floating point types
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<float>(), "float");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<double>(), "double");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<long double>(), "long double");
}

TEST(ArithmeticTypeName, const_type) {
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const bool>(), "const bool");

    // character types
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const char>(), "const char");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const signed char>(), "const signed char");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const unsigned char>(), "const unsigned char");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const char16_t>(), "const char16_t");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const char32_t>(), "const char32_t");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const wchar_t>(), "const wchar_t");

    // integral types
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const short>(), "const short");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const unsigned short>(), "const unsigned short");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const int>(), "const int");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const unsigned int>(), "const unsigned int");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const long>(), "const long");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const unsigned long>(), "const unsigned long");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const long long>(), "const long long");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const unsigned long long>(), "const unsigned long long");

    // floating point types
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const float>(), "const float");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const double>(), "const double");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const long double>(), "const long double");
}

TEST(ArithmeticTypeName, volatile_type) {
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile bool>(), "volatile bool");

    // character types
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile char>(), "volatile char");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile signed char>(), "volatile signed char");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile unsigned char>(), "volatile unsigned char");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile char16_t>(), "volatile char16_t");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile char32_t>(), "volatile char32_t");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile wchar_t>(), "volatile wchar_t");

    // integral types
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile short>(), "volatile short");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile unsigned short>(), "volatile unsigned short");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile int>(), "volatile int");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile unsigned int>(), "volatile unsigned int");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile long>(), "volatile long");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile unsigned long>(), "volatile unsigned long");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile long long>(), "volatile long long");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile unsigned long long>(), "volatile unsigned long long");

    // floating point types
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile float>(), "volatile float");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile double>(), "volatile double");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<volatile long double>(), "volatile long double");
}

TEST(ArithmeticTypeName, const_volatile_type) {
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile bool>(), "const volatile bool");

    // character types
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile char>(), "const volatile char");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile signed char>(), "const volatile signed char");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile unsigned char>(), "const volatile unsigned char");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile char16_t>(), "const volatile char16_t");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile char32_t>(), "const volatile char32_t");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile wchar_t>(), "const volatile wchar_t");

    // integral types
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile short>(), "const volatile short");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile unsigned short>(), "const volatile unsigned short");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile int>(), "const volatile int");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile unsigned int>(), "const volatile unsigned int");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile long>(), "const volatile long");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile unsigned long>(), "const volatile unsigned long");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile long long>(), "const volatile long long");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile unsigned long long>(), "const volatile unsigned long long");

    // floating point types
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile float>(), "const volatile float");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile double>(), "const volatile double");
    EXPECT_EQ(plssvm::detail::arithmetic_type_name<const volatile long double>(), "const volatile long double");
}