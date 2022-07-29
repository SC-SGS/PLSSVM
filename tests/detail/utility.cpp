/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functions in the utility header.
 */

#include "plssvm/default_value.hpp"   // plssvm::default_value
#include "plssvm/detail/utility.hpp"  // plssvm::detail::{always_false, remove_cvref_t, get, to_underlying, erase_if, contains_key}

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

#include <map>            // std::map
#include <set>            // std::set
#include <type_traits>    // std::is_same_v
#include <unordered_map>  // std::unordered_map
#include <unordered_set>  // std::unordered_set
#include <utility>        // std::pair


TEST(Base_Detail, always_false) {
    EXPECT_FALSE(plssvm::detail::always_false_v<void>);
    EXPECT_FALSE(plssvm::detail::always_false_v<int>);
    EXPECT_FALSE(plssvm::detail::always_false_v<double>);
}

TEST(Base_Detail, remove_cvref_t) {
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<double>>));
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<const double>>));
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<volatile double>>));
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<const volatile double>>));
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<double&>>));
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<const double&>>));
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<volatile double&>>));
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<const volatile double&>>));
}

TEST(Base_Detail, get) {
    EXPECT_EQ(plssvm::detail::get<0>(0, 1, 2, 3, 4), 0);
    EXPECT_EQ(plssvm::detail::get<1>(0, 1.5, 2, 3, 4), 1.5);
    EXPECT_EQ(plssvm::detail::get<2>(0, 1, -2, 3, 4), -2);
    EXPECT_EQ(plssvm::detail::get<3>(0, 1, 2, 'a', 4), 'a');
    EXPECT_EQ(plssvm::detail::get<4>(0, 1, 2, 3, "abc"), "abc");
}

TEST(Base_Detail, to_underlying) {
    enum class int_enum { a, b, c = 10 };

    EXPECT_EQ(plssvm::detail::to_underlying(int_enum::a), 0);
    EXPECT_EQ(plssvm::detail::to_underlying(int_enum::b), 1);
    EXPECT_EQ(plssvm::detail::to_underlying(int_enum::c), 10);

    enum class char_enum { a = 'a', b = 'b', c = 'c' };

    EXPECT_EQ(plssvm::detail::to_underlying(char_enum::a), 'a');
    EXPECT_EQ(plssvm::detail::to_underlying(char_enum::b), 'b');
    EXPECT_EQ(plssvm::detail::to_underlying(char_enum::c), 'c');
}

TEST(Base_Detail, to_underlying_default_value) {
    enum class int_enum { a, b, c = 10 };

    plssvm::default_value<int_enum> int_default;
    EXPECT_EQ(plssvm::detail::to_underlying(int_default = int_enum::a), 0);
    EXPECT_EQ(plssvm::detail::to_underlying(int_default = int_enum::b), 1);
    EXPECT_EQ(plssvm::detail::to_underlying(int_default = int_enum::c), 10);

    enum class char_enum { a = 'a', b = 'b', c = 'c' };

    plssvm::default_value<char_enum> char_default;
    EXPECT_EQ(plssvm::detail::to_underlying(char_default = char_enum::a), 'a');
    EXPECT_EQ(plssvm::detail::to_underlying(char_default = char_enum::b), 'b');
    EXPECT_EQ(plssvm::detail::to_underlying(char_default = char_enum::c), 'c');
}

TEST(Base_Detail, erase_if) {
    // std::map
    std::map<int, int> m = { { 0, 0 }, { 1, 1 } };
    EXPECT_EQ(plssvm::detail::erase_if(m, [](const std::pair<int, int> p) { return p.second % 2 == 0; }), 1);
    EXPECT_EQ(m.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(m, [](const std::pair<int, int> p) { return p.second % 2 == 0; }), 0);
    EXPECT_EQ(m.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(m, [](const std::pair<int, int> p) { return p.second % 2 == 1; }), 1);
    EXPECT_TRUE(m.empty());

    // std::unordered_map
    std::unordered_map<int, int> um = { { 0, 0 }, { 1, 1 } };
    EXPECT_EQ(plssvm::detail::erase_if(um, [](const std::pair<int, int> p) { return p.second % 2 == 0; }), 1);
    EXPECT_EQ(um.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(um, [](const std::pair<int, int> p) { return p.second % 2 == 0; }), 0);
    EXPECT_EQ(um.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(um, [](const std::pair<int, int> p) { return p.second % 2 == 1; }), 1);
    EXPECT_TRUE(um.empty());

    // std::set
    std::set<int> s = { 0, 1 };
    EXPECT_EQ(plssvm::detail::erase_if(s, [](const int k) { return k % 2 == 0; }), 1);
    EXPECT_EQ(s.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(s, [](const int k) { return k % 2 == 0; }), 0);
    EXPECT_EQ(s.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(s, [](const int k) { return k % 2 == 1; }), 1);
    EXPECT_TRUE(s.empty());

    // std::unordered_set
    std::unordered_set<int> us = { 0, 1 };
    EXPECT_EQ(plssvm::detail::erase_if(us, [](const int k) { return k % 2 == 0; }), 1);
    EXPECT_EQ(us.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(us, [](const int k) { return k % 2 == 0; }), 0);
    EXPECT_EQ(us.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(us, [](const int k) { return k % 2 == 1; }), 1);
    EXPECT_TRUE(us.empty());
}

TEST(Base_Detail, contains_key) {
    // std::map
    std::map<int, int> m = { { 0, 0 }, { 1, 1 } };
    EXPECT_TRUE(plssvm::detail::contains_key(m, 0));
    EXPECT_TRUE(plssvm::detail::contains_key(m, 1));
    EXPECT_FALSE(plssvm::detail::contains_key(m, 2));
    EXPECT_FALSE(plssvm::detail::contains_key(m, -1));

    // std::unordered_map
    std::unordered_map<int, int> um = { { 0, 0 }, { 1, 1 } };
    EXPECT_TRUE(plssvm::detail::contains_key(um, 0));
    EXPECT_TRUE(plssvm::detail::contains_key(um, 1));
    EXPECT_FALSE(plssvm::detail::contains_key(um, 2));
    EXPECT_FALSE(plssvm::detail::contains_key(um, -1));

    // std::set
    std::set<int> s = { 0, 1 };
    EXPECT_TRUE(plssvm::detail::contains_key(s, 0));
    EXPECT_TRUE(plssvm::detail::contains_key(s, 1));
    EXPECT_FALSE(plssvm::detail::contains_key(s, 2));
    EXPECT_FALSE(plssvm::detail::contains_key(s, -1));

    // std::unordered_set
    std::unordered_set<int> us = { 0, 1 };
    EXPECT_TRUE(plssvm::detail::contains_key(us, 0));
    EXPECT_TRUE(plssvm::detail::contains_key(us, 1));
    EXPECT_FALSE(plssvm::detail::contains_key(us, 2));
    EXPECT_FALSE(plssvm::detail::contains_key(us, -1));
}