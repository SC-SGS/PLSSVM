/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functions in the utility header.
 */

#include "plssvm/detail/utility.hpp"

#include "plssvm/default_value.hpp"  // plssvm::default_value

#include "../naming.hpp"   // naming::{map_types_to_name, set_types_to_name, vector_types_to_name}
#include "../utility.hpp"  // util::{is_map_v, is_unordered_map_v, is_set_v, is_unordered_set_v, is_vector_}

#include "gmock/gmock-matchers.h"  // EXPECT_THAT, ::testing::{HasSubstr, ContainsRegex}
#include "gtest/gtest.h"           // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

#include <map>            // std::map
#include <set>            // std::set
#include <type_traits>    // std::is_same_v
#include <unordered_map>  // std::unordered_map
#include <unordered_set>  // std::unordered_set
#include <vector>         // std::vector

TEST(Utility, always_false) {
    EXPECT_FALSE(plssvm::detail::always_false_v<void>);
    EXPECT_FALSE(plssvm::detail::always_false_v<int>);
    EXPECT_FALSE(plssvm::detail::always_false_v<double>);
}

TEST(Utility, remove_cvref_t) {
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<double>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<const double>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<volatile double>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<const volatile double>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<double &>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<const double &>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<volatile double &>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<const volatile double &>>) );
}

TEST(Utility, get) {
    EXPECT_EQ(plssvm::detail::get<0>(0, 1, 2, 3, 4), 0);
    EXPECT_EQ(plssvm::detail::get<1>(0, 1.5, 2, 3, 4), 1.5);
    EXPECT_EQ(plssvm::detail::get<2>(0, 1, -2, 3, 4), -2);
    EXPECT_EQ(plssvm::detail::get<3>(0, 1, 2, 'a', 4), 'a');
    EXPECT_EQ(plssvm::detail::get<4>(0, 1, 2, 3, "abc"), "abc");
}

TEST(Utility, to_underlying_int) {
    // clang-format off
    enum class int_enum { a, b, c = 10 };
    // clang-format on

    EXPECT_EQ(plssvm::detail::to_underlying(int_enum::a), 0);
    EXPECT_EQ(plssvm::detail::to_underlying(int_enum::b), 1);
    EXPECT_EQ(plssvm::detail::to_underlying(int_enum::c), 10);
}

TEST(Utility, to_underlying_char) {
    // clang-format off
    enum class char_enum : char { a = 'a', b = 'b', c = 'c' };
    // clang-format on

    EXPECT_EQ(plssvm::detail::to_underlying(char_enum::a), 'a');
    EXPECT_EQ(plssvm::detail::to_underlying(char_enum::b), 'b');
    EXPECT_EQ(plssvm::detail::to_underlying(char_enum::c), 'c');
}

TEST(Utility, to_underlying_default_value_int) {
    // clang-format off
    enum class int_enum { a, b, c = 10 };
    // clang-format on

    plssvm::default_value<int_enum> int_default;
    EXPECT_EQ(plssvm::detail::to_underlying(int_default = int_enum::a), 0);
    EXPECT_EQ(plssvm::detail::to_underlying(int_default = int_enum::b), 1);
    EXPECT_EQ(plssvm::detail::to_underlying(int_default = int_enum::c), 10);
}

TEST(Utility, to_underlying_default_value_char) {
    // clang-format off
    enum class char_enum : char { a = 'a', b = 'b', c = 'c' };
    // clang-format on

    plssvm::default_value<char_enum> char_default;
    EXPECT_EQ(plssvm::detail::to_underlying(char_default = char_enum::a), 'a');
    EXPECT_EQ(plssvm::detail::to_underlying(char_default = char_enum::b), 'b');
    EXPECT_EQ(plssvm::detail::to_underlying(char_default = char_enum::c), 'c');
}

// test fixture for map like classes
template <typename T>
class UtilityMapContainer : public ::testing::Test {
  protected:
    void SetUp() override {
        // initialize map
        map = { { 0, 0 }, { 1, 1 } };
    }

    using map_type = T;
    map_type map;
};

// the map container types to test
using map_types = ::testing::Types<std::map<int, int>, std::unordered_map<int, int>>;
TYPED_TEST_SUITE(UtilityMapContainer, map_types, naming::map_types_to_name);

TYPED_TEST(UtilityMapContainer, erase_if) {
    EXPECT_EQ(plssvm::detail::erase_if(this->map, [](const typename TestFixture::map_type::value_type value) { return value.second % 2 == 0; }), 1);
    EXPECT_EQ(this->map.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(this->map, [](const typename TestFixture::map_type::value_type value) { return value.second % 2 == 0; }), 0);
    EXPECT_EQ(this->map.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(this->map, [](const typename TestFixture::map_type::value_type value) { return value.second % 2 == 1; }), 1);
    EXPECT_TRUE(this->map.empty());
}
TYPED_TEST(UtilityMapContainer, contains) {
    EXPECT_TRUE(plssvm::detail::contains(this->map, 0));
    EXPECT_TRUE(plssvm::detail::contains(this->map, 1));
    EXPECT_FALSE(plssvm::detail::contains(this->map, 2));
    EXPECT_FALSE(plssvm::detail::contains(this->map, -1));
}

// test fixture for set like classes
template <typename T>
class UtilitySetContainer : public ::testing::Test {
  protected:
    void SetUp() override {
        // initialize set
        set = { 0, 1 };
    }

    using set_type = T;
    set_type set;
};

// the set container types to test
using set_types = ::testing::Types<std::set<int>, std::unordered_set<int>>;
TYPED_TEST_SUITE(UtilitySetContainer, set_types, naming::set_types_to_name);

TYPED_TEST(UtilitySetContainer, erase_if) {
    EXPECT_EQ(plssvm::detail::erase_if(this->set, [](const typename TestFixture::set_type::value_type value) { return value % 2 == 0; }), 1);
    EXPECT_EQ(this->set.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(this->set, [](const typename TestFixture::set_type::value_type value) { return value % 2 == 0; }), 0);
    EXPECT_EQ(this->set.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(this->set, [](const typename TestFixture::set_type::value_type value) { return value % 2 == 1; }), 1);
    EXPECT_TRUE(this->set.empty());
}
TYPED_TEST(UtilitySetContainer, contains) {
    EXPECT_TRUE(plssvm::detail::contains(this->set, 0));
    EXPECT_TRUE(plssvm::detail::contains(this->set, 1));
    EXPECT_FALSE(plssvm::detail::contains(this->set, 2));
    EXPECT_FALSE(plssvm::detail::contains(this->set, -1));
}

// test fixture for vector like classes
template <typename T>
class UtilityVectorContainer : public ::testing::Test {
  protected:
    void SetUp() override {
        // initialize vector
        vec = { 0, 1 };
    }

    using vector_type = T;
    vector_type vec;
};

// the vector container types to test
using vector_types = ::testing::Types<std::vector<int>>;
TYPED_TEST_SUITE(UtilityVectorContainer, vector_types, naming::vector_types_to_name);

TYPED_TEST(UtilityVectorContainer, erase_if) {
    EXPECT_EQ(plssvm::detail::erase_if(this->vec, [](const typename TestFixture::vector_type::value_type value) { return value % 2 == 0; }), 1);
    EXPECT_EQ(this->vec.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(this->vec, [](const typename TestFixture::vector_type::value_type value) { return value % 2 == 0; }), 0);
    EXPECT_EQ(this->vec.size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(this->vec, [](const typename TestFixture::vector_type::value_type value) { return value % 2 == 1; }), 1);
    EXPECT_TRUE(this->vec.empty());
}
TYPED_TEST(UtilityVectorContainer, contains) {
    EXPECT_TRUE(plssvm::detail::contains(this->vec, 0));
    EXPECT_TRUE(plssvm::detail::contains(this->vec, 1));
    EXPECT_FALSE(plssvm::detail::contains(this->vec, 2));
    EXPECT_FALSE(plssvm::detail::contains(this->vec, -1));
}

TEST(Utility, current_date_time) {
    // test if the current date time matches the pattern
    EXPECT_THAT(plssvm::detail::current_date_time(), ::testing::ContainsRegex("[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}"));
}