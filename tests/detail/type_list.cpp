/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the type_list.
 */

#include "plssvm/detail/type_list.hpp"

#include "gtest/gtest.h"  // TEST, EXPECT_TRUE, EXPECT_FALSE

#include <tuple>  // std::tuple

TEST(TypeList, TypeListContains) {
    using types = std::tuple<int, float, double>;

    // test the type_trait
    EXPECT_TRUE((plssvm::detail::tuple_contains<int, types>::value));
    EXPECT_TRUE((plssvm::detail::tuple_contains<float, types>::value));
    EXPECT_TRUE((plssvm::detail::tuple_contains<double, types>::value));
}
TEST(TypeList, TypeListContainsV) {
    using types = std::tuple<int, float, double>;

    // test the type_trait
    EXPECT_TRUE((plssvm::detail::tuple_contains_v<int, types>) );
    EXPECT_TRUE((plssvm::detail::tuple_contains_v<float, types>) );
    EXPECT_TRUE((plssvm::detail::tuple_contains_v<double, types>) );
}

TEST(TypeList, TypeListContainsNot) {
    using types = std::tuple<int, float, double>;

    // test the type_trait
    EXPECT_FALSE((plssvm::detail::tuple_contains<char, types>::value));
    EXPECT_FALSE((plssvm::detail::tuple_contains<const double, types>::value));
}
TEST(TypeList, TypeListContainsVNot) {
    using types = std::tuple<int, float, double>;

    // test the type_trait
    EXPECT_FALSE((plssvm::detail::tuple_contains_v<char, types>) );
    EXPECT_FALSE((plssvm::detail::tuple_contains_v<const double, types>) );
}