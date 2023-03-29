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

#include "gtest/gtest.h"  // TEST, EXPECT_TRUE, EXPECT_FALSE, ::testing::StaticAssertTypeEq

#include <tuple>  // std::tuple

TEST(TypeList, CartesianTypeProductSingle) {
    using lhs = std::tuple<int>;
    using rhs = std::tuple<float>;

    // create cartesian type product
    using res = plssvm::detail::cartesian_type_product<lhs, rhs>::type;
    // check if the resulting type is correct
    ::testing::StaticAssertTypeEq<res, std::tuple<plssvm::detail::real_type_label_type_combination<int, float>>>();
}
TEST(TypeList, CartesianTypeProductTSingle) {
    using lhs = std::tuple<int>;
    using rhs = std::tuple<float>;

    // create cartesian type product
    using res = plssvm::detail::cartesian_type_product_t<lhs, rhs>;
    // check if the resulting type is correct
    ::testing::StaticAssertTypeEq<res, std::tuple<plssvm::detail::real_type_label_type_combination<int, float>>>();
}

TEST(TypeList, CartesianTypeProductMultiple) {
    using lhs = std::tuple<int, long>;
    using rhs = std::tuple<float, double>;

    // create cartesian type product
    using res = plssvm::detail::cartesian_type_product<lhs, rhs>::type;

    // check if the resulting type is correct
    // clang-format off
    ::testing::StaticAssertTypeEq<res, std::tuple<plssvm::detail::real_type_label_type_combination<int, float>,
                                                  plssvm::detail::real_type_label_type_combination<int, double>,
                                                  plssvm::detail::real_type_label_type_combination<long, float>,
                                                  plssvm::detail::real_type_label_type_combination<long, double>>>();
    // clang-format on
}
TEST(TypeList, CartesianTypeProductTMultiple) {
    using lhs = std::tuple<int, long>;
    using rhs = std::tuple<float, double>;

    // create cartesian type product
    using res = plssvm::detail::cartesian_type_product_t<lhs, rhs>;

    // check if the resulting type is correct
    // clang-format off
    ::testing::StaticAssertTypeEq<res, std::tuple<plssvm::detail::real_type_label_type_combination<int, float>,
                                                  plssvm::detail::real_type_label_type_combination<int, double>,
                                                  plssvm::detail::real_type_label_type_combination<long, float>,
                                                  plssvm::detail::real_type_label_type_combination<long, double>>>();
    // clang-format on
}

TEST(TypeList, TypeListContains) {
    using types = std::tuple<int, float, double>;

    // test the type_trait
    EXPECT_TRUE((plssvm::detail::type_list_contains<int, types>::value));
    EXPECT_TRUE((plssvm::detail::type_list_contains<float, types>::value));
    EXPECT_TRUE((plssvm::detail::type_list_contains<double, types>::value));
}
TEST(TypeList, TypeListContainsV) {
    using types = std::tuple<int, float, double>;

    // test the type_trait
    EXPECT_TRUE((plssvm::detail::type_list_contains_v<int, types>) );
    EXPECT_TRUE((plssvm::detail::type_list_contains_v<float, types>) );
    EXPECT_TRUE((plssvm::detail::type_list_contains_v<double, types>) );
}

TEST(TypeList, TypeListContainsNot) {
    using types = std::tuple<int, float, double>;

    // test the type_trait
    EXPECT_FALSE((plssvm::detail::type_list_contains<char, types>::value));
    EXPECT_FALSE((plssvm::detail::type_list_contains<const double, types>::value));
}
TEST(TypeList, TypeListContainsVNot) {
    using types = std::tuple<int, float, double>;

    // test the type_trait
    EXPECT_FALSE((plssvm::detail::type_list_contains_v<char, types>) );
    EXPECT_FALSE((plssvm::detail::type_list_contains_v<const double, types>) );
}