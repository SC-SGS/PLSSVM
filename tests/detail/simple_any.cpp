/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for simple_any implementation.
 */

#include "plssvm/detail/simple_any.hpp"

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE

#include <string>  // std::string
#include <vector>  // std::vector
#include <memory>  // std::shared_ptr

TEST(SimpleAny, construct) {
    // construct simple_any objects
    const plssvm::detail::simple_any a1{ 42 };
    EXPECT_EQ(a1.get<int>(), 42);

    const plssvm::detail::simple_any a2{ 3.1415 };
    EXPECT_EQ(a2.get<double>(), 3.1415);

    const plssvm::detail::simple_any a3{ 'a' };
    EXPECT_EQ(a3.get<char>(), 'a');

    const plssvm::detail::simple_any a4{ true };
    EXPECT_TRUE(a4.get<bool>());

    const plssvm::detail::simple_any a5{ std::string{ "Test" } };
    EXPECT_EQ(a5.get<std::string>(), std::string{ "Test" });

    const plssvm::detail::simple_any a6{ std::vector<int>{ 1, 2, 3 } };
    EXPECT_EQ(a6.get<std::vector<int>>(), (std::vector<int>{ 1, 2, 3 }));

    const plssvm::detail::simple_any a7{ std::make_shared<float>(1.0f) };
    EXPECT_EQ(*a7.get<std::shared_ptr<float>>(), 1.0f);
}

TEST(SimpleAny, get) {
    // construct simple_any object
    plssvm::detail::simple_any a{ 42 };
    EXPECT_EQ(a.get<int>(), 42);
}
TEST(SimpleAny, get_wrong_type) {
    // construct simple_any object
    plssvm::detail::simple_any a{ 42 };

    // try casting to wrong type
    EXPECT_THROW(std::ignore = a.get<float>(), std::bad_cast);
}

TEST(SimpleAny, get_const) {
    // construct simple_any object
    const plssvm::detail::simple_any a{ 42 };
    EXPECT_EQ(a.get<int>(), 42);
}
TEST(SimpleAny, get_const_wrong_type) {
    // construct simple_any object
    const plssvm::detail::simple_any a{ 42 };

    // try casting to wrong type
    EXPECT_THROW(std::ignore = a.get<std::string>(), std::bad_cast);
}