/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the plssvm::shape struct nad the respective free functions.
 */

#include "plssvm/shape.hpp"

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gtest/gtest.h"  // TEST, EXPECT_TRUE, EXPECT_FALSE

#include <algorithm>   // std::swap
#include <cstddef>     // std::size_t
#include <functional>  // std::hash
#include <sstream>     // std::istringstream

TEST(Shape, DefaultConstruct) {
    // default construct a shape
    const plssvm::shape s{};

    // check set shapes
    EXPECT_EQ(s.x, 0);
    EXPECT_EQ(s.y, 0);
}

TEST(Shape, Construct) {
    // construct a shape with values
    const plssvm::shape s{ 4, 5 };

    // check set shapes
    EXPECT_EQ(s.x, 4);
    EXPECT_EQ(s.y, 5);
}

TEST(Shape, SwapMemberFunction) {
    // construct two shape objects
    plssvm::shape s1{ 4, 5 };
    plssvm::shape s2{};

    // swap both objects
    s1.swap(s2);

    // check swapped contents
    EXPECT_EQ(s1.x, 0);
    EXPECT_EQ(s1.y, 0);
    EXPECT_EQ(s2.x, 4);
    EXPECT_EQ(s2.y, 5);
}

// check whether the plssvm::shape -> std::string conversions are correct
TEST(Shape, ToString) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING((plssvm::shape{ 0, 0 }), "[0, 0]");
    EXPECT_CONVERSION_TO_STRING((plssvm::shape{ 4, 0 }), "[4, 0]");
    EXPECT_CONVERSION_TO_STRING((plssvm::shape{ 0, 5 }), "[0, 5]");
    EXPECT_CONVERSION_TO_STRING((plssvm::shape{ 4, 5 }), "[4, 5]");
}

// check whether the std::string -> plssvm::shape conversions are correct
TEST(Shape, FromString) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("0 0", (plssvm::shape{ 0, 0 }));
    EXPECT_CONVERSION_FROM_STRING("4 0", (plssvm::shape{ 4, 0 }));
    EXPECT_CONVERSION_FROM_STRING("0 5", (plssvm::shape{ 0, 5 }));
    EXPECT_CONVERSION_FROM_STRING("4 5", (plssvm::shape{ 4, 5 }));
}

TEST(Shape, FromStringUnknown) {
    // foo isn't a valid solver_type
    std::istringstream input{ "foo" };
    plssvm::shape shape{};
    input >> shape;
    EXPECT_TRUE(input.fail());
}

TEST(Shape, SwapFreeFunction) {
    // construct two shape objects
    plssvm::shape s1{ 4, 5 };
    plssvm::shape s2{};

    // swap both objects
    using std::swap;
    swap(s1, s2);

    // check swapped contents
    EXPECT_EQ(s1.x, 0);
    EXPECT_EQ(s1.y, 0);
    EXPECT_EQ(s2.x, 4);
    EXPECT_EQ(s2.y, 5);
}

TEST(Shape, Equal) {
    // construct shape objects
    const plssvm::shape s1{ 4, 5 };
    const plssvm::shape s2{};
    const plssvm::shape s3{ 0, 0 };

    // check shapes for equality
    EXPECT_FALSE(s1 == s2);
    EXPECT_FALSE(s1 == s3);
    EXPECT_TRUE(s2 == s3);
}

TEST(Shape, Unequal) {
    // construct shape objects
    const plssvm::shape s1{ 4, 5 };
    const plssvm::shape s2{};
    const plssvm::shape s3{ 0, 0 };

    // check shapes for equality
    EXPECT_TRUE(s1 != s2);
    EXPECT_TRUE(s1 != s3);
    EXPECT_FALSE(s2 != s3);
}

TEST(Shape, Hash) {
    // create a shape object
    const plssvm::shape s{ 2, 3 };

    // hash shape
    const std::size_t hash_value1 = std::hash<plssvm::shape>{}(s);
    const std::size_t hash_value2 = std::hash<plssvm::shape>{}(s);

    // result should be deterministic
    EXPECT_EQ(hash_value1, hash_value2);
}
