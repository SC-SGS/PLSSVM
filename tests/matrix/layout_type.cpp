/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Test the plssvm::layout_type used in the plssvm::matrix class.
 */

#include "plssvm/matrix.hpp"

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE

#include <sstream>  // std::istringstream, std::ostringstream
#include <string>   // std::string

// check whether the plssvm::layout_type -> std::string conversions are correct
TEST(LayoutType, ToString) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::layout_type::aos, "aos");
    EXPECT_CONVERSION_TO_STRING(plssvm::layout_type::soa, "soa");
}

TEST(LayoutType, ToStringUnknown) {
    // check conversions to std::string from unknown layout_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::layout_type>(2), "unknown");
}

// check whether the std::string -> plssvm::layout_type conversions are correct
TEST(LayoutType, FromString) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("aos", plssvm::layout_type::aos);
    EXPECT_CONVERSION_FROM_STRING("AoS", plssvm::layout_type::aos);
    EXPECT_CONVERSION_FROM_STRING("Array-of-Structs", plssvm::layout_type::aos);
    EXPECT_CONVERSION_FROM_STRING("soa", plssvm::layout_type::soa);
    EXPECT_CONVERSION_FROM_STRING("SoA", plssvm::layout_type::soa);
    EXPECT_CONVERSION_FROM_STRING("Struct-of-Arrays", plssvm::layout_type::soa);
}

TEST(LayoutType, FromStringUnknown) {
    // foo isn't a valid layout_type
    std::istringstream input{ "foo" };
    plssvm::layout_type layout{};
    input >> layout;
    EXPECT_TRUE(input.fail());
}

TEST(LayoutType, LayoutTypeToFullString) {
    // check conversion from plssvm::classification_type to a full string
    EXPECT_EQ(plssvm::layout_type_to_full_string(plssvm::layout_type::aos), "Array-of-Structs");
    EXPECT_EQ(plssvm::layout_type_to_full_string(plssvm::layout_type::soa), "Struct-of-Arrays");
}

TEST(LayoutType, LayoutTypeToFullStringUnknown) {
    // check conversion from unknown classification_typ to a full string
    EXPECT_EQ(plssvm::layout_type_to_full_string(static_cast<plssvm::layout_type>(2)), "unknown");
}
