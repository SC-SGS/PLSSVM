/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different classification_types.
 */

#include "plssvm/classification_types.hpp"

#include "custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gtest/gtest.h"  // EXPECT_EQ

#include <sstream>  // std::istringstream

// check whether the plssvm::classification_type -> std::string conversions are correct
TEST(ClassificationType, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::classification_type::oaa, "oaa");
    EXPECT_CONVERSION_TO_STRING(plssvm::classification_type::oao, "oao");
}
TEST(ClassificationType, to_string_unknown) {
    // check conversions to std::string from unknown classification_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::classification_type>(2), "unknown");
}

// check whether the std::string -> plssvm::classification_type conversions are correct
TEST(ClassificationType, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("oaa", plssvm::classification_type::oaa);
    EXPECT_CONVERSION_FROM_STRING("OAA", plssvm::classification_type::oaa);
    EXPECT_CONVERSION_FROM_STRING("one_vs_all", plssvm::classification_type::oaa);
    EXPECT_CONVERSION_FROM_STRING("One_Against_All", plssvm::classification_type::oaa);
    EXPECT_CONVERSION_FROM_STRING("oao", plssvm::classification_type::oao);
    EXPECT_CONVERSION_FROM_STRING("OAO", plssvm::classification_type::oao);
    EXPECT_CONVERSION_FROM_STRING("one_vs_one", plssvm::classification_type::oao);
    EXPECT_CONVERSION_FROM_STRING("One_Against_One", plssvm::classification_type::oao);
}
TEST(ClassificationType, from_string_unknown) {
    // foo isn't a valid classification_type
    std::istringstream input{ "foo" };
    plssvm::classification_type classification{};
    input >> classification;
    EXPECT_TRUE(input.fail());
}

TEST(ClassificationType, classification_type_to_full_string) {
    // check conversion from plssvm::classification_type to a full string
    EXPECT_EQ(plssvm::classification_type_to_full_string(plssvm::classification_type::oaa), "one vs. all");
    EXPECT_EQ(plssvm::classification_type_to_full_string(plssvm::classification_type::oao), "one vs. one");
}