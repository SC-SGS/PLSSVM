/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the performance tracker class.
 */

#include "plssvm/detail/performance_tracker.hpp"

#include "../naming.hpp"         // naming::label_type_to_name
#include "../types_to_test.hpp"  // util::label_type_gtest
#include "../utility.hpp"        // util::redirect_output

#include "fmt/core.h"            // fmt::format
#include "gtest/gtest.h"         // TEST, TYPED_TEST_SUITE, TYPED_TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, ::testing::Test

#include <iostream>              // std::cout
#include <string>                // std::string

template <typename T>
class TrackingEntry : public ::testing::Test, public util::redirect_output<> {};
TYPED_TEST_SUITE(TrackingEntry, util::label_type_gtest, naming::label_type_to_name);

TYPED_TEST(TrackingEntry, construct) {
    using type = TypeParam;

    // construct a tracking entry
    const plssvm::detail::tracking_entry e{ "category", "name", type{} };

    // check the values
    EXPECT_EQ(e.entry_category, "category");
    EXPECT_EQ(e.entry_name, "name");
    EXPECT_EQ(e.entry_value, type{});
}

TYPED_TEST(TrackingEntry, output_operator) {
    using type = TypeParam;

    // construct a tracking entry
    const plssvm::detail::tracking_entry e{ "category", "name", type{} };

    // output the value
    std::cout << e;

    // check the output value
    EXPECT_EQ(this->get_capture(), fmt::format("{}", type{}));
}

TEST(TrackingEntry, is_tracking_entry) {
    // check whether the provided type is a tracking entry or not, ignoring any top-level const, reference, and volatile qualifiers
    EXPECT_TRUE(plssvm::detail::is_tracking_entry<plssvm::detail::tracking_entry<int>>::value);
    EXPECT_TRUE(plssvm::detail::is_tracking_entry_v<plssvm::detail::tracking_entry<int>>);
    EXPECT_TRUE(plssvm::detail::is_tracking_entry<plssvm::detail::tracking_entry<const int &>>::value);
    EXPECT_TRUE(plssvm::detail::is_tracking_entry_v<plssvm::detail::tracking_entry<const int &>>);
    EXPECT_TRUE(plssvm::detail::is_tracking_entry<plssvm::detail::tracking_entry<std::string>>::value);
    EXPECT_TRUE(plssvm::detail::is_tracking_entry_v<plssvm::detail::tracking_entry<std::string>>);
}
TEST(TrackingEntry, is_no_tracking_entry) {
    // the following types are NOT tracking entries
    EXPECT_FALSE(plssvm::detail::is_tracking_entry<int>::value);
    EXPECT_FALSE(plssvm::detail::is_tracking_entry_v<int>);
    EXPECT_FALSE(plssvm::detail::is_tracking_entry<const int &>::value);
    EXPECT_FALSE(plssvm::detail::is_tracking_entry_v<const int &>);
    EXPECT_FALSE(plssvm::detail::is_tracking_entry<std::string>::value);
    EXPECT_FALSE(plssvm::detail::is_tracking_entry_v<std::string>);
}