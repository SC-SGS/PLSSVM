/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom standard layout tuple implementation.
 */

#include "plssvm/backends/SYCL/detail/standard_layout_tuple.hpp"  // plssvm::sycl::detail::standard_layout_tuple

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, ::testing::StaticAssertTypeEq

TEST(StandardLayoutTuple, make_standard_layout_tuple_empty) {
    // create empty tuple
    auto t = plssvm::sycl::detail::make_standard_layout_tuple();

    ::testing::StaticAssertTypeEq<decltype(t), plssvm::sycl::detail::standard_layout_tuple<>>();
}

TEST(StandardLayoutTuple, make_standard_layout_tuple) {
    // create tuple
    auto t = plssvm::sycl::detail::make_standard_layout_tuple(42, 3.1415, .1234);

    ::testing::StaticAssertTypeEq<decltype(t), plssvm::sycl::detail::standard_layout_tuple<int, double, double>>();
}

TEST(StandardLayoutTuple, get) {
    // create tuple
    const auto t = plssvm::sycl::detail::make_standard_layout_tuple(42, 3.1415, 0.1234);

    // check get function
    EXPECT_EQ(plssvm::sycl::detail::get<0>(t), 42);
    EXPECT_EQ(plssvm::sycl::detail::get<1>(t), 3.1415);
    EXPECT_EQ(plssvm::sycl::detail::get<2>(t), 0.1234);
}
