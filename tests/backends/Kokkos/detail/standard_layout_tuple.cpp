/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom standard layout tuple implementation necessary for Kokkos.
 */

#include "plssvm/backends/Kokkos/detail/standard_layout_tuple.hpp"  // plssvm::kokkos::detail::{standard_layout_tuple, make_standard_layout_tuple, get}

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, testing::StaticAssertTypeEq

#include <type_traits>  // std::remove_const_t

TEST(KokkosStandardLayoutTuple, make_standard_layout_tuple) {
    // create a new standard layout tuple
    [[maybe_unused]] const auto tuple = plssvm::kokkos::detail::make_standard_layout_tuple(true, 42, 3.1415);

    // check the tuple type
    ::testing::StaticAssertTypeEq<plssvm::kokkos::detail::standard_layout_tuple<bool, int, double>, std::remove_const_t<decltype(tuple)>>();
}

TEST(KokkosStandardLayoutTuple, get) {
    // create a new standard layout tuple
    const auto tuple = plssvm::kokkos::detail::make_standard_layout_tuple(true, 42, 3.1415);

    // check getter functions
    EXPECT_EQ(plssvm::kokkos::detail::get<0>(tuple), true);
    EXPECT_EQ(plssvm::kokkos::detail::get<1>(tuple), 42);
    EXPECT_EQ(plssvm::kokkos::detail::get<2>(tuple), 3.1415);
}
