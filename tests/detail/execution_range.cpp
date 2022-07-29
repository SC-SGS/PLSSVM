/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the backend independent execution range.
 */

#include "plssvm/detail/execution_range.hpp"  // plssvm::detail::execution_range

#include "gtest/gtest.h"  // TEST, EXPECT_EQ

#include <array>             // std::array
#include <cstddef>           // std::size_t
#include <initializer_list>  // std::initializer_list

// TODO: DEATH TESTs?

void check_execution_range(const plssvm::detail::execution_range range, const std::array<std::size_t, 3> expected_grid, const std::array<std::size_t, 3> expected_block) {
    EXPECT_EQ(range.grid, expected_grid);
    EXPECT_EQ(range.block, expected_block);
}

TEST(Base_Detail, execution_range_initializer_list) {
    using plssvm::detail::execution_range;

    check_execution_range(execution_range{ { 11 }, { 11 } }, std::array<std::size_t, 3>{ 11, 1, 1 }, std::array<std::size_t, 3>{ 11, 1, 1 });
    check_execution_range(execution_range{ { 11 }, { 21, 22 } }, std::array<std::size_t, 3>{ 11, 1, 1 }, std::array<std::size_t, 3>{ 21, 22, 1 });
    check_execution_range(execution_range{ { 11 }, { 31, 32, 33 } }, std::array<std::size_t, 3>{ 11, 1, 1 }, std::array<std::size_t, 3>{ 31, 32, 33 });

    check_execution_range(execution_range{ { 21, 22 }, { 11 } }, std::array<std::size_t, 3>{ 21, 22, 1 }, std::array<std::size_t, 3>{ 11, 1, 1 });
    check_execution_range(execution_range{ { 21, 22 }, { 21, 22 } }, std::array<std::size_t, 3>{ 21, 22, 1 }, std::array<std::size_t, 3>{ 21, 22, 1 });
    check_execution_range(execution_range{ { 21, 22 }, { 31, 32, 33 } }, std::array<std::size_t, 3>{ 21, 22, 1 }, std::array<std::size_t, 3>{ 31, 32, 33 });

    check_execution_range(execution_range{ { 31, 32, 33 }, { 11 } }, std::array<std::size_t, 3>{ 31, 32, 33 }, std::array<std::size_t, 3>{ 11, 1, 1 });
    check_execution_range(execution_range{ { 31, 32, 33 }, { 21, 22 } }, std::array<std::size_t, 3>{ 31, 32, 33 }, std::array<std::size_t, 3>{ 21, 22, 1 });
    check_execution_range(execution_range{ { 31, 32, 33 }, { 31, 32, 33 } }, std::array<std::size_t, 3>{ 31, 32, 33 }, std::array<std::size_t, 3>{ 31, 32, 33 });
}

TEST(Base_Detail, execution_range_array) {
    using plssvm::detail::execution_range;

    check_execution_range(execution_range{ std::array<std::size_t, 1>{ 11 }, std::array<std::size_t, 1>{ 11 } }, std::array<std::size_t, 3>{ 11, 1, 1 }, std::array<std::size_t, 3>{ 11, 1, 1 });
    check_execution_range(execution_range{ std::array<std::size_t, 1>{ 11 }, std::array<std::size_t, 2>{ 21, 22 } }, std::array<std::size_t, 3>{ 11, 1, 1 }, std::array<std::size_t, 3>{ 21, 22, 1 });
    check_execution_range(execution_range{ std::array<std::size_t, 1>{ 11 }, std::array<std::size_t, 3>{ 31, 32, 33 } }, std::array<std::size_t, 3>{ 11, 1, 1 }, std::array<std::size_t, 3>{ 31, 32, 33 });

    check_execution_range(execution_range{ std::array<std::size_t, 2>{ 21, 22 }, std::array<std::size_t, 1>{ 11 } }, std::array<std::size_t, 3>{ 21, 22, 1 }, std::array<std::size_t, 3>{ 11, 1, 1 });
    check_execution_range(execution_range{ std::array<std::size_t, 2>{ 21, 22 }, std::array<std::size_t, 2>{ 21, 22 } }, std::array<std::size_t, 3>{ 21, 22, 1 }, std::array<std::size_t, 3>{ 21, 22, 1 });
    check_execution_range(execution_range{ std::array<std::size_t, 2>{ 21, 22 }, std::array<std::size_t, 3>{ 31, 32, 33 } }, std::array<std::size_t, 3>{ 21, 22, 1 }, std::array<std::size_t, 3>{ 31, 32, 33 });

    check_execution_range(execution_range{ std::array<std::size_t, 3>{ 31, 32, 33 }, std::array<std::size_t, 1>{ 11 } }, std::array<std::size_t, 3>{ 31, 32, 33 }, std::array<std::size_t, 3>{ 11, 1, 1 });
    check_execution_range(execution_range{ std::array<std::size_t, 3>{ 31, 32, 33 }, std::array<std::size_t, 2>{ 21, 22 } }, std::array<std::size_t, 3>{ 31, 32, 33 }, std::array<std::size_t, 3>{ 21, 22, 1 });
    check_execution_range(execution_range{ std::array<std::size_t, 3>{ 31, 32, 33 }, std::array<std::size_t, 3>{ 31, 32, 33 } }, std::array<std::size_t, 3>{ 31, 32, 33 }, std::array<std::size_t, 3>{ 31, 32, 33 });
}