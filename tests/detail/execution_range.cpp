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

#include "../utility.hpp"  // util::convert_to_string

#include "gtest/gtest.h"  // TEST, EXPECT_EQ

#include <array>             // std::array
#include <cstddef>           // std::size_t
#include <initializer_list>  // std::initializer_list

/*
 * @brief Tests whether the grid and block in the plssvm::detail::execution_range @p range match the expected values.
 */
void check_execution_range(const plssvm::detail::execution_range range, const std::array<std::size_t, 3> expected_grid, const std::array<std::size_t, 3> expected_block) {
    EXPECT_EQ(range.grid, expected_grid);
    EXPECT_EQ(range.block, expected_block);
}

TEST(ExecutionRange, initializer_list) {
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

TEST(ExecutionRangeDeathTest, initializer_list_too_few_dimensions) {
    using plssvm::detail::execution_range;

    // empty grid and/or block in execution range
    EXPECT_DEATH((execution_range{ {}, {} }), "The number of grid sizes specified must be between 1 and 3, but is 0!");
    EXPECT_DEATH((execution_range{ {}, { 11 } }), "The number of grid sizes specified must be between 1 and 3, but is 0!");
    EXPECT_DEATH((execution_range{ { 11 }, {} }), "The number of block sizes specified must be between 1 and 3, but is 0!");
}

TEST(ExecutionRangeDeathTest, initializer_list_too_many_dimensions) {
    using plssvm::detail::execution_range;

    // too many dimensions for grid and/or block in execution range
    EXPECT_DEATH((execution_range{ { 41, 42, 43, 44 }, { 41, 42, 43, 44 } }), "The number of grid sizes specified must be between 1 and 3, but is 4!");
    EXPECT_DEATH((execution_range{ { 41, 42, 43, 44 }, { 11 } }), "The number of grid sizes specified must be between 1 and 3, but is 4!");
    EXPECT_DEATH((execution_range{ { 11 }, { 51, 52, 53, 54, 55 } }), "The number of block sizes specified must be between 1 and 3, but is 5!");
}

TEST(ExecutionRange, array) {
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

TEST(ExecutionRange, to_string) {
    using plssvm::detail::execution_range;

    EXPECT_EQ(util::convert_to_string(execution_range{ { 11 }, { 11 } }), "grid: [11, 1, 1]; block: [11, 1, 1]");
    EXPECT_EQ(util::convert_to_string(execution_range{ { 11 }, { 21, 22 } }), "grid: [11, 1, 1]; block: [21, 22, 1]");
    EXPECT_EQ(util::convert_to_string(execution_range{ { 11 }, { 31, 32, 33 } }), "grid: [11, 1, 1]; block: [31, 32, 33]");

    EXPECT_EQ(util::convert_to_string(execution_range{ { 21, 22 }, { 11 } }), "grid: [21, 22, 1]; block: [11, 1, 1]");
    EXPECT_EQ(util::convert_to_string(execution_range{ { 21, 22 }, { 21, 22 } }), "grid: [21, 22, 1]; block: [21, 22, 1]");
    EXPECT_EQ(util::convert_to_string(execution_range{ { 21, 22 }, { 31, 32, 33 } }), "grid: [21, 22, 1]; block: [31, 32, 33]");

    EXPECT_EQ(util::convert_to_string(execution_range{ { 31, 32, 33 }, { 11 } }), "grid: [31, 32, 33]; block: [11, 1, 1]");
    EXPECT_EQ(util::convert_to_string(execution_range{ { 31, 32, 33 }, { 21, 22 } }), "grid: [31, 32, 33]; block: [21, 22, 1]");
    EXPECT_EQ(util::convert_to_string(execution_range{ { 31, 32, 33 }, { 31, 32, 33 } }), "grid: [31, 32, 33]; block: [31, 32, 33]");
}