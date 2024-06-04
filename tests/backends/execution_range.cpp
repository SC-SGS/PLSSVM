/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the execution range related functions and classes.
 */

#include "plssvm/backends/execution_range.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::kernel_launch_resources

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

#include <string>  // std::string

//*************************************************************************************************************************************//
//                                                              dim_type                                                               //
//*************************************************************************************************************************************//

TEST(DimType, default_construct) {
    // a default constructed dim_type should be all ones
    constexpr plssvm::detail::dim_type dim{};

    EXPECT_EQ(dim.x, 1ull);
    EXPECT_EQ(dim.y, 1ull);
    EXPECT_EQ(dim.z, 1ull);
}

TEST(DimType, one_argument) {
    // a default constructed dim_type should be all ones
    constexpr plssvm::detail::dim_type dim{ 64ull };

    EXPECT_EQ(dim.x, 64ull);
    EXPECT_EQ(dim.y, 1ull);
    EXPECT_EQ(dim.z, 1ull);
}

TEST(DimType, two_arguments) {
    // a default constructed dim_type should be all ones
    constexpr plssvm::detail::dim_type dim{ 64ull, 32ull };

    EXPECT_EQ(dim.x, 64ull);
    EXPECT_EQ(dim.y, 32ull);
    EXPECT_EQ(dim.z, 1ull);
}

TEST(DimType, three_arguments) {
    // a default constructed dim_type should be all ones
    constexpr plssvm::detail::dim_type dim{ 64ull, 32ull, 16ull };

    EXPECT_EQ(dim.x, 64ull);
    EXPECT_EQ(dim.y, 32ull);
    EXPECT_EQ(dim.z, 16ull);
}

TEST(DimType, swap_member_function) {
    plssvm::detail::dim_type dim1{ 64ull };
    plssvm::detail::dim_type dim2{ 32ull, 16ull };

    // swap the contents
    dim1.swap(dim2);

    // check the contents
    EXPECT_EQ(dim1.x, 32ull);
    EXPECT_EQ(dim1.y, 16ull);
    EXPECT_EQ(dim1.z, 1ull);

    EXPECT_EQ(dim2.x, 64ull);
    EXPECT_EQ(dim2.y, 1ull);
    EXPECT_EQ(dim2.z, 1ull);
}

TEST(DimType, swap_free_function) {
    plssvm::detail::dim_type dim1{ 64ull };
    plssvm::detail::dim_type dim2{ 32ull, 16ull };

    // swap the contents
    using plssvm::detail::swap;
    swap(dim1, dim2);

    // check the contents
    EXPECT_EQ(dim1.x, 32ull);
    EXPECT_EQ(dim1.y, 16ull);
    EXPECT_EQ(dim1.z, 1ull);

    EXPECT_EQ(dim2.x, 64ull);
    EXPECT_EQ(dim2.y, 1ull);
    EXPECT_EQ(dim2.z, 1ull);
}

TEST(DimType, equality) {
    // create dim types
    constexpr plssvm::detail::dim_type dim1{};
    constexpr plssvm::detail::dim_type dim2{ 64ull };
    constexpr plssvm::detail::dim_type dim3{ 64ull, 32ull };
    constexpr plssvm::detail::dim_type dim4{ 64ull, 32ull, 16ull };
    constexpr plssvm::detail::dim_type dim5{ 32ull };
    constexpr plssvm::detail::dim_type dim6{ 32ull, 16ull };
    constexpr plssvm::detail::dim_type dim7{ 32ull, 16ull, 8ull };

    // check for equality
    EXPECT_TRUE(dim1 == dim1);
    EXPECT_TRUE(dim2 == dim2);
    EXPECT_TRUE(dim3 == dim3);
    EXPECT_TRUE(dim4 == dim4);
    EXPECT_FALSE(dim2 == dim3);
    EXPECT_FALSE(dim2 == dim4);
    EXPECT_FALSE(dim3 == dim4);
    EXPECT_FALSE(dim2 == dim5);
    EXPECT_FALSE(dim3 == dim6);
    EXPECT_FALSE(dim4 == dim7);
}

TEST(DimType, inequality) {
    // create dim types
    constexpr plssvm::detail::dim_type dim1{};
    constexpr plssvm::detail::dim_type dim2{ 64ull };
    constexpr plssvm::detail::dim_type dim3{ 64ull, 32ull };
    constexpr plssvm::detail::dim_type dim4{ 64ull, 32ull, 16ull };
    constexpr plssvm::detail::dim_type dim5{ 32ull };
    constexpr plssvm::detail::dim_type dim6{ 32ull, 16ull };
    constexpr plssvm::detail::dim_type dim7{ 32ull, 16ull, 8ull };

    // check for inequality
    EXPECT_FALSE(dim1 != dim1);
    EXPECT_FALSE(dim2 != dim2);
    EXPECT_FALSE(dim3 != dim3);
    EXPECT_FALSE(dim4 != dim4);
    EXPECT_TRUE(dim2 != dim3);
    EXPECT_TRUE(dim2 != dim4);
    EXPECT_TRUE(dim3 != dim4);
    EXPECT_TRUE(dim2 != dim5);
    EXPECT_TRUE(dim3 != dim6);
    EXPECT_TRUE(dim4 != dim7);
}

TEST(DimType, to_string) {
    constexpr plssvm::detail::dim_type dim{ 64ull, 32ull, 16ull };

    // convert it to a string
    const std::string str = fmt::format("{}", dim);

    // check the string for correctness
    EXPECT_EQ(str, std::string{ "[64, 32, 16]" });
}

//*************************************************************************************************************************************//
//                                                           execution_range                                                           //
//*************************************************************************************************************************************//

TEST(ExecutionRange, construct_single_grid) {
    // create execution range
    const plssvm::detail::execution_range exec{ plssvm::detail::dim_type{ 16ull, 16ull, 2ull }, 1024ull, plssvm::detail::dim_type{ 64ull, 64ull, 64ull }, plssvm::detail::dim_type{ 1024ull, 1024ull, 1024ull } };

    // check the block size
    EXPECT_EQ(exec.block, (plssvm::detail::dim_type{ 16ull, 16ull, 2ull }));

    // check the grids
    EXPECT_EQ(exec.grids.size(), 1);
    EXPECT_EQ(exec.grids.front().first, (plssvm::detail::dim_type{ 64ull, 64ull, 64ull }));
    EXPECT_EQ(exec.grids.front().second, (plssvm::detail::dim_type{ 0ull, 0ull, 0ull }));
}

TEST(ExecutionRange, construct_multiple_grids) {
    // create execution range
    const plssvm::detail::execution_range exec{ plssvm::detail::dim_type{ 16ull, 16ull, 4ull }, 1024ull, plssvm::detail::dim_type{ 128ull, 127ull, 126ull }, plssvm::detail::dim_type{ 64ull, 64ull, 64ull } };

    // check the block size
    EXPECT_EQ(exec.block, (plssvm::detail::dim_type{ 16ull, 16ull, 4ull }));

    // check the grids
    EXPECT_EQ(exec.grids.size(), 8);
    EXPECT_EQ(exec.grids[0].first, (plssvm::detail::dim_type{ 64ull, 64ull, 64ull }));
    EXPECT_EQ(exec.grids[0].second, (plssvm::detail::dim_type{ 0ull, 0ull, 0ull }));
    EXPECT_EQ(exec.grids[1].first, (plssvm::detail::dim_type{ 64ull, 64ull, 62ull }));
    EXPECT_EQ(exec.grids[1].second, (plssvm::detail::dim_type{ 0ull, 0ull, 64ull }));

    EXPECT_EQ(exec.grids[2].first, (plssvm::detail::dim_type{ 64ull, 63ull, 64ull }));
    EXPECT_EQ(exec.grids[2].second, (plssvm::detail::dim_type{ 0ull, 64ull, 0ull }));
    EXPECT_EQ(exec.grids[3].first, (plssvm::detail::dim_type{ 64ull, 63ull, 62ull }));
    EXPECT_EQ(exec.grids[3].second, (plssvm::detail::dim_type{ 0ull, 64ull, 64ull }));

    EXPECT_EQ(exec.grids[4].first, (plssvm::detail::dim_type{ 64ull, 64ull, 64ull }));
    EXPECT_EQ(exec.grids[4].second, (plssvm::detail::dim_type{ 64ull, 0ull, 0ull }));
    EXPECT_EQ(exec.grids[5].first, (plssvm::detail::dim_type{ 64ull, 64ull, 62ull }));
    EXPECT_EQ(exec.grids[5].second, (plssvm::detail::dim_type{ 64ull, 0ull, 64ull }));

    EXPECT_EQ(exec.grids[6].first, (plssvm::detail::dim_type{ 64ull, 63ull, 64ull }));
    EXPECT_EQ(exec.grids[6].second, (plssvm::detail::dim_type{ 64ull, 64ull, 0ull }));
    EXPECT_EQ(exec.grids[7].first, (plssvm::detail::dim_type{ 64ull, 63ull, 62ull }));
    EXPECT_EQ(exec.grids[7].second, (plssvm::detail::dim_type{ 64ull, 64ull, 64ull }));
}

TEST(ExecutionRange, construct_block_zero_threads) {
    // at least one thread must be present!
    EXPECT_THROW_WHAT((plssvm::detail::execution_range{ plssvm::detail::dim_type{ 0ull, 0ull, 0ull }, 16ull, plssvm::detail::dim_type{ 64ull, 64ull, 64ull }, plssvm::detail::dim_type{ 1024ull, 1024ull, 1024ull } }),
                      plssvm::kernel_launch_resources,
                      "At least one thread must be given per block! Maybe one dimension is zero?");
}

TEST(ExecutionRange, construct_block_zero_threads_in_single_dimension) {
    // EACH dimension must at least consist of a single thread!
    EXPECT_THROW_WHAT((plssvm::detail::execution_range{ plssvm::detail::dim_type{ 4ull, 4ull, 0ull }, 16ull, plssvm::detail::dim_type{ 64ull, 64ull, 64ull }, plssvm::detail::dim_type{ 1024ull, 1024ull, 1024ull } }),
                      plssvm::kernel_launch_resources,
                      "At least one thread must be given per block! Maybe one dimension is zero?");
}

TEST(ExecutionRange, construct_block_too_many_threads) {
    // the product of the block dimensions may not exceed to total number of threads allowed in a block
    EXPECT_THROW_WHAT((plssvm::detail::execution_range{ plssvm::detail::dim_type{ 16ull, 16ull, 4ull }, 16ull, plssvm::detail::dim_type{ 64ull, 64ull, 64ull }, plssvm::detail::dim_type{ 1024ull, 1024ull, 1024ull } }),
                      plssvm::kernel_launch_resources,
                      "Not enough work-items allowed for a work-groups of size 16x16x4 (#threads: 1024; max allowed: 16)! Try reducing THREAD_BLOCK_SIZE.");
}

TEST(ExecutionRange, swap_member_function) {
    plssvm::detail::execution_range exec1{ plssvm::detail::dim_type{ 16ull, 16ull, 4ull }, 1024ull, plssvm::detail::dim_type{ 64ull, 64ull, 64ull }, plssvm::detail::dim_type{ 1024ull, 1024ull, 1024ull } };
    plssvm::detail::execution_range exec2{ plssvm::detail::dim_type{ 4ull, 4ull, 4ull }, 1024ull, plssvm::detail::dim_type{ 128ull, 128ull, 128ull }, plssvm::detail::dim_type{ 64ull, 64ull, 64ull } };

    // swap the contents
    exec1.swap(exec2);

    // check the contents
    EXPECT_EQ(exec1.block, (plssvm::detail::dim_type{ 4ull, 4ull, 4ull }));
    EXPECT_EQ(exec1.grids.size(), 8);

    EXPECT_EQ(exec2.block, (plssvm::detail::dim_type{ 16ull, 16ull, 4ull }));
    EXPECT_EQ(exec2.grids.size(), 1);
    EXPECT_EQ(exec2.grids.front().first, (plssvm::detail::dim_type{ 64ull, 64ull, 64ull }));
}

TEST(ExecutionRange, swap_free_function) {
    plssvm::detail::execution_range exec1{ plssvm::detail::dim_type{ 16ull, 16ull, 4ull }, 1024ull, plssvm::detail::dim_type{ 64ull, 64ull, 64ull }, plssvm::detail::dim_type{ 1024ull, 1024ull, 1024ull } };
    plssvm::detail::execution_range exec2{ plssvm::detail::dim_type{ 4ull, 4ull, 4ull }, 1024ull, plssvm::detail::dim_type{ 128ull, 128ull, 128ull }, plssvm::detail::dim_type{ 64ull, 64ull, 64ull } };

    // swap the contents
    using plssvm::detail::swap;
    swap(exec1, exec2);

    // check the contents
    EXPECT_EQ(exec1.block, (plssvm::detail::dim_type{ 4ull, 4ull, 4ull }));
    EXPECT_EQ(exec1.grids.size(), 8);

    EXPECT_EQ(exec2.block, (plssvm::detail::dim_type{ 16ull, 16ull, 4ull }));
    EXPECT_EQ(exec2.grids.size(), 1);
    EXPECT_EQ(exec2.grids.front().first, (plssvm::detail::dim_type{ 64ull, 64ull, 64ull }));
}

TEST(ExecutionRange, equality) {
    // create execution ranges
    const plssvm::detail::execution_range exec1{ plssvm::detail::dim_type{ 16ull, 16ull }, 1024ull, plssvm::detail::dim_type{ 64ull, 64ull }, plssvm::detail::dim_type{ 1024ull, 1024ull } };
    const plssvm::detail::execution_range exec2{ plssvm::detail::dim_type{ 16ull, 16ull }, 1024ull, plssvm::detail::dim_type{ 64ull, 64ull }, plssvm::detail::dim_type{ 64ull, 64ull } };
    const plssvm::detail::execution_range exec3{ plssvm::detail::dim_type{ 32ull, 32ull }, 1024ull, plssvm::detail::dim_type{ 128ull, 128ull }, plssvm::detail::dim_type{ 64ull, 64ull } };
    const plssvm::detail::execution_range exec4{ plssvm::detail::dim_type{ 16ull, 16ull }, 1024ull, plssvm::detail::dim_type{ 128ull, 128ull }, plssvm::detail::dim_type{ 64ull, 64ull } };

    // check for equality
    EXPECT_TRUE(exec1 == exec1);
    EXPECT_TRUE(exec1 == exec2);
    EXPECT_FALSE(exec1 == exec3);
    EXPECT_FALSE(exec1 == exec4);
    EXPECT_FALSE(exec3 == exec4);
    EXPECT_TRUE(exec4 == exec4);
}

TEST(ExecutionRange, inequality) {
    // create execution ranges
    const plssvm::detail::execution_range exec1{ plssvm::detail::dim_type{ 16ull, 16ull }, 1024ull, plssvm::detail::dim_type{ 64ull, 64ull }, plssvm::detail::dim_type{ 1024ull, 1024ull } };
    const plssvm::detail::execution_range exec2{ plssvm::detail::dim_type{ 16ull, 16ull }, 1024ull, plssvm::detail::dim_type{ 64ull, 64ull }, plssvm::detail::dim_type{ 64ull, 64ull } };
    const plssvm::detail::execution_range exec3{ plssvm::detail::dim_type{ 32ull, 32ull }, 1024ull, plssvm::detail::dim_type{ 128ull, 128ull }, plssvm::detail::dim_type{ 64ull, 64ull } };
    const plssvm::detail::execution_range exec4{ plssvm::detail::dim_type{ 16ull, 16ull }, 1024ull, plssvm::detail::dim_type{ 128ull, 128ull }, plssvm::detail::dim_type{ 64ull, 64ull } };

    // check for inequality
    EXPECT_FALSE(exec1 != exec1);
    EXPECT_FALSE(exec1 != exec2);
    EXPECT_TRUE(exec1 != exec3);
    EXPECT_TRUE(exec1 != exec4);
    EXPECT_TRUE(exec3 != exec4);
    EXPECT_FALSE(exec4 != exec4);
}

TEST(ExecutionRange, to_string_single_grid) {
    const plssvm::detail::execution_range exec{ plssvm::detail::dim_type{ 16ull, 16ull }, 1024ull, plssvm::detail::dim_type{ 64ull, 64ull }, plssvm::detail::dim_type{ 1024ull, 1024ull } };

    // convert it to a string
    const std::string str = fmt::format("{}", exec);

    // check the string for correctness
    EXPECT_EQ(str, std::string{ "grid: [64, 64, 1]; block: [16, 16, 1]" });
}

TEST(ExecutionRange, to_string_multiple_grids) {
    const plssvm::detail::execution_range exec{ plssvm::detail::dim_type{ 32ull, 32ull }, 1024ull, plssvm::detail::dim_type{ 128ull, 128ull }, plssvm::detail::dim_type{ 64ull, 64ull } };

    // convert it to a string
    const std::string str = fmt::format("{}", exec);

    // check the string for correctness
    EXPECT_EQ(str, std::string{ "grids: [[64, 64, 1], [64, 64, 1], [64, 64, 1], [64, 64, 1]]; block: [32, 32, 1]" });
}
